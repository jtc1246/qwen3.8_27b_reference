# 昇腾 310P 硬件特性与本项目的设计约束

Atlas 300I Duo (双 310P3) 上与本项目直接相关的硬件数字, 以及它们如何决定 NPU 实现 ([npu_impl.md](npu_impl.md)) 的各项选择。数字均为实测 (ECC 已关, 时钟按 1.08GHz 折算); 完整的硬件特性记录在本机 ascend_usage 仓库 `docs/310P-PERFORMANCE.md`。

## 1. 关键数字

| 项目 | 数值 |
|---|---|
| 显存带宽 (单颗, 流式读) | 204 GB/s (标称 204.8; 开 ECC 只有 ~182) |
| 片上共享缓存 | 容量 ≥16MB, 重复读 **~1010 GB/s** (>64MB 回落 204) |
| cube 算力 | fp16 理论 70.8 TFLOPS (实测含流水 25-35T); 还支持 int8/int4, **无 fp32/bf16** |
| vector 算力 | fp32 乘加折算仅 **~0.55 TFLOPS** (与 cube 差 128 倍) |
| 可用显存 | 单颗 ~46.2GB (48GB 减去设备侧预留; 关 ECC 比开多 3GB) |
| kernel 启动 | ~5 µs/次 (连续提交) |
| 向量指令发射 | ~14 ns/调用; 同 V 管线依赖硬件保序 (零停顿) |

## 2. 这些数字如何决定设计

### decode 上限 = 权重字节 / 204GB/s

decode 每 token 把全部 matmul 权重流过一遍, 共享缓存帮不上忙 (只有一遍)。所以 decode 的一切优化最终都是在**减少字节**或**贴近 204**:
当前重打包后 ~21.3GB → 理论上限 9.6 tok/s; 若换 4-bit 量化 (~14GB) → 上限 ~14 tok/s; 双芯片张量并行每颗减半。

### 反量化的向量预算: 每权重字节约 10 个"向量字节"

带宽摊到每核每周期 ~24B, 而向量单元每周期只处理 256B。反量化链超过约 4-5 条指令/字节, 瓶颈就从带宽变成向量:
- I8 系 (cast + scale 乘) 预算够 → 实测 195 GB/s, 基本贴满
- I4 (nibble 乘法解码 + scale/bias 乘加) 超预算 → 实测只有 95 GB/s
- 这也是为什么 q5_k/q6_k 宁可在 host 展开成 int8 (字节 +37%) 也不在片上解位打包: 310P **没有整数移位指令**, 位解包链太长, 省字节是负收益

### 一切矩阵计算必须上 cube, 且只能走 fp16 反量化

vector fp32 只有 0.55T, 是 310P 版图里的"配角"; 真正算力全在 cube, 而 cube 不支持 fp32。于是:
- matmul: 量化权重片上反量化成 fp16 → mmad (fp32 累加), 即 k_gemm
- cube 直接吃 int8/int4 走不通: mmad 的 fractal k 粒度 (int8 为 32, int4 为 64) 与 k-quant 每 16/32 元素的 scale 组对不齐, 沿 k 累加时无法按组乘 scale
- attention / DeltaNet 目前还是 fp32 向量 (prefill 的主要瓶颈), 被 0.55T 封顶, 后续也要 fp16 化上 cube

### mmad 的两个硬约束 (k_gemm 的布局依据)

- 布局: L0A zZ `[M1][K1][16m][16k]`, L0B `[K1][N1][16n][16k]` (存 [16k][16n] 时用 LoadData 转置), L0C `[N1][M1][16m][16n]`; CO1→UB 拷贝必须显式 `BLOCK_MODE_MATRIX` (默认**静默不拷贝**)
- **m=1 是 gemv 模式**: A 按 k 连续读, 但按 512B 粒度访问, A 片偏移必须 512B 对齐且 L0A 尾部留余量 (否则 507015) — decode 与 prefill 得以共用一个 kernel 的前提

### 广播不可用 → scale 靠布局技巧

Brcb (f16 每块 9.6 周期)、Gather (24 周期/rep) 都慢到不可用, 所以反量化的 per-组 scale 不能展开广播。解法是重打包时把 fractal 存成 [16k][16r] k 主序, 让 scale 恰好按 32B 块变化, 用二元指令 `src1BlkStride=0` 零成本逐 fractal 乘 — 这是 repack 布局长这样的原因。

### 共享缓存 ~1TB/s → prefill 的激进 tiling

prefill 时 x (fp16 ~2.6MB) 被各核各块反复读, 全部命中共享缓存、几乎免费。因此 k_gemm 不需要为 A 的复用在 L1 做大块驻留, 直接 L0C 常驻 [m≤256][n=256] 累加整个 k 维, 权重和 x 都纯流式。

### 启动开销可忽略但次数要有数

decode 每 token ~800 次 kernel 启动 × 5µs ≈ 4ms (占 ~2%); 不构成当前瓶颈, 但 kernel 进一步拆小前要算这笔账。

### KV cache fp32 的代价

每 token 128KB, ctx 4096 共 512MB; 长上下文时 attention 读 KV 的量随 ctx 线性涨, 且 fp32 是 fp16 的两倍流量 — KV 转 fp16 (配合 attention 上 cube) 能同时省显存和带宽。

## 3. 运维相关

- DDR ECC 已关闭 (`npu-smi set -t ecc-enable -i <id> -d 0`, 重启生效): 带宽 +10%, 每颗容量 +3GB; 这是实测 204/200 GB/s 的前提
- 两颗 310P3 显存与带宽完全独立, 双芯并行时互不干扰 (已实测)
