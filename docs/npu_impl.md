# C++ NPU (华为昇腾 310P) 实现说明

`qwen_npu/` 是文本主干在华为 Atlas 300I Duo (Ascend 310P3) 上的实现, 结构对应 [cpu_impl.md](cpu_impl.md) / [cuda_impl.md](cuda_impl.md): 模型常量、权重绑定、前向流程、采样和 C 接口与 CPU / CUDA 版一一对应。

当前状态 (2026-11-15): **v2, matmul 全部走 cube (fp16 反量化 + mmad), 其余计算 fp32 向量**。单芯片、单序列; decode 约 5.1 tok/s, prefill 约 180-210 tok/s (见第 8 节, v1 分别为 1.8 / 1.9)。与 CPU 版贪心逐 token 一致 (fp16 计算, 逐层误差 ~1e-3 级, 阈值 5e-3)。python 侧 (`qwen.py` / `server.py`) 尚未接入该后端, 目前通过 ctypes 直接加载测试。

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `kernels_npu.cpp` | 全部 Ascend C kernel (device 侧) |
| `qwen.cpp` | 权重重打包与上传、前向 (`eval_chunk`)、采样、自测、调试接口、C 接口 |
| `acl.h` | AscendCL 运行时薄封装 (初始化、`ck` 检查、分配、拷贝、同步) |
| `repack.h` | GGUF 权重 → NPU 自定义格式的重打包 (host 侧, 见第 2 节) |
| `gguf.h` / `quants_ref.h` | 与 `qwen_cuda/` 相同 (只改了 include guard) |

构建 (与 CPU / CUDA 版不同, **不用 musl**, 因为必须链接 CANN 的 glibc 动态库):

```bash
export CANN_HOME=/path/to/ascend-toolkit/latest  # 包含 set_env.sh 的目录
./qwen_npu/compile.sh
```

- 需要: CANN toolkit + 与设备匹配的 ops 包、系统 g++ (C++17)、cmake; `compile.sh` 里的 cmake 路径按机器情况通过 `CMAKE_BIN` 覆盖
- kernel 通过 CANN 的 `ascendc.cmake` 框架编译 (SOC `Ascend310P3`, 核函数直调模式), host 通过生成的 `aclrtlaunch_*.h` 启动 kernel
- 产物 `cmake_build/libqwen_npu.so`, 带 RPATH 指向 CANN lib64 和驱动目录; **运行前仍需 `source $CANN_HOME/set_env.sh`** (CANN 二级依赖和环境变量)
- 链接时需要 `-Wl,-rpath-link` 指向 CANN lib64 与驱动目录, 否则 ld 解析 so 的二级依赖 (libascend_hal.so、libruntime.so 的符号) 会失败, CMakeLists 已包含

C 接口与 CPU 版相同 (`qwen_init(path, device, max_ctx)` 第二个参数是 NPU 编号), 直接 ctypes 加载即可:

```python
lib = ctypes.CDLL("qwen_npu/cmake_build/libqwen_npu.so")
lib.qwen_init(b"/path/to/Qwen3.8-27B-Q4_K_M.gguf", 0, 4096)
```

## 2. 权重重打包

matmul 权重 (除 alpha/beta) 全部重打包成 **cube GEMM 的 fractal 量化布局** (`repack.h` 的 `repack2_block`), device 侧向量单元反量化成 fp16 后喂给 cube。量化值用原始整数存, scale/bias 转成 fp16 (fp16 乘加与 CUDA fp16 版同级误差, 不再逐位等于 CPU 的 fp32 反量化)。

| 格式 | 来源 | 字节/权重 | 内容 |
|---|---|---|---|
| I8F | q8_0, q6_k | 1.125 | int8 + 每 16 元素 fp16 scale |
| I8BF | q5_k | 1.25 | int8 + per-16 fp16 scale + bias |
| I4F | q4_k | 0.75 | nibble + per-32 (复制成 per-16) fp16 scale + bias |
| F32 | f32 (alpha / beta) | 4 | 自然行布局 (向量 GEMV 用) |

fractal 布局 (对应 `k_gemm` 的流水):

- 行按 8 核均分 (nr = n/8, 要求 n 为 128 的倍数), 核内按 256 行分块, 块内按 128-k 切片
- 每切片 = `[8 ki][NI ni]` 个 512B fractal + scale 区 `[ki][ni][16 half]` (+bias 区同样); fractal 内为 **[16k][16r] k 主序** — 这样 scale 恰好按 32B 块变化, kernel 里用 `src1BlkStride=0` 的块复用技巧逐 fractal 乘 scale, 完全不需要广播展开 (310P 的 Brcb/Gather 都慢到不可用); L1→L0B 用 `LoadData` 的 `ifTranspose` 转回标准 [16n][16k]
- I4F 切片按半片组织 (半片 = 4 个 ki), 字节 i = 半片前 2 个 ki 的元素 i (低 nibble) | 后 2 个 ki 的元素 i (高 nibble) << 4
- `token_embd.weight` 仍用 v1 的 wave 布局 (旧 I4 格式, f32 scale, 反量化逐位精确), 只给 `k_dq_*` 做 embedding 查行
- 总显存约 23.2GB (v1 为 25.0), 重打包 + 上传约 21s

norm、ssm_a 等小 f32 向量原样上传; `ssm_conv1d.weight` 重排成 `[4][QKV_DIM]` 平面; conv 状态布局 `[3][QKV_DIM]`。

## 3. Kernel

全部在 `kernels_npu.cpp`, block_dim 8 (k_gemv_f32t 为 n/8), 激活全部是 GM 上的 fp32。

| kernel | 说明 |
|---|---|
| `k_gemm_{i8,i8b,i4}` | 核心 matmul, **decode (m=1) 和 prefill (m≤256) 同一个 kernel**。流水: MTE2 (量化切片) → 向量反量化 fp16 (cast + scale 块复用乘, I4F 先做 nibble 乘法解码) → MTE3 (L1) → MTE1 (转置加载 L0B) → mmad (fp16, fp32 累加, L0C 常驻累加整个 k) → 输出。m=1 时 x 在 kernel 内转 fp16 常驻 L0A (mmad gemv 模式); m>1 时 x 由 `k_pack_x` 预打包, 逐切片进 L1/L0A。事件全部用 TPipe `AllocEventID` 分配 |
| `k_pack_x` | f32 [T][k] → fp16 zZ 切片布局 [kc][mi][ki][16m][16k], 三级双缓冲流水 |
| `k_gemv_f32t` | alpha/beta (f32 自然行布局) 的 GEMV, 行常驻 UB (每核 8 行), T 个 token 在 kernel 内循环 |
| `k_dq_{f32,i8,i8b,i4}` | 按行号表反量化整行到自然顺序 f32 (embedding 和自测用, 旧 wave 布局) |
| `k_add_norm` | x += add (可空) + RMSNorm, 保存最后一个 token 的 hidden; token 分核轮转 |
| `k_silu_mul` | g = silu(g)·u, 按核均分逐元素 |
| `k_conv` | 因果 conv1d (核宽 4) + SiLU; 通道均分到核, 状态 3 个平面在 UB 轮转 |
| `k_deltanet` | 每核 6 个 v 头顺序处理, 状态 S[128][128] 常驻 UB (与 v1 相同, 未重构) |
| `k_qk_rope` | (t, 头) 单元分核; RMSNorm + 交错 MRoPE + 写 KV cache; cos/sin 由 host 算好上传 |
| `k_attn` | **两遍扫描版**: 单元 = (16-query 块, q头), K/V 批 (48 key) 在块内所有 query 间复用; 第一遍只读 K 求每 query 全局 max, 第二遍直接 exp 累加 (无在线重缩放); softmax 全向量化 (max/sum 按 [16][8] 块跨步存放 + 块广播), 热循环无任何标量数据读取; 因果/尾批用 score 置 -3e38 屏蔽 |

## 4. 310P 的主要限制与踩过的坑

v1 实测的坑 (向量部分) 依然成立:

1. 向量单元**没有整数移位指令** (`ShiftLeft/Right` 静默输出垃圾) → nibble 用乘法 + RINT 解码
2. Cast 组合不全, 不支持的组合静默输出垃圾; `Reciprocal`/`Rsqrt` 低精度, norm 用 `Sqrt`+`Div`
3. 同 kernel 内循环, 下一轮 DataCopy (MTE2) 不等上一轮向量读取, 覆盖前必须 `PipeBarrier<PIPE_ALL>` 或 TQue
4. 未初始化 UB 可能是 NaN 位型, 固定 repeat 会扫到的 padding 区要清零
5. 标量 `GetValue`/`SetValue` 走 S 管线, 与向量之间要 `PipeBarrier<PIPE_ALL>`
6. Level-2 API 每次重设 mask; 热路径用 Level-0 + `isSetMask=false`

v2 (cube 化) 新增的实测结论:

7. **mmad / 数据通路** (dav_m200): `Mmad` 支持 fp16×fp16→f32; 布局 L0A zZ `[M1][K1][16m][16k]`、L0B `[K1][N1][16n][16k]`、L0C `[N1][M1][16m][16n]`; **m=1 时 mmad 进入 gemv 模式, A 按 k 连续读取且按 512B fractal 粒度访问** (A 片偏移必须 512B 对齐, L0A 末尾要留余量, 否则越界 507015); UB→L1 (`copy_ubuf_to_cbuf`)、L1→L0A/B (`LoadData`, 可转置)、CO1→UB 都可用, 但 CO1→UB 的 DataCopy 必须显式 `BlockMode::BLOCK_MODE_MATRIX` (默认 blockMode 走不到任何分支, **静默不拷贝**), blockLen 单位是 fractal
8. **向量速率** (实测 ns/repeat): Mul/Add f32 和 f16 都是 ~2 周期/repeat (f32 32 元素/周期, f16 64), Cast i8→h ~1.2, BlockReduceSum ~1; **`MulAddDst(f32,h,h)` 5 周期/repeat (很慢)**; **`Brcb` 慢到不可用** (f32 ~3.6 周期/块, f16 ~9.6), **`Gather` 同样慢** (~24 周期/repeat) → 任何按块广播都要用布局技巧 (`src1BlkStride=0`) 而不是展开
9. **同 V 管线内依赖硬件保序** (微基准证实依赖链与独立指令同速), 热循环里的 `PipeBarrier<PIPE_V>` 不需要; 指令发射开销 ~14ns/调用
10. **片上共享缓存 ~16MB, 读带宽 ~1TB/s** (错相反复读 ≤16MB 区域实测 1010GB/s, ≥64MB 回落 ~195GB/s) → prefill 时 x/A 的反复读取几乎免费, k_gemm 用 L0C 常驻 [m][256] 的激进 tiling
11. 手写跨管线事件的 id 必须用 TPipe `AllocEventID` 分配, 自己写死 0/1 会与框架内部冲突 (出现形状相关的"随机"错数)
12. kernel 启动开销 ~5µs/次 (同 kernel 连发或交替都一样), 一个 decode token ~800 次启动 ≈ 4ms

## 5. 显存

- 权重约 21.3GB (第 2 节) + tok_embd 0.95GB; KV cache fp32, 每 token 128KB (ctx 4096 时 512MB); DeltaNet 状态 201MB; 激活按 CHUNK = 256; x 打包缓冲 8.9MB
- 实测 ctx = 4096 时总占用 23.15GB (单芯片可用约 46GB, ECC 已关)

## 6. 调试接口 (正式功能不依赖, 但保留)

- 环境变量 `QWEN_NPU_STOP=<层号>` / `QWEN_NPU_PHASE=<0/1/2/10/11/12>`: 跑完该层 (或该层内某阶段) 直接返回
- `QWEN_NPU_PROF=1` + `qwen_prof_dump()`: 逐 kernel 同步计时 (会拖慢, 小 kernel 数字被同步开销抬高)
- `qwen_dbg(name, out, n)`: 把中间激活缓冲拷回 host
- v1 的 `k_attn` dump 模式 (T=-1) 已随两遍扫描重写移除

## 7. 验证

测试脚本在仓库外的实验目录 (`tests/npu_port`、`tests/npu_attn`、`tests/npu_api`、`tests/npu_gemm`、`tests/npu_vlat`、`tests/npu_cube`、`tests/npu_l2`, 依赖本机绝对路径, 未入库):

- `qwen_selftest`: tok_embd 反量化 vs 标量参考逐位一致; 各 matmul 权重 (含 q4_k/q5_k/q6_k/q8_0/f32) m=1 GEMM vs 保存行的 f32 反量化 double 点积, 相对误差 ~1e-5 (阈值 5e-3)
- 逐层对比 (`vs_cpu1`): 65 层相对误差 ~1e-3 级 (fp16 计算, 阈值 5e-3), 无 NaN
- 300 token prefill (跨 chunk) + 20 步贪心解码: **逐 token 与 CPU 相同**; logits 最大绝对差 ~0.16 (CPU 最大值 10.5)
- `tests/npu_gemm`: k_gemm 独立测试台 (三种格式 × m=1/5/16/33/100/256 × 各形状), 与 host 模拟 fp16 反量化的参考对比
- `tests/npu_vlat`: 向量指令速率/发射开销/启动开销微基准 (第 4 节第 8/9/12 条的出处)
- `tests/npu_cube`: mmad fractal 布局验证 (第 4 节第 7 条的出处)

## 8. 性能 (v2, Atlas 300I Duo 单颗 310P3, ECC 关)

| 项目 | v2 | v1 |
|---|---|---|
| prefill (第一个 256 chunk) | ~230 tok/s | 1.9 tok/s |
| prefill (1024 token) | 178 tok/s | - |
| decode (ctx ~300) | 5.2 tok/s | 1.8 tok/s |
| decode (ctx ~1024 / ~3100) | 5.1 / 4.8 tok/s | - |
| 模型加载 (重打包 + 上传) | 21s | 14s |

分解 (QWEN_NPU_PROF, 数字含同步开销):

- decode ~192ms/token: 基本全是 k_gemm 的权重 DMA。I8F/I8BF 独立压测有效带宽 ~195GB/s (纯读上限 204), I4F 受向量反量化限制只有 ~95GB/s (nibble 解码本身 191GB/s, scale/bias 的逐元素乘加把向量预算打爆); 混合后整体 ~145GB/s, 加上 lm_head 1.43GB 和 ~800 次启动
- prefill 256-chunk ~1.35s: deltanet 430ms + attn 372ms + gemm 337ms (独立压测 25-35 TFLOPS, fp16 峰值 70T) + alpha/beta 92ms + pack_x 76ms + 小 kernel ~90ms。attn/deltanet 是 fp32 向量计算, 向量峰值只有 0.55 TFLOPS, 已接近其理论下限 → 下一步必须上 cube

## 9. 已知不足与下一步

1. **attention 上 cube** (最大的 prefill 瓶颈, 长上下文时也拖 decode): Q·K^T 和 P·V 都改成 fp16 mmad (flash 风格 tile), KV cache 顺势改 fp16 (显存减半); 向量只做 softmax。预计 attn 从 ~23ms/层降到 ~2ms/层
2. **DeltaNet 重构** (第二大 prefill 瓶颈): 至少把 beta/decay/softplus/范数的逐 token 小向量操作在 chunk 内批量预计算, S 递推做双缓冲预取; 彻底方案是 chunk 化 (WY 表示) 上 cube
3. **I4F 反量化向量瓶颈**: And 平面解码 (int16 视图 4 平面 + scale 预除) 约 -14% 向量量; 或者干脆 q4_k 也存成 I8BF (字节 +0.5/权重但 DMA-bound, decode 反而省 ~8ms, 代价显存 +2.8GB)
4. 小 kernel 吞吐 (conv/silu/add_norm/f32t 的串行 barrier 循环)、alpha/beta 批量化
5. 双芯片张量并行 (每颗一半权重, 两倍带宽)、KV cache 量化 (fp16→q8_0, 对应 CUDA 版演进)、与 python 服务接入
6. 自定义采样 (wasm) 未移植; 图片按计划固定走 CPU / CUDA, 不上 NPU
