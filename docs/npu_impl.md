# C++ NPU (华为昇腾 310P) 实现说明

`qwen_npu/` 是文本主干在华为 Atlas 300I Duo (Ascend 310P3) 上的实现, 结构对应 [cpu_impl.md](cpu_impl.md) / [cuda_impl.md](cuda_impl.md): 模型常量、权重绑定、前向流程、采样和 C 接口与 CPU / CUDA 版一一对应。

当前状态 (2026-11-15): **v3, 权重源从 GGUF 换成 RedHatAI/Qwen3.8-27B-INT4 的 `model.safetensors`** (compressed-tensors, int4 对称 group128, AWQ 平滑 + GPTQ; 选型调研见该模型卡与本文第 2 节), matmul 全部走 cube (fp16 反量化 + mmad), 其余计算 fp32 向量。单芯片、单序列, 不含投机解码 (MTP 文件不加载)。**v4: decode 改走 W4A8** — 主干 matmul 在 decode 时用 u8×u8 整数 mmad (x 由 `k_qx8` 动态量化 int8, 权重 int4 原位喂 cube), 反量化向量瓶颈消除; prefill 仍走 fp16 反量化 + mmad (从同一份权重布局五平面拆解)。decode 约 10.6 tok/s (v3 为 6.7), prefill 约 150 tok/s (v3 为 178, w8p 流水未优化); 权重显存 17.5GB (v3 为 19.2)。精度: decode KL vs CPU fp32 = 2.54e-2 / top1 30/32, 与 v3 的 fp16 激活路径完全同级 (A8 增量被 W4 噪声淹没)。与 CPU 版 (GGUF Q4_K_M, 另一种 4bit 量化) 相比 logits 差在量化噪声级 (~1-3, 最大值 ~10-17), 300 tok prefill 后贪心 19/20 一致。python 侧 (`qwen.py` / `server.py`) 尚未接入该后端, 目前通过 ctypes 直接加载测试。

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `kernels_npu.cpp` | 全部 Ascend C kernel (device 侧) |
| `qwen.cpp` | 权重加载 (含 HF→本实现的变换)、重打包与上传、前向 (`eval_chunk`)、采样、自测、调试接口、C 接口 |
| `acl.h` | AscendCL 运行时薄封装 (初始化、`ck` 检查、分配、拷贝、同步) |
| `st.h` | safetensors 加载 (mmap + 手写 JSON 头解析) |
| `repack.h` | checkpoint 权重 → NPU 自定义格式的重打包 (host 侧, 见第 2 节) |

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
lib.qwen_init(b"/path/to/Qwen3.8-27B-INT4/model.safetensors", 0, 4096)
```

## 2. 权重来源与重打包

权重源是 [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) 的 `model.safetensors` (18GB): 主干 linear 全部为 compressed-tensors `pack-quantized` int4 **对称** group128 (`weight_packed` int32 每词 8 个 nibble, nibble = q+8; `weight_scale` bf16 [n][k/128]; 无 zero_point 无 g_idx, `actorder: static` 只是求解期重排, 列序不变), embed/lm_head/norm/conv/α/β 为 bf16。选它的原因: 对称无 bias 项 + g128 的 scale 元数据少, 是对本 kernel 最友好的 4bit 格式; AWQ 平滑 + GPTQ 且模型卡有 6 基准恢复率 98.5~101%。

**HF → 本实现的权重变换** (与 GGUF 转换语义一致, 见 [model.md](model.md) 第 11 节, 在 host 加载时完成):
norm 权重 +1 (`linear_attn.norm` 除外)、`ssm_a = -exp(A_log)`、DeltaNet v 头 grouped→tiled 重排 (tiled 头 j ← HF 头 `(j%16)*3 + j/16`, 涉及 in_proj_qkv 的 v 行 / in_proj_z 行 / α β 行 / A_log / dt_bias / conv v 通道 / out_proj 列)。行重排对量化表示无损; out_proj 的列重排以 128 为块, 恰好与量化组对齐, 也无损。

matmul 权重重打包成 **cube GEMM 的 fractal 量化布局** (`repack.h`), device 侧向量单元反量化成 fp16 后喂给 cube; scale 转 fp16 (fp16 乘与 CUDA fp16 版同级误差):

| 格式 | 来源 | 字节/权重 | 内容 |
|---|---|---|---|
| I4W8 | 主干 int4 | 0.547 | W4A8 布局: 包字节 + per-128 fp16 scale + f32 W2 修正量 (见下) |
| I8F | lm_head (bf16 经 host per-16 对称 RTN) | 1.125 | int8 + 每 16 元素 fp16 scale (fp16 反量化路径) |
| F32 | α / β (bf16 转 f32) | 4 | 自然行布局 (向量 GEMV 用) |

**I4W8 布局** (每核连续; 每 256 行块: 每 sp = 组对 (2u, 2u+1): `[4t][NI][512B 包字节]` + meta `[s fp16 NI·32B ×2组][W2 f32 NI·64B ×2组]`):

- 包字节 (fractal 内 `32r+2n+p`) = `q(组2u, k=t·32+p·16+r)+8` 的低 nibble | 组 2u+1 同 k 的高 nibble。这个顺序是按 **b8 LoadData 转置的硬件语义**(按 u16 做 16×16 转置, u16 两字节分别落到 k 和 k+16) 反推的: `And 0x0F0F`/`0xF0F0` 各一条就得到两组的转置源 fractal (hi 组值天然 ×16), 转置加载后即标准 `[16n][32k]` b8 fractal — **decode 的全部向量解码 = 2 条 And**
- decode (`k_gemm_w8d`): u8(x, 零点128)×u8(w, nib) mmad (dav_m200 仅支持 (i8,i8)/(u8,i8)/(u8,u8) 组合), 每 128-k 组内 s32 **精确**累加, drain 后 f32 域乘 scale 并做两个零点修正: `y=Σ s'·sx·D + s·(−8·Sx·sx) + W2·(−128·sx)`, W2 = h2f(存储 s)·Σ(存储 nib) 用 **f32** 存 (fp16 存会把大数相消误差放大到 0.7%); hi 组的 /16 在 f32 域派生 (fp16 存 s/16 会掉入次正规被向量单元 flush, 曾导致 decode 发散); 次正规 scale 在 repack 时连同 W2 一起钳 0。drain 滞后 2 组软件流水 (drain 的 WaitFlag 会挡 V 队列)
- x 侧 (`k_qx8`): per-token per-128 组对称 int8 (存 u8), f32 域 RINT 与 host lrintf 逐字节一致; 输出 4 份相位旋转副本 (b8 gemv 的 A 偏移粒度是 512B, 组 g 的 128B 放在副本 g%4 的 512·(g/4) 处) + 组 scale/修正量 ([KS][8] 复制 8 份供 src1BlkStride=0 当标量乘)
- prefill (`k_gemm_w8p`): 同一份包字节五平面拆解反量化 fp16: And 0x000F/0x0F00/0x00F0/0x7000+0x8000(符号位单独提再相减) → Cast int16→half → Muls 2⁻ʲ 还原 q+8 → Adds −8 → 乘 s; 每平面每 chunk 恰为一个 [16k][16n] k 主序 fp16 fractal, MTE3 跨步交织成 [8ki][NI] 后走原 fp16 mmad 路径 (x 仍用 `k_pack_x`)

I8F (lm_head) 的 fractal 布局 (老 `k_gemm` fp16 流水):

- 行按 8 核均分 (nr = n/8, 要求 n 为 128 的倍数), 核内按 256 行分块, 块内按 128-k 切片
- 每切片 = `[8 ki][NI ni]` 个 512B fractal + scale 区 `[ki][ni][16 half]`; fractal 内为 **[16k][16r] k 主序** — 这样 scale 恰好按 32B 块变化, kernel 里用 `src1BlkStride=0` 的块复用技巧逐 fractal 乘 scale, 完全不需要广播展开 (310P 的 Brcb/Gather 都慢到不可用); L1→L0B 用 `LoadData` 的 `ifTranspose` 转回标准 [16n][16k]

其它:

- **matmul 合并**: DeltaNet 的 in_proj_qkv+in_proj_z 拼成 [16384][D] 一个 gemm (输出 [T][QKVZ], conv 只处理前 QKV_DIM 通道), 全注意力的 q+k+v 拼成 [14336][D] (输出 [T][QKVC]), α+β 拼成 f32 [96][D] 一次 GEMV
- `embed_tokens` 由 host RTN 成 int8 后存 v1 的 wave 布局 (I8 格式, f32 scale, 反量化与 host 参考逐位一致), 只给 `k_dq_i8` 做 embedding 查行
- 权重显存 17.5GB (v3 为 19.2): 主干 13.3 + lm_head 1.43 + embed 1.59 + α β 等 0.1; decode 每 token 流读 ~14.8GB (v3 为 16.7); 重打包 + 上传约 26s

norm、ssm_a 等小向量转 f32 上传; `conv1d.weight` 重排成 `[4][QKV_DIM]` 平面 (v 通道按 tiled 序); conv 状态布局 `[3][QKV_DIM]`。

## 3. Kernel

全部在 `kernels_npu.cpp`, block_dim 8 (k_gemv_f32t 为 n/8), 激活全部是 GM 上的 fp32。

| kernel | 说明 |
|---|---|
| `k_gemm_w8d` | 主干 matmul 的 **decode (m=1) 路径**: MTE2 (包字节+meta) → And ×2 (两组转置源 fractal) → MTE3 (L1) → b8 转置 LoadData → u8×u8 mmad (组内 s32 精确累加) → drain (滞后 2 组软件流水) + f32 scale/零点修正累加。x 为 `k_qx8` 的 u8 旋转副本, L0A 整段驻留 (k≥16384 分两半重载) |
| `k_gemm_w8p` | 主干 matmul 的 **prefill (m≤256) 路径**: 五平面 And 拆解反量化 fp16 → MTE3 跨步交织 → fp16 mmad (fp32 累加整个 k); x 由 `k_pack_x` 预打包。当前为保守 barrier 写法, 未做流水 (prefill 回退 ~15% 的原因) |
| `k_qx8` | decode 的 x 量化: f32 [k] → per-128 组对称 int8 (u8 存, 零点 128) 4 份旋转副本 + 组 scale/修正量; f32 域 RINT, 与 host lrintf 逐字节一致; 单核, 同一 x 多个 matmul 只做一次 (随 pack 缓存一起失效) |
| `k_gemm_i8` | lm_head (I8F) 的 fp16 反量化 mmad 路径 (老 kernel); GGUF 时代的 i8b/i4/i4s 模板仍在, host 不再使用 |
| `k_pack_x` | f32 [T][k] → fp16 zZ 切片布局 [kc][mi][ki][16m][16k], 三级双缓冲流水 |
| `k_gemv_f32t` | α+β 合并矩阵 ([96][D] f32 自然行布局) 的 GEMV, 行常驻 UB (每块 8 行, block_dim = n/8 = 12), T 个 token 在 kernel 内循环 |
| `k_dq_i8` | 按行号表反量化整行到自然顺序 f32 (embedding 和自测用, wave 布局) |
| `k_add_norm` | x += add (可空) + RMSNorm, 保存最后一个 token 的 hidden; token 分核轮转 |
| `k_silu_mul` | g = silu(g)·u, 按核均分逐元素 |
| `k_conv` | 因果 conv1d (核宽 4) + SiLU, 作用于合并输出 [T][QKVZ] 的前 QKV_DIM 通道; 通道均分到核, 状态 3 个平面在 UB 轮转 |
| `k_deltanet` | 每核 6 个 v 头顺序处理, 状态 S[128][128] 常驻 UB; 输入 qkvz [T][QKVZ] 与 ab [T][96] 合并布局 |
| `k_qk_rope` | (t, 头) 单元分核; 输入 q+k+v 合并输出 [T][QKVC]; RMSNorm + 交错 MRoPE + 写 KV cache; cos/sin 由 host 算好上传 |
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

- 权重约 19.2GB (第 2 节, 含 embed 1.59GB); KV cache fp32, 每 token 128KB (ctx 4096 时 512MB); DeltaNet 状态 201MB; 激活按 CHUNK = 256; x 打包缓冲 8.9MB
- 实测 ctx = 4096 时总占用 19.18GB (单芯片可用约 46GB, ECC 已关)

## 6. 调试接口 (正式功能不依赖, 但保留)

- 环境变量 `QWEN_NPU_STOP=<层号>` / `QWEN_NPU_PHASE=<0/1/2/10/11/12>`: 跑完该层 (或该层内某阶段) 直接返回
- `QWEN_NPU_PROF=1` + `qwen_prof_dump()`: 逐 kernel 同步计时 (会拖慢, 小 kernel 数字被同步开销抬高)
- `qwen_dbg(name, out, n)`: 把中间激活缓冲拷回 host
- v1 的 `k_attn` dump 模式 (T=-1) 已随两遍扫描重写移除

## 7. 验证

测试脚本在仓库外的实验目录 (`tests/npu_port`、`tests/npu_attn`、`tests/npu_api`、`tests/npu_gemm`、`tests/npu_vlat`、`tests/npu_cube`、`tests/npu_l2`, 依赖本机绝对路径, 未入库):

- `qwen_selftest` (`smoke_st.py`): tok_embd 反量化 vs host 参考逐位一致 (差 0); 各 matmul 权重 (I4W8 含行重排/列重排/多段拼接的、I8F、F32) m=1 GEMM vs 参考点积 — I4W8 的参考把 x 按与 `k_qx8` 逐字节一致的规则先量化, 相对误差 ~1e-7 (整数路径精确); int4 解包公式已与 compressed-tensors 库逐位对过
- 与 CPU (GGUF Q4_K_M) 对比 (`vs_cpu_st.py`): 两边是不同的 4bit 量化, logits 只会接近不会相等 — 300 tok prefill 后最大差 0.99 (最大值 10.5), top1 一致; 20 步贪心 19/20 一致 (喂同一轨迹)
- KL 评估 (`w8_eval.py`, 方法同 bench_cmp/prec): 真实文本 1100 prefill + 32 步 teacher forcing, decode KL vs CPU fp32 = 2.54e-2 / top1 30/32 — 与 v3 fp16 激活路径完全同级, A8 增量被 W4 噪声淹没 (预研时的纯 A8 增量实测 2.03e-3)
- 真实生成 (`gen_st.py`): chat 模板 + 贪心, 思考与正文连贯正确 (与 v3 逐字一致)
- 曾踩的数值坑 (都已修): fp16 存 s/16 掉入次正规被向量单元 flush → 个别 z 行错 → decode 发散 (meta 改存原始 s, /16 在 f32 域派生, 次正规 scale 连同 W2 钳 0); W2 用 fp16 存时大数相消误差放大到 0.7% (改 f32 并与存储 scale 精确一致)
- `tests/npu_gemm`: k_gemm 独立测试台 (三种格式 × m=1/5/16/33/100/256 × 各形状), 与 host 模拟 fp16 反量化的参考对比
- `tests/npu_vlat`: 向量指令速率/发射开销/启动开销微基准 (第 4 节第 8/9/12 条的出处)
- `tests/npu_cube`: mmad fractal 布局验证 (第 4 节第 7 条的出处)

## 8. 性能 (v4, Atlas 300I Duo 单颗 310P3, ECC 关)

| 项目 | v4 (W4A8 decode) | v3 (fp16 反量化) | v2 (GGUF) | v1 |
|---|---|---|---|---|
| prefill (300 tok, 256+44 两 chunk) | 161 tok/s | 215 tok/s | - | 1.9 tok/s |
| prefill (1024 token) | 150 tok/s | 178 tok/s | 178 tok/s | - |
| decode (ctx ~300) | 11.3 tok/s | 6.8 tok/s | 5.2 tok/s | 1.8 tok/s |
| decode (ctx ~1024 / ~3100) | 10.6 / 9.6 tok/s | 6.67 / 6.23 | 5.1 / 4.8 | - |
| 模型加载 (重打包 + 上传) | 26s | 23s | 21s | 14s |

分解 (QWEN_NPU_PROF, 数字含同步开销):

- decode ~94ms/token: gemm ~104ms (prof 含同步, 每 token 流读 ~14.8GB, I4W8 原型独立压测 170-192GB/s 接近纯 DMA 204); 解码向量只剩 2 条 And, 反量化瓶颈消除。k_qx8 ~17ms (prof 口径, 大头是同步; 每 token ~192 次调用)
- prefill 150 (v3 为 178): k_gemm_w8p 为保守 barrier 写法 (原型压测 ~22 TFLOPS, 老 fp16 kernel 25-35), gemm 环节变慢; deltanet/attn 的 fp32 向量瓶颈不变
- 原型与微测试在本机 tests/npu_w4a8 (b8 mmad 组合/转置语义/gemv 偏移粒度探测, decode/prefill/qx8 kernel 原型与带宽测试)

## 9. 已知不足与下一步

1. **attention 上 cube** (最大的 prefill 瓶颈, 长上下文时也拖 decode): Q·K^T 和 P·V 都改成 fp16 mmad (flash 风格 tile), KV cache 顺势改 fp16 (显存减半); 向量只做 softmax。预计 attn 从 ~23ms/层降到 ~2ms/层
2. **DeltaNet 重构** (第二大 prefill 瓶颈): 至少把 beta/decay/softplus/范数的逐 token 小向量操作在 chunk 内批量预计算, S 递推做双缓冲预取; 彻底方案是 chunk 化 (WY 表示) 上 cube
3. **k_gemm_w8p (prefill) 流水化**: 把保守 barrier 写法改成老 fp16 kernel 的事件流水 (qbuf 双缓冲、And/MTE3/MTE1/mmad 重叠), 收回 prefill 的 ~15% 回退; decode 侧 W4A8 已落地 (精度实测与 fp16 激活同级)
4. 小 kernel 吞吐 (conv/silu/add_norm 的串行 barrier 循环; α/β 合并与 qkv+z / q+k+v 合并已做)
5. 双芯片张量并行 (每颗一半权重, 两倍带宽)、KV cache 量化 (fp16→q8_0, 对应 CUDA 版演进)、与 python 服务接入
6. 自定义采样 (wasm) 未移植; 图片按计划固定走 CPU / CUDA, 不上 NPU
