# C++ CUDA 实现说明

`qwen_cuda/` 是 [cpu_impl.md](cpu_impl.md) 的 GPU 版本: 模型常量、权重绑定、前向流程、采样和 C 接口都与 CPU 版一一对应, 这里只写不同的地方。单序列、单 stream、按 chunk 批量输入; prefill 的矩阵乘和全部 attention 用 tensor core (`mma.sync`, fp16 输入, fp32 累加)。

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `kernels.cu` | 全部 device kernel (`extern "C"`, 按名字加载) |
| `qwen.cpp` | 权重上传、前向 (`eval_chunk`)、采样、自测、C 接口; 用 `.incbin` 把 fatbin 嵌进 so |
| `cu.h` | driver API 薄封装: 初始化、`ck` 检查、分配、拷贝、`launch` |
| `gguf.h` | 与 `qwen_cpu/gguf.h` 相同 (只改了 include guard) |
| `quants_ref.h` | 量化块结构、`row_bytes`、标量参考反量化 (软件 fp16 转换, 不需要 AVX) |

python 端共用仓库根目录的 `qwen.py` (`load(gguf, "cuda", device=N)`), C 接口与 CPU 版相同, 只是 `qwen_init` 的第二个参数是 GPU 编号而不是线程数。

- **不用 cudart、cuBLAS 或任何 CUDA 库**, 只用驱动自带的 `libcuda.so.1` (driver API)
- `kernels.cu` 用 conda 的 `nvcc` 编译成 fatbin (`sm_120` SASS + `compute_120` PTX), host 编译器用系统 `/usr/bin/g++` (nvcc 预处理需要 glibc 头文件); conda 激活时 `NVCC_PREPEND_FLAGS` 会带 `-ccbin=x86_64-conda-linux-gnu-c++`, 被 CMake 里的 `-ccbin` 覆盖, 有一条 `incompatible redefinition` 警告, 可忽略
- host 代码仍是 musl g++ 编译, 静态链接 libstdc++/libc, 唯一的动态依赖是 `libcuda.so.1`。只 include toolkit 里的 `cuda.h` (driver API 头文件)
- 产物 `cmake_build/libqwen_cuda.so`, 运行时不需要 CUDA toolkit, 只需要驱动 (≥ 编译用的 CUDA 版本, 本机驱动 595.58 / CUDA 13.2, nvcc 13.2.86)

## 2. 显存

- 整个 GGUF 文件 (16.24GB) 原样 `cuMemcpyHtoD` 到显存, 权重在 GPU 上直接按 GGUF 量化格式使用, 不做重排; 上传后释放 host 内存 (之后自测需要的原始字节从显存拷回)
- KV cache fp16, 每 token 64KB (32K 上下文 2GB, 256K 上下文 16GB); DeltaNet 状态 144MiB; 激活按 CHUNK = 1024 (prefill 每批 token 数)
- 实测 ctx = 32768 时显存占用约 19.0GB

### 精度

原则: 两个操作数都是 fp16 的运算一律用 fp16 计算; 其余参照 llama.cpp (CUDA, 默认参数), llama.cpp 用 fp32 的保持 fp32。

- KV cache (llama.cpp 默认 F16)
- matmul 的输入激活: norm 输出 `xn`、DeltaNet 输出 `o`、attention 输出 `att`、`silu(g)·u` (llama.cpp 把它们量化成 q8_1, 即 int8)
- prefill 的 matmul (T ≥ 16): 权重解成 fp16, tensor core `mma` (fp16 输入, 乘积精确, fp32 累加)
- decode 的 matmul (T < 16, GEMV): 权重解成 half2, 每个 lane 每个超块的 8 个乘积及其部分和全用 half, 再转 fp32 跨超块累加和 warp 归约 (llama.cpp 的 f16 权重 mmvf 默认整行用 half2 累加)
- attention (prefill 和 decode): q、p 转 fp16, q·Kᵀ 和 P·V 用 tensor core `mma` (fp32 累加); softmax (max/exp/sum) 和跨 warp / 跨段的合并是 fp32 (llama.cpp 的 flash attention 强制 F32 累加)

本卡实测 (`/root/local_llm/tests/mma_bench`): mma fp16 输入 fp32 累加 142~148 TFLOPS, fp16 累加 140 TFLOPS (fp32 累加没有变慢), CUDA core HFMA2 97 TFLOPS, FFMA 72 TFLOPS。

保持 fp32 的: 残差流 `x`、所有 norm 的计算、RoPE、DeltaNet 的状态和递推、conv 状态、matmul 输出、logits (与 llama.cpp 相同)。反量化中 fp16 scale × 整数量化值的乘法在 fp32 中做 (乘积在 fp32 中是精确的, 转 half 只舍入一次, 与直接用 half 乘结果相同)。实测真实推理时 fp16 部分的最大绝对值: matmul 输入约 437 (第 63 层 silu·up), KV 约 113, q (rope 后) 约 29.5, 远小于 fp16 上限 65504; 残差流最大约 996。

## 3. Kernel

| kernel | grid × block | 说明 |
|---|---|---|
| `k_gemm_<type>` | (T/64, n/64) × 128 | prefill 的矩阵乘 (T ≥ 16)。block tile 64 行 × 64 token, 4 个 warp 2×2, 每个 warp 32×32; 沿 k 每步 64, 权重用 `dq8h` 反量化成 fp16 写 smem, x (fp16) 写 smem, 行跨度补 8 个 half 防 bank 冲突; `ldmatrix` + `mma.m16n8k16`。实测 120~143 TFLOPS |
| `k_gemv` / `k_gemv_glu` (`k_gemv1*`: T = 1 特化) | 总行数/2 × 64 | decode 的矩阵乘 (T < 16): 一个 warp 一行, block 只有 2 个 warp 使各 SM 负载均衡; 一组共享输入的矩阵一次 launch (qkv+z+alpha+beta, q+k+v), 量化类型运行时分发; glu 版同一 warp 算 gate 和 up 的同一行, 直接输出 silu(gate)·up (fp16)。权重解成 half2 (`__byte_perm` 拼成 1024+q 再减偏移, 精确), 8 个乘积的部分和用 half, 再 fp32 累加; 超块循环 unroll 4 |
| `k_dq_<type>` / `k_embed_<type>` | 行数 × 256 | 反量化整行 (自测用) / token embedding (mask 为 1 时拷贝外部 embedding) |
| `k_add_norm` | T × 1024 | `x += add` (可为空) + RMSNorm, 顺便保存最后一个 token 的 hidden。上一层 FFN 的残差在下一层开头的这个 kernel 里加 |
| `k_dn_prep` | (T, 16) × 128 | DeltaNet 中与状态无关的部分, 各 token 并行: 因果 conv1d + SiLU、q/k L2 归一化 (q 乘 1/√128)、q·k、beta = sigmoid(b)、decay = exp(ssm_a·softplus(a + dt_bias)) |
| `k_dn_rec` | (48, 4) × 128 | DeltaNet 递推, 每个 block 一个 v 头的 32 列; lane = 4·列 + 行段, 线程在寄存器持有 S 的 32 行 × 1 列。用旧状态算 a = kᵀS、b = qᵀS (行段间 2 次 shuffle), delta = (v − decay·a)·beta, o = decay·b + (q·k)·delta, S = decay·S + k·delta, 每个 token 没有 `__syncthreads`; 每 16 个 token 的输入用 `cp.async` 双缓冲预取到 smem; 开头顺便更新 conv 状态 |
| `k_dn_post` | (T, 48) × 128 | DeltaNet 输出的门控 RMSNorm, 输出 fp16 |
| `k_qk_rope` | (T, 28) × 256 | q/k 头的 RMSNorm + 交错 MRoPE, k/v 写入 KV cache |
| `k_attn_pf` | (T/64, 24) × 128 | prefill (T > 1) 的 flash attention: 每个 block 64 个 query × 1 个 q 头, 每个 warp 16 个 query; 每轮 32 个 key 的 K/V 用 `cp.async` 双缓冲预取到 smem (动态 smem 101376 字节), S = Q·Kᵀ 用 `ldmatrix` + `mma`, 因果 mask + online softmax (fp32), P 由 S 的累加器转 fp16 直接作为 A, V 用 `ldmatrix.trans` 得到 B; 最后乘 sigmoid(gate) |
| `k_attn_dec` | (4, 78) × 128 | decode (T = 1) 的 flash-decoding: 每个 block 一个 kv 头的 6 个 q 头 (K/V 只读一次) × 一段 key (每段 ≥ 128 个, 最多 78 段, 4 × 78 = 312 = 每 SM 2 个 block 一波), 4 个 warp 轮流处理 16 个 key 一组; mma 的 A 有 16 行, 前 6 行是 q 头; head 维重排后 K/V 每行每 32 维 16 字节连续读并直接作为 B fragment, V 用 `movmatrix` 转置; 段内 4 个 warp 在 smem 合并后输出 (max, sum, 未归一化 o) |
| `k_attn_merge` | 24 × 256 | 合并 `k_attn_dec` 各段, 乘 sigmoid(gate) |
| `k_silu_mul` | 逐元素 | `silu(g) · u`, 输出 fp16 |

各量化类型的 lane 映射和对齐 (`dq8<TYPE>`):

- **q4_k / q5_k**: 超块 144 / 176 字节, 都是 16 的倍数, qs/qh 可以按 8 字节读; lane 的子块 = lane / 4, 64 元素组 = lane / 8
- **q6_k**: 超块 210 字节, 只保证 2 字节对齐, 8 字节按 4 个 uint16 读
- **q8_0**: 34 字节一块, lane / 4 是块号, 同样按 uint16 读
- 所有矩阵的行数都是 8 的倍数, k 都是 256 的倍数

lm_head 用同一个 `k_mm_q6_k` (T = 1); logits 拷回 pinned host 内存后, 采样复用 CPU 版的代码。

## 4. 其他

- 所有 kernel 和拷贝都在一个非阻塞 stream 上顺序执行 (不与同进程 torch 的 legacy stream 隐式同步), 只在拷回 logits 时同步
- decode (T = 1, 无外部 embedding) 第一次运行时捕获成 CUDA Graph, 之后每个 token 只拷 token/位置/n_past 再重放; n_past 放在显存里, `k_attn_dec` 的 grid 固定为 4 × ATT_SPLIT, 多余的段直接退出
- CUDA context 是每线程的, 每个 C 接口入口都会 `cuCtxSetCurrent` (server.py 会从不同的 HTTP 线程调用)
- 用的是 device 的 primary context, 与同进程的 torch (`server.py --type cuda` 时视觉塔在 GPU 上) 共用同一个 context
- CUDA 出错直接打印并 `_Exit(1)`

## 5. 验证

- `qwen_selftest`: 6 个真实 tensor (覆盖 f32 / q8_0 / q4_k / q5_k / q6_k 和 lm_head), GPU 反量化与标量参考对比 (实测逐位一致), T = 13 (GEMV) 和 T = 100 (tensor core GEMM, 含不满 64 的 tile) 的 matmul (输入取 fp16 可精确表示的值) 与 double 点积对比 (相对误差 1~5e-5, 阈值 2e-4; 全 fp32 时约 1.5e-8)
- `tests/cuda_vs_cpu.py`: 同一进程加载 CPU 和 CUDA 两个库, 对比每层 hidden 和 logits: 1100 token prefill (跨 chunk; `--n 4000` 也测过)、逐 token decode、图片 embedding + MRoPE 位置。hidden 相对误差约 1e-3, logits 约 5e-4~1e-3 (阈值 5e-3; 全 fp32 时都在 1e-6 量级), top1 全部一致, top-10 重合 9~10 个, 贪心生成结果相同
- `tests/test_server.py` 对 `server.py --type cuda` 全部通过

## 6. 性能 (RTX 6000D, `tests/bench.py --type cuda`)

| 上下文 | prefill | decode |
|---|---|---|
| 256 | 1606 tok/s | 66 tok/s |
| 4096 | 2410 tok/s | 65 tok/s |
| 16384 | 2294 tok/s | 63 tok/s |
| 65536 | 1922 tok/s | 53 tok/s |
| 262144 | 996 tok/s (263s) | 30 tok/s (GEMV 优化前测的) |

bench 的 decode 包含 python 和采样开销, 单纯 `qwen_eval` 约 15.0ms/token; 采样: 贪心 0.16ms, top_k=20 约 0.09ms (单遍扫描维护前 k 个, 原来 nth_element 要 1.5ms)。prefill 的 prompt 不满一个 chunk (256) 时 block 数少, 速度偏低。

理论上限 (实测只读带宽 1339GB/s, mma 145 TFLOPS): decode 短上下文 79 tok/s、256K 39 tok/s (每 token 读权重 16.7GB + KV); prefill 只算 matmul 约 2980 tok/s (48.7 GFLOP/token), 256K 平均 (attention 计算量与 matmul 相当) 约 1430 tok/s。

历史 (16K 上下文): 全 fp32 版 prefill 141 / decode 20 → 改 fp16 后 247 / 26 → decode 改 flash-decoding 54 → prefill 改 tensor core GEMM 900 → flash attention + CHUNK 1024 1980 → DeltaNet 拆分 + 预取 2293。

## 7. 可改进的方向

- **decode**: 约为带宽理论值 (12.5ms/token) 的 84%; GEMV 带宽利用率 gate/up、qkv 约 88%, down / ssm_out 约 80%, lm_head 98%; 长上下文 attention 约 80~89%
- **prefill**: GEMM 已接近 tensor core 峰值, 剩下的是 DeltaNet 递推 (按 token 串行) 和小 n 的 matmul; 递推可以改成 chunked (WY) 形式用 mma 计算
- 仍然是单序列, 没有 batch / 并发 / 前缀缓存
