# C++ CUDA 实现说明

`qwen_cuda/` 是 [cpu_impl.md](cpu_impl.md) 的 GPU 版本: 模型常量、权重绑定、前向流程、采样和 C 接口都与 CPU 版一一对应, 这里只写不同的地方。4 个序列槽 (见 4.1 节), prefill 按 chunk 批量输入; prefill 的矩阵乘和全部 attention 用 tensor core (`mma.sync`, fp16 输入, fp32 累加)。

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `kernels.cu` | 全部 device kernel (`extern "C"`, 按名字加载) |
| `qwen.cpp` | 权重上传、前向 (`eval_chunk`)、采样、自测、C 接口; 用 `.incbin` 把 fatbin 嵌进 so |
| `cu.h` | driver API 薄封装: 初始化 (含两个 green context 和 4 个 stream)、`ck` 检查、分配、拷贝、`launch` |
| `gguf.h` | 与 `qwen_cpu/gguf.h` 相同 (只改了 include guard) |
| `quants_ref.h` | 量化块结构、`row_bytes`、标量参考反量化 (软件 fp16 转换, 不需要 AVX) |

python 端共用仓库根目录的 `qwen.py` (`load(gguf, "cuda", device=N)`), C 接口与 CPU 版相同 (直接模式, 用槽 0), 只是 `qwen_init` 的第二个参数是 GPU 编号而不是线程数; 另有 CUDA 独有的引擎模式接口 (4.1 节, `qwen.engine_*`)。

- **不用 cudart、cuBLAS 或任何 CUDA 库**, 只用驱动自带的 `libcuda.so.1` (driver API)
- `kernels.cu` 用 conda 的 `nvcc` 编译成 fatbin (`sm_120` SASS + `compute_120` PTX), host 编译器用系统 `/usr/bin/g++` (nvcc 预处理需要 glibc 头文件); conda 激活时 `NVCC_PREPEND_FLAGS` 会带 `-ccbin=x86_64-conda-linux-gnu-c++`, 被 CMake 里的 `-ccbin` 覆盖, 有一条 `incompatible redefinition` 警告, 可忽略
- host 代码仍是 musl g++ 编译, 静态链接 libstdc++/libc, 唯一的动态依赖是 `libcuda.so.1`。只 include toolkit 里的 `cuda.h` (driver API 头文件)
- 产物 `cmake_build/libqwen_cuda.so`, 运行时不需要 CUDA toolkit, 只需要驱动 (≥ 编译用的 CUDA 版本, 本机驱动 595.58 / CUDA 13.2, nvcc 13.2.86)

## 2. 显存

- 整个 GGUF 文件 (16.24GB) 原样 `cuMemcpyHtoD` 到显存, 之后 `k_repack` 把 q8_0 / q6_k 矩阵按行原地重排 (字节数不变, 使 decode GEMV 对齐读取), 其余类型直接按 GGUF 量化格式使用; 上传后释放 host 内存 (之后自测需要的原始字节从显存拷回, 在 host 上还原成 GGUF 布局再对比)
  - q8_0: 每 8 块 (256 元素) 一组 272 字节 = [2][8][16] 量化值 (每块前 16 个 / 后 16 个) + 8 个 fp16 scale, 量化值 16 字节对齐
  - q6_k: 每 4 个超块一组 840 字节 = 4 × 208 (ql, qh, scales) + 4 个 fp16 d; 行起点只保证 8 字节对齐
- KV cache q8_0 (见下面精度一节), 每 token 34KB (32K 上下文 1.06GB, 256K 上下文 8.5GB; fp16 时是 64KB / 2GB / 16GB); 4 个槽在启动时全部分配, 每槽 `max_ctx` 行, 布局每层 `[槽][max_ctx][1024]`
- 每槽 DeltaNet + conv 状态连续存放 `[48 层 DeltaNet 状态][48 层 conv 状态]`, 144MiB + 5.6MiB, 4 个槽共 598MiB
- 激活: prefill 按 CHUNK = 1024 (每批 token 数), decode 另有一份按 4 个 token 的 (两者同时运行)
- 实测 4 个槽每槽 262144 (server 的 cuda 默认) 时本进程显存约 52GB (含视觉塔), 每槽 4096 时约 18.3GB

### 精度

原则: 两个操作数都是 fp16 的运算一律用 fp16 计算; 其余参照 llama.cpp (CUDA, 默认参数), llama.cpp 用 fp32 的保持 fp32。

- matmul 的输入激活: norm 输出 `xn`、DeltaNet 输出 `o`、attention 输出 `att`、`silu(g)·u` (llama.cpp 把它们量化成 q8_1, 即 int8)
- prefill 的 matmul (T ≥ 16): 权重解成 fp16, tensor core `mma` (fp16 输入, 乘积精确, fp32 累加)
- decode 的 matmul (T < 16, GEMV): 每个 lane 负责一个 32 元素子块。q4_k/q5_k/q6_k 的量化值 (整数, half 精确) 直接与 x 用 half2 相乘累加, scale/min 提到子块外 (d1·Σ(x·q) − m1·Σx), 每 4 项转 fp32 (激活最大 ~437 时不溢出, q6_k 余量最小, 上限约 511); q8_0/f32 仍是 w = q·d (half) 再乘 x, 每 8 个乘积 half 累加后转 fp32; 之后 fp32 跨子块累加和 warp 归约 (llama.cpp 的 f16 权重 mmvf 默认整行用 half2 累加)
- attention (prefill 和 decode): q、p 转 fp16, q·Kᵀ 和 P·V 用 tensor core `mma` (fp32 累加); softmax (max/exp/sum) 和跨 warp / 跨段的合并是 fp32 (llama.cpp 的 flash attention 强制 F32 累加)

本卡实测 (`/root/local_llm/tests/mma_bench`): mma fp16 输入 fp32 累加 142~148 TFLOPS, fp16 累加 140 TFLOPS (fp32 累加没有变慢), CUDA core HFMA2 97 TFLOPS, FFMA 72 TFLOPS。

KV cache 是 q8_0 (llama.cpp 默认 F16, q8_0 是可选的 `-ctk q8_0 -ctv q8_0`): 每 32 维一块, scale d = amax/127 (fp16), q = round(x/d), 与 llama.cpp `quantize_row_q8_0` 相同; 读取时 int8 精确转 half 再乘 d (舍入一次, 等于 half(q·d)), 之后与原来一样 fp16 进 tensor core。存储与 GGUF 不同: 每层 K、V 各一个 int8 `[ctx][1024]` 和一个 fp16 scale `[ctx][32]` (4 个头 × 8 块), 都是 16 字节对齐。

保持 fp32 的: 残差流 `x`、所有 norm 的计算、RoPE、DeltaNet 的状态和递推、conv 状态、matmul 输出、logits (与 llama.cpp 相同)。反量化中 fp16 scale × 整数量化值的乘法在 fp32 中做 (乘积在 fp32 中是精确的, 转 half 只舍入一次, 与直接用 half 乘结果相同)。实测真实推理时 fp16 部分的最大绝对值: matmul 输入约 437 (第 63 层 silu·up), KV 约 113, q (rope 后) 约 29.5, 远小于 fp16 上限 65504; 残差流最大约 996。

## 3. Kernel

| kernel | grid × block | 说明 |
|---|---|---|
| `k_gemm_<type>` | (T/64, n/64) × 128 | prefill 的矩阵乘 (T ≥ 16)。block tile 64 行 × 64 token, 4 个 warp 2×2, 每个 warp 32×32; 沿 k 每步 64, 权重用 `dq8h` 反量化成 fp16 写 smem, x (fp16) 写 smem, 行跨度补 8 个 half 防 bank 冲突; `ldmatrix` + `mma.m16n8k16`。实测 120~143 TFLOPS |
| `k_gemv` / `k_gemv_glu` (`k_gemv1*`: T = 1 特化) | 总行数/2 × 64 | decode 的矩阵乘 (T < 16): 一个 warp 一行, block 只有 2 个 warp 使各 SM 负载均衡; 一组共享输入的矩阵一次 launch (qkv+z+alpha+beta, q+k+v), 量化类型运行时分发; glu 版同一 warp 算 gate 和 up 的同一行, 直接输出 silu(gate)·up (fp16)。lane = 超块内第 lane%8 个 32 元素子块, warp 每轮 4 个超块 (要求 k 是 1024 的倍数); 量化值 `__byte_perm` 拼成 1024+q 再减偏移 (精确), 子块 scale/min 无分支解码 (`scale_min_k4_bl`), q8_0/q6_k 读重排后的对齐布局; T < 16 时输入 x 是 `xperm` 布局 (每个 token 一行, 行内重排; 由产生它的 `k_add_norm`、`k_dn_post`、`k_attn_pf` / `k_attn_merge`、`k_gemv*_glu` 直接写成), 每条 load 指令 32 个 lane 读连续 512 字节; `k_gemv1` 限制每 SM 18 个 block (寄存器 ≤ 56), n = 5120 的 2560 个 block 一波放下; 精度见上 |
| `k_dq_<type>` / `k_embed_<type>` | 行数 × 256 | 反量化整行 (自测用) / token embedding (mask 为 1 时拷贝外部 embedding) |
| `k_add_norm` | T × 1024 | `x += add` (可为空) + RMSNorm, 顺便保存最后一个 token 的 hidden。上一层 FFN 的残差在下一层开头的这个 kernel 里加 |
| `k_dn_prep` | (T, 16) × 128 | DeltaNet 中与状态无关的部分, 各 token 并行: 因果 conv1d + SiLU、q/k L2 归一化 (q 乘 1/√128)、q·k、beta = sigmoid(b)、decay = exp(ssm_a·softplus(a + dt_bias)) |
| `k_dn_rec` | (48, 4) × 128 | DeltaNet 递推, 每个 block 一个 v 头的 32 列; lane = 4·列 + 行段, 线程在寄存器持有 S 的 32 行 × 1 列。用旧状态算 a = kᵀS、b = qᵀS (行段间 2 次 shuffle), delta = (v − decay·a)·beta, o = decay·b + (q·k)·delta, S = decay·S + k·delta, 每个 token 没有 `__syncthreads`; 每 16 个 token 的输入用 `cp.async` 双缓冲预取到 smem; 开头顺便更新 conv 状态 |
| `k_dn_post` | (T, 48) × 128 | DeltaNet 输出的门控 RMSNorm, 输出 fp16 |
| `k_qk_rope` | (T, 28) × 256 | q/k 头的 RMSNorm + 交错 MRoPE, k/v 量化成 q8_0 写入 KV cache (一个 warp 正好一个 32 维块, shuffle 求 amax) |
| `k_attn_pf` | (T/64, 24) × 128 | prefill (T > 1) 的 flash attention: 每个 block 64 个 query × 1 个 q 头, 每个 warp 16 个 query; 每轮 32 个 key 的 K/V 反量化成 fp16 放在 smem (双缓冲, 动态 smem 101376 字节): 算本轮 Q·Kᵀ 时把下一轮 K 的 int8 读进寄存器, 算完写入另一个缓冲, V 同理在 softmax + P·V 期间读取 (K、V 分开以控制寄存器, 241 个无溢出), S = Q·Kᵀ 用 `ldmatrix` + `mma`, 因果 mask + online softmax (fp32), P 由 S 的累加器转 fp16 直接作为 A, V 用 `ldmatrix.trans` 得到 B; 最后乘 sigmoid(gate) |
| `k_attn_dec` | (4, 78) × 128 | decode (T = 1) 的 flash-decoding: 每个 block 一个 kv 头的 6 个 q 头 (K/V 只读一次) × 一段 key (每段 ≥ 128 个, 最多 78 段, 4 × 78 = 312 = 每 SM 2 个 block 一波), 4 个 warp 轮流处理 16 个 key 一组; mma 的 A 有 16 行, 前 6 行是 q 头; head 维重排后 K/V 每行每 32 维 (正好一个 q8_0 块) 读 8 字节 int8, 乘该块 scale 反量化成 fp16 作为 B fragment, V 用 `movmatrix` 转置; 每组开头用 `prefetch.global.L2` 预取本组 V 和下一组 K (否则 K、V 两次显存延迟串行, 256K 时 decode 39 → 45 tok/s); 段内 4 个 warp 在 smem 合并后输出 (max, sum, 未归一化 o) |
| `k_attn_merge` | 24 × 256 | 合并 `k_attn_dec` 各段, 乘 sigmoid(gate) |
| `k_silu_mul` | 逐元素 | `silu(g) · u`, 输出 fp16 |
| `k_dn_prep_b` / `k_dn_rec_b` / `k_qk_rope_b` / `k_attn_dec_b` / `k_attn_merge_b` | 同上, 多一维 B | 多序列 decode: B 个 token 分属不同槽 (`slot[t]`), 按槽偏移状态 / KV 指针、按 token 偏移输入输出后调用与单序列相同的 device 函数, 计算完全相同 |

各量化类型的 lane 映射和对齐 (`dq8<TYPE>`):

- **q4_k / q5_k**: 超块 144 / 176 字节, 都是 16 的倍数, qs/qh 可以按 8 字节读; lane 的子块 = lane / 4, 64 元素组 = lane / 8
- **q6_k**: 超块 210 字节, 重排后每 4 个超块一组 (d 移到组尾), 8 字节对齐, 按 uint2 读
- **q8_0**: 34 字节一块, 重排后每 8 块一组 (scale 移到组尾), 量化值按 uint2 / uint4 读
- 所有矩阵的行数都是 8 的倍数, k 都是 256 的倍数

lm_head 用同一个 `k_mm_q6_k` (T = 1); logits 拷回 pinned host 内存后, 采样复用 CPU 版的代码。

## 4. 其他

- 4 个非阻塞 stream (不与同进程 torch 的 legacy stream 隐式同步): prefill 整卡 / prefill 分区 / decode 整卡 / decode 分区; 整卡的在 primary context (与同进程 torch 共用), 分区的在两个 green context (`cuDevSmResourceSplit`, SM 不相交, decode 分区的 SM 数用 `qwen_set_dec_sm` 在 init 前设置; server.py 用 `--dec-ratio` 比例换算 (默认 0.31, 取最近的偶数个且两边都至少 8 个), 本卡为 decode 48 个、prefill 108 个)。kernel 用 `cuLibraryLoadData` 加载 (与 context 无关, launch 时按 stream 所属的 context 执行), 显存在 primary context 分配, 各 context 都能访问; pinned 内存用 `CU_MEMHOSTALLOC_PORTABLE`
- decode 每步 B (1~4) 个槽各 1 个 token 拼成 T = B 一起算: 读一遍权重给 B 个 token 用。整卡 / 分区 × B = 1..4 共 8 个 CUDA Graph 在初始化时全部捕获, 之后每步只拷 token / 位置 / n_past / 槽号再重放; n_past 放在显存里, `k_attn_dec_b` 的 grid 固定为 4 × ATT_SPLIT × B, 多余的段直接退出。直接模式的 T = 1 decode 也走这里 (B = 1)
- CUDA context 是每线程的, prefill / decode 选 stream 时同时 `cuCtxSetCurrent` (server.py 会从不同的 HTTP 线程调用)
- CUDA 出错直接打印并 `_Exit(1)`

### 4.1 多序列 (引擎模式)

- 4 个槽, 最多 4 个请求同时进行 (prefill + decode)。更多的请求在 python (`server.py` 的 `SlotPool`) 按到达顺序排队, 不进 C++; 拿到槽位时按顺序发 ticket
- `qwen_prefill(槽, ticket, ...)`: 在调用线程 (请求的 HTTP 线程) 上执行, 先按 ticket 顺序等前面的 prefill 算完 (FIFO, 同时只有一个 prefill), 清空槽状态, 逐 chunk prefill, 采样第一个 token 后把槽交给 decode 循环。每个 chunk 重新选 stream: 有 decode 在跑时用 prefill 分区, 否则整卡; 换 stream 前同步上一个 chunk
- `qwen_decode_loop`: 由一个 python 线程调用, 不返回。每步把所有 decode 中的槽一起算 (进度对齐), 有 prefill 在算时用 decode 分区, 否则整卡; 采样 (各槽自己的参数、seed、历史) 后写入槽的输出数组, 遇到停止 token / 长度上限 / 取消时结束该槽
- `qwen_next(槽, buf, cap)`: 等新 token, 返回已生成但还没读的 (可能多个), 结束且读完返回 0; `qwen_cancel` 设取消标志, 之后仍要读到 0 才能复用槽
- 等待都用 futex (直接 `syscall`, 不经过 libc: musl 的 errno 在 glibc 创建的线程里不可用); 并发路径上不分配内存 (musl 在没有自己创建的线程时 malloc 不加锁, 这里的线程都是 python 的)
- 直接模式 (`qwen_eval` 等, 测试脚本用) 固定用槽 0, 不能与引擎模式同时用

## 5. 验证

- `qwen_selftest`: 6 个真实 tensor (覆盖 f32 / q8_0 / q4_k / q5_k / q6_k 和 lm_head), GPU 反量化与标量参考对比 (实测逐位一致), T = 13 (GEMV) 和 T = 100 (tensor core GEMM, 含不满 64 的 tile) 的 matmul (输入取 fp16 可精确表示的值) 与 double 点积对比 (相对误差 1~5e-5, 阈值 2e-4; 全 fp32 时约 1.5e-8)
- `tests/cuda_vs_cpu.py`: 同一进程加载 CPU 和 CUDA 两个库, 对比每层 hidden 和 logits: 1100 token prefill (跨 chunk; `--n 4000` 也测过)、逐 token decode、图片 embedding + MRoPE 位置。hidden 相对误差约 2e-3, logits 约 1e-3~2e-3 (阈值 5e-3; KV fp16 时约 1e-3 / 5e-4~1e-3; 全 fp32 时都在 1e-6 量级), top1 全部一致, top-10 重合 9~10 个, 贪心生成结果相同
- `tests/test_server.py` 对 `server.py --type cuda` 全部通过
- KV q8_0 vs fp16 (同一 CUDA 实现, 真实长文本: llama.cpp 文档 + 本仓库文档): prefill 后 teacher forcing 128 步, 4K 上下文 KL 平均 5.1e-5 (最大 7.9e-4), top1 一致 127/128, PPL 2.0723 → 2.0717; 32K 上下文 KL 平均 5.7e-6 (最大 9.3e-5), top1 128/128, PPL 1.4415 → 1.4413; 30K token 的 needle 检索 (深度 10% / 50% / 90%) 贪心输出逐字相同。q8_0 存储与 "fp16 存储 q8_0 还原值" 的假量化版本逐位一致

## 6. 性能 (RTX 6000D, `tests/bench.py --type cuda`)

| 上下文 | prefill | decode | decode (KV fp16 时) |
|---|---|---|---|
| 256 | 1556 tok/s | 66 tok/s | 66 tok/s |
| 4096 | 2413 tok/s | 66 tok/s | 65 tok/s |
| 16384 | 2298 tok/s | 64 tok/s | 63 tok/s |
| 65536 | 1923 tok/s | 59 tok/s | 53 tok/s |
| 131072 | 1577 tok/s (83s) | 54 tok/s | 45 tok/s |
| 262144 | 1161 tok/s (226s) | 45 tok/s | 34 tok/s |

KV 改 q8_0 后 prefill 不变 (fp16 时 4K / 64K / 256K 分别为 2409 / 1921 / 1164 tok/s)。

bench 的 decode 包含 python 和采样开销, 单纯 `qwen_eval` 约 15.0ms/token; 采样: 贪心 0.16ms, top_k=20 约 0.09ms (单遍扫描维护前 k 个, 原来 nth_element 要 1.5ms)。prefill 的 prompt 不满一个 chunk (256) 时 block 数少, 速度偏低。

理论上限 (实测只读带宽 1339GB/s, mma 145 TFLOPS): decode 短上下文 79 tok/s、256K 52 tok/s (每 token 读权重 16.7GB + KV 34KB × 上下文长度; KV fp16 时 256K 为 39 tok/s); prefill 只算 matmul 约 2980 tok/s (48.7 GFLOP/token), 256K 平均 (attention 计算量与 matmul 相当) 约 1430 tok/s。

历史 (16K 上下文): 全 fp32 版 prefill 141 / decode 20 → 改 fp16 后 247 / 26 → decode 改 flash-decoding 54 → prefill 改 tensor core GEMM 900 → flash attention + CHUNK 1024 1980 → DeltaNet 拆分 + 预取 2293。

### 少 SM 时的 decode

用 green context (`cuDevSmResourceSplit`, 测试副本里, 正式代码不含) 限制可用 SM 数, 模拟算力 / 带宽比更低的卡 (3090 的比例约相当于本卡 80 个 SM)。decode GEMV 改为每 lane 一个 32 元素子块、scale/min 提到子块外、无分支解码、x 用 `xperm` 布局、q8_0/q6_k 重排之后 (每 token, 上下文 4K):

| SM 数 | 改之前 | 现在 |
|---|---|---|
| 32 | 34.9ms | 21.8ms |
| 78 | 17.5ms | 15.0ms |
| 156 (整卡) | 15.1ms | 15.0ms |

改之前少 SM 时慢的原因: 反量化每个权重的指令多 (每 8 个元素解一次子块 scale, 且有分支发散), 32 SM 时即使权重全在 L1 (只计算) q4_k 也只能处理约 680GB/s; 改成每 lane 32 个连续元素后又出现读 x 的跨步访问 (每条 load 指令跨 16 个 cache line), 用 `xperm` 布局解决。

### 多序列与并发 (每槽 prompt 4096, 随机 token, 贪心)

decode 单独 (整卡) 合计 tok/s: B = 1 / 2 / 3 / 4 为 66 / 115 / 155 / 177; 每步 eval 14.9 / 17.0 / 18.7 / 20.3ms, 采样每个槽约 0.16ms。长时间满载时 GPU 到 85°C 后 SM 时钟从 2422 降到约 1930MHz, B = 4 每步升到约 22.4ms。

prefill 与 decode 同时跑 (prefill 4096 token, 括号内是相对各自单独跑的比例; prefill 单独 2400 tok/s):

| decode 分区 SM 数 | 1 个 decode: prefill / decode tok/s | 3 个 decode: prefill / decode 合计 tok/s |
|---|---|---|
| 48 (默认) | 1261 (53%) / 52.6 (80%) | 1354 (56%) / 64.4 (43%) |
| 78 | 671 (28%) / 62.7 (95%) | 967 (40%) / 102 (68%) |
| 108 | 367 (15%) / 63.5 (96%) | 546 (23%) / 120 (80%) |

表中 SM 数对应比例约 0.31 / 0.5 / 0.69。decode 每 token 最大间隔: 78 时 1 个 decode 约 17ms, 3 个约 30ms; 48 时 3 个约 47ms。只有 decode 时 B = 1 的 decode 基本吃满显存带宽, 分区后 prefill 被拖得很重 (显存带宽 / L2 争用)。

不足 16 个 token 的 prompt 走 GEMV (T < 16), 16 个以上走 tensor core GEMM, 后者在 token 少时 block 数少, 16~100 个 token 的 prefill 都要约 110ms (改动前就是这样)。

## 7. 可改进的方向

- **decode**: 约为带宽理论值 (12.5ms/token) 的 84%; 单个 GEMV (真实形状, 不命中 L2) 整卡 1070~1270GB/s (带宽的 80~95%), 32 SM 时 720~1210GB/s (q8_0 最高, q4_k 最低); 长上下文 attention 约 80~89%
- **prefill**: GEMM 已接近 tensor core 峰值, 剩下的是 DeltaNet 递推 (按 token 串行) 和小 n 的 matmul; 递推可以改成 chunked (WY) 形式用 mma 计算
- 没有前缀缓存; SM 分区比例固定, 没有按 decode 数量调整; B > 1 的 decode GEMV 用 NT = 8 的通用模板, 可以按 B 特化; T = 16~100 的短 prefill 偏慢
