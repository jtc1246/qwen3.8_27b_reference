# C++ CUDA 实现说明

`qwen_cuda/` 是 [cpu_impl.md](cpu_impl.md) 的 GPU 版本: 模型常量、权重绑定、前向流程、采样和 C 接口都与 CPU 版一一对应, 这里只写不同的地方。当前是第一版, 目标是简单、正确: 单序列、单 stream、按 chunk 批量输入, 不用 tensor core。

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
- KV cache fp32, 每 token 128KB (32K 上下文 4GB); DeltaNet 状态 144MiB; 激活按 CHUNK = 256
- 实测 ctx = 32768 时显存占用 21.0GB

## 3. Kernel

| kernel | grid × block | 说明 |
|---|---|---|
| `k_mm_<type>` | n/8 × 256 | 反量化 + 矩阵乘。一个 warp 算一行, 每次解一个 256 元素超块, lane 负责其中连续 8 个; 每 8 个 token 一组累加 (8 个寄存器累加器), token 多于 8 时该行权重重新解码; 最后 warp shuffle 求和 |
| `k_dq_<type>` / `k_embed_<type>` | 行数 × 256 | 反量化整行 (自测用) / token embedding (mask 为 1 时拷贝外部 embedding) |
| `k_add_norm` | T × 1024 | `x += add` (可为空) + RMSNorm, 顺便保存最后一个 token 的 hidden。上一层 FFN 的残差在下一层开头的这个 kernel 里加 |
| `k_conv` | 40 × 256 | 因果 conv1d + SiLU, 一个线程一个通道 |
| `k_deltanet` | 48 × 128 | 一个 block 一个 v 头, 线程 j 在寄存器中持有状态的第 j 列 (128 个 float), 递推不需要跨线程归约; q/k L2 归一化、beta/g、门控 RMSNorm 都在这个 kernel 里。寄存器 255 个, 有约 300 字节溢出 |
| `k_qk_rope` | (T, 28) × 256 | q/k 头的 RMSNorm + 交错 MRoPE, k/v 写入 KV cache |
| `k_attn` | (T, 24) × 256 | 一个 block 一个 (token, q 头), 8 个 warp 各自用 online softmax 处理 1/8 的 key, 最后在 shared memory 合并, 乘 sigmoid(gate) |
| `k_silu_mul` | 逐元素 | `g = silu(g) · u` |

各量化类型的 lane 映射和对齐 (`dq8<TYPE>`):

- **q4_k / q5_k**: 超块 144 / 176 字节, 都是 16 的倍数, qs/qh 可以按 8 字节读; lane 的子块 = lane / 4, 64 元素组 = lane / 8
- **q6_k**: 超块 210 字节, 只保证 2 字节对齐, 8 字节按 4 个 uint16 读
- **q8_0**: 34 字节一块, lane / 4 是块号, 同样按 uint16 读
- 所有矩阵的行数都是 8 的倍数, k 都是 256 的倍数

lm_head 用同一个 `k_mm_q6_k` (T = 1); logits 拷回 pinned host 内存后, 采样复用 CPU 版的代码。

## 4. 其他

- 所有 kernel 在默认 stream 上顺序执行, 只在拷回 logits 时同步
- CUDA context 是每线程的, 每个 C 接口入口都会 `cuCtxSetCurrent` (server.py 会从不同的 HTTP 线程调用)
- 用的是 device 的 primary context, 与同进程的 torch (`server.py --type cuda` 时视觉塔在 GPU 上) 共用同一个 context
- CUDA 出错直接打印并 `_Exit(1)`

## 5. 验证

- `qwen_selftest`: 6 个真实 tensor (覆盖 f32 / q8_0 / q4_k / q5_k / q6_k 和 lm_head), GPU 反量化与标量参考对比 (实测逐位一致), T = 13 的 matmul 与 double 点积对比 (相对误差约 1.5e-8)
- `tests/cuda_vs_cpu.py`: 同一进程加载 CPU 和 CUDA 两个库, 对比每层 hidden 和 logits: 300 token prefill (跨 chunk)、逐 token decode、图片 embedding + MRoPE 位置, 相对误差都在 1e-6 量级, top-10 完全一致, 贪心生成结果相同
- `tests/test_server.py` 对 `server.py --type cuda` 全部通过

## 6. 性能 (RTX 6000D, `tests/bench.py --type cuda`)

| 上下文 | prefill | decode |
|---|---|---|
| 256 | 156 tok/s | 54 tok/s |
| 1024 | 162 tok/s | 50 tok/s |
| 4096 | 157 tok/s | 38 tok/s |
| 16384 | 141 tok/s | 20 tok/s |

短 prompt 的 decode 约 62 tok/s (CPU 版 3.6 tok/s), prefill 约是 CPU 版的 7.5 倍。

## 7. 可改进的方向

- **长上下文 decode**: `k_attn` 在 T = 1 时只有 24 个 block, 而且 6 个共享 kv 头的 q 头各读一遍 KV; 可以按 kv 头合并、沿序列切分 (split-K) 再合并, KV cache 改 fp16/bf16
- **prefill**: matmul 没有 shared memory 分块、没用 tensor core, 每个 block 都从 L2 读整块输入; 可以改成 `mma` (fp16/int8) 分块 GEMM
- **decode**: 每 token 约 800 次 kernel launch, 可以用 CUDA Graph 或融合 (gate/up 合并、silu 融进 down 的输入); matmul 可以把 x 放 shared memory、调整每 warp 行数
- **k_deltanet** 寄存器溢出
- 仍然是单序列, 没有 batch / 并发 / 前缀缓存
