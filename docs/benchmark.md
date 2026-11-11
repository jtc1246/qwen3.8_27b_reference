# 性能对比: 本实现 vs llama.cpp vs vLLM

同一张 GPU、同一个模型 (Qwen3.8-27B, 4bit 权重), 单序列 (batch = 1) 的 prefill / decode 速度对比, 以及 llama.cpp 为了速度在精度上的"偷工减料"。测试日期 2026-11-10。

## 1. 结论

- **decode**: 按每 token 实际读取的字节数换算后, 本实现在 16bit 和 8bit KV 下、所有上下文长度都最快。短上下文比 llama.cpp 快约 2%, 比 vLLM 快约 5%; 64K 上下文 + 8bit KV 时比 llama.cpp 快 13%
- **prefill**: 本实现最慢 (约 2400 tok/s)。llama.cpp 快约 50% (约 3600), vLLM 快约 12% (约 2700)
- **llama.cpp 的 prefill 优势来自降低精度**: 所有量化权重的矩阵乘都先把激活量化成 int8 (q8_1), 再用 int8 tensor core 计算 (本卡 int8 吞吐是 fp16 的约 2 倍)。以全 fp32 的 CPU 实现为基准, llama.cpp 的 logits KL 散度是本实现的约 100 倍 (见第 6 节)

## 2. 测试环境

| 项目 | 版本 / 参数 |
|---|---|
| GPU | NVIDIA RTX 6000D, 85651 MiB 显存, sm_120, 156 SM, L2 112MB; 最大 SM 频率 2430 MHz, 显存频率 12481 MHz, 功耗上限 600W, PCIe 5.0 x16 |
| GPU 实测能力 | 只读带宽 1339 GB/s; tensor core fp16 输入 fp32 累加 142~148 TFLOPS, int8 273 TOPS |
| 驱动 | 595.58.03 (支持 CUDA 13.2) |
| CPU / 内存 | AMD EPYC 9115 16 核 32 线程, 124GB |
| 系统 | Rocky Linux 9.3, 内核 5.14.0-570.58.1.el9_6.x86_64 |
| conda | 25.9.1, 环境 `local_llm`: python 3.13.18, cmake 4.4.3, cuda-nvcc 13.2.86 (conda-forge) |
| local_llm 的 pip 包 | torch 2.15.0+cu132, transformers 5.22.2, tokenizers 0.23.3, numpy 2.5.4, nvidia-cublas 13.4.1.3 |
| 编译器 | 系统 gcc/g++ 11.5.0 (Red Hat 11.5.0-11); musl gcc/g++ 11.2.1 (本实现的 host 代码) |

测试期间同一张卡上还有另一个本实现的服务进程, 一直空闲 (占约 22GB 显存, GPU 利用率 0%), 三家测试都一样。

## 3. 参与对比的实现

### 3.1 本实现

| 版本 | 提交 | KV cache |
|---|---|---|
| KV 16bit | `1bcf3f8` (2026-11-10 14:55 UTC) | fp16 |
| KV 8bit | `7814e25` (2026-11-10 15:58 UTC) | q8_0 |

编译方式见 [README](../README.md) (`./qwen_cuda/compile.sh`, nvcc 13.2.86, 只编译 sm_120)。两个提交只有 KV cache 相关的 kernel 不同, 精度规则见 [cuda_impl.md](cuda_impl.md)。

### 3.2 llama.cpp

- 源码: https://github.com/ggml-org/llama.cpp, 提交 `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5` (2026-11-09 15:48 +0200), 版本 0.9.0 / ggml 0.29.0
- 编译: CUDA 13.2.86 (conda 的 nvcc + cudart), cuBLAS 13.4.1.3 (pip 包 nvidia-cublas), host 编译器系统 g++ 11.5

```bash
cmake -S llama.cpp -B llama_build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCUDAToolkit_ROOT=$CONDA_PREFIX -DCUDA_cublas_LIBRARY=<libcublas.so> -DCUDA_cublasLt_LIBRARY=<libcublasLt.so> \
    -DCMAKE_CUDA_FLAGS="-I<cublas 头文件目录>" -DCMAKE_CXX_FLAGS="-I<cublas 头文件目录>"
cmake --build llama_build -j 32 --target llama-bench
```

- 生效的 CUDA 选项: `GGML_CUDA_GRAPHS=ON` (默认), `GGML_CUDA_FORCE_MMQ=OFF`, `GGML_CUDA_FORCE_CUBLAS=OFF`
- `GGML_NATIVE=OFF` 只影响 CPU 后端: 系统 binutils 的汇编器不认识 `-march=native` 生成的 AVX-VNNI 指令 (`vpdpbusd`)。模型全部层都在 GPU 上, 对测试没有影响
- cuBLAS 在 pip 包里的布局不是标准 CUDA 目录, 所以用软链接把头文件和库接给 CMake; conda 激活时 `NVCC_PREPEND_FLAGS` 带的 `-ccbin` 指向没安装的 conda 编译器, 编译前 `unset`

### 3.3 vLLM

- vLLM 0.33.0 (PyPI), 装在独立的 venv 里 (用 `local_llm` 的 python 3.13.18 创建), 不和 `local_llm` 共用
- 主要依赖: torch 2.14.0 (cu130), flashinfer-python 0.7.2.post1, triton 3.8.0, transformers 5.20.2, compressed-tensors 0.17.0, nvidia-cuda-runtime 13.0.96, nvidia-cublas 13.1.1.3
- 实际使用的 kernel (见启动日志): 线性层 `MarlinLinearKernel` (W4A16), attention 后端 `FLASHINFER` (decode 用 `xqa`), DeltaNet prefill 用 FlashInfer GDN kernel, V2 Model Runner, CUDA Graph (PIECEWISE)

能运行起来需要的环境修正 (都只在这个 venv 内):

| 问题 | 原因 | 处理 |
|---|---|---|
| `libcuda.so cannot found!` | 容器里的 ldconfig 缓存没有收录驱动的 libcuda (文件在 /usr/lib64) | `TRITON_LIBCUDA_PATH=/usr/lib64` |
| `Could not find nvcc` | FlashInfer 需要 JIT 编译 kernel, 默认找 /usr/local/cuda | `CUDA_HOME` 指向 pip 包 `nvidia/cu13` 拼成的目录, PATH 加上它的 bin 和 venv 的 bin (ninja) |
| `CUDA compiler and CUDA toolkit headers are incompatible` | vLLM 的依赖装了 nvcc 13.5.37, 但 runtime 头文件是 13.0.96 | venv 内把 nvidia-cuda-nvcc / nvidia-cuda-crt / nvidia-nvvm 降到 13.0.88 (同时也不超过驱动支持的 13.2) |
| `cannot find -lcudart` | FlashInfer 用 `-L$CUDA_HOME/lib64 -lcudart` 链接, pip 包是 `lib/libcudart.so.13` | 用软链接拼出 `lib64/libcudart.so` 的标准布局 |

## 4. 模型

| | 本实现 / llama.cpp | vLLM |
|---|---|---|
| 来源 | [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 的 `Qwen3.8-27B-Q4_K_M.gguf` (仓库 revision `87410e7b`) | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) (revision `e45c27e8f463817db788d141a73a3c235a0a7c0c`) |
| 校验 | sha256 `880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673` (与 HF 上的 LFS oid 一致) | — |
| 量化 | Q4_K_M 混合精度 + imatrix: 主体 q4_k, 部分 q5_k / q6_k / q8_0 (分布见 [model.md](model.md) 第 13 节) | compressed-tensors W4A16: AWQ + GPTQ, int4 对称, group 128; lm_head、embedding、DeltaNet 的 in_proj_a/b、视觉塔不量化 (bf16) |
| 文件大小 | 17.431 GB (含 MTP 层) | 19.453 GB (含视觉塔和 MTP 层) |
| **decode 每 token 读取的权重** | **16.477 GB** (lm_head q6_k 1.043 GB) | **15.139 GB** (lm_head bf16 2.543 GB) |
| 参数量 | 27.32B (llama-bench 报告, 不含视觉塔) | 同一个模型 |

"decode 每 token 读取的权重" = 文本主干的全部权重 (含 lm_head), 不含 token embedding (每 token 只读一行)、视觉塔和 MTP 层; 按 GGUF / safetensors 头里每个 tensor 的实际字节数统计。vLLM 的模型少读 8.1% (int4 group128 约 4.13 bit/权重, 比 Q4_K_M 的混合精度省, 抵消了 bf16 lm_head 多出的 1.5GB)。

RedHat 这个模型的量化配置自带 `kv_cache_scheme` (fp8, 每个 tensor 一个静态 scale), **vLLM 用默认的 `kv_cache_dtype=auto` 时实际是 fp8 KV**。16bit 测试要显式指定 `kv_cache_dtype=bfloat16` (启动日志 `GPU KV cache size` 的 token 数会减半, 可以据此确认)。

## 5. 速度

### 5.1 测试方法

三家用同一套口径:

- **prefill n**: 从空上下文开始处理 n 个 token, n = 512 / 4096 / 16384 / 65536
- **decode @n**: 先 prefill n 个 token, 再在这个上下文上生成 128 个 token
- 单序列 (batch = 1), prompt 用随机 token, flash attention 开启, 全部层在 GPU 上

| 实现 | 工具与参数 |
|---|---|
| 本实现 | `tests/bench.py --lens 512,4096,16384,65536 --decode 128`: 预热 64 token 后, 每个长度测 1 次。decode 包含 python ctypes 调用和贪心采样 |
| llama.cpp | `llama-bench -ngl 99 -fa 1 -ctk <kv> -ctv <kv> -r 2`: prefill 用 `-p 512,4096,16384,65536 -n 0`, decode 用 `-p 0 -n 128 -d 512,4096,16384,65536`。n_batch 2048 / n_ubatch 512 (默认; ubatch 试过 512 / 1024 / 2048 / 4096, pp4096 分别是 3609 / 3600 / 3511 / 3278, 默认最快)。decode 不做采样 |
| vLLM | offline `LLM` API: `max_num_seqs=1`, `enable_prefix_caching=False`, `gpu_memory_utilization=0.6`, `max_model_len=65728`, `limit_mm_per_prompt={"image": 0, "video": 0}`; 调度默认 chunked prefill, `max_num_batched_tokens=16384`。prefill = n / (`max_tokens=1` 的耗时), decode = 128 / (`max_tokens=129` 的耗时 − `max_tokens=1` 的耗时), 每次用新的随机 prompt, 各测 2 次取最好。包含调度和采样 (temperature = 0) |

KV cache 分两组, 每组内三家都是同样的位数:

| 组 | 本实现 | llama.cpp | vLLM |
|---|---|---|---|
| 16bit | fp16 | f16 | bf16 |
| 8bit | q8_0 (每 32 个元素一个 fp16 scale) | q8_0 (同左) | fp8 e4m3 (每个 tensor 一个静态 scale) |

**换算方法**: decode 是显存带宽瓶颈, 换算到本实现的读取量:

换算后速度 = 实测 × (该实现每 token 读取字节 / 本实现每 token 读取字节)

每 token 读取字节 = 权重 + 每位置 KV 字节 × 上下文长度。每位置 KV 字节 (16 层 × 4 个 kv 头 × 256 维 × K/V): 16bit 65536, q8_0 34816, fp8 32768。

本实现与 llama.cpp 用同一个 GGUF, 每 token 读取字节相同, 换算后不变。有效带宽 = 实测速度 × 每 token 读取字节, 是最直接的可比指标。

prefill 是计算瓶颈, 三家计算量相同 (同一个模型), 不按字节换算, 只给实测值。

### 5.2 KV 16bit

| 实现 | prefill 512 / 4K / 16K / 64K (tok/s) | decode @512 / 4K / 16K / 64K 实测 (tok/s) | decode 换算后 (tok/s) | decode 有效带宽 (GB/s) |
|---|---|---|---|---|
| 本实现 (fp16) | 2315 / 2409 / 2292 / 1920 | **66.1** / **65.5** / **62.6** / 53.1 | **66.1 / 65.5 / 62.6** / 53.1 | 1091 / 1096 / 1100 / 1103 |
| llama.cpp (f16) | **3616 / 3610 / 3325 / 2517** | 64.9 / 64.0 / 61.6 / 52.7 | 64.9 / 64.0 / 61.6 / 52.7 | 1071 / 1072 / 1081 / 1094 |
| vLLM (bf16) | 2619 / 2702 / 2586 / 2195 | 68.5 / 67.8 / 65.7 / 56.9 | 62.9 / 62.3 / 60.7 / **53.3** | 1039 / 1044 / 1066 / 1107 |

### 5.3 KV 8bit

| 实现 | prefill 512 / 4K / 16K / 64K (tok/s) | decode @512 / 4K / 16K / 64K 实测 (tok/s) | decode 换算后 (tok/s) | decode 有效带宽 (GB/s) |
|---|---|---|---|---|
| 本实现 (q8_0) | 2320 / 2408 / 2297 / 1921 | 66.1 / 65.7 / 64.1 / 58.7 | **66.1 / 65.7 / 64.1 / 58.7** | 1091 / 1091 / 1092 / 1101 |
| llama.cpp (q8_0) | **3645 / 3609 / 3325 / 2509** | 64.2 / 63.6 / 61.0 / 52.1 | 64.2 / 63.6 / 61.0 / 52.1 | 1060 / 1057 / 1040 / 978 |
| vLLM (fp8 e4m3) | 2623 / 2703 / 2581 / 2180 | 67.7 / 68.0 / 67.3 / 62.9 | 62.2 / 62.5 / 61.8 / 57.9 | 1026 / 1038 / 1054 / 1087 |

### 5.4 说明

- llama.cpp 把 KV 换成 q8_0 后 decode 反而略慢 (64K 时有效带宽从 1094 降到 978 GB/s), 本实现和 vLLM 换成 8bit 后长上下文明显变快
- vLLM 的实测 decode 最高, 是因为它的模型每 token 少读 8%, 按读取量换算后低于本实现
- 本实现的 decode 计时包含 python 调用和采样, llama-bench 不做采样, 这一项对本实现不利 (约 0.4ms/token, 见 [cuda_impl.md](cuda_impl.md) 第 6 节)

## 6. llama.cpp 的"偷工减料"

### 6.1 做法

llama.cpp 在 CUDA 上对**所有量化权重**的矩阵乘, 都先把输入激活 (fp32) 量化成 q8_1 (每 32 个元素一个 scale, int8), 再做整数点积:

- **prefill** (token 数较多): `ggml_cuda_should_use_mmq` 在有 tensor core 的卡上 (`turing_mma_available`) 对 q4_k / q5_k / q6_k / q8_0 等全部返回 true (`ggml/src/ggml-cuda/mmq.cu:389-443`); `mmq.cu:285` 的 `quantize_mmq_q8_1_cuda` 把激活量化成 q8_1, kernel 里用 int8 tensor core `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`mma.cuh:1096`)
- **decode** (token 数 ≤ 5~8, Blackwell 上 q4_k 是 ≤ 5): 走 MMVQ (`mmvq.cu:319-347`), 同样先 `quantize_row_q8_1_cuda` (`mmvq.cu:1558`) 再做 int8 点积 (dp4a)
- **KV 为 q8_0 时**, decode attention 的 vec kernel 还会把 Q 也量化成 q8_1 再和 K 做点积 (`fattn-vec.cuh:99`)

int8 tensor core 在本卡的吞吐是 fp16 的约 2 倍 (273 TOPS vs 145 TFLOPS), 这就是 llama.cpp prefill 比本实现快 50% 的主要原因。decode 是带宽瓶颈, 激活量化不带来速度优势, 只带来误差。

作为对比, 本实现的 matmul 输入激活是 fp16 (prefill: 权重解成 fp16, tensor core fp16 输入、fp32 累加; decode: half2 乘加, 规则见 [cuda_impl.md](cuda_impl.md) 第 2 节); vLLM 的 Marlin W4A16 激活也是 16bit, 不做激活量化。

### 6.2 精度实测

基准: 本实现的 CPU 版本。它读同一个 GGUF, 权重反量化后全部用 fp32 计算, 与 CUDA 版逐层对比的误差在 1e-6 量级 (全 fp32 的 CUDA 版), 可以当作"这个 GGUF 的精确结果"。

输入: 真实文本 (llama.cpp 的 docs/*.md + 本仓库 docs/*.md, 用 HF tokenizer 分词, 取第 5000 个 token 起的一段)。一次 prefill 1100 个 token (本实现跨过 1024 的 chunk 边界), 再 teacher forcing 逐个 decode 32 个 token, 每一步都比较最后位置的 logits。

llama.cpp 用 libllama 的 API 跑同样的 token: 全部层在 GPU, flash attention 开启, n_batch / n_ubatch 为默认的 2048 / 512, prefill 一次 `llama_decode` 提交。

| 实现 | prefill 后: KL / logits 相对误差 / top1 | decode 32 步 KL: 平均 (最大) | decode 32 步 logits 相对误差: 平均 (最大) | decode top1 一致 |
|---|---|---|---|---|
| 本实现 CUDA (KV fp16) | **2.45e-5** / **4.6e-3** / ✅ | **5.5e-6** (3.4e-5) | **5.3e-3** (1.8e-2) | 32/32 |
| llama.cpp CUDA (KV f16) | 2.78e-3 / 4.2e-2 / ✅ | 4.9e-4 (2.8e-3) | 2.8e-2 (5.5e-2) | 31/32 |
| 本实现 CUDA (KV q8_0) | 8.6e-5 / 7.8e-3 / ✅ | 2.2e-5 (1.5e-4) | 9.2e-3 (3.3e-2) | 32/32 |
| llama.cpp CUDA (KV q8_0) | 2.74e-3 / 5.5e-2 / ✅ | 6.1e-4 (4.0e-3) | 3.1e-2 (5.3e-2) | 31/32 |

KL = KL(p_cpu ‖ p_x), 相对误差 = ‖logits_x − logits_cpu‖ / ‖logits_cpu‖。

- 同为 16bit KV, llama.cpp 的 KL 散度是本实现的约 100 倍 (prefill 后 113 倍, decode 平均 89 倍), logits 误差是 5~9 倍
- 本实现即使用 q8_0 KV, 误差也只有 llama.cpp 用 f16 KV 时的约 1/20 到 1/30
- 绝对值上, llama.cpp 的 KL 散度 (约 5e-4 ~ 3e-3) 仍远小于 4bit 权重量化本身带来的损失, 日常使用不一定能察觉; 但它的 prefill 速度优势是用这部分精度换来的, 不是同等精度下的比较

## 7. 其他注意事项

- 两种 4bit 权重的量化方法不同 (Q4_K_M + imatrix vs AWQ + GPTQ int4 group128), 模型质量也会不同, 本文只比较速度; 第 6 节的精度对比只在本实现和 llama.cpp 之间进行 (同一个 GGUF, 有共同的 fp32 基准)
- vLLM 的 8bit KV 是 fp8 e4m3 + 每 tensor 静态 scale, 本实现和 llama.cpp 是 q8_0 (每 32 个元素一个 scale), 位数相同但格式不同
- 只测了单序列; vLLM 的强项 (连续批处理、多请求并发、prefix cache) 不在本次对比范围
- 没有测投机解码 (MTP), 三家都没开
