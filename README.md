# Qwen3.8-27B 参考实现 (CPU / CUDA)

从零实现的 Qwen3.8-27B 推理。C++ 部分没有第三方依赖, 直接读取 GGUF 量化模型 (目前使用 [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 的 `Q4_K_M`), 完成文本主干的前向和采样; python 部分负责 tokenizer、图片和 OpenAI 兼容服务。

## 文档

- [docs/model.md](docs/model.md): 模型结构、计算公式、GGUF 格式与 Q4_K_M 量化方式
- [docs/cpu_impl.md](docs/cpu_impl.md): C++ 实现细节
- [docs/cuda_impl.md](docs/cuda_impl.md): CUDA 实现 (`qwen_cuda/`, 只依赖驱动的 driver API) 与 CPU 版的差异、验证和性能
- [docs/benchmark.md](docs/benchmark.md): 与 llama.cpp、vLLM 的速度对比 (环境、版本、方法), 以及 llama.cpp 激活量化的精度实测
- [docs/python_impl.md](docs/python_impl.md): python 部分 (服务、路由、tokenizer、拼 prompt、缓存) 的完整实现和内部机制
- [docs/custom_sampler.md](docs/custom_sampler.md): 自定义采样 (CUDA) 的使用方式 (接口、代码写法、token 范围、样例) 和内部实现

## 环境准备

以下假设是一台普通的 x86_64 Linux, 没有 musl, 已安装 conda。

**1. 确认 CPU 支持 AVX-512** (需要 avx512f/bw/dq/vl, 输出 4 行即可):

```bash
grep -o -w -E 'avx512f|avx512bw|avx512dq|avx512vl' /proc/cpuinfo | sort -u
```

**2. musl 工具链** (gcc 11.2.1 + musl libc, 编译器本身是静态链接的, 解压即可用, 不依赖系统 glibc 版本):

- 下载: https://musl.cc/x86_64-linux-musl-native.tgz (约 85MB)

```bash
mkdir -p ~/opt && cd ~/opt
wget https://musl.cc/x86_64-linux-musl-native.tgz
tar -xzf x86_64-linux-musl-native.tgz
echo 'export MUSL_BIN=$HOME/opt/x86_64-linux-musl-native/bin' >> ~/.bashrc
source ~/.bashrc
$MUSL_BIN/g++ --version
```

`compile.sh` 通过环境变量 `MUSL_BIN` 找编译器, CMake 中写的是 `$ENV{MUSL_BIN}/g++`。

**3. conda 环境** (环境名需要是 `local_llm`, `compile.sh` 会激活它; 用 conda-forge, 不需要 Anaconda 默认 channel):

```bash
conda create -n local_llm -c conda-forge --override-channels python=3.13 cmake
conda activate local_llm
# 二选一:
pip install -r requirements_cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu # 只用 --type cpu
pip install -r requirements_cuda.txt --extra-index-url https://download.pytorch.org/whl/cu132 # 要用 --type cuda (也能跑 cpu)
```

只编译 C++ 和跑冒烟测试只需要 cmake; python 包是 OpenAI 服务 (tokenizer、图片) 用的。

两个 requirements 文件是实际测试通过的环境的 `pip freeze`, 版本全部固定。区别只在 torch:

- `requirements_cpu.txt`: `torch==2.15.0+cpu`, `torchvision==0.29.2+cpu`, 视觉塔只能在 CPU 上跑
- `requirements_cuda.txt`: `torch==2.15.0+cu132`, `torchvision==0.29.2+cu132`, 外加 CUDA 运行时相关的 `nvidia-*`、`cuda-*`、`triton` (约 2.5GB)。cu132 对应 CUDA 13.2, 不能高于驱动支持的 CUDA 版本 (`nvidia-smi` 右上角), 驱动较旧时要换成对应版本的 torch

`+cpu` / `+cu132` 版本只在 PyTorch 自己的源上有, 所以需要 `--extra-index-url`。其余主要版本: python 3.13, transformers 5.22.2, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.4, pillow 13.0.0。

**4. 自定义采样 (可选, 只有 cuda)**: 不用这个功能可以跳过, 其余功能不受影响。需要两样东西:

- WAMR 源码: 仓库里的 `resources/WAMR-2.4.5.tar.gz` (编译 CUDA 版时用)
- 编译镜像: [sampler_rootfs.tar.7z](https://github.com/jtc1246/qwen3.8_27b_reference/releases/download/wasm/sampler_rootfs.tar.7z) (运行服务时用, 解压后是 `sampler_rootfs.tar`)

```bash
# 在仓库根目录执行
mkdir -p ~/opt
tar -xzf resources/WAMR-2.4.5.tar.gz -C ~/opt
echo 'export WAMR_DIR=$HOME/opt/wasm-micro-runtime-WAMR-2.4.5' >> ~/.bashrc
wget https://github.com/jtc1246/qwen3.8_27b_reference/releases/download/wasm/sampler_rootfs.tar.7z
7z x sampler_rootfs.tar.7z
```

运行服务的机器需要有 `tar`、`chroot`、`unshare` 命令, 并且有权限使用 `chroot` 和 `unshare --net`。

## 编译

```bash
./qwen_cpu/compile.sh
```

产物是 `qwen_cpu/cmake_build/libqwen.so` (musl 全静态, 无外部 so 依赖)。

CUDA 版本 (需要 NVIDIA 驱动, 编译时还需要 nvcc, 版本不能高于驱动支持的 CUDA 版本; 目前 kernel 只编译 sm_120):

```bash
conda install -n local_llm --override-channels -c conda-forge cuda-nvcc=13.2.86
export WAMR_DIR=/path/to/wasm-micro-runtime-WAMR-2.4.5 # 可选 (环境准备第 4 步), 不设置则编译出的 so 不支持自定义采样
./qwen_cuda/compile.sh
```

产物是 `qwen_cuda/cmake_build/libqwen_cuda.so`, 唯一的动态依赖是驱动的 `libcuda.so.1`。

## 运行

### 需要的文件

- GGUF: `Qwen3.8-27B-Q4_K_M.gguf`
- 官方仓库 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 tokenizer/processor 配置文件 (`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`preprocessor_config.json`、`config.json`、`model.safetensors.index.json` 等), 以及包含视觉塔权重的 `model-00001-of-00018.safetensors` (只需要这一个分片)

### OpenAI 兼容服务

```bash
python -u server.py --type cuda --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf --hf-dir /path/to/Qwen3.8-27B --port 8000
python -u tests/test_server.py --base http://127.0.0.1:8000/v1
```

`--type` 选择后端: `cpu` (默认, `--threads` 指定线程数) 或 `cuda` (`--device` 指定 GPU, 文本主干和视觉塔都在 GPU 上)。`--port 0` 由系统自动分配端口, 实际端口打印在启动完成的日志里。`--ctx` 默认 cpu 32768, cuda 262144 (cuda 是每个槽的上下文, 启动时分配 4 个槽的 KV, 约 52GB 显存)。`--cache-dir <目录>` 启用磁盘前缀缓存 (cpu 和 cuda 都支持, 直接使用该目录, 见 docs/cpu_impl.md 9.1 节和 docs/cuda_impl.md 4.2 节); 通过 router 启动时由 router 统一传入。`--sampler-rootfs` 见下面的自定义采样。完整的参数和行为见 [docs/python_impl.md](docs/python_impl.md)。

### 自定义采样 (只有 cuda)

请求里带一段 C++ 代码 (`sampler_cpp`, 可选二进制参数 `sampler_data`), 每生成一个 token 调用一次, 代替内置采样。启动时加 `--sampler-rootfs /path/to/sampler_rootfs.tar` (镜像见环境准备第 4 步), 同时需要 `--cache-dir`; 经 router 启动时同样直接传给 router。没有配置时只有带 `sampler_cpp` 的请求报错, 其余请求不受影响。

接口、代码写法、token 范围和样例见 [docs/custom_sampler.md](docs/custom_sampler.md)。

```bash
python -u tests/test_wasm_sampler.py --base http://127.0.0.1:8000/v1 --hf-dir /path/to/Qwen3.8-27B
```

### 多后端路由

```bash
python -u router.py --mask 11 --port 8000 --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf --hf-dir /path/to/Qwen3.8-27B --cache-dir /path/to/cache_root
python -u tests/test_router.py --base http://127.0.0.1:8000/v1
python -u tests/test_server.py --base http://127.0.0.1:8000/v1 --model qwen3.8-27b-cuda # 经路由测某个后端
```

同时启动多个后端, 对外是一个 OpenAI 兼容服务, 按请求里的 `model` 转发。

### 冒烟测试 (只用标准库)

```bash
python -u tests/smoke_test.py --type cuda --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf --think --temp 1.0
```

## python 部分

python 代码在仓库根目录, 与具体后端无关 (各后端只有 C++ 库, C 接口相同)。完整的实现和内部机制见 [docs/python_impl.md](docs/python_impl.md)。

- `server.py`: OpenAI 兼容服务 (单个后端), 用法见上面的"运行"
  - 接口: `POST /v1/chat/completions` (支持 `stream`), `GET /v1/models`, `GET /health`; 模型名是 `qwen3.8-27b-{cpu,cuda}`
  - 支持的参数: `messages` (system / user / assistant, 文本 + `image_url` 图片, 只支持 base64 的 data URL), `temperature`, `seed`, 思考控制 (`enable_thinking` 或 `chat_template_kwargs.enable_thinking`; `reasoning_effort`: none / minimal 关闭思考, low / medium / high / xhigh), 自定义采样 (`sampler_cpp` / `sampler_data`, 只有 cuda), 其余参数忽略
  - 返回: 思考内容在 `reasoning_content`, 正文在 `content`, 都是模型原样生成的文本 (不做 strip); `usage` 里另有前缀缓存的命中 / 写入统计
  - 默认采样参数与官方推荐一致, `temperature` 只覆盖温度; cuda 最多 4 个请求同时进行, 更多的排队; cpu 串行处理
- `router.py`: 同时启动多个后端, 对外是一个服务, 按请求里的 `model` 转发 (用法见上面的"多后端路由")
- `qwen.py`: C++ 库的 ctypes 封装, 可以不起服务直接在 python 里推理: `qwen.load(gguf, backend)` 后用 `qwen.generate(token 序列, thinking=...)` 逐个得到输出 token (例子见 `tests/smoke_test.py`)
- `qwen_tok.py`: tokenizer, 只用标准库: `qwen_tok.load("tokenizer.json")`, `encode(text)` / `decode(ids)` / `token_bytes(id)`
- `wasm_sampler.py`: 自定义采样代码的编译和缓存 (server 内部使用)
- 测试 (都需要先启动对应的服务, `--base` 指定地址; 经路由测单个后端时用 `--model` 指定模型名):
  - `tests/test_server.py`: 服务端到端测试
  - `tests/test_router.py`: 路由测试
  - `tests/test_wasm_sampler.py`: 自定义采样测试 (另需 `--hf-dir`)
- 其他测试 (直接加载 C++ 库, 不需要服务): `tests/smoke_test.py` (冒烟测试), `tests/cuda_vs_cpu.py` (CPU 与 CUDA 逐层数值对比), `tests/bench.py` (prefill / decode 速度)
