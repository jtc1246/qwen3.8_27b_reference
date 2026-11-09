# Qwen3.8-27B 参考实现 (纯 CPU)

从零实现的 Qwen3.8-27B 推理。C++ 部分没有第三方依赖, 直接读取 GGUF 量化模型 (目前使用 [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 的 `Q4_K_M`), 完成文本主干的前向和采样; python 部分负责 tokenizer、图片和 OpenAI 兼容服务。

## 文档

- [docs/model.md](docs/model.md): 模型结构、计算公式、GGUF 格式与 Q4_K_M 量化方式
- [docs/cpu_impl.md](docs/cpu_impl.md): C++ 实现细节

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers safetensors jinja2 numpy pillow
```

只编译 C++ 和跑冒烟测试只需要 cmake; python 包是 OpenAI 服务 (tokenizer、图片) 用的。

## 编译

```bash
./qwen_cpu/compile.sh
```

产物是 `qwen_cpu/cmake_build/libqwen.so` (musl 全静态, 无外部 so 依赖)。

## 运行

### 需要的文件

- GGUF: `Qwen3.8-27B-Q4_K_M.gguf`
- 官方仓库 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 tokenizer/processor 配置文件 (`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`preprocessor_config.json`、`config.json`、`model.safetensors.index.json` 等), 以及包含视觉塔权重的 `model-00001-of-00018.safetensors` (只需要这一个分片)

### OpenAI 兼容服务

```bash
python -u qwen_cpu/server.py --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf --hf-dir /path/to/Qwen3.8-27B --port 8000
python -u tests/test_server.py --base http://127.0.0.1:8000/v1
```

### 冒烟测试 (只用标准库)

```bash
python -u tests/smoke_test.py --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf --think --temp 1.0
```

## python 部分

- `qwen_cpu/qwen.py`: `libqwen.so` 的 ctypes 封装。输入 token 序列 (可附带图片 embedding 和 3D 位置), `generate()` 逐个返回采样出的 token id
- `qwen_cpu/server.py`: OpenAI 兼容服务
  - 接口: `POST /v1/chat/completions` (支持 `stream`), `GET /v1/models`
  - 支持的参数: `messages` (system/user/assistant, 文本 + `image_url` 图片, 支持 data URL 和 http URL), `temperature`, `seed`, 思考控制 (`enable_thinking` 或 `chat_template_kwargs.enable_thinking`; `reasoning_effort`: none/minimal 关闭思考, low/medium/high/xhigh), 其余参数忽略
  - 默认采样参数与官方推荐一致 (思考 / 非思考两套), `temperature` 只覆盖温度; 思考内容在 `reasoning_content` 字段返回
  - tokenizer、chat 模板、图片预处理用 transformers 的官方 processor, 视觉塔用 transformers 的模块加载官方权重
  - 启动时先加载完所有内容再开始监听; 推理是单序列, 并发请求排队串行处理, 不会报错; 流式请求的客户端断开会立即停止生成
  - 图片默认最多 1048576 像素 (约 1024 token), `--max-image-pixels 0` 恢复官方默认
- `tests/smoke_test.py`: 不依赖第三方库, 用 GGUF 词表做贪心匹配分词 (不是真正的 BPE), 只用于验证能跑
- `tests/test_server.py`: 服务端到端测试 (system、思考开关、流式、图片、seed、并发)
