# Qwen3.8-27B 参考实现 (CPU / CUDA)

从零实现的 Qwen3.8-27B 推理。C++ 部分没有第三方依赖, 直接读取 GGUF 量化模型 (目前使用 [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 的 `Q4_K_M`), 完成文本主干的前向和采样; python 部分负责 tokenizer、图片和 OpenAI 兼容服务。

## 文档

- [docs/model.md](docs/model.md): 模型结构、计算公式、GGUF 格式与 Q4_K_M 量化方式
- [docs/cpu_impl.md](docs/cpu_impl.md): C++ 实现细节
- [docs/cuda_impl.md](docs/cuda_impl.md): CUDA 实现 (`qwen_cuda/`, 只依赖驱动的 driver API) 与 CPU 版的差异、验证和性能
- [docs/benchmark.md](docs/benchmark.md): 与 llama.cpp、vLLM 的速度对比 (环境、版本、方法), 以及 llama.cpp 激活量化的精度实测

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

## 编译

```bash
./qwen_cpu/compile.sh
```

产物是 `qwen_cpu/cmake_build/libqwen.so` (musl 全静态, 无外部 so 依赖)。

CUDA 版本 (需要 NVIDIA 驱动, 编译时还需要 nvcc, 版本不能高于驱动支持的 CUDA 版本; 目前 kernel 只编译 sm_120):

```bash
conda install -n local_llm --override-channels -c conda-forge cuda-nvcc=13.2.86
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

`--type` 选择后端: `cpu` (默认, `--threads` 指定线程数) 或 `cuda` (`--device` 指定 GPU, 文本主干和视觉塔都在 GPU 上)。`--port 0` 由系统自动分配端口, 实际端口打印在启动完成的日志里。`--ctx` 默认 cpu 32768, cuda 262144 (cuda 是每个槽的上下文, 启动时分配 4 个槽的 KV, 约 52GB 显存)。`--cache-dir <目录>` 启用磁盘前缀缓存 (cpu 和 cuda 都支持, 直接使用该目录, 见 docs/cpu_impl.md 9.1 节和 docs/cuda_impl.md 4.2 节); 通过 router 启动时由 router 统一传入。

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

python 代码在仓库根目录, 与具体后端无关; 各后端 (`qwen_cpu/`、`qwen_cuda/`) 只有 C++ 库, C 接口相同。

- `qwen.py`: C++ 库的 ctypes 封装。`load(gguf, backend, n_threads, device, max_ctx)` 按 `backend` 选择 `LIBS` 里的 so (新增后端只需在 `LIBS` 加一项); 输入 token 序列 (可附带图片 embedding 和 3D 位置), `generate()` 逐个返回采样出的 token id。一个模块实例只加载一个后端, 需要同时用两个时用 importlib 把 `qwen.py` 加载成两个模块 (见 `tests/cuda_vs_cpu.py`)。cuda 另有引擎模式 `engine_*` (server.py 用), 与 `eval_tokens` 等不能混用
- `qwen_tok.py`: tokenizer 自己实现, 只用标准库: 读 `tokenizer.json` 的 vocab / merges / added token; `encode` = NFC → 预分词 (手写, 等价于官方正则) → byte-level → BPE, 只查 BPE 词表, 不会产生控制 token; `decode` 把 id 还原成字节再按 utf-8 解码 (不完整的字节换成 U+FFFD)。与 HF tokenizers 对比: 真实文本 22 万 token、全部 Unicode 码位 (单独 / 带上下文)、随机 id 的 decode 都相同; 只有字符表版本不同的码位有差别 (Python 3.13 标准库是 Unicode 15.1: 4433 个 Unicode 16 新增字符在这里不算字母; 59 个罕见组合符号的 NFC 重排 HF 没做)。19 万 token 首次 0.25s, 重复内容有缓存
- `server.py`: OpenAI 兼容服务 (单个后端)
  - 接口: `POST /v1/chat/completions` (支持 `stream`), `GET /v1/models`, `GET /health`
  - 模型名: `{--model-name}-{--type}`, 默认 `qwen3.8-27b-cpu` / `qwen3.8-27b-cuda`, 只用于 `/v1/models` 和响应里的 `model` 字段, 请求里的 `model` 不校验。`/v1/models` 返回 `{"object": "list", "data": [{"id": "qwen3.8-27b-cuda", "object": "model", "created": <请求时的时间戳>, "owned_by": "local"}]}`
  - 支持的参数: `messages` (system/user/assistant, 文本 + `image_url` 图片, 只支持 base64 的 data URL `data:<类型>;base64,<数据>`, 不支持链接, 不会因为图片发出网络请求), `temperature`, `seed`, 思考控制 (`enable_thinking` 或 `chat_template_kwargs.enable_thinking`; `reasoning_effort`: none/minimal 关闭思考, low/medium/high/xhigh), 其余参数忽略
  - 默认采样参数与官方推荐一致 (思考 / 非思考两套), `temperature` 只覆盖温度; 思考内容在 `reasoning_content` 字段返回。文中控制 token 都按 id 记: im_start 248045、im_end 248046、endoftext 248044、思考开始 248068、思考结束 248069、图片开始 / 图片占位 / 图片结束 248053 / 248056 / 248054。返回的文本都是模型原样生成的, 不做任何 strip: 开思考时 `reasoning_content` 是思考结束 token 之前生成的全部内容 (通常以模型生成的 `\n` 结尾), `content` 是它之后生成的全部内容 (通常以模型生成的 `\n\n` 开头); 思考结束 token 本身只用来划分两个字段, 停止 token 只用来决定 finish_reason
  - tokenizer 和 chat 模板自己实现 (`qwen_tok.py` + `server.py` 的 `build_tokens`, 不用任何第三方库); 图片预处理用 transformers 的官方 image processor, 视觉塔用 transformers 的模块加载官方权重
  - 启动时先加载完所有内容再开始监听; 流式请求的客户端断开会立即停止生成
  - 流式: 开头一个 role chunk, 之后每个输出 token (含结尾的停止 token) 正好一个 chunk, 没有新增文本 (停止 token、思考结束 token、多字节字符还不完整) 时发空串, 最后一个 chunk 带 finish_reason。解码按字节 (标准库 utf-8 增量解码器): 还不完整的字节先攒着, 遇到思考结束 token、停止 token 或输出截断时把剩下的吐出来 (不完整的换成 U+FFFD, 不丢)
  - usage (所有字段始终存在, 没有缓存时为 0): `prompt_tokens` 输入总数, `completion_tokens` 输出数 (含停止 token), `total_tokens`; `prompt_tokens_details`: `cached_tokens` (命中的缓存, 到所用 state 为止), `uncached_tokens` (= prompt_tokens - cached_tokens), `cache_written_tokens` (输入部分新写入缓存的 KV token 数, 只到最后一个缓存点, 之后的 prompt 末尾算在输出缓存点里), `cache_written_states` (输入部分新写入的 state 数); `completion_tokens_details.cache_written_states` (输出缓存点写入的 state 数, 0 或 1)。写入数是实际写成功的。流式: 请求带 `stream_options.include_usage` 时按 OpenAI 规范在 `[DONE]` 前单独发一个 `choices` 为空的 usage chunk, 否则 usage 附在最后一个带 finish_reason 的 chunk 上
  - cpu: 推理是单序列, 并发请求排队串行处理, 不会报错
  - cuda: 最多 4 个请求同时进行, 更多的按到达顺序排队; 输入 (prefill) 按到达顺序逐个算, 输出 (decode) 所有请求每步一起算; prefill 和 decode 同时运行时按 SM 分区 (`--dec-ratio`, decode 默认占 0.31, 本卡 156 SM 时为 48 个, 其余给 prefill), 只有一种时用整卡。同一请求 (固定 seed) 单独跑和并发跑的输出完全相同
  - 图片大小上限默认与官方相同 (最多 16777216 像素, 约 16384 token), `--max-image-pixels` 可以调小
  - 拼 prompt (`build_tokens`): 按 chat_template.jinja 的结构 (不含 tools) 在 python 里直接拼 token 序列, 控制 token (im_start、im_end、思考开始 / 结束、图片开始 / 结束等) 只由结构代码按 id 插入, 消息正文 (content、文本 part、reasoning_content) 只用 `qwen_tok.encode` 按文本分词, 正文里任何字符串都只是普通文本, 不可能变成控制 token; 图片占位 token 按图片 grid 展开。与官方模板的区别: 正文不做 trim (原样), 也不补模板为 trim 加的换行; 不做任何空字符串判断 (有 system 消息就有 system 块, 内容为空也一样; text part 必须有字符串 text, 否则 400); 历史 assistant 消息还原成当时的生成提示 + 原样内容: 有 `reasoning_content` 字段 (字符串, 空串也算) 时是 [思考开始] `\n` + reasoning_content + [思考结束] + content, 否则是 [思考开始] `\n\n` [思考结束] `\n\n` + content。客户端原样回传本服务返回的内容时, 历史 assistant 段与生成时的 token 相同 (多轮能命中输出缓存点); 不回传 reasoning_content 时那一轮回复对不上 (当轮命中不到它), 但之后各轮的还原方式一致, 能用上前一轮写下的缓存点
  - 前缀缓存 (`--cache-dir`, cpu 和 cuda 都支持, 实现分别见 docs/cpu_impl.md 9.1 节和 docs/cuda_impl.md 4.2 节): 缓存点在每条消息末尾的 im_end token 之后, 存到磁盘, 进程重启后仍可用; 同一目录同时只能被一个进程使用 (文件锁)。多轮对话命中 assistant 消息的前提是客户端原样回传它 (开 thinking 时包括 `reasoning_content`), 且重新分词后的 token 与生成时相同 (实测 4323 个生成 token 用 BPE 重新分词只有 2 个不同)
  - 视觉塔 fp32, `--type cuda` 时在 GPU 上, 否则在 CPU 上
  - 图片缓存: 视觉塔的输出按图片原始字节 (base64 解码后) 的 sha256 缓存, 只在内存里, 不写文件; 一个请求里命中的图片直接用缓存, 未命中的图片一起过视觉塔再按图片拆开存入。按 LRU 淘汰, 上限 `--image-cache-mb` (默认 4096MB, 每张 1024 token 的图约 20MB), 0 表示不缓存。图片预处理 (resize、切 patch) 每次仍会执行, 因为需要它确定图片 token 数
- `router.py`: 多后端路由, 后端列表都取自 `qwen.LIBS`, 新增后端不需要改路由
  - `--mask`: 0/1 串, 第 i 位对应 `qwen.LIBS` 的第 i 个后端 (目前依次是 `cpu`、`cuda`), 1 表示启动; 默认全 1 (现在是 `11`)。位数不对、一个都没选都直接报错
  - 每个选中的后端启动一个 `server.py --type <后端> --host 127.0.0.1 --port 0` 子进程, 实际端口从子进程日志中解析; 除 `--host`、`--port`、`--mask`、`--start-timeout`、`--cache-dir` 外的参数 (`--gguf`、`--hf-dir`、`--ctx`、`--threads`、`--device` 等) 原样传给每个子进程, 不能传 `--type`
  - `--cache-dir` (必填, 没有默认值): 前缀缓存根目录, 不存在则创建 (上级目录必须存在); 支持缓存的后端 (`qwen.CACHE_BACKENDS`, 目前是 cpu 和 cuda) 收到 `--cache-dir <根目录>/<后端名>`, 其余后端不传; 子目录由 server 自己检查和创建
  - 子进程日志加 `[后端名]` 前缀转发到路由的 stderr
  - 启动阶段同时等待所有子进程, 任何一个启动失败 (没有 GPU、显存不够、参数错误等, 子进程退出) 或超过 `--start-timeout` (默认 600s) 未就绪, 立即结束全部子进程并报错退出; 运行中任一子进程退出, 路由也整体退出。子进程设置了 `PR_SET_PDEATHSIG`, 路由被 kill 时子进程也会被结束
  - `GET /v1/models`: 返回所有已启动后端的模型 (各子进程自己报告的名字), 例如 `--mask 11` 时是 `qwen3.8-27b-cpu` 和 `qwen3.8-27b-cuda`
  - `POST /v1/chat/completions`: 把 `model` 转小写后, 检查包含哪些后端名; 恰好包含一个且该后端已启动时原样转发 (流式边收边回, 客户端断开会传到后端并停止生成), 一个都不包含、包含多个或后端没启动都返回 400
- `tests/smoke_test.py`: 不依赖第三方库, 用 GGUF 词表做贪心匹配分词 (不是真正的 BPE), 只用于验证能跑; `--type` 选择后端
- `tests/cuda_vs_cpu.py`: CPU 与 CUDA 后端逐层数值对比
- `tests/bench.py`: 不同上下文长度的 prefill / decode 速度 (`--type`, 默认 cuda)
- `tests/test_server.py`: 服务端到端测试 (system、思考开关、流式、图片、图片缓存、多图、seed、并发); `--model` 指定请求里的模型名 (经路由测试时用)
- `tests/test_router.py`: 路由测试 (模型列表、按模型名转发、大小写、不匹配 / 匹配多个时报错)
