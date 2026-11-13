# Python 部分实现

python 代码在仓库根目录, 与具体后端无关; 各后端 (`qwen_cpu/`、`qwen_cuda/`) 只有 C++ 库, C 接口相同。用法见 README, 自定义采样见 [custom_sampler.md](custom_sampler.md)。

文中控制 token 都按 id 记: im_start 248045、im_end 248046、endoftext 248044、思考开始 248068、思考结束 248069、图片开始 / 图片占位 / 图片结束 248053 / 248056 / 248054。

## 1. qwen.py

C++ 库的 ctypes 封装。

- `load(gguf, backend, n_threads, device, max_ctx)` 按 `backend` 选择 `LIBS` 里的 so (新增后端只需在 `LIBS` 加一项); 输入 token 序列 (可附带图片 embedding 和 3D 位置), `generate()` 逐个返回采样出的 token id
- 一个模块实例只加载一个后端, 需要同时用两个时用 importlib 把 `qwen.py` 加载成两个模块 (见 `tests/cuda_vs_cpu.py`)
- cuda 另有引擎模式 `engine_*` (server.py 用), 与 `eval_tokens` 等不能混用: `engine_args` 预先准备好 `qwen_prefill` 的全部参数 (拿到 ticket 之后不能再出错), `engine_prefill` 按 ticket 顺序 prefill 并采样第一个 token, `engine_next` 阻塞读取新 token, `engine_cancel` 取消, `engine_cache_stats` / `engine_error` 在请求结束后取缓存统计和自定义采样的错误

## 2. qwen_tok.py

tokenizer 自己实现, 只用标准库。

- 读 `tokenizer.json` 的 vocab / merges / added token
- `encode` = NFC → 预分词 (手写, 等价于官方正则) → byte-level → BPE, 只查 BPE 词表, 不会产生控制 token
- `decode` 把 id 还原成字节再按 utf-8 解码 (不完整的字节换成 U+FFFD); `token_bytes(id)` 返回一个 id 的原始字节
- 与 HF tokenizers 对比: 真实文本 22 万 token、全部 Unicode 码位 (单独 / 带上下文)、随机 id 的 decode 都相同; 只有字符表版本不同的码位有差别 (Python 3.13 标准库是 Unicode 15.1: 4433 个 Unicode 16 新增字符在这里不算字母; 59 个罕见组合符号的 NFC 重排 HF 没做)
- 19 万 token 首次 0.25s, 重复内容有缓存

## 3. server.py

OpenAI 兼容服务 (单个后端)。

### 3.1 接口和参数

- 接口: `POST /v1/chat/completions` (支持 `stream`), `GET /v1/models`, `GET /health`
- 模型名: `{--model-name}-{--type}`, 默认 `qwen3.8-27b-cpu` / `qwen3.8-27b-cuda`, 只用于 `/v1/models` 和响应里的 `model` 字段, 请求里的 `model` 不校验。`/v1/models` 返回 `{"object": "list", "data": [{"id": "qwen3.8-27b-cuda", "object": "model", "created": <请求时的时间戳>, "owned_by": "local"}]}`
- 支持的参数: `messages` (system/user/assistant, 文本 + `image_url` 图片, 只支持 base64 的 data URL `data:<类型>;base64,<数据>`, 不支持链接, 不会因为图片发出网络请求), `temperature`, `seed`, 思考控制 (`enable_thinking` 或 `chat_template_kwargs.enable_thinking`; `reasoning_effort`: none/minimal 关闭思考, low/medium/high/xhigh), 自定义采样 (`sampler_cpp` / `sampler_data`, 只有 cuda, 见 [custom_sampler.md](custom_sampler.md)), 其余参数忽略
- 启动时先加载完所有内容再开始监听; 流式请求的客户端断开会立即停止生成

### 3.2 采样和返回文本

- 默认采样参数与官方推荐一致 (思考 / 非思考两套), `temperature` 只覆盖温度; 只采样普通文本 token, 加上思考中的思考结束 token 或不在思考中时的 im_end (思考中不会直接 im_end, 所以开 thinking 的回复总是 思考 → 思考结束 → 正文 → im_end, 多轮还原时与生成的 token 一致), 其余控制 token (包括 endoftext) 和输出层的填充位都不会被采样; 思考内容在 `reasoning_content` 字段返回
- 返回的文本都是模型原样生成的, 不做任何 strip: 开思考时 `reasoning_content` 是思考结束 token 之前生成的全部内容 (通常以模型生成的 `\n` 结尾), `content` 是它之后生成的全部内容 (通常以模型生成的 `\n\n` 开头); 思考结束 token 本身只用来划分两个字段, 停止 token 只用来决定 finish_reason
- 流式: 开头一个 role chunk, 之后每个输出 token (含结尾的停止 token) 正好一个 chunk, 没有新增文本 (停止 token、思考结束 token、多字节字符还不完整) 时发空串, 最后一个 chunk 带 finish_reason。解码按字节 (标准库 utf-8 增量解码器): 还不完整的字节先攒着, 遇到思考结束 token、停止 token 或输出截断时把剩下的吐出来 (不完整的换成 U+FFFD, 不丢)
- usage (所有字段始终存在, 没有缓存时为 0): `prompt_tokens` 输入总数, `completion_tokens` 输出数 (含停止 token), `total_tokens`; `prompt_tokens_details`: `cached_tokens` (命中的缓存, 到所用 state 为止), `uncached_tokens` (= prompt_tokens - cached_tokens), `cache_written_tokens` (输入部分新写入缓存的 KV token 数, 只到最后一个缓存点, 之后的 prompt 末尾算在输出缓存点里), `cache_written_states` (输入部分新写入的 state 数); `completion_tokens_details.cache_written_states` (输出缓存点写入的 state 数, 0 或 1)。写入数是实际写成功的。流式: 请求带 `stream_options.include_usage` 时按 OpenAI 规范在 `[DONE]` 前单独发一个 `choices` 为空的 usage chunk, 否则 usage 附在最后一个带 finish_reason 的 chunk 上

### 3.3 并发

- cpu: 推理是单序列, 并发请求排队串行处理, 不会报错
- cuda: 最多 4 个请求同时进行, 更多的按到达顺序排队; 输入 (prefill) 按到达顺序逐个算, 输出 (decode) 所有请求每步一起算; prefill 和 decode 同时运行时按 SM 分区 (`--dec-ratio`, decode 默认占 0.31, 本卡 156 SM 时为 48 个, 其余给 prefill), 只有一种时用整卡。同一请求 (固定 seed) 单独跑和并发跑的输出完全相同

### 3.4 拼 prompt (`build_tokens`)

- tokenizer 和 chat 模板自己实现 (`qwen_tok.py` + `build_tokens`, 不用任何第三方库)
- 按 chat_template.jinja 的结构 (不含 tools) 在 python 里直接拼 token 序列, 控制 token (im_start、im_end、思考开始 / 结束、图片开始 / 结束等) 只由结构代码按 id 插入, 消息正文 (content、文本 part、reasoning_content) 只用 `qwen_tok.encode` 按文本分词, 正文里任何字符串都只是普通文本, 不可能变成控制 token; 图片占位 token 按图片 grid 展开
- 与官方模板的区别: 正文不做 trim (原样), 也不补模板为 trim 加的换行; 不做任何空字符串判断 (有 system 消息就有 system 块, 内容为空也一样; text part 必须有字符串 text, 否则 400)
- 历史 assistant 消息还原成当时的生成提示 + 原样内容: 有 `reasoning_content` 字段 (字符串, 空串也算) 时是 [思考开始] `\n` + reasoning_content + [思考结束] + content, 否则是 [思考开始] `\n\n` [思考结束] `\n\n` + content。客户端原样回传本服务返回的内容时, 历史 assistant 段与生成时的 token 相同 (多轮能命中输出缓存点); 不回传 reasoning_content 时那一轮回复对不上 (当轮命中不到它), 但之后各轮的还原方式一致, 能用上前一轮写下的缓存点

### 3.5 前缀缓存和输出 token 缓存

- 前缀缓存 (`--cache-dir`, cpu 和 cuda 都支持, 实现分别见 [cpu_impl.md](cpu_impl.md) 9.1 节和 [cuda_impl.md](cuda_impl.md) 4.2 节): 缓存点在每条消息末尾的 im_end token 之后, 存到磁盘, 进程重启后仍可用; 同一目录同时只能被一个进程使用 (文件锁)。多轮对话命中 assistant 消息的前提是客户端原样回传它 (开 thinking 时包括 `reasoning_content`)
- 输出 token 缓存: 每次生成结束 (含客户端断开) 把思考和正文各存一行 (key 是文本 utf-8 的 sha256, value 是生成时的 token 序列, 每个 3 字节小端; 不存类型, 相同文本覆盖), 放在缓存目录的 `output_tokens.sqlite3` (cpu / cuda 各自独立, 进程内只打开一次)
- 拼 prompt 时任何消息 (system、user、回复、tool、文本 part) 的每段正文先查输出 token 缓存, 查到就直接用生成时的 token 序列 (之前的结构文本先单独分词), 查不到才按文本分词并和两边的结构文本整段分词 (普通输入的分词结果不存)。所以原样回传的回复与生成时的 token 逐个相同, 不受重新分词的影响 (长 thinking 回复也能命中输出缓存点)
- 自定义采样的编译缓存也放在这个目录下的 `wasm/` (见 [custom_sampler.md](custom_sampler.md)), 前缀缓存只认两位十六进制的子目录, 互不影响

### 3.6 图片

- 图片预处理用 transformers 的官方 image processor, 视觉塔用 transformers 的模块加载官方权重; 视觉塔 fp32, `--type cuda` 时在 GPU 上, 否则在 CPU 上
- 图片大小上限默认与官方相同 (最多 16777216 像素, 约 16384 token), `--max-image-pixels` 可以调小
- 图片缓存: 视觉塔的输出按图片原始字节 (base64 解码后) 的 sha256 缓存, 只在内存里, 不写文件; 一个请求里命中的图片直接用缓存, 未命中的图片一起过视觉塔再按图片拆开存入。按 LRU 淘汰, 上限 `--image-cache-mb` (默认 4096MB, 每张 1024 token 的图约 20MB), 0 表示不缓存。图片预处理 (resize、切 patch) 每次仍会执行, 因为需要它确定图片 token 数

## 4. router.py

多后端路由, 后端列表都取自 `qwen.LIBS`, 新增后端不需要改路由。

- `--mask`: 0/1 串, 第 i 位对应 `qwen.LIBS` 的第 i 个后端 (目前依次是 `cpu`、`cuda`), 1 表示启动; 默认全 1 (现在是 `11`)。位数不对、一个都没选都直接报错
- 每个选中的后端启动一个 `server.py --type <后端> --host 127.0.0.1 --port 0` 子进程, 实际端口从子进程日志中解析; 除 `--host`、`--port`、`--mask`、`--start-timeout`、`--cache-dir` 外的参数 (`--gguf`、`--hf-dir`、`--ctx`、`--threads`、`--device`、`--sampler-rootfs` 等) 原样传给每个子进程, 不能传 `--type`
- `--cache-dir` (必填, 没有默认值): 前缀缓存根目录, 不存在则创建 (上级目录必须存在); 支持缓存的后端 (`qwen.CACHE_BACKENDS`, 目前是 cpu 和 cuda) 收到 `--cache-dir <根目录>/<后端名>`, 其余后端不传; 子目录由 server 自己检查和创建
- 子进程日志加 `[后端名]` 前缀转发到路由的 stderr
- 启动阶段同时等待所有子进程, 任何一个启动失败 (没有 GPU、显存不够、参数错误等, 子进程退出) 或超过 `--start-timeout` (默认 600s) 未就绪, 立即结束全部子进程并报错退出; 运行中任一子进程退出, 路由也整体退出。子进程设置了 `PR_SET_PDEATHSIG`, 路由被 kill 时子进程也会被结束
- `GET /v1/models`: 返回所有已启动后端的模型 (各子进程自己报告的名字), 例如 `--mask 11` 时是 `qwen3.8-27b-cpu` 和 `qwen3.8-27b-cuda`
- `POST /v1/chat/completions`: 把 `model` 转小写后, 检查包含哪些后端名; 恰好包含一个且该后端已启动时原样转发 (流式边收边回, 客户端断开会传到后端并停止生成), 一个都不包含、包含多个或后端没启动都返回 400

## 5. wasm_sampler.py

自定义采样代码的编译和缓存, 以及编译镜像的生成 (`build-rootfs`), 见 [custom_sampler.md](custom_sampler.md) 第二部分。

## 6. 测试

- `tests/smoke_test.py`: 不依赖第三方库, 用 GGUF 词表做贪心匹配分词 (不是真正的 BPE), 只用于验证能跑; `--type` 选择后端
- `tests/cuda_vs_cpu.py`: CPU 与 CUDA 后端逐层数值对比
- `tests/bench.py`: 不同上下文长度的 prefill / decode 速度 (`--type`, 默认 cuda)
- `tests/test_server.py`: 服务端到端测试 (system、思考开关、流式、图片、图片缓存、多图、seed、并发); `--model` 指定请求里的模型名 (经路由测试时用)
- `tests/test_router.py`: 路由测试 (模型列表、按模型名转发、大小写、不匹配 / 匹配多个时报错)
- `tests/test_wasm_sampler.py`: 自定义采样测试, 覆盖范围见 [custom_sampler.md](custom_sampler.md) 第二部分第 6 节
