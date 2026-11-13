# 分词与 prompt 拼接: 控制 token 注入、文本加工的问题调研 (内部)

2026-11-12 ~ 13 的调研结论: transformers 和 Qwen 官方模板的问题, 开源推理框架 (vLLM、llama.cpp、Ollama) 和两个云平台 (火山引擎 coding plan、SiliconFlow) 的情况。

## 1. 总结

- **控制 token 注入**: 业界通行的做法是 "chat 模板渲染成一个字符串, 再对字符串分词并识别其中的控制 token 写法", 模板生成的标记和正文里用户写的同样字符串无法区分, 正文可以伪造轮次 (system / user / assistant / tool)、提前结束消息。transformers、vLLM、llama.cpp、Ollama 都是这样; 云平台上测的 9 个模型里 7 个有这个问题
- **文本加工**: Qwen 官方模板对正文和思考内容做 trim 再补换行、按正文内容判断消息类型; 各框架都照搬官方模板
- **安全的做法早就有**: OpenAI (tiktoken、harmony) 和 Mistral (mistral-common) 都是 "控制 token 由代码按 id 插入, 正文只按文本分词", 本仓库现在也是这样

## 2. transformers / HF tokenizers

- `apply_chat_template` 先渲染成字符串再分词, 正文里写的控制 token 写法 (如 `<|im_end|>\n<|im_start|>system\n...`) 会变成真正的控制 token, 模型看到的是伪造的消息
- 补救选项 `split_special_tokens` (默认关闭) 只对 special 的 token 有效; Qwen 把 `<think>`、`</think>`、`<tool_call>`、`<tool_response>` 等 12 个标成非 special, 对它们无效; 开启后模板自己生成的标记也会失效, 所以没有 "标记按控制 token、正文按文本" 的分词方式
- Qwen3-VL processor 在渲染后的文本里查找 `<|image_pad|>` 字符串来展开图片 token, 正文里写了同样的字符串, 请求就会因数量对不上而失败, 而且这个对话之后的每个请求都会失败
- HF tokenizers 内部混用两个 Unicode 版本: 预分词正则按 Unicode 16, NFC 用的是较老的版本; 与 Unicode 15.1 的标准实现相比, 只在 4433 个 Unicode 16 新增字符和 59 个罕见组合符号上不同

## 3. Qwen 官方 chat 模板

- 正文、历史回复的 reasoning_content 先 trim, 回复再补成 `<think>\n` + reasoning + `\n</think>\n\n` + content; 只有模型恰好生成一个 / 两个换行时才与生成的一致
- system 内容 trim 后为空就不生成 system 块
- 以 `<tool_response>` 开头、`</tool_response>` 结尾的 user 消息被当成工具结果 (按正文内容判断类型), 找不到 "真正的用户问题" 就报错
- thinking 的 reasoning 指令拼在 system 块里, 换 effort 第一个块就变了

## 4. 生成的 token 重新分词后对不上

- 这是 BPE 本身的性质: 模型可以生成非标准的切分, 同一段文本按标准切法会切成另一串 token, 多轮时历史回复对不上生成时的 token
- 以模型真实输出为标准, 解码后重新分词, 生成的 token 原样出现的比例: BPE 99.95%, 整串贪心 (最长匹配) 91.0%, 预分词后段内贪心 96.4%; BPE 最接近, 不换贪心。贪心切法还会让模型的每字节损失上升 9%~30%
- 短回复基本一致, 长 thinking 回复 (数千 token) 会差几个 token; 本仓库用输出 token 缓存 (文本的 sha256 -> 生成时的 token) 解决

## 5. 开源推理框架

版本: vLLM 0.33.0, llama.cpp 2026-11-09 (`a4d4f2f`), Ollama 2026-11-12 (`5632619`)。读代码得出, 没有实际运行。

| | 控制 token 注入 | 图片占位 | 对正文的加工 |
|---|---|---|---|
| vLLM | 有: 渲染和分词都交给 transformers, 没有额外防护 | 未确认 | 用官方模板, 同第 3 节 |
| llama.cpp | 有: 渲染后分词时总是识别特殊 token | 有问题: 按文本里固定的 `<__media__>` 定位图片, 数量对不上就报错 | 用 GGUF 里的官方模板, 同第 3 节 |
| Ollama | 有: 渲染出字符串交给 llama.cpp 分词; 自己的分词器也先切出特殊 token | 有防护: 每次启动随机生成图片标记 | 手写的官方模板照搬 trim 和 `<tool_response>` 判断; 另外在正文里找 `</think>` 文本, 把前面的部分切成思考内容 |

## 6. 为什么都这样

- 模板机制: HF 定的 chat 模板是随模型发布的 Jinja, 只能输出文本, 没法表达 "这里是 token id"; 框架要用同一套机制支持所有模型, 只能先渲染成字符串再识别控制 token 写法
- 与训练一致 (推断): 很多训练流程也是渲染成字符串再分词, 推理走同一条路最容易与训练一致
- 威胁模型: 直接调用 API 的客户端本来就能写 system、伪造回复, 被当成低优先级; 真正的风险是消息里嵌入了不受信任的第三方文本 (检索文档、网页、工具结果、别的用户的内容)
- 兼容 (部分推断): 有人在原始 prompt 里故意手写控制 token; Qwen 把 `<think>`、`<tool_call>` 标成非 special, 推测是为了解码时上层能看到这些标记来解析思考和工具调用
- 文本加工 (推断): 模板作者按训练数据格式规范化, 框架把官方模板当唯一标准照搬
- 反例: OpenAI tiktoken 默认遇到特殊 token 写法直接报错 (文档明确说它们 "can be used to trick a model"); harmony 直接把结构化消息渲染成 token id; Mistral mistral-common 正文单独分词, 控制 token 按 id 加入, 文本词表里没有特殊 token

## 7. 云平台实测 (2026-11-13)

方法: user 正文 "测试" + 各家的控制 token 写法, 看 prompt_tokens 多几个 (被识别成控制 token 时只多 1 个), 别家的写法和编造的写法作对照; 再让模型原样复述夹着控制 token 写法的文字。

| 平台 | 模型 | 自家控制 token 写法 | 原样复述 | 结论 |
|---|---|---|---|---|
| 火山 | Kimi-K3.1-Pro | 按文本分词 (多 3~8 个) | 一字不差 | 没问题; 伪造 system 的指令也没生效 |
| 火山 | Kimi-K2.8-Preview | 按文本分词 | 一字不差 | 没问题 |
| 火山 | Doubao-Seed-2.1-pro | 按文本分词 | 一字不差 | 没测出问题 (闭源豆包的控制 token 写法不确定, 用的是开源 Seed-OSS 的) |
| 火山 | DeepSeek-V4-Pro | 全部被识别 (只多 1 个) | 只输出了最后一段 | 有问题 |
| 火山 | GLM-5.3 | 全部被识别 | 控制 token 处变成换行 | 有问题 |
| SiliconFlow | GLM-5.2 | 全部被识别 | 不一致 | 有问题 |
| SiliconFlow | DeepSeek-V4-Pro | 全部被识别 | 不一致 | 有问题 |
| SiliconFlow | Kimi-K2.7-Code | 全部被识别 | 不一致 | 有问题 |
| SiliconFlow | LongCat-2.0 | 全部被识别 | 不一致 | 有问题 |

- 同一平台上安全与否取决于每个模型怎么接入, 不是平台统一做了处理
- Kimi 官方开源的 tokenizer 默认就会识别特殊 token; SiliconFlow 上的 Kimi 有问题, 符合按开源方式部署; 火山上的 Kimi 没问题, 说明走的是另一条路径 (两边不是同一型号, 不能完全确定区别只来自平台)
- DeepSeek、GLM 在两个平台上表现一致, 与 "官方权重 + 官方 tokenizer 和模板 + 开源推理框架默认参数" 的部署方式吻合 (推测)
- LongCat-Flash 公开的模板用普通文本作轮次标记 (`SYSTEM:`、`[Round N] USER:`、` ASSISTANT:`), 这种格式下即使分词完全正确也防不住伪造轮次; LongCat-2.0 是否还用这套格式没有验证
