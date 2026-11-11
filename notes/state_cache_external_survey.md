# 循环状态缓存 / 量化的外部调研 (内部)

2026-11-11。调研混合模型 (attention + 线性注意力 / SSM) 里固定大小循环状态的前缀缓存和压缩量化: 宏观策略、推理引擎的做法、官方建议、论文、个人实现。我们自己的量化实验见 [state_quant_experiments.md](state_quant_experiments.md)。

## 0. 搜索方式和可信度

- 谷歌: serper.dev (`POST https://google.serper.dev/search`, 头 `X-API-KEY`, 返回 JSON, `organic` 里是标题 / 链接 / 摘要); GitHub: REST API 带 token (`/root/local_llm/github_token.txt`), issue / PR 搜索必须带 `is:issue` 或 `is:pr`, 代码搜索每分钟 10 次。Claude Code 内置的 WebSearch 只返回服务端整理过的摘要, 很多搜不到, 这次没用
- 原文: arXiv 摘要用 arXiv API, GitHub issue / PR 用 API 取正文, 博客 / 文档用 curl 转纯文本; OpenReview (HiServe, WEAVE, LeapQuant 的 OpenReview 版) 的 API 和网页都要浏览器验证, 只看到谷歌摘要
- 脚本和原始结果在仓库外 `/root/local_llm/tests/search/` (`run_search.py` 批量搜索, `fetch.py` 取原文, `r*.txt` / `f*.txt` 输出, `pages/` 原文)
- 大部分来源是 2026 年的预印本、RFC、issue, 结论没有复现; 论文只看了摘要; issue / RFC / 博客看了正文或相关段落

## 1. 问题本身

KV 按 token 追加, 任何前缀都能截出来复用; 循环状态是原地更新的, 只在 "恰好处理到某个位置" 时有效, 不能回退。所以前缀复用必须在复用位置正好有一份状态快照, 而每份快照都是一整份状态 (Qwen3.8-27B: 144MiB DeltaNet + 5.6MiB conv; SGLang 的 Qwen3.8 cookbook 写 153.9MB/槽 fp32, 78.4MB bf16)。

## 2. 宏观策略

### 2.1 引擎默认: 只在固定网格上存

- **vLLM** (`--enable-prefix-caching` 时 Mamba cache mode = `align`): 只在 block 边界存状态, 且 attention 的页大小被撑到和状态一样大, block 很大 (Qwen3.8-27B 约 1584~1600 token, GLM-5.3-Flash 4352)。后果: 比一个 block 短的 prompt 完全不缓存 ([#40696](https://github.com/vllm-project/vllm/issues/40696), Reddit 实测约 5K 以下不命中); 多轮对话下一轮从上一轮 prompt 的最后一个 block 边界开始, 要重算剩下的 prompt 和整个上一轮输出 ([#66844](https://github.com/vllm-project/vllm/issues/66844) 的例子: 第二轮 10800 token 要重算 2096); 页面膨胀和内存估算偏大 ([#54458](https://github.com/vllm-project/vllm/issues/54458) 7808 token/页, [#37121](https://github.com/vllm-project/vllm/issues/37121) 约 7 倍)
  - `prefix_cache_retention_interval`: 0 (默认) 只留 "语义检查点" (最近的重放边界、共享前缀分叉点), 正数再加固定间隔的点, 不设则所有合格的点都留 ([fanout](https://fanout.sh/blog/prefix-cache-retention-interval-hybrid-models), [#53595](https://github.com/vllm-project/vllm/issues/53595): 默认 0 且不打日志, 很多点存了但永远不会命中)
  - `--enable-mamba-shared-prefix-checkpoint`: 在之前请求停下的分叉点额外存一份 (要求 EAGLE/MTP 等条件, [文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/))
- **SGLang MambaRadixCache** ([PyTorch 博客](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)): radix tree 存 KV 指针 + 完整状态快照, 命中时复制状态; 状态和 KV 两个 LRU; prefill 只存最后一个对齐点, decode 每 `mamba_track_interval` 存一份 (`extra_buffer` 模式); [#22326](https://github.com/sgl-project/sglang/issues/22326) 提议 prefill 也每 1024 token 存一份
- **llama.cpp**: `--ctx-checkpoints N` (每个 slot 最多几份) + 间隔 (`--checkpoint-every-n-tokens`, 新版 `--checkpoint-min-step`), 只在内存 (`--cache-ram`)。对 Qwen3.5/3.6 一批 bug: 检查点总失效 ([#24055](https://github.com/ggml-org/llama.cpp/issues/24055)), slot 存盘不含检查点 ([#28194](https://github.com/ggml-org/llama.cpp/issues/28194)), 模板空白不能往返导致整段重算 ([#30357](https://github.com/ggml-org/llama.cpp/issues/30357), 和我们遇到的 `<think>\n` 与 `<think>\n\n</think>` 分词不一致是同一类)

### 2.2 准入 / 淘汰看 "复用可能性 × 省下的计算"

- **[Marconi](https://arxiv.org/abs/2411.19379)** (MLSys 2025, [代码](https://github.com/ruipeterpan/marconi)): 按历史判断复用可能性, 只缓存可能被用到的状态; 淘汰时除了最近使用, 还看每字节能省多少 FLOP。token 命中率最高 34.4 倍
- **[Sparse Prefix Caching](https://arxiv.org/abs/2605.05219)** ([代码](https://github.com/arabel1a/sparse-prefix-caching)): 已知请求重叠深度的分布时, 用动态规划选检查点位置, 输出精确, 点更少
- vLLM RFC [#64150](https://github.com/vllm-project/vllm/issues/64150): 淘汰策略可插拔 (现在是共用一个 LRU 队列, 各种修复都在改它的顺序)

### 2.3 每会话只留一个 / 精确结尾 / 状态和 KV 一起淘汰

- SGLang [#20144](https://github.com/sgl-project/sglang/issues/20144): agent 每轮都插入新状态但只需要最新的; 提议请求头 `X-Overwrite-Cache-ID`, 插入时删掉同会话的旧状态
- vLLM RFC [#66844](https://github.com/vllm-project/vllm/issues/66844) (论文 [FlashStates](https://www.alphaxiv.org/abs/2611.flashstate)): 请求结束时按精确长度存状态 (`--mamba-frontier-checkpoint`), 不对齐 block; 状态和它要的 KV 一起淘汰, 避免只剩一半; 论文把检查点位置和 KV block 边界解耦 (Arbitrary State Checkpointing), 用按成本的在线策略决定存哪些
- SGLang 统一 radix 树 ([LMSYS 博客](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/)): 按会话淘汰, 多级存储 (HiCache: GPU / CPU / SSD)

### 2.4 分层存储: 检查点放 SSD

- **[HiState](https://arxiv.org/abs/2610.41508)**: 每次命中要读全部匹配的 KV, 但只读一份检查点, 且检查点大小与上下文长度无关 → 历史检查点放 SSD, 快的存储留给 KV; 自适应调检查点间隔, 统一淘汰 (检查点稀疏化 + 回收过期分支); p99 首 token 延迟比 vLLM+Marconi 快 5.73 倍

### 2.5 应用指定边界

- vLLM RFC [#55697](https://github.com/vllm-project/vllm/issues/55697): 输入里插 `<|mamba_checkpoint|>` 标记
- [splash #361](https://github.com/incoai/splash/issues/361): 在最后一条 user 消息之前主动存一份

### 2.6 不存完整状态, 近似重建

- **[SuffixReplay](https://arxiv.org/abs/2609.33477)**: 不存检查点, 存每层最近一段后缀的输入隐藏状态, 复用时重放近似状态 (依据: 状态会衰减, 远处输入影响小); 质量为全量 prefill 的 91.4~100%, 存储是 SGLang 默认每 8192 token 一个检查点的 0.36~0.51 倍; 有损
- DASC (见 3.1) 的 "后缀刷新" 也是这个思路

## 3. 状态压缩 / 量化

两种情况要分开: (a) 运行时每步都低精度存储, 误差每步累积 (大部分论文和引擎选项); (b) 只压缩缓存快照, 只舍入一次 (我们的情况, 相关工作少)。

### 3.1 只压缩快照

- **HiServe** ([OpenReview](https://openreview.net/pdf?id=ByZKXmIAKy), 只看到摘要): 检查点里的状态矩阵存成低秩因子, 复用时重建; KV 和 conv 不动; 摘要提到 "每个检查点 51.5MB"
- **[DASC](https://arxiv.org/abs/2608.30386)**: 由权重里的 decay 推出各头 / 通道的 "保持时长", 只存保持久的部分 (ragged 布局, 跨 TP 均衡); 复用时缺的填 0 或用一段后缀重算; Kimi-Linear (KDA) 压 2.63 倍接近全量, Qwen (GDN, 按头) 趋势类似
- **WEAVE** ([OpenReview](https://openreview.net/forum?id=ULyXCdVPZI), 只看到标题和片段): 沿状态的自然演化压缩状态变化量, 用于 agent 场景的检查点
- 没找到和我们相同的做法 (每块 32 个、对数编码、按块搜覆盖范围的 8 bit)

### 3.2 引擎和官方 (都是运行时精度)

| | 选项 | 默认 |
|---|---|---|
| vLLM | `--mamba-ssm-cache-dtype` auto / float32 / float16 / bfloat16; `--enable-mamba-cache-stochastic-rounding`, `--mamba-cache-philox-rounds` | auto: 跟随 conv 状态类型 (即模型精度, Qwen3.5 是 bf16; 依据是 [mratsim 笔记](https://github.com/mratsim/vllm-glm53f-exl3-sm120-turbo/issues/2) 引用的 `mamba_utils.py` 代码, 没看官方源码) |
| SGLang | `--mamba-ssm-dtype` float32 / bfloat16 / float16 (+随机舍入); FP8 提案 [#40706](https://github.com/sgl-project/sglang/issues/40706) (每 tile 一个 fp32 scale = amax/240, Qwen3.8-27B 153.9MB → 38.8MB) | float32 |
| llama.cpp | 无 (`recurrent_type_s` 写死 F32) | F32 |
| TensorRT-LLM | [#8763](https://github.com/NVIDIA/TensorRT-LLM/issues/8763): 为准确度要 float32 | — |

- Qwen: Qwen3.5-397B 模型卡的 vLLM 部署命令有一个 [PR](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/discussions/2/files) 加上 `--mamba-ssm-cache-dtype float32`, 理由是 auto 解析成 bf16 会影响质量 (HF 讨论页, 没核实是否合并)
- NVIDIA Nemotron 3: fp16 + 随机舍入 (Super 报告: 直接 fp16 时 LiveCodeBench 输出 token 多 36.95%, 加 W8A8 时多 40.27%, 通过率不变, 引自 fanout 博客; Ultra 报告: 分块 int8 + 随机舍入基本保持, fp8 E4M3 掉点, 这条来自内置搜索的摘要, 没读原文); NVIDIA 论坛上有人因为不能开随机舍入而保持 fp32

### 3.3 论文 (运行时每步量化)

| 论文 | 结论 / 方法 |
|---|---|
| [DAMP](https://arxiv.org/abs/2608.27513) | 均匀 INT8 / FP8 就会掉点: Qwen3.6-35B AIME 2026 FP32 85.46 → FP8 29.27 / INT8 18.48 (Hadamard + INT8 44.21); 离群值集中在特定 key 通道和 value 维度, decay 决定误差保留多少; 慢衰减的 key 通道存 FP16, 其余 INT8, 平均 9.9 bit 接近 FP32; SGLang 里状态存储减 69.1% |
| [Lost in Rounding](https://arxiv.org/abs/2611.12127) | bf16 最近舍入让 6 个混合推理模型里 5 个推理变长 (1.07~1.27 倍), Qwen3.5-9B 准确率 85.6% → 70.8%, 达到 32K 上限的题 118 → 223; 同宽度随机舍入修复 (主要是舍入偏差, 不是分辨率不够) |
| [LeapQuant](https://arxiv.org/abs/2609.38166) | 每 16 token 量化一次 (窗口内低精度状态 + 高精度缓冲的更新), 最大离群值存成少量 fp16 "补偿 token", 残差平滑; INT8 约 1.19 字节/元素接近 FP32 (Qwen3.5-9B AIME: FP32 87.9, 每步 BF16 72.1, 每步 FP8 14.6, LeapQuant 87.9); vLLM RFC [#64725](https://github.com/vllm-project/vllm/issues/64725), vllm-ascend RFC [#21097](https://github.com/vllm-project/vllm-ascend/issues/21097) |
| [STEPQuant](https://arxiv.org/abs/2609.38169) ([代码](https://github.com/Dreamer-Toby/STEPQuant)) | 按误差大小和记忆时长分配精度, key 行 / value 列联合定 scale; Qwen3.8-27B 等上 6 bit 接近 FP32, 4 bit 优于均匀 INT8 |
| [LoReS](https://arxiv.org/abs/2611.01864) | value 轴分组 scale + 按衰减混合精度 + 随机舍入, 8 bit 稳定 |
| [Low-Bit Recurrent States](https://arxiv.org/abs/2609.30950) | 按可观测性 Gramian 分配位宽, decay 对数量化; 6 bit 时 NLL 与 FP32 差 < 0.005 nats |
| [GGA + PW](https://arxiv.org/abs/2610.39931) | 按梯度敏感度分配头的位宽 + 最近的写入单独用 FP8 维护 |
| [Preserving Updates](https://arxiv.org/abs/2610.15709) | 误差跨步高度相关, 主因是小更新在舍入中丢失; 低秩旁路缓冲小更新 + 有界方向舍入 |
| [Low-Discrepancy Dither](https://arxiv.org/abs/2609.39185) | 确定性 Weyl 抖动比随机舍入好; 最近舍入短评测看起来最好, 长生成落后最多 |
| [When Dynamic Scaling Fails](https://arxiv.org/abs/2610.15139) | 量化 scale 和状态之间的反馈导致范数爆炸; 32K 时分组 INT4 不比丢掉状态好 |
| [SPARE](https://arxiv.org/abs/2611.07552) | 状态有低秩冗余, 运行时直接在固定低维子空间上读写 |
| [Why GDN Survives 4-Bit](https://arxiv.org/abs/2609.04098) | Qwen3.8-27B 全部线性层 NVFP4 W4A4 (含 GDN); delta-rule 递推把注入的噪声维持在平台上, 对状态的一次扰动几百步内被忘掉 (与我们 4.5 节 "误差随后续输入衰减" 一致) |

### 3.4 个人实现

- [KohakuvLLM `state8.py`](https://github.com/KohakuBlueleaf/KohakuvLLM): vLLM 插件, GDN 状态按行存 FP8 / INT8 + 每行 fp32 scale (amax/448 或 amax/127), 每步反量化 → fp32 算 → 重新量化写回
- [yanghaocsg/LeapQuant-NPU](https://github.com/yanghaocsg/LeapQuant-NPU), [awesome_quant_algo](https://github.com/yanghaocsg/awesome_quant_algo): LeapQuant 在昇腾上的复现, TurboQuant 对比
- [lattice #484](https://github.com/ohdearquant/lattice/issues/484): 计划测 "在轮次边界量化状态, 误差是否跨轮次超线性累积" (和我们 4.5 节的问题相同), 还没有结果
- [mratsim 笔记](https://github.com/mratsim/vllm-glm53f-exl3-sm120-turbo/issues/2): KDA 状态每会话 0.9GB, 2 张 RTX Pro 6000 并发只到 4

## 4. 计算精度: 低精度存储, fp32 计算

查到的实现都是 "低精度存储, 读出来转 fp32 算, 算完舍入写回", 没找到递推本身用 fp16 / bf16 算的:

| 实现 | 依据 |
|---|---|
| vLLM / SGLang (FLA kernel) | [`fla/ops/gated_delta_rule/fused_recurrent.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py): 状态累加器 `b_h` 是 `tl.float32` (104/106 行), 初始状态 `.to(tl.float32)` (112 行), q/k/v/门控都转 fp32, 写回时 `.to(p_ht.dtype.element_ty)` (181 行) |
| SGLang fp16 + 随机舍入 | 参数说明: "写入 FP16 Mamba SSM 状态时随机舍入" |
| SGLang FP8 提案 | "写入时量化成 e4m3, 读取时反量化" |
| KohakuvLLM | 文档原文: 反量化后用 fp32 算一步, 再量化写回 |
| LeapQuant | 窗口内用低精度状态 + 高精度缓冲的更新算输出 |
| HF transformers / llama.cpp | fp32 存储和计算 |
| DAMP / STEPQuant / LoReS 等 | 从摘要看是同样做法, 没查代码 |

FLA 的 decode 每个 token 调一次 kernel, 所以 "每步低精度" = 每个 token 舍入写回一次, 误差累积就来自这里; fp32 计算只保证同一次调用内部 (如 prefill 的一个 chunk) 不损失。

## 5. 为什么外面这么重视 (显存 / 带宽账)

### 5.1 显存

状态大小只和并发数有关, 和上下文长度无关。一个槽 149.6MiB 相当于约 4.5K token 的 q8_0 KV (每 token 34KB), 或约 2.4K token 的 bf16 KV (每 token 64KB, vLLM 默认):

| 上下文 | KV (q8_0) | 状态 / KV (显存) | 每步 decode 流量: 状态读写 / KV 读 |
|---|---|---|---|
| 4K | 0.14GB | 110% | 约 220% |
| 16K | 0.56GB | 28% | 约 56% |
| 50K | 1.74GB | 9% | 约 18% |
| 100K | 3.5GB | 4.5% | 约 9% |

单个长请求时确实不算什么。外面重视的原因:

1. 实际负载很多是短请求 (聊天、RL 采样、大量并发 agent 会话), 几 K token 时状态和 KV 一样大甚至更大
2. 前缀缓存要复制状态, KV 不用: 同一对话的 KV 只存一份、各检查点共用, 但每个检查点要一整份状态; 100K 对话按 SGLang 默认每 8192 token 一个点要 12 份 ≈ 1.9GB, 接近这段对话 q8 KV 的 3.5GB; 有分叉时每个分支一份。缓存论文 (Marconi、HiState、DASC 等) 解决的主要是这个
3. 状态按槽整块预留 (请求再短也占满), KV 按页按需分配; vLLM 把 attention 页大小撑到和状态一样, 每页几千 token
4. 线性层比例更高或 attention KV 很小的模型, 状态在更长上下文里仍是大头 (KDA 模型每会话 0.9GB; DAMP: Qwen3.6-35B 并发 256 时 fp32 状态 15GB)

### 5.2 带宽和计算

- decode 每个序列每步把整个状态读一遍、写一遍: 我们约 300MB, 约 0.22ms/步 (多序列 decode 实验拆出来的; 按带宽算 300MB / 1.3TB/s ≈ 0.23ms), 递推计算本身很小
- 权重每步 16.7GB (约 12.5ms), B 个序列共用一次读取; 状态是每个序列各一份。改 16 位存储每序列每步省约 0.11ms: B = 1 约 0.7%, B = 4 约 2%; prefill 每 chunk 才读写一次, 不受影响
- 只有大并发 (比如 256) 时权重读取被分摊, 状态读写才成为大头: DAMP 测到此时状态更新占 decode 时间 24.3%

### 5.3 本卡读写带宽 (RTX 6000D, `tests/search/bw_rw.py`, 2GB 缓冲, GPU 空闲)

| 操作 | 带宽 |
|---|---|
| 纯写 (`fill_`) | 1336 GB/s |
| 纯读 (`sum`) | 1339 GB/s |
| 读一块写到另一块 (`copy_`) | 1280 GB/s (读写合计) |
| 读出再原地写回 (`mul_`, 同状态更新) | 1279 GB/s (读写合计) |

写不比读慢, 但读写共用同一份带宽, 同时读写时合计比纯读低约 4% (推测是总线读写切换的开销, 没验证)。状态每步 "读 150MB + 写 150MB" 相当于读约 310MB。

## 6. 对我们的结论

- 运行时状态保持 fp32 存储和计算: 4 个槽只占 0.6GB (KV 36GB), 改 16 位只快 1~2%, 还会引入每步累积的误差
- 量化只用在缓存快照 (内存 / 磁盘), 只舍入一次, 最接近 HiServe / DASC 这一类; 外部 "8 bit 会崩" 的结论都是每步量化的场景, 不能直接套用; 我们实测误差随后续输入衰减、KL 接近噪声底
- 宏观策略可以借鉴: 每个对话只留最新状态, 有了后面的就删中间的 (SGLang #20144, FlashState 精确结尾); 上一轮 prompt 结束处作为语义检查点 (vLLM retention = 0, splash); 检查点放磁盘、KV 放快的存储 (HiState, 与 "写盘 + 后台清理" 一致); 按 "能省多少计算" 决定是否缓存, 短的不存 (Marconi); 状态和对应的 KV 一起淘汰 (FlashState)
