# 槽状态 (DeltaNet + conv) 压缩 / 量化实验记录 (内部)

2026-11-11, 为前缀缓存做准备: 每个缓存点要存一份槽状态 (`[48 层 DeltaNet 48×128×128][48 层 conv 10240×3]`, fp32, 149.6MiB), 看能压到多小、对输出影响多大。用户的要求: 量化可以慢 (CPU 5~10 秒都行), 还原要快 (≤ 0.5 秒)。测试脚本都在仓库外 `/root/local_llm/tests/state_cmp/` (见最后一节), 正式代码不动; 测试副本 `qwen_cuda_state` 只多了 `qwen_get_state` / `qwen_set_state` (槽状态整块拷出 / 写回)。

## 1. 无损压缩: 基本没用

真实状态 (对话 97 token prefill / 生成 400 后, 4K 文档, 16K 代码, 36K 长代码, 共 6 份) 用命令行工具压 (占原大小的比例):

| 排列 | lz4 | zstd -1 | zstd -19 | xz -6 |
|---|---|---|---|---|
| 原始字节 | 100% | 93.8% | 93.9% | 93.5% |
| 按字节平面 (fp32 的 4 个字节各自连续) | 94% | 85% | 85% | 84.5% |

- 低 3 个字节 (尾数) 完全压不动 (100%), 只有最高字节 (符号 + 7 位指数) 能压到约 45%; 指数熵约 3.5 bit, 无损极限约 (1 + 3.5 + 23) / 32 ≈ 86%, 已经到了
- 各份状态之间差别不到 1 个百分点, DeltaNet 和 conv 差不多; zstd -19 / xz 每次 8~16 秒, 比 zstd -1 没有改善

数值分布 (重尾): DeltaNet 状态 |x| 中位数 7e-4, 99% 0.04, 99.99% 0.9, 最大 24~60; conv 状态中位数 0.5, 99% 5.5, 最大 58~66。

## 2. 为什么状态用 fp32

- 参考实现都是 fp32: llama.cpp `recurrent_type_s` 写死 `GGML_TYPE_F32` (KV 可以选 q8_0, 循环状态不能选); HF `modeling_qwen3_5.py` 递推前把 q/k/v 转 fp32, 状态 fp32 存进 cache; 我们按 "llama.cpp 用 fp32 的保持 fp32" 的原则, 没有自己测过低精度递推
- 递推 `S = decay·S + k·delta`, `delta = (v − decay·kᵀS)·beta`: 每个 token 舍入一次且误差会累积 (decay 接近 1), 增量远小于状态本身 (低精度下被吃掉), delta 是两个接近的数相减
- 缓存只在存快照时舍入一次, 恢复后仍是 fp32 递推, 与 "每步都低精度" 不是一回事

## 3. 公开实现调研

- 带缓存的: SGLang MambaRadixCache (radix tree 存 KV 指针 + 完整状态快照, 命中时复制状态); vLLM `mamba_cache_mode` (每个 block 一份状态快照, GDN 只支持 `align`, 实验性, 短于 block 的 prompt 不命中, 结果不逐位一致); llama.cpp `--ctx-checkpoints` (只在内存, 对 Qwen3.5/3.6 有快照不恢复等 bug)
- 快照精度都等于运行时状态精度, 没找到 "运行时 fp32、只量化快照" 的公开实现
- 运行时状态降精度: SGLang `--mamba-ssm-dtype` 默认 float32, 可选 bfloat16 / float16 (Qwen3.8 每槽 153.9MB → 78.4MB); vLLM `mamba_ssm_cache_dtype` 默认 auto, 可选 float32 / float16; NVIDIA Nemotron 3 报告 (Mamba, 每步低精度存储): 直接转 fp16 会让输出变长最多 40%, fp16 + 随机舍入或 int16 分块 scale 可以, Ultra 的分块 int8 + 随机舍入基本保持, fp8 E4M3 掉点

## 4. 量化方案比较

都是每 32 个一块、fp32 scale (q8_1 的做法, 但 scale 用 fp32, 不存 s), 每块 36 字节 = fp32 的 28.1%, 一个槽 42.1MiB。对输出的测法: 快照点精确状态往后贪心生成 128 步作参考; 重新 prefill (状态和 logits 与第一次逐位相同, 已验证) 后把状态换成量化还原的, 用同样的 token teacher forcing, 比较 KL / top1。参照: KV cache q8_0 当初测的 KL 平均 4K 5.1e-5、32K 5.7e-6。

### 4.1 int8 线性 (第一版)

| prompt | KL 平均 | KL 最大 | top1 | 量化 / 还原 (numpy 单线程) |
|---|---|---|---|---|
| 对话 97 | 1.1e-4 | 4.5e-3 | 126/128 | 0.15s / 0.02s |
| 文档 4K | 5.1e-5 | 7.7e-4 | 128/128 | 0.15s / 0.03s |
| 代码 16K | 1.6e-4 | 1.8e-3 | 127/128 | 0.15s / 0.02s |
| 长代码 36K | 6.5e-5 | 6.4e-4 | 127/128 | 0.15s / 0.02s |

### 4.2 误差来源和改进方案 (CPU 上算还原误差)

- 整体 L2 误差的 91~92% 来自 conv 状态 (只占 5.6MiB); DeltaNet 每个头的输出之后有 RMSNorm, 头之间的绝对大小无关, 所以 DeltaNet 要按头算相对误差
- 误差高度集中: 平方误差最大的 1% 的块占 98%; 能量最大的 1% 的值占 99%

| DeltaNet 方案 | 大小 | 各头相对误差 平均 / 最大 |
|---|---|---|
| q8 b32 线性 (基线) | 28.1% | 9.7e-3 / 1.4e-2 |
| b16 | 31.2% | 7.2e-3 / 1.0e-2 |
| 沿 k 方向分块 | 28.1% | 8.4e-3 / 1.3e-2 |
| b128 | 25.8% | 1.7e-2 / 2.7e-2 |
| 非均匀: 量化 sign·(\|x\|/amax)^0.5, 还原时平方 | 28.1% | 5.3e-3 / 7.2e-3 |
| 非均匀 ^0.75 | 28.1% | 6.2e-3 / 7.5e-3 |
| 每块最大 1 个单独存 (fp32 + 1 字节位置) | 32.0% | 3.8e-3 / 7.1e-3 |
| 每块最大 2 个单独存 | 35.9% | 2.8e-3 / 4.7e-3 |
| \|x\| > 3·rms 的单独存 (1.85% 的值, 每个 8 字节) | 31.8% | 3.8e-3 / 6.3e-3 |
| amax/rms > 4 的块整块存 fp32 (27% 的块) | 47.3% | 4.0e-3 / 8.1e-3 |
| 两级 (q8 + 残差 q8) | 56.2% | 3.6e-5 / 5.0e-5 |

conv q8 b32 相对误差 6.3e-3, b16 5.1e-3。

### 4.3 对输出的影响 (文档 4K / 长代码 36K, KL 平均)

| 方案 | 大小 | 文档 4K | 长代码 36K | top1 |
|---|---|---|---|---|
| q8 线性 (conv 也 q8) | 42.1MiB | 5.4e-5 | 6.0e-5 | 128 / 127 |
| q8 线性, conv fp32 | 46.1MiB | 5.4e-5 | 7.0e-5 | 128 / 127 |
| sqrt, conv fp32 | 46.1MiB | 9.6e-6 | 1.2e-5 | 128 / 128 |
| 每块最大 1 个单独存, conv fp32 | 51.8MiB | 1.3e-5 | 1.5e-5 | 128 / 127 |
| sqrt + 最大 1 个单独存, conv fp32 | 51.8MiB | 8.7e-6 | 9.2e-6 | 128 / 128 |
| 两级 q8, conv fp32 | 86.6MiB | 9.6e-6 | 8.7e-6 | 128 / 128 |

- sqrt 不增加大小, KL 降到线性的约 1/5, 与误差小 270 倍的两级 q8 相当: 约 1e-5 是噪声底 (推测是实现本身 fp16 计算的扰动), 状态误差再小也压不下去
- conv 存 fp32 没有改善 (conv 状态只是最近 3 个输入, 误差 3 步后冲掉)
- 随机舍入不适合: 只量化一次时最近舍入的均方误差最小, 随机舍入约翻倍 (它适合反复量化、误差累积的场景)

### 4.4 真实对话 (`/root/local_llm/chats` 两份 Open WebUI 导出) 上的 8 bit 编码比较

对话 1 (40 条, 2.4 万 token, 开思考) / 对话 2 (70 条, 3.0 万 token, 不开思考, 图片跳过); 每份 2 个快照点 (某轮 assistant 结束), 还原后把下一条真实 user 消息作为一段 prefill, 再贪心 128 步, 比 129 个位置。

| 方案 (每 32 个一块, fp32 scale) | 4 个快照点的 KL 平均 | 平均 | 各头相对误差 |
|---|---|---|---|
| fp32 原样 (检查) | 0 / 0 / 0 / 0 | 0 | 0 |
| int8 线性 | 7.0e-5 / 1.9e-4 / 2.7e-4 / 6.3e-5 | 1.5e-4 | 9.9e-3 |
| fp8 E4M3 (最大值映射到 448) | 1.8e-5 / 3.6e-5 / 2.8e-5 / 3.3e-5 | 2.9e-5 | 1.5e-2 |
| int8 对数 (1 位符号 + 7 位 log2, 每级 1/8 倍频程) | 1.6e-5 / 4.3e-5 / 9.7e-5 / 3.3e-5 | 4.7e-5 | 1.4e-2 |
| int8 sqrt | 8.0e-6 / 2.5e-5 / 2.4e-5 / 1.6e-5 | 1.8e-5 | 5.2e-3 |
| int8 Lloyd-Max (按该状态分布训练 128 级幅度, 4.8s) | 8.1e-6 / 2.4e-5 / 1.3e-5 / 1.8e-5 | 1.6e-5 | 5.2e-3 |
| 两级 int8 (噪声底参照) | 4.8e-6 / 1.5e-5 / 1.7e-5 / 1.2e-5 | 1.2e-5 | 3.7e-5 |

- fp8 E5M2 只算了还原误差 (各头 3.1e-2, 是 E4M3 的 2 倍), 没上 GPU
- top1 全部 129/129 (包括线性)
- fp8 / 对数在各数量级的相对精度一样 (约 6%), 不如 sqrt; Lloyd-Max 与 sqrt 基本持平 (平均略好, 有一个快照点的单步最大 KL 更好), sqrt 已接近这种分布的最优

### 4.5 误差在后续输入中的传播

快照点还原 (精确 / 量化) 后, 继续输入对话里后面的真实 token L 个, 然后接 `<|im_end|>\n<|im_start|>assistant\n<think>\n` 开始生成, teacher forcing 精确路径的贪心输出 64 步。对话 1 的快照点在第一条 user 之后的 prompt 结束处 (6932 token; 它第一个 assistant 结束处后面不够 16K), 对话 2 在 assistant 结束处 (13530 token)。对话 2 不开思考, prompt 结束处的 `<think>\n` 与历史里的 `<think>\n\n</think>` 分词不同, 不能作为前缀, 这类候选点要排除。

DeltaNet 各头相对误差平均 (括号内为最大的头), 对话 1 / 对话 2:

| 方案 | 刚还原 | 500 后 | 4K 后 | 16K 后 |
|---|---|---|---|---|
| int8 线性 | 9.8e-3 | 1.0e-2 / 7.2e-3 (最大 5.5e-2 / 7.0e-2) | 3.3e-3 / 4.3e-3 | 1.5e-3 / 1.9e-3 |
| int8 sqrt | 5.3e-3 | 2.3e-3 / 2.1e-3 (最大 1.0e-2 / 8.0e-3) | 1.2e-3 / 1.5e-3 | 0.8e-3 / 0.9e-3 |
| Lloyd-Max | 5.3e-3 | 3.3e-3 / 2.1e-3 (最大 3.7e-2 / 1.4e-2) | 1.2e-3 / 1.6e-3 | 0.8e-3 / 0.9e-3 |

之后 65 步输出 KL 平均, 对话 1 / 对话 2:

| 方案 | 500 后 | 4K 后 | 16K 后 |
|---|---|---|---|
| int8 线性 | 6.7e-5 / 7.8e-5 | 1.9e-5 / 6.4e-5 | 2.0e-5 / 2.7e-5 |
| int8 sqrt | 2.3e-5 / 2.5e-5 | 7.5e-6 / 3.1e-5 | 1.3e-5 / 2.1e-5 |
| Lloyd-Max | 3.5e-5 / 3.0e-5 | 5.7e-6 / 2.9e-5 | 1.1e-5 / 2.5e-5 |

- 误差不累积, 随输入衰减 (decay < 1), 16K 后降到刚还原时的 1/6~1/10; 停在约 1e-3 不再降, 推测是误差已经进入之后 token 的激活、写进了新的 KV (不衰减); conv 状态误差 (7e-4~1e-3) 也是这个原因
- 线性量化在 500 个输入后个别头会放大到 5.5e-2 / 7.0e-2 (比刚还原的最大 1.4e-2 还大), 之后才下降; sqrt 没有这种情况
- Lloyd-Max 按整体均方误差优化, 部分头更差 (500 后最大 3.7e-2)
- top1: 24 组中 3 组有 1 次不一致 (64/65), 其余 65/65; fp32 原样各项都严格为 0

### 4.6 对数量化改成每块的 a·b^q

4.4 的对数方案是全局固定 b = 2^(1/8), 每块从最大值往下覆盖 15.75 个倍频程, 实际的块用不了这么宽 (见下), 大部分级浪费了。改成 |x| = a·b^q (q = 0..127, 符号 1 位, 精确为 0 的单独记位置), a / b 每块不同:

- torch 版 (`quant_logab.py`): 顶级 q = 127 固定为块内最大值, 搜覆盖的倍频程数 o = 1.0, 1.25, ..., 20.0 (77 档), a = 最大值·2^(-o), b = 2^(o/127); 每档内 q 按线性距离取最近, 比 a 小的取 q = 0; 选块内平方误差最小的一档 (另一种目标 "相对误差平方和" 更差)
- 选中的 o: 中位数 7, 10%~90% 为 5.5~9.5
- 还原误差: DeltaNet 各头 5.2e-3 (sqrt 5.3e-3), 单个值相对误差 90% 分位 1.85e-2 (sqrt 3.4e-2); 16 线程 8.5 秒 (未优化)
- 真实对话 4 个快照点 KL 平均 5.8e-6 / 1.7e-5 / 1.6e-5 / 1.5e-5 (平均 1.4e-5, sqrt 1.8e-5), 单步最大 2.7e-4 (sqrt 9.4e-4); 相对误差目标 2.4e-5

### 4.7 C++ 候选搜索 (两点定位)

`state_codec/` (独立 so, musl 静态, `std::thread` + 原子计数器分块): 最大值对到 q = 127, 再取一个块内的值对到指定的级, 由两点定出 a, b, 选块内平方误差最小的候选。

- 用户给的 23 种: 第 1~8 小 (非零值升序) 对到 q = 0; 第 1~8 小对到 0 和 1 两级正中间 (0.5); 中位数对到 16 / 32 / ... / 112
- 扩展 281 种: 第 1~16 小 × 级 {0, 0.25, 0.5, 0.75}, 1/8 ~ 7/8 分位数 × 级 {4, 8, ..., 124}
- 每块存 a, b 两个 fp32 (40 字节, 31.25%); b 是连续值, 存成 8 bit 会有损

| 候选 | DeltaNet 各头相对误差 平均 | 编码 (16 线程) | 解码 | 4 个快照点 KL 平均 | 单步最大 |
|---|---|---|---|---|---|
| 23 种 | 5.6e-3 | 0.89s | 0.021s | 1.1e-5 / 2.2e-5 / 1.5e-5 / 1.6e-5 (1.6e-5) | 2.8e-4 |
| 281 种 | 4.6e-3 | 7.1s | 0.021s | 7.3e-6 / 1.9e-5 / 1.6e-5 / 1.7e-5 (1.5e-5) | 3.4e-4 |

### 4.8 按倍频程 + 8 bit 档位 (最终方案), AVX2

4.6 的形式里 a, b 只由两个量决定: 块内最大值 amax 和覆盖的倍频程数 o (o 取自固定的表)。所以每块只存 amax (fp32) 和 o 在表里的下标 (uint8), 每块 4 + 1 + 32 = 37 字节 (fp32 的 28.9%), 解码按同一公式重算每一级, 与编码时算误差用的值逐位相同。

档数 / 范围 (标量 C++ 版, 16 线程; 4K 文档 / 36K 长代码两份状态, 平均误差两份只差第 4 位):

| 档 (o 的候选) | 编码 | DeltaNet 各头相对误差 平均 / 最大 | conv |
|---|---|---|---|
| 77 档: 1.0, 1.25, ..., 20.0 | 2.1s (单线程 24.2s) | 5.17e-3 / 8.1e-3~9.3e-3 | 6.9e-3 |
| 128 档: 1~20 均分 | 3.3s | 4.94e-3 / 8.0e-3~9.1e-3 | 6.7e-3 |
| 256 档: 1~20 均分 | 6.3s | 4.71e-3 / 7.7e-3~8.7e-3 | 6.4e-3 |
| 256 档: 2~12 均分 | 8.4s | 4.60e-3 / 7.5e-3~8.7e-3 | 6.3e-3 |

实际选中的 o 99% 落在 4.3~12.5, 所以 256 档集中到 2~12 (间距 0.039 倍频程) 比 1~20 (间距 0.075) 好。2~12 比 1~20 慢是因为档密, 各档误差接近, 提前放弃更晚触发 (标量版有提前放弃, AVX2 版没有)。

AVX2 版 (每块 4 个 __m256, exp2 换成多项式, 无提前放弃) 误差统计与标量版到第 4 位有效数字相同:

| 档 | 编码 标量 → AVX2 (16 线程) | 解码 标量 → AVX2 |
|---|---|---|
| 77 档 (1~20) | 2.05s → 0.24s | 0.021s → 0.006s |
| 256 档 (1~20) | 6.3s → 0.47~0.59s | 0.021s → 0.006s |
| 256 档 (2~12) | 8.3s → 0.53~0.57s | 0.021s → 0.006s |

真实对话 4 个快照点 (同 4.4) KL:

| 方案 | KL 平均 | 平均 | 单步最大 | 各头相对误差 |
|---|---|---|---|---|
| int8 sqrt | 8.0e-6 / 2.5e-5 / 2.4e-5 / 1.6e-5 | 1.8e-5 | 9.4e-4 | 5.2e-3 |
| AVX2 77 档 (1~20) | 6.0e-6 / 2.0e-5 / 1.2e-5 / 1.7e-5 | 1.4e-5 | 1.9e-4 | 5.0e-3 |
| AVX2 256 档 (1~20) | 7.9e-6 / 2.0e-5 / 4.0e-5 / 1.6e-5 | 2.1e-5 | 1.6e-3 | 4.5e-3 |
| AVX2 256 档 (2~12) | 5.5e-6 / 1.9e-5 / 1.2e-5 / 1.7e-5 | 1.3e-5 | 1.4e-4 | 4.4e-3 |
| 两级 int8 (噪声底参照) | 4.8e-6 / 1.5e-5 / 1.7e-5 / 1.2e-5 | 1.2e-5 | — | 3.7e-5 |

- top1 全部 129/129
- 256 档 (1~20) 在对话 2 第一个快照点出现 KL 4.0e-5 / 单步 1.6e-3, 但它的状态误差比 77 档还低: 噪声底附近单个快照点的 KL 会因个别位置波动, 比较方案时要同时看状态误差, 4 个快照点不够稳
- 256 档 (2~12) 平均 KL 只比噪声底高约 10%

## 5. 结论: 最终方案 (256 档 2~12, AVX2)

### 5.1 数据格式

输入是一个槽的状态, 按内存顺序展平: `[48 层 DeltaNet 48×128×128][48 层 conv 10240×3]`, 共 n = 39,223,296 个 fp32, 每 32 个连续的值为一块, nb = n / 32 = 1,225,728 块 (n 必须是 32 的倍数, 这里正好整除)。编码结果:

| 数组 | 类型 / 长度 | 含义 |
|---|---|---|
| `amax` | fp32 × nb | 块内 \|x\| 的最大值 |
| `idx` | uint8 × nb | 覆盖倍频程数 o 在档位表里的下标 |
| `code` | int8 × n | 第 7 位: 符号 (1 = 负); 低 7 位: q (0..127) |
| `zero_idx` | uint32 × nz | 精确为 0 的值的下标 (在整个 n 里, 升序) |

大小 nb × 37 + 4·nz 字节 = 45,351,936 B (43.25MiB) + 0 的下标 (实测每份状态 nz = 777, 约 3KB)。

档位表 (编码解码共用, 固定不存): 256 个, `oct[k] = (float)(k == 255 ? 12.0 : 2.0 + (double)k * (10.0 / 255.0))`, 与 numpy `np.linspace(2, 12, 256).astype(np.float32)` 逐位相同 (已验证), 即 2.0, 2.0392158, 2.0784314, ..., 12.0。

### 5.2 数学定义

- 第 k 块 st = oct[idx] / 127 (float 除法), 第 q 级的值 lev(q) = amax · 2^((q − 127)·st), 即 q = 127 是 amax, q = 0 是 amax·2^(-oct), 相邻两级之比 2^st
- 编码: 对每个档 c, 每个非零值 u = |x| 取线性距离最近的级 (log 域里与相邻两级的线性中点比较), 块内平方误差 Σ(u − lev(q))² 最小的档胜出
- 解码: x = ±lev(q), 之后把 `zero_idx` 里的位置写成 0

### 5.3 实现细节 (要逐位一致必须照做)

档位表的派生量 (编码前算一次, 标量 float, musl libm 的 `log2f` / `exp2f`):

```cpp
st[c] = oct[c] / 127.0f;
inv[c] = 1.0f / st[c];
mid[c] = log2f((1 + exp2f(st[c])) / 2); // 相邻两级 l0 < l1 的线性中点 = l0·(1 + 2^st)/2, 在 log2 域相对 l0 的偏移
```

每块的编码 / 解码 (`state_codec.cpp` 原样):

```cpp
// AVX2 版 (按倍频程): 每块 32 个 = 4 个 __m256; exp2 用多项式 (编码解码共用, 结果与标量版 exp2f 在最后几位不同)
inline __m256 exp2v(__m256 x) { // 2^x: x = n + f, f ∈ [-0.5, 0.5], 2^f 用 6 阶泰勒 (相对误差 < 1e-6), 2^n 拼指数位
    __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(x, n);
    __m256 p = _mm256_set1_ps(1.5403530e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

inline __m256 lev_oct_v(__m256 amax, __m256 st, __m256 q) { return _mm256_mul_ps(amax, exp2v(_mm256_mul_ps(_mm256_sub_ps(q, _mm256_set1_ps(127)), st))); } // q 是 float 形式的整数

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

void encode_block_oct_avx(const float* x, const float* st_t, const float* inv_t, const float* mid_t, int n_oct, float* amax_out, uint8_t* idx_out, int8_t* code) { // st = o/127, inv = 1/st, mid = log2((1 + 2^st)/2)
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), zero = _mm256_setzero_ps();
    __m256 u[4], lr[4], nzm[4], bq[4];
    float amax = 0;
    for (int k = 0; k < 4; k++) {
        u[k] = _mm256_and_ps(_mm256_loadu_ps(x + k * 8), absm);
        float t[8];
        _mm256_storeu_ps(t, u[k]);
        for (int j = 0; j < 8; j++) amax = std::max(amax, t[j]);
    }
    int best_o = 0;
    for (int k = 0; k < 4; k++) bq[k] = _mm256_set1_ps(127);
    if (amax > 0) {
        for (int k = 0; k < 4; k++) { // log2(|x|/amax), 各档共用; 0 处填 0, 之后用 nzm 屏蔽
            float t[8];
            _mm256_storeu_ps(t, u[k]);
            for (int j = 0; j < 8; j++) t[j] = t[j] > 0 ? log2f(t[j] / amax) : 0;
            lr[k] = _mm256_loadu_ps(t);
            nzm[k] = _mm256_cmp_ps(u[k], zero, _CMP_GT_OQ);
        }
        const __m256 va = _mm256_set1_ps(amax), c127 = _mm256_set1_ps(127), c126 = _mm256_set1_ps(126), one = _mm256_set1_ps(1);
        float best_e = INFINITY;
        for (int c = 0; c < n_oct; c++) {
            __m256 st = _mm256_set1_ps(st_t[c]), inv = _mm256_set1_ps(inv_t[c]), mid = _mm256_set1_ps(mid_t[c]), acc = zero, q[4];
            for (int k = 0; k < 4; k++) {
                __m256 q0 = _mm256_min_ps(_mm256_max_ps(_mm256_floor_ps(_mm256_fmadd_ps(lr[k], inv, c127)), zero), c126);
                __m256 r = _mm256_fnmadd_ps(_mm256_sub_ps(q0, c127), st, lr[k]); // 相对 q0 级的 log2 偏移
                q[k] = _mm256_add_ps(q0, _mm256_and_ps(_mm256_cmp_ps(r, mid, _CMP_GT_OQ), one)); // 超过线性中点取 q0 + 1
                __m256 d = _mm256_and_ps(_mm256_sub_ps(u[k], lev_oct_v(va, st, q[k])), nzm[k]);
                acc = _mm256_fmadd_ps(d, d, acc);
            }
            float e = hsum(acc);
            if (e < best_e) {
                best_e = e, best_o = c;
                for (int k = 0; k < 4; k++) bq[k] = _mm256_and_ps(q[k], nzm[k]);
            }
        }
    }
    *amax_out = amax;
    *idx_out = (uint8_t)best_o;
    for (int k = 0; k < 4; k++) {
        int qi[8];
        _mm256_storeu_si256((__m256i*)qi, _mm256_cvtps_epi32(bq[k]));
        for (int j = 0; j < 8; j++) code[k * 8 + j] = (int8_t)((x[k * 8 + j] < 0 ? 0x80 : 0) | qi[j]);
    }
}

void decode_block_oct_avx(float amax, float st, const int8_t* code, float* out) {
    const __m256 va = _mm256_set1_ps(amax), vst = _mm256_set1_ps(st);
    for (int k = 0; k < 4; k++) {
        __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(code + k * 8)));
        __m256 v = lev_oct_v(va, vst, _mm256_cvtepi32_ps(_mm256_and_si256(c, _mm256_set1_epi32(127))));
        __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(c, _mm256_set1_epi32(128)), 24); // 第 7 位移到 float 符号位
        _mm256_storeu_ps(out + k * 8, _mm256_xor_ps(v, _mm256_castsi256_ps(sgn)));
    }
}
```

线程和 0 的处理:

```cpp
template <class F> void parallel(int64_t nb, int n_threads, F f) { // 块 [0, nb): n_threads 个线程用原子计数器每次领 256 块
    const int64_t STEP = 256;
    std::atomic<int64_t> next{0};
    vector<thread> ths;
    for (int t = 0; t < n_threads; t++)
        ths.emplace_back([&] {
            for (int64_t b0; (b0 = next.fetch_add(STEP)) < nb;)
                for (int64_t i = b0; i < std::min(nb, b0 + STEP); i++) f(i);
        });
    for (auto& th : ths) th.join();
}

// AVX2 版, 接口同 sc_encode_oct / sc_decode_oct (两个版本的 exp2 不同, 编码和解码要配对使用)
int64_t sc_encode_oct_avx(const float* x, int64_t n, const float* oct, int n_oct, float* amax, uint8_t* idx, int8_t* code, uint32_t* zero_idx, int64_t zero_cap, int n_threads) {
    vector<float> st(n_oct), inv(n_oct), mid(n_oct);
    for (int c = 0; c < n_oct; c++) {
        st[c] = oct[c] / 127.0f;
        inv[c] = 1.0f / st[c];
        mid[c] = log2f((1 + exp2f(st[c])) / 2);
    }
    parallel(n / QK, n_threads, [&](int64_t i) { encode_block_oct_avx(x + i * QK, st.data(), inv.data(), mid.data(), n_oct, amax + i, idx + i, code + i * QK); });
    int64_t nz = 0;
    for (int64_t i = 0; i < n; i++) {
        if (x[i] != 0) continue;
        if (nz >= zero_cap) return -1;
        zero_idx[nz++] = (uint32_t)i;
    }
    return nz;
}

void sc_decode_oct_avx(const float* amax, const uint8_t* idx, const float* oct, const int8_t* code, int64_t n, const uint32_t* zero_idx, int64_t n_zero, float* out, int n_threads) {
    parallel(n / QK, n_threads, [&](int64_t i) { decode_block_oct_avx(amax[i], oct[idx[i]] / 127.0f, code + i * QK, out + i * QK); });
    for (int64_t i = 0; i < n_zero; i++) out[zero_idx[i]] = 0;
}
```

逐条说明:

1. **exp2v**: x 用 `_mm256_round_ps` (就近, 偶数优先) 拆成整数 n 和 f ∈ [-0.5, 0.5]; 2^f 用 6 阶泰勒 (系数是 ln2^k / k! 截成上面的 float 常数, 按 Horner 从高阶到低阶 fmadd), 相对误差 < 1e-6; 2^n 直接拼 float 指数位 ((n + 127) << 23)。这里 n ≥ −12 (st ≤ 12/127), 不会出现非规格化; amax 极小 (< 约 1e-34) 时乘积可能变成非规格化数, 实际不存在
2. **lev_oct_v**: 先 (q − 127) 再乘 st (两次舍入, 不是 fma), exp2v 之后乘 amax。编码算误差和解码用的是同一个函数, 解码结果与编码时选中的级逐位相同
3. **lr**: 每个非零值 log2f(u / amax) (标量, float 除法后 musl `log2f`), 各档共用; 0 处填 0, 之后用 nzm (u > 0) 屏蔽
4. **q 的选择**: q0 = clamp(floor(fma(lr, inv, 127)), 0, 126); r = fnmadd(q0 − 127, st, lr) = lr − (q0 − 127)·st (一次舍入); r > mid 取 q0 + 1, 否则 q0。比最小级还小的值 q0 = 0, 仍按中点在 0 / 1 之间选; q 最大 127
5. **误差**: d = (u − lev(q)) & nzm (0 值不计), acc 逐 lane fmadd(d, d, acc), 按 k = 0..3 的顺序; 最后 hsum 的加法顺序固定: 低 128 位 + 高 128 位, 再 [0]+[2] 和 [1]+[3], 再相加
6. **选档**: best_e 初始为 INFINITY, `e < best_e` 严格小于才替换, 误差相同时保留下标小的档; 第 0 档必定被选中一次 (输入有限时 e 不会是 NaN)
7. **编码字节**: `(x < 0 ? 0x80 : 0) | q`; 0 值的 q 取 0, 且 -0.0 < 0 为假, 所以 0 值 (包括 -0.0) 的编码都是 0x00
8. **全 0 块** (amax = 0): idx = 0, 每个值的编码是 `(x < 0 ? 0x80 : 0) | 127` = 0x7F, 解码 lev = 0·2^0 = 0; 这些值同样进 `zero_idx`
9. **0 的处理**: 编码完后主线程按下标从小到大扫一遍, `x == 0` (含 -0.0) 的下标写入 `zero_idx`; 解码先按块正常解码 (0 值的位置会得到 lev(0) ≠ 0), 再按 `zero_idx` 覆盖成 +0.0 (-0.0 的符号丢失, 不影响计算)
10. **线程**: `n_threads` 个 `std::thread`, 原子计数器每次领 256 块, 各块独立, 结果与线程数和调度顺序无关
11. **输入要求**: n 是 32 的倍数; 值有限 (不处理 NaN / Inf); `zero_cap` ≥ 0 的个数, 否则返回 -1
12. **编译**: musl g++ 11.2.1 (musl 1.2.2, `log2f` / `exp2f` 静态链接自 musl libm), `-O3 -mavx2 -mfma`, C++23。解码只用到 exp2v (纯算术, 无 libm), 在任意支持 AVX2 + FMA 的机器上逐位相同; 编码选出的 idx / q 还依赖 musl 的 `log2f` (lr) 和 `log2f` / `exp2f` (mid), 换 libm 可能在打平的个别位置选出不同的级 (误差统计不变)
13. **与标量版 (`sc_encode_oct`) 的区别**: 标量版用 libm `exp2f` 算级、有提前放弃; 两版的编码结果不能混用 (误差统计相同)

### 5.4 性能 (16 线程, 一个槽 3922 万个值)

- 编码 0.53~0.57 秒 (含起线程和单线程扫 0), 解码 0.006 秒 (含置 0)
- 量化后走缓存的输出与不走缓存的不再逐位一致; 真实对话上 KL 平均约 1.3e-5 (噪声底约 1.2e-5), top1 全部一致

## 6. 测试文件 (都在 `/root/local_llm/tests/state_cmp/`, 不入库)

- `patch.py` → `qwen_cuda_state/` (测试副本, 加 `qwen_state_floats` / `qwen_get_state` / `qwen_set_state`)
- `dump_states.py` → `states/*.f32` (6 份真实状态), `compress_test.py` (无损压缩, `compress.log`; 只跑完了前 3 份, 太慢中断)
- `quant_test.py` (第一版 q8, `quant.log`), `outlier_analysis.py` (误差来源和改进方案的还原误差), `quant_test2.py` (方案对输出的影响, `quant2.log`)
- `quant_chats.py` (真实对话上的 8 bit 编码比较, `quant_chats.log`; 量化函数和对话读取也被下面的脚本复用), `quant_after.py` (后续输入中的误差传播, `quant_after.log`)
- `quant_logab.py` (torch 版每块 a·b^q, `quant_logab.log`)
- `state_codec/` (C++ 独立 so: `sc_encode` / `sc_decode` 两点定位候选搜索, `sc_encode_oct` / `sc_decode_oct` 按倍频程标量版, `sc_encode_oct_avx` / `sc_decode_oct_avx` 最终方案), `test_codec.py` (候选搜索, `test_codec_gpu.log`), `test_codec_oct.py` (档数 / 范围, 标量), `test_codec_avx.py` (AVX2 与标量对比 + KL, `test_codec_avx_gpu.log`)
