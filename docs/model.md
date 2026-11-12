# Qwen3.8-27B 模型结构与量化格式

本文只描述模型本身 (结构、计算公式、权重布局) 和 GGUF Q4_K_M 量化格式, 不涉及本仓库的实现方式。实现见 [cpu_impl.md](cpu_impl.md)。

信息来源: 官方 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 `config.json` / 权重 shape / model card, transformers 的 `modeling_qwen3_5.py`, llama.cpp 的 `conversion/qwen.py` / `src/models/qwen35.cpp` / `ggml-quants.c`, 以及 [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 的 `Q4_K_M` 文件头。

---

## 1. 概览

| 项目 | 值 |
|---|---|
| 发布 | 阿里 Qwen 团队, 2026-08, Apache 2.0 |
| 类型 | dense 视觉语言模型 (文本 + 图片 + 视频), 可开关思考模式 |
| HF 架构名 | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`), 结构与 Qwen3.5 / Qwen3.6-27B 相同 |
| llama.cpp 架构名 | `qwen35` |
| 上下文 | 原生 262,144, 用 YaRN 可扩展到约 1M |
| 总参数 | 27.78B (BF16 权重 55.56GB) |

参数分布 (按权重 shape 计算):

| 部分 | 参数量 |
|---|---|
| 文本主干 64 层 | 24.35B |
| └ 48 个 Gated DeltaNet 层 (每层 token mixer 115.9M) | 5.56B |
| └ 16 个 Gated Attention 层 (每层 token mixer 104.9M) | 1.68B |
| └ 64 个 FFN (每层 267.4M) | 17.11B |
| token embedding `[248320, 5120]` | 1.27B |
| lm_head `[248320, 5120]` (与 embedding 不共享) | 1.27B |
| 视觉塔 | 0.46B |
| MTP 头 (1 层) | 0.42B |

## 2. 文本模型超参数

| 参数 | 值 | 说明 |
|---|---|---|
| hidden_size | 5120 | |
| num_hidden_layers | 64 | |
| layer_types | 每 4 层: 3 个 `linear_attention` + 1 个 `full_attention` | `full_attention_interval = 4`, 第 3, 7, 11, ..., 63 层 (从 0 计) 是全注意力 |
| intermediate_size | 17408 | SwiGLU FFN |
| vocab_size | 248320 | 含 padding |
| rms_norm_eps | 1e-6 | |
| hidden_act | silu | |
| **Gated Attention** | | |
| num_attention_heads | 24 | |
| num_key_value_heads | 4 | GQA, 每 6 个 q 头共用 1 个 kv 头 |
| head_dim | 256 | |
| attn_output_gate | true | 输出乘 sigmoid 门控 |
| partial_rotary_factor | 0.25 | 只有前 64 维做 RoPE |
| rope_theta | 1e7 | |
| mrope_section | [11, 11, 10], interleaved | t / h / w 三组位置 |
| **Gated DeltaNet** | | |
| linear_num_key_heads | 16 | q、k 头数 |
| linear_num_value_heads | 48 | v 头数, 每个 qk 头对应 3 个 v 头 |
| linear_key_head_dim / linear_value_head_dim | 128 / 128 | |
| linear_conv_kernel_dim | 4 | 因果 depthwise 卷积 |
| mamba_ssm_dtype | float32 | 递推状态精度 |
| **MTP** | | |
| mtp_num_hidden_layers | 1 | 多 token 预测头, 用于投机解码 |

## 3. 整体前向

```
x = embed_tokens[input_ids]                         # [T, 5120], 图片位置替换为视觉塔输出
for layer in 0..63:
    h = x + Mixer(RMSNorm(x))                       # Mixer = Gated DeltaNet 或 Gated Attention
    x = h + FFN(RMSNorm(h))
logits = lm_head(RMSNorm(x))                        # [T, 248320]
```

- **RMSNorm** (所有层的 input/post-attention norm、最终 norm、attention 的 q_norm/k_norm):
  `y = x / sqrt(mean(x²) + eps) * (1 + w)`, 在 fp32 计算。HF 权重存的是零中心化的 `w` (初始化为 0), 所以要加 1。
- **FFN** (SwiGLU): `down( silu(gate(x)) * up(x) )`, gate/up `[17408, 5120]`, down `[5120, 17408]`, 无 bias。
- 所有 Linear 都没有 bias。

## 4. Gated DeltaNet (线性注意力层)

权重 (HF 命名, shape 为 `[out, in]`):

| 权重 | shape | 作用 |
|---|---|---|
| `in_proj_qkv` | [10240, 5120] | 输出按顺序是 q (16×128=2048)、k (2048)、v (48×128=6144) |
| `in_proj_z` | [6144, 5120] | 输出门 z, 每个 v 头 128 维 |
| `in_proj_a` | [48, 5120] | 每个 v 头的衰减输入 a |
| `in_proj_b` | [48, 5120] | 每个 v 头的写入强度 b |
| `conv1d.weight` | [10240, 1, 4] | 对 qkv 全部 10240 个通道的 depthwise 卷积 |
| `A_log` | [48] | 衰减速率 (log 形式) |
| `dt_bias` | [48] | 衰减输入偏置 |
| `norm.weight` | [128] | 门控 RMSNorm 权重 (**不是** 1+w 形式) |
| `out_proj` | [5120, 6144] | 输出投影 |

对每个 token t:

1. `qkv_t = in_proj_qkv(x_t)`, `z_t = in_proj_z(x_t)`, `a_t = in_proj_a(x_t)`, `b_t = in_proj_b(x_t)`
2. **因果卷积**: 每个通道 c 独立, `qkv'_t[c] = silu( Σ_{j=0..3} w[c][j] · qkv_{t-3+j}[c] )`, t<0 的输入视为 0 (推理时需要保存每个通道最近 3 个输入)
3. 切分: q (16 头)、k (16 头)、v (48 头), 每头 128 维
4. q、k 各自按头做 L2 归一化: `x / sqrt(Σx² + 1e-6)`, 然后 `q *= 1/sqrt(128)`
5. 门控:
   - `β_t = sigmoid(b_t)` (每个 v 头一个标量)
   - `g_t = -exp(A_log) · softplus(a_t + dt_bias)` (log 空间的衰减, ≤ 0)
6. **v 头与 qk 头的对应** (grouped value attention): HF 的顺序是按 qk 头分组, 第 j 个 v 头使用第 `j // 3` 个 qk 头 (等价于 `repeat_interleave(3)`)
7. **递推** (每个 v 头一个状态 `S ∈ R^{128×128}`, 行是 key 维、列是 value 维, fp32, 初始为 0):
   ```
   S   = exp(g_t) · S            # 衰减
   u   = Sᵀ k_t                  # 当前状态对 k 的"预测"
   δ   = β_t · (v_t − u)         # delta rule: 只写入预测的误差
   S   = S + k_t δᵀ
   o_t = Sᵀ q_t                  # [128]
   ```
   训练和长 prefill 时可以用分块 (chunked) 形式并行计算, 数学上与逐 token 递推等价。
8. **门控 RMSNorm** (按每个 v 头的 128 维): `o = o / sqrt(mean(o²) + eps) · w · silu(z)`
9. `out = out_proj(o)` (o 拼成 48×128 = 6144 维)

推理需要保存的状态 (与上下文长度无关): 每层 48×128×128 fp32 递推状态 (3MB) + 10240×3 卷积历史, 48 层合计约 150MB。

## 5. Gated Attention (全注意力层)

| 权重 | shape | 说明 |
|---|---|---|
| `q_proj` | [12288, 5120] | 输出 24 个头, **每个头是 [q(256), gate(256)] 交错排列** |
| `k_proj` / `v_proj` | [1024, 5120] | 4 个 kv 头 × 256 |
| `q_norm` / `k_norm` | [256] | 按头的 RMSNorm (1+w 形式) |
| `o_proj` | [5120, 6144] | |

对每个 token:

1. `qg = q_proj(x)` reshape 成 `[24, 512]`, 前 256 维是 q, 后 256 维是 gate
2. `k = k_proj(x)`, `v = v_proj(x)`, reshape 成 `[4, 256]`
3. q、k 按头做 RMSNorm (`q_norm` / `k_norm`)
4. RoPE (只作用于每个头的前 64 维, 后 192 维不变), 见下节
5. 因果注意力, GQA: 第 h 个 q 头用第 `h // 6` 个 kv 头, scale = `1/sqrt(256) = 1/16`, softmax 在 fp32
6. 输出门控: `attn_out = attn_out · sigmoid(gate)` (逐元素)
7. `out = o_proj(attn_out)`

注: `config.json` 里有 `"output_gate_type": "swish"`, 但 transformers 和 llama.cpp 都没有读取这个字段, 实际用的都是 sigmoid。

KV cache: 只有 16 个全注意力层需要, 每 token `16 × 2 × 4 × 256` 个数, BF16 下 64KB/token (32K 上下文 2GB, 262K 上下文 16GB)。

## 6. 位置编码 (交错 MRoPE)

- 旋转维度 64, 频率 `inv_freq_i = θ^(-2i/64)`, i = 0..31, θ = 1e7
- rotate_half 方式: 第 i 维与第 i+32 维配对旋转
  `x'_i = x_i cos − x_{i+32} sin`, `x'_{i+32} = x_{i+32} cos + x_i sin`
- 每个 token 有三个位置 (t, h, w)。32 个频率中, 第 i 个频率使用的位置:
  - `i % 3 == 1` 且 `i < 33`: h (共 11 个)
  - `i % 3 == 2` 且 `i < 30`: w (共 10 个)
  - 其余: t (共 11 个)
- **纯文本**: t = h = w = 序号, 此时等价于普通 1D RoPE
- **图片** (与 HF `get_rope_index` 一致): 设图片开始前当前位置为 p, 合并后网格为 `lh × lw`, 图片的第 (r, c) 个 token 位置为 `(p, p + r, p + c)`; 图片之后的文本从 `p + max(lh, lw)` 继续递增。因此生成阶段的位置 = 已有最大位置 + 1, 不等于序列下标。

## 7. 视觉塔

| 参数 | 值 |
|---|---|
| 层数 / hidden / 头数 / MLP | 27 / 1152 / 16 / 4304 (gelu_pytorch_tanh) |
| patch | 16×16 像素, 时间维 2 帧 (图片复制成 2 帧) |
| patch embed | Conv3d(3 → 1152, kernel = stride = (2, 16, 16)), 有 bias |
| 位置编码 | 可学习 48×48 位置表, 按图片网格双线性插值后相加; 注意力里另有轴向 2D RoPE (h、w 各占一半频率) |
| 注意力 | 每张图内部双向注意力 (不跨图) |
| block | pre-LayerNorm (eps 1e-6), 所有 Linear 有 bias |
| merger | LayerNorm(1152) → 每 2×2 个 patch 拼成 4608 维 → Linear(4608, 4608) → GELU → Linear(4608, 5120) |

一个图片 token 对应 32×32 像素 (16×16 patch 再 2×2 合并)。官方 processor 把图片缩放到 32 的整数倍, 像素数限制在 65,536 ~ 16,777,216 之间, 归一化 mean = std = 0.5。

图片在 prompt 里写成 `<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>`, `<|image_pad|>` 的个数等于该图的视觉 token 数, 这些位置的 embedding 被视觉塔输出替换。

## 8. MTP 头

1 层, 结构是一个完整的全注意力 decoder 层 (attention + FFN)。输入是下一个 token 的 embedding (经 `pre_fc_norm_embedding`) 与主干输出的 hidden (经 `pre_fc_norm_hidden`) 按 [embedding, hidden] 拼接成 10240 维, 经 `fc [5120, 10240]` 投影后进入该层, 再经 `norm` 和共享的 lm_head 预测再下一个 token。只用于投机解码, 不影响正常生成的结果。

## 9. Tokenizer 与 chat 模板

- byte-level BPE (`Qwen2Tokenizer`), 词表 248,320, merges 247,587, 不加 BOS
- pre-tokenize 正则: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

| token | id |
|---|---|
| `<\|endoftext\|>` | 248044 |
| `<\|im_start\|>` | 248045 |
| `<\|im_end\|>` | 248046 |
| `<\|vision_start\|>` / `<\|vision_end\|>` | 248053 / 248054 |
| `<\|image_pad\|>` / `<\|video_pad\|>` | 248056 / 248057 |
| `<think>` / `</think>` | 248068 / 248069 (非 special) |

生成的停止 token: 248046, 248044。

chat 模板要点:

```
<|im_start|>system
{reasoning 指令}\n\n{system 内容}<|im_end|>
<|im_start|>user
{内容}<|im_end|>
<|im_start|>assistant
<think>
```

- `enable_thinking` 默认 true: 生成提示以思考开始 token (248068) + `\n` 结尾, 模型先生成思考内容, 再生成思考结束 token (248069), 之后通常是 `\n\n` 和正文
- `enable_thinking=false`: 生成提示以 `<think>\n\n</think>\n\n` 结尾
- `reasoning_effort` (仅思考模式, 默认 xhigh, 可选 medium / low): xhigh 和 low 会在 system 开头插入一段指令文字 (即使用户没有给 system 消息也会生成一个 system 段), medium 不插入
- 历史 assistant 消息可以带 `reasoning_content`, 官方模板会 trim 后放进思考开始 / 结束 token 之间 (本仓库 server.py 不 trim, 按原样还原, 见 README)

官方推荐采样参数:

| 模式 | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| 思考 | 1.0 | 0.95 | 20 | 0 | 0 |
| 非思考 | 0.7 | 0.80 | 20 | 0 | 1.5 |

---

## 10. GGUF 格式

文件布局 (v3, 小端):

```
magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
n_kv 个 metadata:  key(string) | type u32 | value
n_tensors 个 tensor 信息: name(string) | n_dims u32 | dims u64[n_dims] | type u32 | offset u64
padding 到 general.alignment (缺省 32)
tensor 数据区 (offset 相对数据区起点)
```

- string = `u64 长度 + 字节`; metadata 值类型: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array (元素类型 u32 + 个数 u64 + 元素), 10 u64, 11 i64, 12 f64
- **维度顺序与 PyTorch 相反**: `dims[0]` 是最内层 (连续) 维度。HF 的 `[out, in]` 矩阵在 GGUF 里是 `dims = [in, out]`, 数据仍是 out 行、每行 in 个元素, 量化块沿 in 方向
- tensor 类型编号: 0 F32, 1 F16, 2 Q4_0, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 30 BF16

本模型的主要 metadata (`qwen35.*`): `block_count = 65` (64 层 + 1 个 MTP 层), `embedding_length = 5120`, `feed_forward_length = 17408`, `attention.head_count = 24`, `attention.head_count_kv = 4`, `attention.key_length = value_length = 256`, `rope.dimension_count = 64`, `rope.freq_base = 1e7`, `rope.dimension_sections = [11, 11, 10, 0]`, `ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`, `ssm.time_step_rank = 48`, `ssm.inner_size = 6144`, `full_attention_interval = 4`, `attention.recurrent_layers` (每层是否 DeltaNet), `nextn_predict_layers = 1`。GGUF 里还带了完整的词表、merges 和 chat 模板。

## 11. HF → GGUF 的转换

llama.cpp 转换时对权重做了以下变换, 读 GGUF 时必须按变换后的语义计算:

1. **RMSNorm 权重加 1**: 除 DeltaNet 内部的 `linear_attn.norm` 外, 所有 `*norm.weight` (input/post-attention/最终 norm、q_norm、k_norm、MTP 的 norm) 存的都是 `1 + w`, 直接 `x / rms(x) * w` 即可
2. **`A_log` → `ssm_a = -exp(A_log)`**, 于是 `g = ssm_a · softplus(a + dt_bias)`
3. **`dt_bias` → `ssm_dt.bias`**, conv1d 权重 squeeze 成 `[10240, 4]` (GGUF dims `[4, 10240]`)
4. **DeltaNet 的 V 头重排 (grouped → tiled)**: HF 中 48 个 v 头按 qk 头分组 `[G0v0, G0v1, G0v2, G1v0, ...]`, 转换后变成 `[G0v0, G1v0, ..., G15v0, G0v1, G1v1, ...]`。涉及 `in_proj_qkv` 的 v 行、`in_proj_z` 的行、`in_proj_a/b` 的行、`A_log`、`dt_bias`、conv1d 的 v 通道、`out_proj` 的列, 全部一致重排。重排后 **第 j 个 v 头使用第 `j % 16` 个 qk 头**, 计算结果与原模型完全相同
5. MTP 层改名为第 64 层 (`blk.64.*`), `fc → nextn.eh_proj`, `pre_fc_norm_embedding → nextn.enorm`, `pre_fc_norm_hidden → nextn.hnorm`, `norm → nextn.shared_head_norm`
6. 视觉塔不在主 GGUF 里, 单独放在 `mmproj-*.gguf`

tensor 名对照 (第 N 层):

| HF | GGUF |
|---|---|
| `model.language_model.embed_tokens` | `token_embd` |
| `lm_head` | `output` |
| `model.language_model.norm` | `output_norm` |
| `layers.N.input_layernorm` | `blk.N.attn_norm` |
| `layers.N.post_attention_layernorm` | `blk.N.post_attention_norm` |
| `layers.N.mlp.gate_proj / up_proj / down_proj` | `blk.N.ffn_gate / ffn_up / ffn_down` |
| `linear_attn.in_proj_qkv` | `blk.N.attn_qkv` |
| `linear_attn.in_proj_z` | `blk.N.attn_gate` |
| `linear_attn.in_proj_a` | `blk.N.ssm_alpha` |
| `linear_attn.in_proj_b` | `blk.N.ssm_beta` |
| `linear_attn.A_log` | `blk.N.ssm_a` (已变成 `-exp(A_log)`) |
| `linear_attn.dt_bias` | `blk.N.ssm_dt.bias` |
| `linear_attn.conv1d` | `blk.N.ssm_conv1d` |
| `linear_attn.norm` | `blk.N.ssm_norm` |
| `linear_attn.out_proj` | `blk.N.ssm_out` |
| `self_attn.q_proj / k_proj / v_proj / o_proj` | `blk.N.attn_q / attn_k / attn_v / attn_output` |
| `self_attn.q_norm / k_norm` | `blk.N.attn_q_norm / attn_k_norm` |

## 12. 量化块格式

所有格式都是对一行 (沿输入维度) 按块量化, fp16 的 scale 存在块内。下面的公式与 ggml 的 `dequantize_row_*` 一致。

### Q8_0 (8.5 bit/权重)

块 32 个权重, 34 字节: `d: fp16`, `qs: int8[32]`。`x_j = d · qs[j]`。

### Q4_0 (4.5 bit/权重, 本文件只用于 MTP 层)

块 32 个权重, 18 字节: `d: fp16`, `qs: u8[16]`。
`x_j = d · ((qs[j] & 0xF) − 8)`, `x_{j+16} = d · ((qs[j] >> 4) − 8)`, j = 0..15。

### Q4_K (4.5 bit/权重)

超块 256 个权重, 分 8 个子块 × 32, 144 字节:

```
d: fp16, dmin: fp16       # 超块级的 scale 和 min 的 scale
scales: u8[12]            # 8 个子块的 6bit scale 和 6bit min 打包
qs: u8[128]               # 4bit 量化值
```

子块 j 的 `(sc, m)` 解包:
- j < 4: `sc = scales[j] & 63`, `m = scales[j+4] & 63`
- j ≥ 4: `sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)`, `m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)`

权重: 每 64 个权重用 32 字节 qs, 低 4 bit 是前 32 个 (子块 2k), 高 4 bit 是后 32 个 (子块 2k+1):
`x = d·sc · q − dmin·m` (q 为 0..15)

### Q5_K (5.5 bit/权重)

超块 256, 176 字节: `d, dmin: fp16`, `scales: u8[12]` (同 Q4_K), `qh: u8[32]` (第 5 bit), `qs: u8[128]` (低 4 bit)。
第 k 组 64 个权重 (k = 0..3) 的高位分别取 `qh[l]` 的第 2k 位 (前 32 个) 和第 2k+1 位 (后 32 个):
`x = d·sc · (q_low + 16·bit) − dmin·m`

### Q6_K (6.5625 bit/权重)

超块 256, 分 16 个子块 × 16, 210 字节:

```
ql: u8[128]      # 低 4 bit
qh: u8[64]       # 高 2 bit
scales: i8[16]   # 每个子块一个 8bit 有符号 scale
d: fp16
```

每 128 个权重为一半 (使用 ql 的 64 字节、qh 的 32 字节、scales 的 8 个), l = 0..31:

```
q1 = (ql[l]    & 0xF) | ((qh[l] >> 0 & 3) << 4)   → 第 l      个,  scale = scales[l/16 + 0]
q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2 & 3) << 4)   → 第 l+32   个,  scale = scales[l/16 + 2]
q3 = (ql[l]    >> 4)  | ((qh[l] >> 4 & 3) << 4)   → 第 l+64   个,  scale = scales[l/16 + 4]
q4 = (ql[l+32] >> 4)  | ((qh[l] >> 6 & 3) << 4)   → 第 l+96   个,  scale = scales[l/16 + 6]
x = d · scale · (q − 32)
```

## 13. Q4_K_M 文件 (bartowski) 的类型分布

文件 17,442,399,968 字节 (16.24GiB), 866 个 tensor, 用 imatrix 校准量化。`Q4_K_M` 是混合精度, 不同 tensor 用不同类型:

| tensor | 个数 | 类型分布 |
|---|---|---|
| `token_embd` | 1 | q4_k |
| `output` (lm_head) | 1 | q6_k |
| 所有 norm、`ssm_a`、`ssm_dt.bias`、`ssm_conv1d`、`ssm_alpha`、`ssm_beta` | — | f32 |
| `attn_qkv` (DeltaNet) | 48 | q4_k 36, q6_k 11, q8_0 1 |
| `attn_gate` (DeltaNet z) | 48 | q4_k 35, q6_k 12, q8_0 1 |
| `ssm_out` | 48 | q6_k 36, q8_0 12 |
| `attn_q` | 16 | q4_k 15, q6_k 1 |
| `attn_k` / `attn_v` / `attn_output` | 16 / 16 / 16 | q8_0 |
| `ffn_gate` | 64 | q4_k 63, q6_k 1 |
| `ffn_up` | 64 | q4_k 47, q6_k 16, q8_0 1 |
| `ffn_down` | 64 | q4_k 60, q5_k 2, q6_k 1, q8_0 1 |
| MTP 层 (`blk.64.*`) 的矩阵 | — | q4_0 |

规律: 靠前的层 (0~15) 和最后一层用更高精度, 全注意力的 k/v/o 和 DeltaNet 的输出投影整体用 8bit/6bit, FFN 主体用 q4_k。

正常生成时每个 token 需要读取的权重约 16.5GB (总大小减去 token embedding 0.72GB 和 MTP 层约 0.24GB; embedding 每个 token 只读一行)。
