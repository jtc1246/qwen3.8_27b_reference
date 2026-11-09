# C++ CPU 实现说明

本文描述 `qwen_cpu/` 下 C++ 推理库的实现。模型结构和 GGUF 量化格式本身见 [model.md](model.md), 这里只讲代码怎么做。

范围: 只实现文本主干 (64 层 + lm_head) 的前向和采样, 输入是 token 序列 (可选地附带图片 embedding 和 3D 位置), 输出是采样出的 token id。tokenizer、chat 模板、视觉塔都在 python 侧, MTP 层没有使用。

---

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `qwen.cpp` | 模型常量、权重绑定、前向 (`linear_attn` / `full_attn` / `eval_chunk`)、采样、自测、全部 `extern "C"` 接口 |
| `gguf.h` | GGUF v3 解析, 把整个文件读进内存, 建立 tensor 表 |
| `quants.h` | 量化块结构体, 标量参考反量化 (`Quants::Ref`), AVX-512 反量化 (`dequant_row`) |
| `ops.h` | `Weight` 结构、矩阵乘 `matmul`、`dot`、`rmsnorm`、`silu`/`sigmoid`/`softplus` |
| `pool.h` | 线程池 `Pool::run` |
| `CMakeLists.txt` / `compile.sh` | 构建 |

- 所有辅助模块写在 `.h` 里并用匿名 namespace 包裹, 只有 `qwen.cpp` 一个编译单元, 对外只暴露 `extern "C"` 函数
- 编译器: musl gcc 11.2.1 (`$MUSL_BIN/g++`), C++23, `-O3 -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl`
- 产物 `cmake_build/libqwen.so`: `-nodefaultlibs` + 静态链接 `libstdc++ / libgcc / libgcc_eh / libc (musl)`, 没有任何外部 so 依赖, python 用 `ctypes.CDLL` 加载。musl 的线程在 glibc 的 python 进程中运行做过压力测试, 没有问题
- `compile.sh` 每次 `rm -rf cmake_build` 后重新配置; 如果 `MUSL_BIN` 为空会自己 source `/share/develop/softwares/rc`
- 运行要求: x86_64 + AVX-512 (F/BW/DQ/VL) + F16C

## 2. 常量

模型结构写死为常量 (只支持这一个模型), 加载时用 GGUF metadata 校验:

```
N_LAYER = 64, D = 5120, FF = 17408, VOCAB = 248320
N_HEAD = 24, N_KV = 4, HD = 256, ROT = 64             # 全注意力
LK = 16, LV = 48, LHD = 128                           # DeltaNet
KEY_DIM = 2048, V_DIM = 6144, QKV_DIM = 10240
Q_DIM = 12288 (q + gate), KV_DIM = 1024, ATT_DIM = 6144
CHUNK = 64                                            # prefill 每批 token 数
EPS = 1e-6, THETA = 1e7
```

## 3. 加载 (`qwen_init`)

1. **读文件** (`Gguf::open`): `malloc` 整个文件大小 (16.24GB), 用 `pread` 每次 1GB 全部读入。不用 mmap, 权重在自己分配的匿名内存里, 进程生命周期内不释放
2. **解析**: 依次读 header、metadata、tensor 信息。数值型标量存入 `kv_num`, 字符串存入 `kv_str`, 数组 (词表、merges 等) 直接跳过。数据区起点 = header 结束位置向上对齐到 `general.alignment` (缺省 32), 每个 tensor 的 `data = 数据区起点 + offset`
3. **校验**: `general.architecture == "qwen35"`, 并逐个检查 `embedding_length`、`feed_forward_length`、`head_count`、`head_count_kv`、`key_length`、`rope.dimension_count`、`rope.freq_base`、`ssm.state_size`、`ssm.group_count`、`ssm.time_step_rank`、`ssm.conv_kernel`、`full_attention_interval` 与常量一致, 不一致直接退出
4. **绑定权重**: `get_w(name, k, n)` 检查 2D shape `[k, n]` 并生成 `Weight {data, type, k, n, rb}` (rb = 每行字节数); `get_f(name, numel)` 检查是 f32 且元素数正确, 返回 `const float*`。第 l 层 `(l + 1) % 4 != 0` 时是 DeltaNet。权重不做任何预处理或重排, 直接用文件里的数据
5. **分配状态和缓冲**: 全部通过 `alloc_f` 分配: `malloc` + `memset(0)` + 每 4KB 一次 volatile 写。最后这一步是必要的: gcc 会把 `malloc + memset(0)` 优化成 `calloc`, 大块 calloc 走 mmap 的零页, 不写就不占物理内存; volatile 写保证初始化结束时所有页都已实际占用。之后同样不释放
6. **其他**: 预计算 RoPE 频率 `inv_freq[i] = 1 / powf(1e7, 2i/64)` (与 HF 一样用 fp32), 启动线程池, 分配每线程缓冲

内存占用 (ctx = 32768 时, 进程实测 RssAnon 约 22.5GiB, 其中 1.7GiB 是 python 侧的视觉塔):

| 内容 | 大小 |
|---|---|
| GGUF 文件 | 16.24GiB |
| KV cache: 16 层 × K/V × max_ctx × 1024 × fp32 | 128KB/token, 32K 上下文 4GiB |
| DeltaNet 递推状态: 48 层 × 48 × 128 × 128 × fp32 | 144MiB |
| DeltaNet 卷积状态: 48 层 × 10240 × 3 × fp32 | 5.6MiB |
| 激活缓冲 (按 CHUNK = 64 个 token) | 约 24MB |
| 每线程: 反量化缓冲 4 × 17408 fp32 + 注意力分数 (max_ctx + 16) fp32 | 272KB + 128KB |

## 4. 线程池 (`pool.h`)

- `Pool::init(n)` 启动 n-1 个 detach 的工作线程, 主线程是第 0 号线程, 也参与计算
- `Pool::run(items, f)`: 把任务 `f(item, tid)` 分成 `items` 份。主线程设置函数指针和任务数, 重置原子计数器 `G_next`, 然后 `G_gen` 加 1 并 `notify_all`。所有线程 (包括主线程) 循环 `G_next.fetch_add(1)` 取任务, 直到取完。主线程自己做完后自旋等待 `G_pending` 归零 (每个工作线程完成后减 1)
- 工作线程等待新任务时, 先 `_mm_pause` 自旋 20000 次, 之后进入 `std::atomic::wait` (futex) 休眠, 所以空闲时不会占满 CPU
- 任务数为 1 或单线程时直接在主线程顺序执行
- 不支持嵌套调用 `run`; `tid` 用来索引每线程缓冲

## 5. 反量化 (`quants.h`)

- `Quants::Ref::{q8_0, q4_k, q5_k, q6_k}`: 标量版本, 与 ggml `dequantize_row_*` 逐行对应, 用于自测
- `Quants::dequant_row(type, src, y, k)`: 把一行量化数据解成 k 个 fp32
  - **f32**: memcpy
  - **q8_0**: 每块 32 个 int8 分两半 `cvtepi8_epi32` → `cvtepi32_ps` → 乘 d
  - **q4_k**: 每个超块先用标量解出 8 个子块的 6bit (sc, m); 每 64 个权重一次 256bit 读取 qs, `& 0xF` 得低半, `srli_epi16(4) & 0xF` 得高半, `cvtepu8_epi32` 转浮点后 `fmsub(d·sc, q, dmin·m)`
  - **q6_k**: 每 128 个权重读 ql 的 64 字节和 qh 的 32 字节, 在 256bit 寄存器里按字节先 and 再移位拼出 4 组 6bit 值 (先 and 掩码再 `slli_epi16(4)`, epi16 移位不会把相邻字节的位带进来), `sub_epi8(32)` 变成有符号, 每 16 个乘各自的 `d · scales[i]`
  - **q5_k**: 只有 2 个 tensor (`blk.51/55.ffn_down`) 使用, 直接用标量版本
- fp16 → fp32 用 F16C 的 `_cvtsh_ss`
- 所有矩阵的 k (5120 / 6144 / 17408) 都是 256 的整数倍, 不需要处理尾块

## 6. 矩阵乘 (`Ops::matmul`)

`y[t][n] = Σ_k W[n][k] · x[t][k]`, x 是 T×k 的 fp32 行主序, y 是 T×n。同一份代码处理 decode (T = 1) 和 prefill (T ≤ 64)。

- 按输出行并行, 每个任务 (item) 负责连续 4 行 (`ROWS = 4`), 任务数 n/4。所有矩阵的 n 都是 4 的倍数
- 每个任务先把这 4 行反量化到本线程的 fp32 缓冲 (`G_tbuf[tid]`, 4 × K_MAX); f32 权重直接用原始指针, 不复制
- 然后对 token 每 4 个一组调用 micro kernel `micro_kernel<NT>` (NT = 4, 不足 4 个的尾部用 NT = 3/2/1):
  - 4 行 × NT 个 token 共 4·NT 个 `__m512` 累加器
  - 沿 k 每次 16 个: 读 4 行权重各 16 个, 再对每个 token 读 16 个输入, 做 4 个 FMA
  - 最后 `_mm512_reduce_add_ps` 水平求和写入 y
- 一行权重在一次 matmul 中只反量化一次, 被该批所有 token 复用; 累加全程 fp32
- 自测里与 double 精度的逐行点积对比, 相对误差约 2e-8

## 7. 前向 (`eval_chunk`)

`qwen_eval` 把输入按 CHUNK = 64 切块, 依次调用 `eval_chunk`, 只有最后一块计算 logits。每块:

1. **embedding**: 对每个 token, 如果 `embd_mask[t]` 为 1 就拷贝调用方给的 `embd[t]` (图片), 否则把 `token_embd` 的第 token 行 (q4_k) 反量化成 fp32
2. **位置**: 有 `pos3` 就用调用方给的 (t, h, w); 没有则三个位置都是 `G_next_pos + t`
3. **64 层循环**, 每层:
   - 对每个 token 做 `attn_norm` (GGUF 里已经是 1+w, 直接 `x / rms · w`)
   - DeltaNet 层调用 `linear_attn`, 全注意力层调用 `full_attn`, 结果在 `G_out`
   - 残差 `x += out`
   - `post_attention_norm` → `ffn_gate`、`ffn_up` 两次 matmul → `silu(g) · u` → `ffn_down` → 残差
   - 把最后一个 token 的 hidden 存到 `G_hidden[l + 1]` (调试用)
4. **位置更新**: `G_next_pos = max(之前的 G_next_pos − 1, 本块所有位置) + 1`, 与 HF `get_rope_index` 的规则一致 (图片之后从最大位置 + 1 继续); `G_n_past += T` (KV cache 的写入下标, 等于序列长度, 与位置无关)
5. **logits** (只在最后一块): 最后一个 token 做 `output_norm`, 与 `output` (q6_k, 248320 行) 做 T = 1 的 matmul, 写入 `G_logits`

### 7.1 Gated DeltaNet (`linear_attn`)

1. 4 次 matmul: `attn_qkv` → `G_qkv [T][10240]`, `attn_gate` → `G_z [T][6144]`, `ssm_alpha` → `G_a [T][48]`, `ssm_beta` → `G_b [T][48]` (后两个是 f32 权重)
2. **因果卷积 + SiLU**: 按通道并行 (每任务 256 个通道, 共 40 个任务)。每个通道从卷积状态 `conv_state[c][0..2]` (旧的在前) 开始, 对 T 个 token 顺序计算 `w0·s0 + w1·s1 + w2·s2 + w3·x`, 然后状态左移、填入当前输入, 原地写回 `silu` 结果。块结束时状态里正好是最后 3 个输入, 下一块/下一个 token 接着用
3. **归一化和门控** (单线程, 计算量很小): 对每个 token 的 16 个 q 头和 16 个 k 头做 L2 归一化 (eps 1e-6), q 再乘 1/√128; 原地把 `G_b` 变成 `β = sigmoid(b)`, `G_a` 变成 `g = ssm_a · softplus(a + dt_bias)` (`ssm_a` 已经是 `-exp(A_log)`; softplus 在 x > 20 时直接返回 x, 与 torch 一致)
4. **递推**: 按 48 个 v 头并行, 第 h 个 v 头用第 `h % 16` 个 qk 头 (GGUF 的 tiled 顺序)。状态 `S[h]` 是 128×128 fp32 (行是 key 维, 列是 value 维, 64KB, 在 L2 里), 对 T 个 token 顺序做:
   - 第一遍遍历 S 的 128 行: 每行 (8 个 `__m512`) 乘 `exp(g)` 写回, 同时累加 `u += k[i] · S[i]`
   - `δ = β · (v − u)` (8 个寄存器)
   - 第二遍遍历 128 行: `S[i] += k[i] · δ` 写回, 同时累加 `o += q[i] · S[i]`
   - 把 o 写到 `G_o[t][h·128..]`, 紧接着做门控 RMSNorm: `o / rms(o) · ssm_norm · silu(z)` (ssm_norm 不含 +1)

   prefill 也是逐 token 递推, 不用分块算法: 每 token 每层的递推约 5.5 MFLOP (48 头 × 两遍 128×128), 48 层合计约 0.27 GFLOP, 相对每 token 约 51 GFLOP 的 matmul 可以忽略
5. `ssm_out` matmul → `G_out`

### 7.2 Gated Attention (`full_attn`)

1. 3 次 matmul: `attn_q` → `G_qg [T][12288]`, `attn_k` → `G_k [T][1024]`, `attn_v` → `G_v [T][1024]`
2. 对每个 token (单线程):
   - `rope_cs` 按该 token 的 (t, h, w) 一次算好 32 组 cos/sin。第 i 个频率选用哪个位置: `i % 3 == 1 && i < 33` 用 h, `i % 3 == 2 && i < 30` 用 w, 其余用 t
   - 24 个 q 头: 每个头在 `G_qg` 中占 512 个数, 前 256 是 q, 后 256 是 gate; 对 q 原地做 `q_norm` (RMSNorm) 和 RoPE (前 64 维, 第 i 与 i+32 维配对)
   - 4 个 k 头同样做 `k_norm` 和 RoPE
   - k、v 以 fp32 写入 KV cache 的第 `G_n_past + t` 行
3. **注意力**: 任务数 T × 24, 每个任务是一个 (token t, q 头 h), kv 头为 `h / 6`, 可见长度 `n = G_n_past + t + 1` (因果)。用本线程的分数缓冲:
   - 对 n 个 key 算 `q·k / 16`, 记录最大值
   - `exp(s − max)` 求和
   - 用 AVX-512 累加 `Σ p_j · v_j` (16 个 `__m512` 覆盖 256 维)
   - 结果乘 `sigmoid(gate)` 写入 `G_att [T][6144]`
4. `attn_output` matmul → `G_out`

KV cache 和所有中间结果都是 fp32 (HF 参考实现里 KV 是 bf16)。

## 8. 采样 (`qwen_sample`)

基于 `G_logits` (最后一个 token) 的拷贝:

1. **presence_penalty**: 对 `hist` 中出现过的每个 token (去重) 减去 penalty。由调用方传入已生成的 token, 与 vLLM 只统计输出 token 的做法一致
2. **temperature ≤ 0**: 直接返回 argmax (greedy)
3. **top_k**: `nth_element` 选出最大的 k 个, 再按 logit 降序排序 (top_k ≤ 0 表示不限制, 对全部 248320 个排序)
4. 在 double 下计算 `p_i = exp((l_i − l_max) / temperature)`
5. **min_p**: 去掉 `p_i < min_p · p_max` 的候选 (至少保留 1 个)
6. **top_p**: 按降序累加归一化概率, 累加值第一次 ≥ top_p 时截断 (包含该 token)
7. 在剩余候选上按 p 抽样, 随机数用 splitmix64 (`G_rng`), `qwen_seed` 设置状态

顺序是 penalty → temperature → top_k → top_p, 与 vLLM 一致。

## 9. C 接口

| 函数 | 说明 |
|---|---|
| `int qwen_init(const char* gguf_path, int n_threads, int max_ctx)` | 加载并分配全部内存; 0 成功, -1 文件或架构错误; 形状/参数不符时直接退出进程。重复调用直接返回 0 |
| `int qwen_eval(const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3)` | 追加 n 个 token 到当前序列并计算最后一个的 logits。`embd` `[n][5120]`、`embd_mask` `[n]`、`pos3` `[n][3]` 都可为 NULL。返回 0 成功, -1 未初始化或 n ≤ 0, -2 超出 max_ctx, -3 token id 越界 (mask 为 1 的位置不检查) |
| `int qwen_sample(float temperature, int top_k, float top_p, float min_p, float presence_penalty, const int* hist, int n_hist)` | 从当前 logits 采样一个 token id |
| `void qwen_seed(uint64_t seed)` | 设置采样随机数状态 |
| `void qwen_reset()` | 清零 DeltaNet 的递推和卷积状态, `n_past = next_pos = 0`。KV cache 不需要清零 (按 n_past 覆盖写) |
| `int qwen_n_past()` | 当前序列长度 |
| `float* qwen_logits()` | 最后一次 eval 的 logits, 248320 个 |
| `float* qwen_hidden()` | `[65][5120]`: 0 是 embedding, i 是第 i 层输出, 都是最后一个 token |
| `int qwen_selftest()` | 对 q8_0 / q4_k / q5_k / q6_k / f32 各取一个真实 tensor, 比较 AVX-512 反量化与标量参考 (要求逐位一致), 以及 T = 5 的 matmul 与 double 点积 (相对误差 < 1e-5), 返回失败项数 |

全局只有一个序列状态, 所有函数都不是线程安全的, 调用方需要串行调用。

## 10. 精度

- 权重按 GGUF 原样反量化, 结果与 ggml 参考逐位一致
- 激活、KV cache、DeltaNet 状态、softmax 全部 fp32; 采样概率用 double
- RoPE 频率和角度用 fp32 计算 (与 HF 一致)
- 还没有与 transformers 做逐层数值对比; 目前的验证是自测 + 实际生成的正确性 (问答、数学推理、长文本、图片描述都正常)

## 11. 性能

测试机: AMD EPYC 9115 (16 核 32 线程, Zen5, 2 个 CCD), 实测内存读带宽约 117GB/s。

| 场景 | 速度 |
|---|---|
| prefill (CHUNK = 64) | 约 20 ~ 21 tok/s |
| decode, 16 线程 | 约 3.4 ~ 3.7 tok/s |
| decode, 32 线程 | 约 3.7 tok/s |
| 初始化 (文件已在 page cache) | 约 3 秒 |

decode 每个 token 要读约 16.5GB 权重, 带宽上限约 7 tok/s, 目前只用到一半带宽。32 线程几乎没有提升, 说明瓶颈在 kernel 本身 (先反量化成 fp32 写到缓冲再做 fp32 点积)。prefill 受计算限制。

## 12. 已知限制和可改进的方向

- **单序列**: 没有 batch, 没有并发, 每次新请求都要 `qwen_reset` 后重新 prefill 全部历史; 没有前缀缓存 (DeltaNet 状态不能回退, 需要保存快照才能实现)
- **decode 带宽利用率**: 直接在量化数据上做点积 (激活量化成 int8, 用 AVX-512 VNNI 的 `vpdpbusd`), 省掉 fp32 中间缓冲; 预取; 减少每层的线程同步次数
- **prefill**: 只有简单的 4×4 寄存器分块; 可以做更大的分块、int8 GEMM
- **注意力**: decode 时只有 24 个并行任务 (每个 q 头一个), 长上下文时并行度不够; KV cache 用 fp32, 可以改 fp16/bf16 省一半内存
- **Q5_K** 还是标量反量化
- 不支持 MTP 投机解码, 不支持 YaRN, 上下文上限由 `max_ctx` 固定, 只支持 x86 AVX-512
