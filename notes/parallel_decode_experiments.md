# 并行与 decode GEMV 优化实验记录 (内部)

2026-11-11 一次会话内的实验, 包括没用 / 放弃的尝试。数据都在 RTX 6000D (156 SM, sm_120, 只读带宽实测 1339GB/s, L2 112MB, 驱动 595.58 / CUDA 13.2) 上测得。测试脚本都在仓库外 `/root/local_llm/tests/` 下 (见最后一节), 除第 7 节外都在副本里改, 正式代码不动。

会话开始时的基线 (正式版): decode 15.0~15.1ms/token (4K 上下文, 66 tok/s), prefill 0→64K 平均 1921 tok/s。

## 1. 前缀缓存的前提 (只分析, 未实现)

- 每个缓存点要存 DeltaNet 状态快照: 48 层 × 48 头 × 128×128 fp32 = 144MiB, 加 conv 状态 48 × 10240 × 3 × 4B ≈ 5.6MiB, 约 150MB, 相当于 4.4K token 的 KV (q8_0 每 token 34KB)。KV 只追加, 可以按前缀共享, 不需要每个点单独存
- 不需要每条消息一个快照: `chat_template.jinja` 中 `preserve_thinking` 默认 true, 历史思考保留, 下一轮请求 = 上一轮 prompt + 上一轮完整输出 + 新消息, 每个对话只需 "上一轮生成结束处" 1 份; 客户端不回传 `reasoning_content` 或空白不一致时, 再加 "上一轮 prompt 结束处" 1 份 (约 300MB / 对话)
- 分叉 (重新生成、改历史消息) 用稀疏检查点 + 重算: 每 8K 一个, 重算最多约 3.4s, 100K 上下文约 13 份 ≈ 2GB
- `reasoning_effort` 写进 system prompt, 换 effort 从开头就不匹配
- 不建议快照降 bf16 (结果会和不走缓存时不同); 可放 host 内存 (PCIe 往返时间未实测)

## 2. prefill 和 decode 同时跑 (旧代码, 两份权重)

测试: 同一进程两个 so 副本 (各自全局状态, 各传一份 16GB 权重), decode 从 4K 开始连续贪心, prefill 0→64K 每段 4096。百分比相对整卡单独 (decode 65.6, prefill 1921)。

| 方式 | decode 单独 (分区内) | prefill 单独 (分区内) | 同时 decode | 同时 prefill | 合计 |
|---|---|---|---|---|---|
| 自由抢占 (两个 stream, 默认优先级) | — | — | 18.9 (29%) | 1576 (82%) | 111% |
| decode stream 最高优先级 (-5) | — | — | 25.4 (39%) | 1358 (71%) | 109% |
| green context 78/78 | 56.7 | 1015 | 46.9 (71%) | 807 (42%) | 113% |
| green context decode 32 / prefill 124 | 28.5 | 1469 | 21.2 (32%) | 1198 (62%) | 94% |

- 自由抢占时 decode 每 token 稳定约 50ms (高优先级约 40ms), 不饿死; 推断原因: 同 context 内调度单位是 block, 不抢占正在跑的 block, decode 约 580 个串行小 kernel, 每个结束后 SM 被 prefill 的 block 占走, 要等它们结束。优先级只影响待派发 block 的顺序 (`cuda.h` 文档原文: "do not preempt already-running work")
- green context: `cuDevSmResourceSplit` 分组, 本卡粒度 2 SM (设备 minSmPartitionSize 8 / alignment 8, 指定 coscheduledSmCount = 2 可以按 2 分)。`verify_smid.py` 实测两次独立 split 的 group 0 / group 1 实际 `%smid` 不相交, 并集 156 (请求 78 / 32 / 16 都对)
- 分区后同时跑, 两边相对各自分区单独都还损失约 20% (只剩 L2 / 显存共享)
- 多进程不开 MPS 是整卡时间片, 没测

## 3. 显存带宽 vs SM 数 (纯读)

`smbw/bw_sm.py`, green context 限定 SM 数, 4GB 缓冲, 普通 16B load (每线程 1/4/8/16 个在途, evict-first) 与 TMA 批量拷贝 (多种 chunk / 级数) 取最好:

| SM | 8 | 16 | 24 | 32 | 48 | 78 | 156 |
|---|---|---|---|---|---|---|---|
| GB/s | 619 | 1217 | 1344 | 1343 | 1354 | 1358 | 1348 |

每 SM 最高约 77GB/s, 24~32 SM 就能跑满; 普通 load 与 TMA 差别不大。

## 4. 旧代码 decode 逐 kernel (green context N 个 SM)

`prof_dec.py` + `kprof_patch.py` (每次 launch 前后记 event, 关 CUDA Graph, eval 开头插 50ms host sleep 让 host 先排完 launch), 每 token ms:

| | 156 | 78 | 32 | 16 |
|---|---|---|---|---|
| GEMV (`k_gemv1` 193 次 + `k_gemv1_glu` 64 次) | 14.3 | 16.6 | 33.7 | 63.9 |
| 其余 kernel 合计 | 2.2 | 2.3 | 2.5 | 3.0 |
| 总计 | 16.5 | 18.9 | 36.2 | 66.9 |

GEMV 每 SM 只拉到约 16GB/s (硬件 77GB/s)。按 grid 拆开 (156 SM): glu (gate/up) 5.79, n=5120 (ssm_out / wo / down, 128 次) 4.83, qkv 组 2.22, lm_head 0.80, q/k/v 组 0.66。

## 5. 放弃的尝试 A: TMA 常驻 block 的 decode GEMV

`pd_concurrent/lowsm_patch.py`: 每 SM 1 个常驻 block (16 warp), TMA 把连续若干行按 4 级 × 24KB 拷到 smem, warp 从 smem 用 `dq8h` 反量化。中途 bug: 本 GGUF 的 ffn gate 是 q4_k、up 是 q6_k, 行宽不同, 按 gate 行宽算 tile 导致 smem 越界。数值与原版一致 (logits 相对误差 ≤ 4.3e-3, argmax 一致)。

| decode 单独 tok/s | 32 SM | 48 SM | 78 SM | 156 SM |
|---|---|---|---|---|
| 原写法 | 28.5 | 41.0 | 56.7 | 65.6 |
| TMA 版 | 17.9 | 26.2 | 40.2 | 61.2 |

诊断 (`dec_speed.py`, 每 token): 只搬不算 32 SM 20.4ms / 156 SM 14.5ms; 只算不搬 32 SM 51.4ms / 156 SM 12.4ms。慢在计算 (16 warp + 每 tile 两次同步, 比原版 48 warp 更差)。与 prefill 同时跑也全部更差 (例: 整卡自由抢占时 decode 只剩 9.7 tok/s)。

## 6. 放弃的尝试 B: 按 "小卡" 写 launch 方式 (去掉 block 反复调度)

用户怀疑少 SM 慢是 "切换" 造成的。`pd_concurrent/small_patch.py`: GEMV grid = 32 × 每 SM 驻留数 (10 / glu 9), warp 在 kernel 内循环多行, block 不退出; attention 段数 16。最后按用户要求改成写死常量 (不读运行环境)。结果与原版逐位一致, 但更慢 (每 token ms):

| SM | 原写法 | 固定跨步分行 | 原子计数器动态领行 |
|---|---|---|---|
| 32 | 34.9 | 41.0 | 40.4 |
| 48 | 24.4 | 27.8 | 29.1 |
| 78 | 17.5 | 19.3 | 20.8 |
| 156 | 15.1 | 15.7 | 19.5 |

32 SM 逐 kernel: 非 glu GEMV 18.6 → 24.6 (静态) / 21.5 (动态), glu 15.1 → 15.4 / 17.9。glu (行均匀、静态分配不会不均) 几乎不变, 说明去掉 block 调度没有收益; 非 glu 变慢是负载不均 (qkv 组里 f32 行 20480B vs q8_0 5440B, down 只有 5120 行), 动态版是上千个 warp 抢一个原子计数器。SM 内 warp 切换是零开销的, 没有可去掉的切换。

## 7. 少 SM 慢的真正原因: 反量化指令多

`pd_concurrent/gemv_bench.py`: 原版 `k_gemv1` 单独测, k = 5120, 1GB 权重, "只计算" 是把每行字节数传 0 (所有行读同一行, 在 L1), 按真实字节折算等效带宽 (GB/s):

| SM | q8_0 读 | q8_0 只算 | q4_k 读 | q4_k 只算 | q6_k 读 | q6_k 只算 |
|---|---|---|---|---|---|---|
| 16 | 571 | 683 | 235 | 338 | 315 | 345 |
| 32 | 1080 | 1361 | 472 | 675 | 617 | 689 |
| 78 | 1333 | 3287 | 1081 | 1638 | 1198 | 1669 |
| 156 | 1335 | 6444 | 1329 | 3238 | 1305 | 3322 |

整卡时计算能力是带宽的 2.4 倍, 掩盖了问题; 32 SM 时不读显存也只能处理约 680GB/s 的 q4_k。

对比其他卡 (算力 / 带宽): 本卡整卡约 54 FLOP/B, 3090 约 38, M4 Pro 约 33 (算力为约值), 本卡 32 SM 约 11。3090 的比例约相当于本卡 80 个 SM; 78 SM 时旧代码 decode 已是整卡的 86%, q4_k GEMV 81%。

llama.cpp (CUDA mmvq, `vecdotq.cuh`) 的做法: 激活先量化成 q8_1 (int8), 权重不还原成浮点, `dp4a` 一条指令 4 个乘加; scale/min 每 32 个元素在整数和之后乘一次, min 用 `dp4a(0x01010101, u)` 得到 Σx; 子块 scale 无分支解码; q6_k 的 −32 用 `dp4a(0xE0E0E0E0, u)`。激活 int8 这一条没有采用 (之前测过 llama.cpp 激活量化 KL 约为我们的 100 倍), 其余思路照搬。

## 8. decode GEMV 改写 (已进正式代码, commit d37652f)

逐步过程 (都在正式代码里改, 每步都跑了自测 / `cuda_vs_cpu`):

**v1**: 每个 lane 负责一个 32 元素子块 (warp 每轮 4 个超块); q4_k/q5_k/q6_k 量化值直接与 x 用 half2 乘加, scale/min 提到子块外 (d1·Σ(x·q) − m1·Σx), 每 4 项转 fp32; q8_0/f32 不提 (q 到 127, 单个乘积 127 × 437 ≈ 5.5 万, 两项就超 fp16); scale 无分支解码; 2 字节对齐的 q8_0/q6_k 用对齐 32 位读 + funnel shift。溢出余量 (激活最大实测约 437): q4_k 4 × 15 × 437 ≈ 2.6 万, q5_k 5.4 万, q6_k 5.6 万 (上限 |x| ≤ 511, 余量最小)。

- 整卡 decode 不变 (67.1 / 67.2 / 66.7 / 65.1 tok/s, 上下文 256 / 1K / 4K / 16K); 156 SM 只算反而下降: q8_0 6444 → 4599, q4_k 3238 → 2735, q6_k 3322 → 3254
- 32 SM 读权重: q8_0 1080 → 798, q4_k 472 → 556, q6_k 617 → 580; decode 32 SM 34.9 → 31.6ms, 78 SM 17.5 → 15.6ms
- 诊断: lane 读 32 个连续元素后, 读 x 时一条 load 指令 32 个 lane 相隔 64 字节、跨 16 个 cache line。只把 x 改成连续读 (结果错误, 仅测速, `gemv_opt/xtest`): 32 SM q8_0 1265, q4_k 839, q6_k 825

**v2**: decode 时 x 用 `xperm` 布局 (由产生 x 的 4 个 kernel 直接写); q8_0 / q6_k 上传后 `k_repack` 原地重排 (字节数不变)。32 SM 读权重: q8_0 1112, q4_k 714, q6_k 859。但整卡 decode 回退到 64.4 / 64.4 / 64.0 / 62.5: `k_gemv1` 寄存器 56 → 62, 每 SM 18 → 16 个 block, n = 5120 的 2560 个 block > 156 × 16 = 2496, 要两波; n = 5120 的矩阵在整卡上慢 8~10%。

**v3 (提交版)**: `k_gemv1` 加 `__launch_bounds__(64, 18)` (56 寄存器, 8 字节溢出), `k_gemv1_glu` `(64, 16)` (64 寄存器, 32 字节溢出)。`gemv_opt/gemv_bench2.py` (真实形状, 每次换权重区域避免 L2 命中), 读权重 GB/s:

| 矩阵 | 原 156 SM | 现 156 SM | 原 32 SM | 现 32 SM |
|---|---|---|---|---|
| q8_0 k=6144 n=5120 | 1079 | 1072 | 879 | 1059 |
| q4_k k=17408 n=5120 | 1098 | 1188 | 438 | 769 |
| q5_k k=17408 n=5120 | 1084 | 1218 | 394 | 840 |
| q6_k k=17408 n=5120 | 1137 | 1214 | 568 | 879 |
| q8_0 k=5120 n=16480 | 1259 | 1271 | 1012 | 1214 |
| q4_k k=5120 n=17408 | 1123 | 1192 | 447 | 723 |
| q6_k k=5120 n=17408 | 1175 | 1205 | 584 | 869 |

结果: 与 v2 数值逐位一致, 显存不变; 整卡 decode 66.8 / 66.9 / 66.4 / 64.8, prefill 1625 / 2428 / 2406 / 2290 (都不变); decode 32 SM 34.9 → 21.8ms (45.8 tok/s), 78 SM 15.0ms (= 原整卡); `test_server.py` 全过。遗留: T = 2~15 的 `k_gemv` 72 寄存器 + 16 字节溢出, `k_gemv_glu` 90 寄存器, 速度没测。

## 9. 新代码下 prefill / decode 同时跑 (一份权重)

`pd_concurrent2/pd_test2.py`: 第二个实例用 `qwen_set_shared` 直接用第一个实例的权重显存 (跨 green context 也可以访问, 两实例 logits 完全一致, 只多约 1GB)。相对整卡单独 (decode 66.4, prefill 1918):

| 方式 | decode 单独 (分区内) | prefill 单独 (分区内) | 同时 decode | 同时 prefill | 合计 |
|---|---|---|---|---|---|
| 自由抢占 | — | — | 24.8 (37%) | 1465 (76%) | 113% |
| decode 高优先级 | — | — | 30.1 (45%) | 1187 (62%) | 107% |
| green 78/78 | 66.0 | 1019 | 57.7 (87%) | 589 (31%) | 118% |
| green 48/108 | 59.7 | 1348 | 44.4 (67%) | 1001 (52%) | 119% |
| green 32/124 | 45.5 | 1498 | 33.5 (50%) | 1147 (60%) | 110% |

decode 每 token 最大间隔: 78/78 约 17~18ms, 自由抢占约 39~50ms。decode 分区后基本吃满显存带宽, prefill 被拖得很重 (78 SM 分区内单独 1019 → 同时 589)。推测是显存带宽 / L2 争用 (decode 流式读挤掉 prefill GEMM 在 L2 里复用的权重 tile), 没用 profiler 验证。候选: decode 读权重加 evict-first / `ld.global.cs` 提示 (没试)。

## 10. 多序列同步 decode (测试副本)

`batch_dec/batch_patch.py` → `qwen_cuda_batch`: 4 个序列槽启动时全部分配 (KV / DeltaNet 状态 / conv 状态 / 位置); 每步 B 个槽各 1 个 token 拼成 T = B 一起算, GEMV 读一遍权重给 B 个 token 用; 序列相关 kernel (dn_prep / dn_rec / qk_rope / attn_dec / attn_merge) 主体抽成 device 函数, 批量版按槽偏移指针后按单 token 调用; 每个 B 一个 CUDA Graph; T < 16 都用 xperm (`k_dn_post` 改条件时漏过一次行偏移)。验证 (`check_batch.py`): 4 个不同长度 prompt, B = 1/2/3/4 批量 16 步与正式版逐个单序列 logits 逐位相同。

速度 (`bench_batch.py`, 只计 eval, numpy 贪心采样约 0.1ms 另算):

| 情况 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 短 4 × 4K, 每步 ms | 14.92 | 16.99 | 18.55 | 20.15 |
| 短, 合计 tok/s | 67 | 118 | 162 | 198 (2.96x) |
| 长加短 (槽 0 64K, 其余 4K), 每步 ms | 16.82 | 18.92 | 20.40 | 22.02 |
| 长加短, 合计 tok/s | 59 | 106 | 147 | 182 (3.06x) |
| 长 4 × 64K, 每步 ms | 16.90 | 20.75 | 23.89 | 27.37 |
| 长, 合计 tok/s | 59 | 96 | 126 | 146 (2.47x) |

模型: 每步 t = a + Σ b(L_i), 最小二乘 a ≈ 13.3~13.6ms (共享: 读权重 + 固定开销), b(4K) ≈ 1.73ms, b(64K) ≈ 3.46ms, 残差 ≤ 0.25ms。b(L) = b0 + c·L: c ≈ 28.2ns / 上下文 token (34KB / 28.2ns ≈ 1.23TB/s, 即读 KV), b0 ≈ 1.6ms (DeltaNet 状态读写约 0.22ms + GEMV 多一个 token 等, B > 1 走 `k_gemv` 的 NT = 8 模板, 推测可按 B 特化再压)。"长加短" 没参与拟合, 预测 16.9 / 18.6 / 20.4 / 22.1 vs 实测 16.8 / 18.9 / 20.4 / 22.0; 单序列 4K → 256K 每 token 也正好 28.2ns。

外推 256K (未实测, 4 个 256K 的 KV 约 36GB): 1 个 22.4ms (44.6 tok/s), 2 个 31.4ms (合计 64), 3 个 40.4ms (74), 4 个 49.4ms (81, 1.81x); 1 个 256K + 3 个 4K 27.6ms (合计 145)。

B = 1 走批量路径与正式版单序列性能相同 (只计 `qwen_eval`): 4K 14.91 vs 14.92ms, 64K 16.74 vs 16.82~16.90ms (差 < 1%, 接近噪声), 以后可以只保留批量一套代码。

## 11. 后续计划 (用户方向, 具体未定)

- 输入允许 4 个并行; 输入和输出同时进行; SM 按比例分配 (具体方案未定)
- 启动时直接分配 4 份 KV / 状态, 不考虑其他程序的显存 (不够时可以 kill 其他进程)
- 已知可做: GEMV 按 B 特化 (B > 1); decode 读权重加 L2 淘汰提示减少对 prefill 的干扰; 前缀缓存 (状态快照)

## 12. 测试文件位置 (都在 `/root/local_llm/tests/`, 不入库)

- `pd_concurrent/`: `pd_test.py` (两实例同时跑, `--so-a` / `--green`), `verify_smid.py`, `smbw/` (纯读带宽), `prof_dec.py` + `kprof_patch.py` (逐 kernel 计时), `dec_speed.py` (decode 速度, 可指定 SM 数和上下文), `gemv_bench.py`, `lowsm_patch.py` (尝试 A), `small_patch.py` (尝试 B), `check_lowsm.py`
- `gemv_opt/`: `green_patch.py` (正式代码加 green context 的测试副本), `gemv_bench2.py`, `xtest/` (x 连续读测速), 各步日志
- `pd_concurrent2/`: `test_patch.py` (green / 优先级 / 共享权重), `pd_test2.py`
- `batch_dec/`: `batch_patch.py`, `check_batch.py`, `bench_batch.py`
