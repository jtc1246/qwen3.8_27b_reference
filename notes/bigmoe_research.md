# GLM-5.3 / Kimi K3.1-Pro 大 MoE 在 RTX 6000D + 多张 300I Duo 上的可行性调研

时间: 2026-11-17 ~ 11-20。目标: 评估用 1x RTX 6000D + 若干 300I Duo 跑 700B~2.8T 级 MoE 的可行性、速度与工程风险。调研原始材料在 /root/local_llm/glm53_research/ (不入库), 结论基于官方权重文件逐张量统计 + 本机实测 + 文献, 不是二手数据。

## 1. 两个模型的精确结构 (来自官方 safetensors 分片头部, 逐张量求和)

### GLM-5.3 (zai-org/GLM-5.3, FP8 原生, 文件 703.7GiB)

- 753.33B 总参数 / 每 token 激活 ~40.3B; 78 层 (3 dense + 75 MoE) + 1 MTP 层
- 路由专家 724.78B (96.2%): 75 层 x 256 专家 x 37.76M (gate/up/down, 6144<->2048), 每 token 选 8 + 1 共享专家
- MLA: q_lora 2048 / kv_lora 512, 64 头, qk 256 (nope192+rope64), v 256; 压缩 KV 仅 576B/token/层 (fp8 下 45KB/token 全模型)
- DSA 稀疏注意力: indexer 只在 22 层 (0,1,2,6,10..74,78), 选 top-2048, 相邻层复用索引
- 注意力 12.87B, indexer 0.197B, 共享专家 2.83B, dense 0.68B, gate 0.12B, embed/lm_head 各 0.95B (bf16)
- 每 token W4 读数 ~20.6GB; MTP 可做投机解码 (~x1.6)

### Kimi K3.1-Pro (moonshotai/Kimi-K3.1-Pro, 原生 MXFP4 专家, 文件 1454GiB)

- 2780.1B 总参数 / 每 token 激活 ~104B; 93 层 = 69 KDA 线性注意力 + 24 MLA+DSA 全注意力 (config 层号 1-based)
- 路由专家 2723.0B: 92 层 x 896 专家 x 33.0M, 每 token 选 16; **latent MoE**: down[3584,7168]/up[7168,3584] bf16 每层全激活, 专家在 3584<->3072 潜空间; 专家原生 MXFP4 (g32 u8 scale, 0.531B/w, 免量化)
- KDA: q/k/v/g/o 各 [12288,7168], 96 头 x128, conv4 + delta 递推 (与 Qwen3.8 DeltaNet 同族); 状态固定 ~0.4GB/序列, 与 ctx 无关
- 全激活部分合计 55.8B (注意力 36.4 + 共享 12.2 + latent 4.7 + dense 0.73 + gate 0.59 + lm_head 1.17); 每 token W4 读数 ~54.6GB (GLM 的 2.65 倍)
- 缓存极小: 24 层 MLA KV 13.8KB/token + KDA 固定状态, 1M ctx 仅 ~14GB; 无 MTP; 带 0.45B 视觉塔

55.8B 全激活的量化分配方案 (每 token 读数 ~31.1GB): F32 只放 norm/conv/bias (~0.04GB); BF16 放 KDA 门控链 f_a/f_b/b_proj + router gate (1.6GB, 绝对不能动); INT8 放 lm_head + MLA q_a/q_b/kv_a/kv_b (2.5GB); 其余 52.2B 全 INT4 g128 (26.9GB)。

## 2. 硬件实测基准 (本机)

- RTX 6000D: 85651MiB 显存, 纯读 1338GB/s, bf16 TC 143T / int8 273T, PCIe 5.0 x16
- 310P3 单芯: 显存 47301/46717MB (关 ECC, 开 ECC 各 -3GB), 纯读 204GB/s, w4a8 GEMV 实测 195GB/s, fp16 cube 持续 25-35T, 向量 fp32 仅 0.55T
- 芯间无 P2P, 全走 host: pinned d2h 25.1 / h2d 28.6 GB/s; 同步往返 sync 26µs / doorbell 12.2µs (v9 实测, gated 预下发是收益来源); 4B 同步 memcpy ~6µs; **310P 不支持 host 内存映射** (HostRegister 207000)
- host: EPYC 9115 16C/32T, AVX512+bf16, 内存带宽 ~117GB/s, 124GB RAM

## 3. 方案评估结论 (单序列 decode 为目标)

| 方案 | 单流 decode | prefill | 要点 |
|---|---|---|---|
| GLM-5.3 @ 4 duo + GPU, 全驻留 | **~50 tok/s, +MTP ~80** | 1000-3000 | 容量余 ~12% (443.6 vs 396GB), 全部环节有同源实测, **推荐方案** |
| K3.1 @ 17-18 duo 纯 NPU, 全驻留 | 33-40 | 1000-2500 | 专家 1347GiB 是 17 卡下限; 单机插不下, 要专用整机/多机 |
| K3.1 @ 16 duo + GPU | 45-47 | 同上 | GPU 加分 ~15-20%: 每层汇合点 4->2 + 独吞 MLA/DSA |
| K3.1 @ 4 duo + GPU 四级缓存 (GPU/NPU/RAM/盘) | 初判 28-32, **按路由均匀性修正后 ~5-6; 全压 1.6-1.75bpw 免盘后 ~20** | 200-400 | 见第 5 节, 分层量化在 K3.1 上失效 |
| 基线: GPU 当 NPU 用的均匀 TP | GLM ~40 | — | 任何异构设计必须赢它; GPU 值不值得特殊对待 = 它占系统总带宽比例 (GLM 45% / K3.1@16+ 15%) |

关键设计原则 (多轮修正后):
1. **显存按 GPU+NPU 一个池子算**, 先放固定项再分配, 不要先定死归属再喊不够
2. **单流下 GPU/NPU 读数不可重叠** (层内依赖串行), 时间是加法; 真并行只有 >=2 错峰 micro-batch (attention/FFN 两批重叠, B=4 拆 2x2 约 +40%)
3. **专家切分用 TP 不用 EP**: 并行度上限 TP=芯数 vs EP=激活专家数; 且 gated 预下发下 EP 连扇出都省不全 (空铃也要敲)
4. **同步是成本主宰**: hop 数是结构常数 (GLM 2/层, K3.1 带 GPU 2/层 = 中段 o/norm/down/gate/up 全在 GPU 本地; 纯 NPU 4/层), 乘以单价 (doorbell 后 ~25-40µs/点); 32+ 芯时 ~6µs x N 的 PCIe 小拷贝扇出是新的规模墙, 待验证: 铃嵌广播数据尾字 / 多线程并发小 memcpy 是否被驱动串行化
5. CPU 角色 = qwen hostx 原则放大: 只做无权重向量操作 (归约/norm/量化/topk/采样), ~2ms/token; 16 核需 poller 合并; naive row-split allreduce 会让 host RAM 把 prefill 封顶 ~400 tok/s, 必须 gather 式切法 (o_proj 回传头输出 1.5KB/芯而非 partial 28.7KB, 省 19 倍)
6. decode 的 PCIe 流量是 ctx 无关常数 (GLM ~13MB/token, K3.1 带 GPU ~55MB, 纯 NPU ~270-310MB); 随 ctx 增长的只有 indexer 全历史扫描 (设备本地, 1M ctx 2.3ms) 和 KV 存储; KDA 零增长
7. DSA indexer 计算量 = ctx x 8192 FLOP/层: ctx<=32k 可放 CPU (零通信), 128k 是分水岭, 更长只有 GPU 全包干净

## 4. Reddit Strata 方案解析 (Grawnch/Kimi-K3.1-Pro-Coder, 775GB)

单卡 96GB + 176GB RAM + NVMe 跑 K3.1: batch 36.8 / 单流 10.8 tok/s。机制: 按激活频率三档量化 (2.81/2.56/1.44bpw, 平均 1.92) + 静态三级驻留 (VRAM 134 颗/层 pin, RAM 269 pin, NVMe 493 mmap, **无 LRU**, 预取 staging 用完即弃) + 折叠 lookahead (预合成线性预测矩阵, 一次 GEMV 预测后 1-2 层路由, 有效率 97.9%) + dr-triton 打包传输 GPU 内解压。约 5-10k 行自研代码架在现成 Strata 引擎上。

- prefill 慢 (36.8) 的原因: 每 chunk 几乎点亮全部专家, 等于把温+冷 500GB 重新流一遍, NVMe 是关键路径; chunk 开大是线性免费加速
- 内存里的专家是**传到 GPU 算**的 (batch 导向); 单流下 CPU 原地算 (117GB/s) 比过 PCIe (64GB/s) 快近 2 倍, llama.cpp 式做法更优
- 对本项目可直接复用: 折叠 lookahead 做法, kimi_tier_profile.json 的每层专家排序, dr-triton 的 fp4 解压思路

## 5. 最重要发现: K3.1 路由在统计上几乎均匀

模型自带的 evaluation_results_93layer.json (本目录有扫描 eval_json_scan.txt) 披露了帖子没写的分层命中实测:

| 档位 (占专家数) | AIME 数学命中 | LCB 代码命中 |
|---|---|---|
| 热 VRAM (14.96%) | 16.56% | 15.64% |
| 温 RAM (30.02%) | 29.94% | 28.75% |
| 冷 NVMe (55.02%) | 53.50% | 55.61% |
| 深层 88-92 热+温 (45%) | 66.28% | 61.56% |

**按频率排序的热档命中只比均匀基线高 ~1 个百分点** — noaux_tc 负载均衡把全局频率训得接近均匀, 只有最深几层略集中。连锁结论:
- 静态频率热集对 K3.1 无意义 (top-24/32/48 覆盖率 ~= 2.7/3.6/5.4% 均匀基线); 他的 10.8 tok/s 靠的是 lookahead 预取 + page cache + batch 摊销
- 四级缓存方案下修: 命中按容量比例摊, 盘上 18% 专家 = 18% 命中 = 2.2GB/token NVMe 读 -> ~5-6 tok/s; **均匀路由下激进均匀量化 (1.6-1.75bpw 全放进 GPU+RAM+NPU) 严格优于分层量化**, 可回到 ~20 tok/s
- 对全驻留多卡方案反而是好消息: TP 天然均衡无热点
- 文献交叉印证 (arxiv 2505.16056, 测了 20 个 MoE): 带共享专家 + dense 层的 DeepSeek 构型局部路由一致性显著低于无共享专家构型 (Qwen3/OLMoE/Mixtral); 缓存甜点 ~2x 激活专家数; llama.cpp #24528 实测 Qwen3.5-122B decode top 10% 专家吃 ~80% 命中 (Gini 0.76) — **Qwen 系的偏斜数字不能套到 DeepSeek 构型上**; prefill 路由远比 decode 平

## 6. 调研材料文件索引 (/root/local_llm/glm53_research/)

- glm53_config.json / config_Kimi-K3.1-Pro.json / config_Kimi-K3.json: 官方 config
- glm53_index_page.txt / k31_index.json + headers/ k31_headers/: safetensors index 与全部分片头部 (range 请求抓取, fetch_headers.py / k31_fetch.py)
- analyze.py / k31_analyze.py: 逐张量统计脚本; tensors.json / k31_tensors.json: 全量张量表
- modeling_glm_moe_dsa.py / indexer_code.py: GLM 官方 DSA indexer 实现 (打分公式: score = sum_h w_h * ReLU(q_h . k_j /sqrt(128)), 128 维 key 缓存, top-2048)
- strata_readme.md / tiered_moe_loader.py / engine.py / decoder.py / kimi_tier_profile.json: Strata fork 代码与分层 profile (含每层 896 颗完整优先级排序, 无原始频率)
- runtime/strata_coder/: 可运行的 strata_coder 包, engine.py 已加 2 行 ROUTE_REC 插桩
- route_stats.py / analyze_routes.py: 路由统计与分析脚本 (未跑, 缺 tiktoken/blobfile + 语料; 权重已删, 重跑需先 python -u download_k31coder.py 重下 775GB, sha 校验 verify_sha.py)
- download_k31coder.py / verify_sha.py / dl_log.txt / sha_log.txt: 下载与校验 (115/115 sha256 全过后按决定删除了 96 个权重分片, /root/local_llm/models/kimi_k31_pro_coder 只留小文件)
- eval_json_scan.txt: 第 5 节数据来源
- serp_*.json / arxiv_*.html / gh_*.json: 文献检索原始材料

## 7. 买卡前 POC 清单 (未执行项)

1. 多线程并发 4B memcpy 是否被驱动串行化 (半天, 决定 16+ 卡方案生死)
2. ACL+CUDA 同进程三方汇合点单价实测 (预算 ~40µs)
3. 铃嵌进广播数据尾字 (省独立 gate 写, 验 h2d 写序)
4. decode 骨架: 2 芯装目标系统单芯真实切片 (GLM 19.5MB/层 x78) 跑真 kernel + doorbell + 层循环, 验收线 <=22ms/token
5. 阶梯买卡: 先 1 张, 测 2->4 芯汇合点单价/skew 斜率再补齐
6. GLM 真实专家权重逐层 W4A8 误差 (GPU 可做); K3.1 专家原生 fp4 免验只验 A8
7. (可选) 路由统计实测: 重下权重 + 装 tiktoken/blobfile + 两类语料各 2-5 万 token, 预期确认均匀
