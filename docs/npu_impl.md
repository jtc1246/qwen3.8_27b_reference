# C++ NPU (华为昇腾 310P) 实现说明

`qwen_npu/` 是文本主干在华为 Atlas 300I Duo (Ascend 310P3) 上的实现, 结构对应 [cpu_impl.md](cpu_impl.md) / [cuda_impl.md](cuda_impl.md): 模型常量、权重绑定、前向流程、采样和 C 接口与 CPU / CUDA 版一一对应。

当前状态 (2026-11-15): **v3, 权重源从 GGUF 换成 RedHatAI/Qwen3.8-27B-INT4 的 `model.safetensors`** (compressed-tensors, int4 对称 group128, AWQ 平滑 + GPTQ; 选型调研见该模型卡与本文第 2 节), matmul 全部走 cube (fp16 反量化 + mmad)。单芯片、单序列, 不含投机解码 (MTP 文件不加载)。**v4: decode 改走 W4A8** — 主干 matmul 在 decode 时用 u8×u8 整数 mmad (x 由 `k_qx8` 动态量化 int8, 权重 int4 原位喂 cube), 反量化向量瓶颈消除; prefill 仍走 fp16 反量化 + mmad (从同一份权重布局五平面拆解)。**v5: attention 上 cube + KV cache q8** — KV cache 由 fp32 改为 u8 (零点 128) + per-128 fp16 scale (存储量化、计算 fp16, 与 CUDA 版 q8_0 同构), `k_attn` 重写为按 kv 头合并单元的 fp16 mmad 版 (见第 3 节), decode 的 KV 读冗余从 ~9× 降到 1×, 斜率 4.9→~0.4µs/ctx token (64K 外推 ~8.8 tok/s)。KV 为 i8 对称量化 (无零点), decode 的 V 反量化只剩一条 Cast (scale 折进 P)。**v6 (2026-11-16): prefill 向量侧优化** — ① `k_gemm_w8p` 事件流水化 (qbuf 双缓冲、se/so 两半 V→MTE3 乒乓、L1 w/x 槽按组奇偶双缓冲、MTE1/M 事件串接), 单 kernel 34~43 TFLOPS (原 ~21); ② DeltaNet 拆出 `k_dn_pre` (按 token 分核批量算 β/decay/softplus + q/k L2 归一化原位写回), `k_deltanet` 去掉全部逐 token 小向量操作与标量同步 (β/decay 表常驻 UB 直接 GetValue, 门控 1/sqrt 走向量广播, 输入/输出事件双缓冲); ③ `k_gemv_f32t`/`k_conv`/`k_silu_mul` 事件流水化。全部改动与 v5 在正确同步下逐位一致。**同时修复一个 v1 起就存在的宿主竞态**: `aclrtMemcpy` H2D 不进 stream 队列, 多 chunk prefill 时下一 chunk 的 cos/sin/tok 上传会在上一 chunk 的 kernel (qk_rope 等) 执行前覆盖数据 (无 sync 的 chunk 边界即 >256 token 都受影响, >512 token 可稳定观测到 build 相关差异), `eval_chunk` 开头加 `Acl::sync()` 修复, 修复后精度还略有提升。性能: prefill 1024 为 295 tok/s (v5 206), prefill 300 为 256 (v5 176); decode 与 v5 持平 (同条件对照一致, ~10.9-11.3 区间随机器状态波动, dn_pre 额外启动开销 ~0.3%)。精度 (竞态修复后新基线, 与修复后的 v5 逐位一致): decode KL vs CPU fp32 = 2.42e-2 / top1 29/32, 1100 tok prefill KL 7.82e-2 (修复前 9.53e-2); vs CPU (GGUF Q4_K_M) 300 tok prefill 后贪心 19/20, top10 10/10。python 侧 (`qwen.py` / `server.py`) 尚未接入该后端, 目前通过 ctypes 直接加载测试。

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `kernels_npu.cpp` | 全部 Ascend C kernel (device 侧) |
| `qwen.cpp` | 权重加载 (含 HF→本实现的变换)、重打包与上传、前向 (`eval_chunk`)、采样、自测、调试接口、C 接口 |
| `acl.h` | AscendCL 运行时薄封装 (初始化、`ck` 检查、分配、拷贝、同步) |
| `st.h` | safetensors 加载 (mmap + 手写 JSON 头解析) |
| `repack.h` | checkpoint 权重 → NPU 自定义格式的重打包 (host 侧, 见第 2 节) |

构建 (与 CPU / CUDA 版不同, **不用 musl**, 因为必须链接 CANN 的 glibc 动态库):

```bash
export CANN_HOME=/path/to/ascend-toolkit/latest  # 包含 set_env.sh 的目录
./qwen_npu/compile.sh
```

- 需要: CANN toolkit + 与设备匹配的 ops 包、系统 g++ (C++17)、cmake; `compile.sh` 里的 cmake 路径按机器情况通过 `CMAKE_BIN` 覆盖
- kernel 通过 CANN 的 `ascendc.cmake` 框架编译 (SOC `Ascend310P3`, 核函数直调模式), host 通过生成的 `aclrtlaunch_*.h` 启动 kernel
- 产物 `cmake_build/libqwen_npu.so`, 带 RPATH 指向 CANN lib64 和驱动目录; **运行前仍需 `source $CANN_HOME/set_env.sh`** (CANN 二级依赖和环境变量)
- 链接时需要 `-Wl,-rpath-link` 指向 CANN lib64 与驱动目录, 否则 ld 解析 so 的二级依赖 (libascend_hal.so、libruntime.so 的符号) 会失败, CMakeLists 已包含

C 接口与 CPU 版相同 (`qwen_init(path, device, max_ctx)` 第二个参数是 NPU 编号), 直接 ctypes 加载即可:

```python
lib = ctypes.CDLL("qwen_npu/cmake_build/libqwen_npu.so")
lib.qwen_init(b"/path/to/Qwen3.8-27B-INT4/model.safetensors", 0, 4096)
```

## 2. 权重来源与重打包

权重源是 [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) 的 `model.safetensors` (18GB): 主干 linear 全部为 compressed-tensors `pack-quantized` int4 **对称** group128 (`weight_packed` int32 每词 8 个 nibble, nibble = q+8; `weight_scale` bf16 [n][k/128]; 无 zero_point 无 g_idx, `actorder: static` 只是求解期重排, 列序不变), embed/lm_head/norm/conv/α/β 为 bf16。选它的原因: 对称无 bias 项 + g128 的 scale 元数据少, 是对本 kernel 最友好的 4bit 格式; AWQ 平滑 + GPTQ 且模型卡有 6 基准恢复率 98.5~101%。

**HF → 本实现的权重变换** (与 GGUF 转换语义一致, 见 [model.md](model.md) 第 11 节, 在 host 加载时完成):
norm 权重 +1 (`linear_attn.norm` 除外)、`ssm_a = -exp(A_log)`、DeltaNet v 头 grouped→tiled 重排 (tiled 头 j ← HF 头 `(j%16)*3 + j/16`, 涉及 in_proj_qkv 的 v 行 / in_proj_z 行 / α β 行 / A_log / dt_bias / conv v 通道 / out_proj 列)。行重排对量化表示无损; out_proj 的列重排以 128 为块, 恰好与量化组对齐, 也无损。

matmul 权重重打包成 **cube GEMM 的 fractal 量化布局** (`repack.h`), device 侧向量单元反量化成 fp16 后喂给 cube; scale 转 fp16 (fp16 乘与 CUDA fp16 版同级误差):

| 格式 | 来源 | 字节/权重 | 内容 |
|---|---|---|---|
| I4W8 | 主干 int4 | 0.547 | W4A8 布局: 包字节 + per-128 fp16 scale + f32 W2 修正量 (见下) |
| I8F | lm_head (bf16 经 host per-16 对称 RTN) | 1.125 | int8 + 每 16 元素 fp16 scale (fp16 反量化路径) |
| F32 | α / β (bf16 转 f32) | 4 | 自然行布局 (向量 GEMV 用) |

**I4W8 布局** (每核连续; 每 256 行块: 每 sp = 组对 (2u, 2u+1): `[4t][NI][512B 包字节]` + meta `[s fp16 NI·32B ×2组][W2 f32 NI·64B ×2组]`):

- 包字节 (fractal 内 `32r+2n+p`) = `q(组2u, k=t·32+p·16+r)+8` 的低 nibble | 组 2u+1 同 k 的高 nibble。这个顺序是按 **b8 LoadData 转置的硬件语义**(按 u16 做 16×16 转置, u16 两字节分别落到 k 和 k+16) 反推的: `And 0x0F0F`/`0xF0F0` 各一条就得到两组的转置源 fractal (hi 组值天然 ×16), 转置加载后即标准 `[16n][32k]` b8 fractal — **decode 的全部向量解码 = 2 条 And**
- decode (`k_gemm_w8d`): u8(x, 零点128)×u8(w, nib) mmad (dav_m200 仅支持 (i8,i8)/(u8,i8)/(u8,u8) 组合), 每 128-k 组内 s32 **精确**累加, drain 后 f32 域乘 scale 并做两个零点修正: `y=Σ s'·sx·D + s·(−8·Sx·sx) + W2·(−128·sx)`, W2 = h2f(存储 s)·Σ(存储 nib) 用 **f32** 存 (fp16 存会把大数相消误差放大到 0.7%); hi 组的 /16 在 f32 域派生 (fp16 存 s/16 会掉入次正规被向量单元 flush, 曾导致 decode 发散); 次正规 scale 在 repack 时连同 W2 一起钳 0。drain 滞后 2 组软件流水 (drain 的 WaitFlag 会挡 V 队列)
- x 侧 (`k_qx8`): per-token per-128 组对称 int8 (存 u8), f32 域 RINT 与 host lrintf 逐字节一致; 输出 4 份相位旋转副本 (b8 gemv 的 A 偏移粒度是 512B, 组 g 的 128B 放在副本 g%4 的 512·(g/4) 处) + 组 scale/修正量 ([KS][8] 复制 8 份供 src1BlkStride=0 当标量乘)
- prefill (`k_gemm_w8p`): 同一份包字节五平面拆解反量化 fp16: And 0x000F/0x0F00/0x00F0/0x7000+0x8000(符号位单独提再相减) → Cast int16→half → Muls 2⁻ʲ 还原 q+8 → Adds −8 → 乘 s; 每平面每 chunk 恰为一个 [16k][16n] k 主序 fp16 fractal, MTE3 跨步交织成 [8ki][NI] 后走原 fp16 mmad 路径 (x 仍用 `k_pack_x`)

I8F (lm_head) 的 fractal 布局 (老 `k_gemm` fp16 流水):

- 行按 8 核均分 (nr = n/8, 要求 n 为 128 的倍数), 核内按 256 行分块, 块内按 128-k 切片
- 每切片 = `[8 ki][NI ni]` 个 512B fractal + scale 区 `[ki][ni][16 half]`; fractal 内为 **[16k][16r] k 主序** — 这样 scale 恰好按 32B 块变化, kernel 里用 `src1BlkStride=0` 的块复用技巧逐 fractal 乘 scale, 完全不需要广播展开 (310P 的 Brcb/Gather 都慢到不可用); L1→L0B 用 `LoadData` 的 `ifTranspose` 转回标准 [16n][16k]

其它:

- **matmul 合并**: DeltaNet 的 in_proj_qkv+in_proj_z 拼成 [16384][D] 一个 gemm (输出 [T][QKVZ], conv 只处理前 QKV_DIM 通道), 全注意力的 q+k+v 拼成 [14336][D] (输出 [T][QKVC]), α+β 拼成 f32 [96][D] 一次 GEMV
- `embed_tokens` 由 host RTN 成 int8 后存 v1 的 wave 布局 (I8 格式, f32 scale, 反量化与 host 参考逐位一致), 只给 `k_dq_i8` 做 embedding 查行
- 权重显存 17.5GB (v3 为 19.2): 主干 13.3 + lm_head 1.43 + embed 1.59 + α β 等 0.1; decode 每 token 流读 ~14.8GB (v3 为 16.7); 重打包 + 上传约 26s

norm、ssm_a 等小向量转 f32 上传; `conv1d.weight` 重排成 `[4][QKV_DIM]` 平面 (v 通道按 tiled 序); conv 状态布局 `[3][QKV_DIM]`。

## 3. Kernel

全部在 `kernels_npu.cpp`, block_dim 8 (k_gemv_f32t 为 n/8), 激活全部是 GM 上的 fp32。

| kernel | 说明 |
|---|---|
| `k_gemm_w8d` | 主干 matmul 的 **decode (m=1) 路径**: MTE2 (包字节+meta) → And ×2 (两组转置源 fractal) → MTE3 (L1) → b8 转置 LoadData → u8×u8 mmad (组内 s32 精确累加) → drain (滞后 2 组软件流水) + f32 scale/零点修正累加。x 为 `k_qx8` 的 u8 旋转副本, L0A 整段驻留 (k≥16384 分两半重载) |
| `k_gemm_w8p` | 主干 matmul 的 **prefill (m≤256) 路径**: 五平面 And 拆解反量化 fp16 → MTE3 跨步交织 → fp16 mmad (fp32 累加整个 k); x 由 `k_pack_x` 预打包。v6 事件流水: qbuf 双缓冲 (sp 奇偶), se/so 两半 V→MTE3 乒乓 (每组按偶/奇 byte 平面分两半反量化, 半片粒度重叠), L1 w/x 槽按组奇偶双缓冲, MTE1/M 单缓冲事件串接; 实测 34~43 TFLOPS (独立测试台, 原 barrier 版约 21) |
| `k_qx8` | decode 的 x 量化: f32 [k] → per-128 组对称 int8 (u8 存, 零点 128) 4 份旋转副本 + 组 scale/修正量; f32 域 RINT, 与 host lrintf 逐字节一致; 单核, 同一 x 多个 matmul 只做一次 (随 pack 缓存一起失效) |
| `k_gemm_i8` | lm_head (I8F) 的 fp16 反量化 mmad 路径 (老 kernel); GGUF 时代的 i8b/i4/i4s 模板仍在, host 不再使用 |
| `k_pack_x` | f32 [T][k] → fp16 zZ 切片布局 [kc][mi][ki][16m][16k], 三级双缓冲流水 |
| `k_gemv_f32t` | α+β 合并矩阵 ([96][D] f32 自然行布局) 的 GEMV, 行常驻 UB (每块 8 行, block_dim = n/8 = 12, 物理 8 核跑两波——受输出 32B 粒度限制暂无法改 8 块), T 个 token 在 kernel 内循环; v6: x 双缓冲预取 + 输出 [T][8] 整段驻留最后一次跨步写出 |
| `k_dq_i8` | 按行号表反量化整行到自然顺序 f32 (embedding 和自测用, wave 布局) |
| `k_add_norm` | x += add (可空) + RMSNorm, 保存最后一个 token 的 hidden; token 分核轮转 |
| `k_silu_mul` | g = silu(g)·u, 按核均分逐元素; v6: 段双缓冲事件流水 (重载守卫用 MTE3_MTE2, 见第 4 节第 15 条) |
| `k_conv` | 因果 conv1d (核宽 4) + SiLU, 作用于合并输出 [T][QKVZ] 的前 QKV_DIM 通道; 通道均分到核; v6: 6 缓冲轮转 (状态 3 + xn + 预取 2, token t 占用槽 t..t+3 mod 6, 预取目标 t+4), 输出双缓冲, 无逐 token barrier |
| `k_dn_pre` | **v6 新增**, DeltaNet 预计算 (按 token 分核): β=sigmoid(b)、decay=exp(a·softplus) 批量算好写 bd [T][96]; q/k 行 L2 归一化(含 QSCALE)原位写回 qkvz。数值链与原逐 token 版逐位一致 (两半分别树求和再相加、先 1/x 再乘 QSCALE) |
| `k_deltanet` | 每核 6 个 v 头顺序处理, 状态 S[128][128] 常驻 UB; v6 重写: 输入为 `k_dn_pre` 归一化后的 qkvz + bd 表 (常驻 UB, β/decay 标量直接 GetValue, 无逐 token V↔S 同步), 门控 RMSNorm 的 1/sqrt 走 Brcb+Div 向量广播, 输入 4 行/输出行事件双缓冲; 剩余为递推本身的向量下限 (decay Muls S + 两次 S⊙广播 + 两棵列求和树 + MulAddDst ≈ 3.5µs/头/token) |
| `k_qk_rope` | (t, 头) 单元分核; 输入 q+k+v 合并输出 [T][QKVC]; RMSNorm + 交错 MRoPE + **K/V 量化写入 q8 cache** (i8 对称 per-128 组, 量化用 fp16 舍入后的 scale, RINT 后钳 ±127, half→int8 Cast 实测可用); cos/sin 由 host 算好上传 |
| `k_attn` | **cube 版 (v5)**: KV q8 反量化 fp16 (Cast+Adds−128+Mul, scale 以 16 复制存放配合 src1BlkStride=0 块乘) 后 Q·K^T 与 P·V 都走 fp16 mmad (fp32 累加)。scores 按转置方向算 (scoresT[j][m]=K·Q^T, A=K 流式 48-key tile, B=Q^T 常驻 L1), softmax 的行归约全部变成整列向量操作, 无行广播。decode (T=1): 单元 = (kv头, ctx 两段) 共 8 个, tile = 96 key, 单遍在线 softmax (sT 用 [96][16] 紧凑布局, 减 max 的广播靠 tmax 复制 4 份对齐整 repeat), staging 单缓冲 + 全事件流水 (反量化后立即释放, DMA 与余下链重叠)。**输出按转置方向累加 (outT[d][m] = V^T·P'_g 双 mmad, 每 d-half 一个)**: V 只 Cast 不乘 scale (scale 以 [j][2][16] 隔块跨步折进 P'), V 的 L0A 一次转置加载 48 fractal, OaccT 的重缩放/累加因 m 在 lane 维而各只需 1 条整 repeat 指令; 部分结果 (OaccT/m/l) 写 attp; prefill: 单元 = (kv头, 16-query 块), m = 6头×16token = 96 行, 两遍扫描 (K 二读走 16MB 共享缓存), 输出在 L0C 跨 tile 累加 (无重缩放), 门控在 kernel 内。因果边界行按 16-lane 周期掩码屏蔽 |
| `k_attn_merge` | decode 两段部分结果合并 (标准 flash-decoding 合并: exp(m−M) 加权 + 1/l, OaccT 布局下全是整 repeat 广播) + **单位阵 mmad 做 [d][m]→[m][d] 转置** (fp16 一次舍入, 后续喂 a8 量化的 wo 无感) + sigmoid 门控, 单元 = kv 头 |

## 4. 310P 的主要限制与踩过的坑

v1 实测的坑 (向量部分) 依然成立:

1. 向量单元**没有整数移位指令** (`ShiftLeft/Right` 静默输出垃圾) → nibble 用乘法 + RINT 解码
2. Cast 组合不全, 不支持的组合静默输出垃圾; `Reciprocal`/`Rsqrt` 低精度, norm 用 `Sqrt`+`Div`
3. 同 kernel 内循环, 下一轮 DataCopy (MTE2) 不等上一轮向量读取, 覆盖前必须 `PipeBarrier<PIPE_ALL>` 或 TQue
4. 未初始化 UB 可能是 NaN 位型, 固定 repeat 会扫到的 padding 区要清零
5. 标量 `GetValue`/`SetValue` 走 S 管线, 与向量之间要 `PipeBarrier<PIPE_ALL>`
6. Level-2 API 每次重设 mask; 热路径用 Level-0 + `isSetMask=false`

v2 (cube 化) 新增的实测结论:

7. **mmad / 数据通路** (dav_m200): `Mmad` 支持 fp16×fp16→f32; 布局 L0A zZ `[M1][K1][16m][16k]`、L0B `[K1][N1][16n][16k]`、L0C `[N1][M1][16m][16n]`; **m=1 时 mmad 进入 gemv 模式, A 按 k 连续读取且按 512B fractal 粒度访问** (A 片偏移必须 512B 对齐, L0A 末尾要留余量, 否则越界 507015); UB→L1 (`copy_ubuf_to_cbuf`)、L1→L0A/B (`LoadData`, 可转置)、CO1→UB 都可用, 但 CO1→UB 的 DataCopy 必须显式 `BlockMode::BLOCK_MODE_MATRIX` (默认 blockMode 走不到任何分支, **静默不拷贝**), blockLen 单位是 fractal
8. **向量速率** (实测 ns/repeat): Mul/Add f32 和 f16 都是 ~2 周期/repeat (f32 32 元素/周期, f16 64), Cast i8→h ~1.2, BlockReduceSum ~1; **`MulAddDst(f32,h,h)` 5 周期/repeat (很慢)**; **`Brcb` 慢到不可用** (f32 ~3.6 周期/块, f16 ~9.6), **`Gather` 同样慢** (~24 周期/repeat) → 任何按块广播都要用布局技巧 (`src1BlkStride=0`) 而不是展开
9. **同 V 管线内依赖硬件保序** (微基准证实依赖链与独立指令同速), 热循环里的 `PipeBarrier<PIPE_V>` 不需要; 指令发射开销 ~14ns/调用
10. **片上共享缓存 ~16MB, 读带宽 ~1TB/s** (错相反复读 ≤16MB 区域实测 1010GB/s, ≥64MB 回落 ~195GB/s) → prefill 时 x/A 的反复读取几乎免费, k_gemm 用 L0C 常驻 [m][256] 的激进 tiling
11. 手写跨管线事件的 id 必须用 TPipe `AllocEventID` 分配, 自己写死 0/1 会与框架内部冲突 (出现形状相关的"随机"错数)
12. kernel 启动开销 ~5µs/次 (同 kernel 连发或交替都一样), 一个 decode token ~800 次启动 ≈ 4ms

v5 (attention cube 化) 新增:

13. **Level-0 API 的 mask 参数传单个 uint64 时是 counter 模式 (每 repeat 处理的元素个数), 不是位掩码**: 传 0xFFFF 意为 65535 个 → 被钳到整 repeat (64 个 f32), 而不是 16 lane。要位掩码必须用 `SetVectorMask<T>(hi, lo)` + `isSetMask=false` (或传 uint64 数组)。踩坑表现: 看似局部的 masked 写实际整 repeat 生效, 多次调用互相重叠踩踏, 且最后一个 repeat 越界写坏相邻 UB 缓冲 (本例 Oacc 累加越界 48 f32 恰好覆写 runm, 表现为"softmax 正确但导出的 max 是垃圾"这种矛盾现象)
14. `LoadData2DParams` 的 `srcStride` (源 fractal 间隔) 和 `dstGap` (目的 fractal 间隙, 0=连续) 都可用, 配合可把 L1 的 [K1][M1]/[N1][K1] 序加载成 L0 需要的序; L1→L0A 也支持 `ifTranspose` (m200 实测)

v6 (流水化) 新增:

15. **`WaitFlag<SRC_DST>` 由 DST 管线执行, 事件类型必须按"谁等谁"选**: 要挡 MTE2 重载必须用 `MTE3_MTE2`/`V_MTE2`; 误用 `MTE3_V` 时等待发生在 V 管线, MTE2 完全不被挡, 跑飞覆盖在用缓冲。踩坑表现: k_silu_mul 用 `WaitFlag<MTE3_V>` 想挡重载, 结果大 T 时输入被下一段覆盖, 全模型表现为 prefill logits 大错而 decode (单段) 正常
16. **宿主 `aclrtMemcpy` H2D 是同步拷贝但不排队到 stream**: 对已入队未执行的 kernel 要读的缓冲 (cos/sin/tok) 写入会直接覆盖 (WAR), 多 chunk prefill 必现; 且 corruption 随 build 的时序不同而不同 (每个 build 内确定, 跨 build 不同), 表现为"两个应当逐位一致的 build 在 >512 token 时结果不同"。修复: 覆盖宿主写之前 `aclrtSynchronizeStream` (eval_chunk 开头, decode 时 stream 已空闲, 代价可忽略)

## 5. 显存

- 权重约 17.5GB (第 2 节, 含 embed 1.59GB); KV cache q8: 每 token 每层 2560B (K/V 各 1024B u8 + 256B scale), 16 层共 40KB/token (fp32 时代为 128KB), ctx 4096 时 160MB; DeltaNet 状态 201MB; 激活按 CHUNK = 256; x 打包缓冲 8.9MB
- cache 布局 (每层): kq/vq `[N_KV][max_ctx][256]` i8 (按头分平面, attention tile 读取连续), ksc/vsc `[N_KV][max_ctx][2][16]` half (每组 scale 复制 16 份); 分配末尾留 16KB, 允许固定 96-key tile 越量读
- 实测 ctx = 4096 时总占用 17.15GB (单芯片可用约 46GB, ECC 已关)

## 6. 调试接口 (正式功能不依赖, 但保留)

- 环境变量 `QWEN_NPU_STOP=<层号>` / `QWEN_NPU_PHASE=<0/1/2/10/11/12>`: 跑完该层 (或该层内某阶段) 直接返回 (v6 起 STOP 的提前返回也推进 n_past/G_next_pos, 多 chunk 下与完整运行的位置语义一致; PHASE 返回路径未改)
- `QWEN_NPU_PROF=1` + `qwen_prof_dump()`: 逐 kernel 同步计时 (会拖慢, 小 kernel 数字被同步开销抬高)
- `qwen_dbg(name, out, n)`: 把中间激活缓冲拷回 host
- v1 的 `k_attn` dump 模式 (T=-1) 已随两遍扫描重写移除

## 7. 验证

测试脚本在仓库外的实验目录 (`tests/npu_port`、`tests/npu_attn2`、`tests/npu_api`、`tests/npu_gemm`、`tests/npu_vlat`、`tests/npu_cube`、`tests/npu_l2`, 依赖本机绝对路径, 未入库; `tests/npu_attn` 是 v2 fp32 版 k_attn 的旧测试台, 已随 v5 重写过时):

- `qwen_selftest` (`smoke_st.py`): tok_embd 反量化 vs host 参考逐位一致 (差 0); 各 matmul 权重 (I4W8 含行重排/列重排/多段拼接的、I8F、F32) m=1 GEMM vs 参考点积 — I4W8 的参考把 x 按与 `k_qx8` 逐字节一致的规则先量化, 相对误差 ~1e-7 (整数路径精确); int4 解包公式已与 compressed-tensors 库逐位对过
- 与 CPU (GGUF Q4_K_M) 对比 (`vs_cpu_st.py`): 两边是不同的 4bit 量化, logits 只会接近不会相等 — 300 tok prefill 后最大差 0.99 (最大值 10.5), top1 一致; 20 步贪心 19/20 一致 (喂同一轨迹)
- KL 评估 (`w8_eval.py`, 方法同 bench_cmp/prec): 真实文本 1100 prefill + 32 步 teacher forcing, decode KL vs CPU fp32 = 2.42e-2 / top1 29/32, prefill KL 7.82e-2 (v6 竞态修复后的新基线; 修复前为 2.53e-2 / 30/32 / 9.53e-2, 老数字被宿主竞态轻微污染) — 与 v3 fp16 激活路径完全同级, A8 增量被 W4 噪声淹没 (预研时的纯 A8 增量实测 2.03e-3)。v6 的全部 kernel 改动在修复竞态后与 v5 逐位一致 (md5 对比验证, 300/524/1100 token)
- 真实生成 (`gen_st.py`): chat 模板 + 贪心, 思考与正文连贯正确 (与 v3 逐字一致)
- 曾踩的数值坑 (都已修): fp16 存 s/16 掉入次正规被向量单元 flush → 个别 z 行错 → decode 发散 (meta 改存原始 s, /16 在 f32 域派生, 次正规 scale 连同 W2 钳 0); W2 用 fp16 存时大数相消误差放大到 0.7% (改 f32 并与存储 scale 精确一致)
- `tests/npu_gemm`: k_gemm 独立测试台 (三种格式 × m=1/5/16/33/100/256 × 各形状), 与 host 模拟 fp16 反量化的参考对比
- `tests/npu_vlat`: 向量指令速率/发射开销/启动开销微基准 (第 4 节第 8/9/12 条的出处)
- `tests/npu_cube`: mmad fractal 布局验证 (第 4 节第 7 条的出处)
- `tests/npu_attn2` (v5): `attn2_test` 对 k_attn/k_attn_merge/k_qk_rope 与 host double 参考全面对比 (decode n_past 扫 0~3100 含段/tile 边界, prefill 各 (T, n_past) 含因果边界, 全部相对误差 ~2e-4); `dbg_test`/`k_dbg.cpp` 是逐环节隔离测试台 (反量化链/两条 mmad 通路/P^T cast/双 mmad 共存/全 tile 复刻), 第 4 节第 13 条就是用它定位的

## 8. 性能 (v6, Atlas 300I Duo 单颗 310P3, ECC 关)

| 项目 | v6 (prefill 流水化) | v5 (q8 KV + cube attn) | v4 (W4A8 decode) | v3 | v2 | v1 |
|---|---|---|---|---|---|---|
| prefill (300 tok, 256+44 两 chunk) | 256 tok/s | 176 | 161 | 215 | - | 1.9 |
| prefill (1024 token) | 295 tok/s | 206 | 150 | 178 | 178 | - |
| decode (ctx ~1024 / ~3100) | 持平 v5 | 11.3 / 11.1 tok/s | 10.6 / 9.6 | 6.67 / 6.23 | 5.1 / 4.8 | - |
| 模型加载 (重打包 + 上传) | 25s | 26s | 26s | 23s | 21s | 14s |

分解与说明:

- decode ~89ms/token 几乎全是权重 DMA (每 token 流读 ~14.8GB, I4W8 原型独立压测 170-192GB/s 接近纯 DMA 204), 绝对值随机器状态在 ~10.9-11.3 波动, v6 与 v5 同条件对照持平 (dn_pre 额外 48 次启动 ≈0.3%); attention 斜率实测 ~0.39µs/ctx token (演进见 v5 说明)
- prefill 1024 分解 (v6, 开 prof 含同步开销, 总 3.95s / 实际 3.59s): gemm 1328ms (流水化后 ~40T, v5 2361), deltanet+dn_pre 1393ms (v5 1724, 已接近 fp32 向量递推下限 ~3.5µs/头/token), gemv_f32t 312 (两波调度所限), attn 310 (未流水), pack_x 305, 其余小 kernel ~300
- 继续往上的空间: deltanet chunk 化上 cube (大头, 涉及混合精度需实测 KL), prefill 版 k_attn 流水化, pack_x 合并进上游 kernel; 全做完单芯估计 ~350-400
- KV q8 对精度的增量 ~7e-4 KL, 低于预研时纯 A8 的增量 (2.03e-3), 被 W4 本底淹没
- 原型与微测试在本机 tests/npu_w4a8 (W4A8 + w8p 流水版 pf_test) 和 tests/npu_attn2 (v5 attention, 见第 7 节); 向量/标量除法逐位一致性微测试在 tests/npu_div (排查 v6 差异时写的, 结论: 一致)

## 9. 已知不足与下一步

1. **DeltaNet chunk 化上 cube** (prefill 剩余大头 1.4s): 向量侧修补已做完 (v6, 接近递推下限); 彻底方案是 chunk 化 (WY 表示), chunk 内两个大矩阵乘走 fp16 mmad、S 的累加链保 fp32 (fp16 舍入只发生在单次读出操作数, 不跨 token 累积), 需实测 KL 验证
2. **prefill 版 k_attn 流水化** (310ms, 当前整段 PIPE_ALL 串行); **k_pack_x 合并进上游 kernel** (add_norm/silu 直接多产出一份 fp16 zZ, 省 305ms 的独立 pass)
3. **decode attention 斜率余量**: 实测 ~0.39µs/ctx token, DMA 理想 ~0.2; 已做: i8 对称 (省零点 Adds)、V scale 折进 P (V 反量化只剩 Cast)、outT 方向累加 (累加/重缩放各 1 条指令)。剩余是 K 的 Cast+Mul、K/V 各 16 次列片 copy 与残余不可重叠, 继续压空间不大
4. **k_gemv_f32t 两波调度** (312ms): block_dim 12 > 物理 8 核; 改 8 块受输出 32B 对齐粒度限制 (每核 12 行 = 48B), 需要换输出布局才有解
5. 双芯片张量并行 (每颗一半权重, 两倍带宽)、与 python 服务接入
6. 自定义采样 (wasm) 未移植; 图片按计划固定走 CPU / CUDA, 不上 NPU
