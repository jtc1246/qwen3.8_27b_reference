# C++ NPU (华为昇腾 310P) 实现说明

`qwen_npu/` 是文本主干在华为 Atlas 300I Duo (Ascend 310P3) 上的实现, 结构对应 [cpu_impl.md](cpu_impl.md) / [cuda_impl.md](cuda_impl.md) 的第一版: 模型常量、权重绑定、前向流程、采样和 C 接口与 CPU / CUDA 版一一对应。

当前状态 (2026-11-15): **v1, 只保证正确性, 性能未优化**。单芯片、单序列、全 fp32 计算、只用向量单元 (未用 cube)。与 CPU 版的数值对齐已全部通过; decode 约 1.8 tok/s, prefill 约 1.9 tok/s (见第 8 节, 优化方向见第 9 节)。python 侧 (`qwen.py` / `server.py`) 尚未接入该后端, 目前通过 ctypes 直接加载测试。

## 1. 文件与构建

| 文件 | 内容 |
|---|---|
| `kernels_npu.cpp` | 全部 Ascend C kernel (device 侧) |
| `qwen.cpp` | 权重重打包与上传、前向 (`eval_chunk`)、采样、自测、调试接口、C 接口 |
| `acl.h` | AscendCL 运行时薄封装 (初始化、`ck` 检查、分配、拷贝、同步) |
| `repack.h` | GGUF 权重 → NPU 自定义格式的重打包 (host 侧, 见第 2 节) |
| `gguf.h` / `quants_ref.h` | 与 `qwen_cuda/` 相同 (只改了 include guard) |

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
lib.qwen_init(b"/path/to/Qwen3.8-27B-Q4_K_M.gguf", 0, 4096)
```

## 2. 权重重打包

310P 向量单元没有整数移位指令, k-quant 的位打包格式无法在 device 高效解码, 因此加载时在 host 把每个矩阵重打包成以下格式再上传 (`repack.h`)。所有格式都能**逐位还原原始反量化值**: scale / bias 用 f32 存 (d×sc 等乘积在 fp32 下是精确的, 与 CPU 版的 fp32 中间值一致), 量化值用原始整数存。

| 格式 | 来源 | 行布局 | 字节/权重 |
|---|---|---|---|
| I4 | q4_k | nibble 两平面 + per-32 f32 scale + f32 bias | 0.75 |
| I8 | q8_0, q6_k | int8 + per-16 f32 scale | 1.25 |
| I8B | q5_k | int8 + per-16 f32 scale + f32 bias | 1.5 |
| F32 | f32 (alpha / beta) | 原样 | 4 |

- 值部分是 **wave 布局**: 自然顺序的 8-float 块按轮转重排 (I8 系 2 路, I4 4 路), 使得 kernel 里 `Mul`/`MulAddDst` 逐元素乘加后, 一次 `BlockReduceSum` (per-8 求和) 直接得到每个量化组的部分和; x 在 kernel 内用带块跨步的 `Adds` 做一次对应重排
- I4 的 nibble 解码用乘法技巧: 字节 b → half, 高 nibble = `CAST_RINT(b×0.0625 − 0.46875)` (小数部分恒在 ±0.47 内, 结果精确), 低 nibble = b − 16×高
- 组数补齐到 64 的倍数, 补齐区 scale / bias 全零 (kernel 侧还会把对应 UB 清零, 否则 0 × 未初始化可能得 NaN)
- 行跨度 64B 对齐; 总显存约 22GB (原始 GGUF 16.2GB), 重打包 + 上传约 12s
- norm、ssm_a 等小 f32 向量原样上传; `ssm_conv1d.weight` 重排成 `[4][QKV_DIM]` 平面; conv 状态布局改为 `[3][QKV_DIM]` (与 CPU / CUDA 的 `[c][3]` 不同, 为了向量化)

## 3. Kernel

全部在 `kernels_npu.cpp`, block_dim 8 (单核场景除外), 激活全部是 GM 上的 fp32。

| kernel | 说明 |
|---|---|
| `k_gemv_{f32,i8,i8b,i4}` | 单 token 矩阵乘 y[n] = W·x。每核 n/8 行; 行加载 TQue 双缓冲与计算重叠, 小行 8 行一批拷贝; 反量化 + `Mul`/`MulAddDst` 逐元素乘加 + `BlockReduceSum` 得组和 ×scale (+bias×组内 x 和), 两级 `WholeReduceSum` 规约, 16 行一批写出 (无标量操作); 热路径全部 Level-0 调用 + mask 只设一次 |
| `k_dq_{f32,i8,i8b,i4}` | 按行号表反量化整行到自然顺序 f32 (embedding 和自测用); scale 用 `Brcb` 展开成 per-8, wave→自然顺序用 DataCopy 的 32B 块跨步写 GM |
| `k_add_norm` | x += add (可空) + RMSNorm, 保存最后一个 token 的 hidden; token 分核轮转 |
| `k_silu_mul` | g = silu(g)·u, 按核均分逐元素 |
| `k_conv` | 因果 conv1d (核宽 4) + SiLU; 通道均分到核, 状态 3 个平面在 UB 轮转 |
| `k_deltanet` | 每核 6 个 v 头顺序处理, 状态 S[128][128] 常驻 UB; S^T·k 和 S^T·q 用 `Brcb` 行标量广播 + 两次 Level-0 `Mul` (src1 块跨步 0) + 7 层二分树求和; S 的 rank-1 更新用同样跨步的 `MulAddDst`; β/decay/softplus 每 token 用 48 长度的向量算; q/k 归一化、门控 RMSNorm、silu(z) 都在同一 kernel |
| `k_qk_rope` | (t, 头) 单元分核; RMSNorm + 交错 MRoPE + 写 KV cache。**cos/sin 不在 device 算** (无 Cos 指令): host 按 pos3 和分段规则算好 `[T][32]` 的 cos/sin 行随 chunk 上传 (与 CPU 版同用 host cosf, 数值一致) |
| `k_attn` | (t, q头) 单元分核; 64 个 key 一批在线 softmax: 分数用 4 个四分之一段的跨步 `Mul`/`MulAddDst` + `WholeReduceSum`; 合并用 exp 槽位技巧 (sc 第 65 个位置放旧 max, 和分数一起减新 max 求 exp 得到校正系数 c); p·V 用 `Brcb` + 跨步 `Mul` + 树求和; 尾批把 V 的多余行清零 |

Prefill 目前是**占位方案**: host 对每个 token 逐个调用 GEMV kernel (每 token 约 650 次 launch, 每次约 4µs)。正确性没问题但速度差, 真正的 prefill 需要 cube (fp16 GEMM), 见第 9 节。

## 4. 310P 的主要限制与踩过的坑

实现中实测确认的 310P (dav_m200) 限制, 直接影响上面的设计:

1. 向量单元**没有整数移位指令** (`ShiftLeft/Right` 是 NOT_SUPPORT 且运行不报错输出垃圾) → nibble 用乘法 + RINT 解码
2. Cast 组合不全: 无 int16→float 直转、half→int16 无 CAST_FLOOR; **不支持的组合同样静默输出垃圾**
3. `Reciprocal`/`Rsqrt` 是低精度估计指令 (误差 ~4e-3), norm 必须用 `Sqrt`+`Div` (~1e-7)
4. 同一 kernel 内循环中, **下一轮 DataCopy (MTE2) 不会等上一轮向量读取完成**, `PipeBarrier<PIPE_V>` 拦不住 MTE2; 覆盖正在读的缓冲前必须 `PipeBarrier<PIPE_ALL>` 或用 TQue。这类 bug 在各行数据相似的测试里完全隐形, 必须用随机数据测
5. UB 未初始化区域可能是 inf/NaN 位型, 0 乘它得 NaN → 所有固定 repeat 数会扫到的 padding 区要清零
6. 标量 `GetValue`/`SetValue` 走 S 管线, 与向量之间要 `PipeBarrier<PIPE_ALL>`; 同一向量管线内的依赖硬件保序, 无需 barrier
7. Level-2 API 每次调用重设 mask, 标量开销大; 热路径用 Level-0 + `isSetMask=false`, kernel 开头 `SetVectorMask` 设满 (Level-2 调用结束会把 mask 恢复成满, 混用安全)
8. `aicore` 函数里禁止 float 与无符号整数直接互转 (编译报错), 经 int64 中转

## 5. 显存

- 权重约 22GB (第 2 节); KV cache fp32, 每 token 128KB (ctx 4096 时 16 层 × 2 × 16MB = 512MB); DeltaNet 状态 201MB; 激活按 CHUNK = 256
- 实测 ctx = 4096 时总占用 25.0GB (单芯片可用约 46GB, ECC 已关)

## 6. 调试接口 (正式功能不依赖, 但保留)

- 环境变量 `QWEN_NPU_STOP=<层号>`: eval 跑完该层后直接返回; `QWEN_NPU_PHASE=<0/1/2/10/11/12>`: 进一步在该层的 FFN (gate/up/silu 后) 或 attention (QKV 矩阵乘 / rope / attn 后) 中途停下
- `qwen_dbg(name, out, n)`: 把中间激活缓冲 (x / xn / qkv / z / a / b / o / qg / k / v / att / g / u / out) 拷回 host
- `k_attn` 以 `T=-1` 启动进入 dump 模式 (当 T=1 跑, 每批的 p、树结果、ob 快照写到 att 缓冲之后), 配合独立测试台使用

## 7. 验证

测试脚本在仓库外的实验目录 (`tests/npu_port`、`tests/npu_attn`、`tests/npu_api`、`tests/npu_gemv`, 依赖本机绝对路径, 未入库):

- `qwen_selftest`: 4 种格式各取真实 tensor, kernel 反量化 vs 标量参考**逐位一致** (最大差 0), GEMV vs 反量化行的 double 点积相对误差 ~1e-9
- 逐层对比 (同进程加载 CPU 和 NPU 两个库, 对比 `qwen_hidden`): 6 token 和 100 token 下 65 层全部 ~1e-6 级
- 300 token prefill (跨 chunk, 覆盖 conv/DeltaNet/KV 状态延续): logits 最大差 7.6e-6, top10 完全一致; 随后 20 步贪心解码**逐 token 与 CPU 相同**
- `k_attn` 独立扫测: 随机 K/V/q, n = 1..192 (覆盖整批/尾批/多批在线 softmax 合并) 全部 < 1e-4; 另有一组"手术"模式 (均匀/批间不同/批内不同分数 + 指示器 V) 用于归因
- Ascend C 基础 API 行为 (`tests/npu_api`): cast 链、reduce 语义、GatherMask、Brcb、MulAddDst 混合精度、nibble 解码等逐项与 host 参考对比

## 8. 性能 (v1, Atlas 300I Duo 单颗 310P3, ECC 关)

| 项目 | 速度 |
|---|---|
| prefill (300 token) | 1.9 tok/s |
| decode (ctx ~320) | 1.8 tok/s |
| 模型加载 (重打包 + 上传) | 14s |

对比: CPU 版 decode 3.6 tok/s (16 线程), CUDA 第一版 62 tok/s。单颗 310P3 纯读显存带宽实测 204GB/s, 权重 22GB → decode 理论上限约 9-10 tok/s。目前 GEMV 有效带宽只有约 40GB/s: 纯向量管线约 70-93GB/s (独立压测), 加上每 token 约 650 次 kernel launch (~4µs/次) 和 attention/DeltaNet 的标量同步开销进一步拉低。

## 9. 已知不足与下一步

1. **prefill 换 cube**: 逐 token GEMV 只是正确性占位, cube fp16 GEMM 是数量级提升 (对应 CUDA 版 fp16+TC 的演进路线)
2. **decode GEMV 优化**: R 行分组布局摊薄指令发射、x 转 half 走 `MulAddDst(float, half, half)`、kernel 融合减少 launch 数; 目标把有效带宽从 ~40 提到 150+ GB/s, decode 到 7-10 tok/s
3. 双芯片张量并行 (每颗一半权重, 利用两倍显存带宽)、KV cache 量化、与 python 服务 (`qwen.py` / `server.py` / `router.py`) 的接入
4. 自定义采样 (wasm) 未移植; 图片按计划固定走 CPU / CUDA, 不上 NPU
