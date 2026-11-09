# Qwen3.8-27B 参考实现 (纯 CPU)

从零实现的 Qwen3.8-27B (llama.cpp 架构名 `qwen35`) 推理, C++ 无第三方依赖, 读取 GGUF (目前测试的是 [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 的 `Q4_K_M`)。

- C++ (`qwen_cpu/`): GGUF 解析、Q4_K/Q5_K/Q6_K/Q8_0 反量化 (AVX-512)、Gated DeltaNet + Gated Attention 前向、采样, 编译为 musl 全静态 so, 通过 `extern "C"` 给 python ctypes 调用
- 输入是 token 序列, 输出采样好的 token id; tokenizer、图片编码由 python 侧负责 (接口已支持传入图片 embedding 和 MRoPE 3 维位置)

## 编译

需要 x86_64 + AVX-512, musl 编译器 (`$MUSL_BIN`), cmake。

```bash
./qwen_cpu/compile.sh
```

## 运行

```bash
python -u tests/smoke_test.py --gguf /path/to/Qwen3.8-27B-Q4_K_M.gguf --think --temp 1.0
```

`smoke_test.py` 只用标准库, 用 GGUF 词表做贪心匹配分词 (不是真正的 BPE), 仅用于验证能跑。

## 文件

- `qwen_cpu/qwen.cpp`: 模型、前向、采样、自测、C 接口
- `qwen_cpu/gguf.h` / `quants.h` / `ops.h` / `pool.h`: GGUF 解析 / 量化格式 / 计算函数 / 线程池
- `qwen_cpu/qwen.py`: ctypes 封装, `generate()` 逐个返回 token id

## 与 HF 权重的差异 (GGUF 转换时已处理)

- 除 DeltaNet 内的 `ssm_norm` 外, 所有 RMSNorm 权重已 +1
- `ssm_a` 存的是 `-exp(A_log)`
- DeltaNet 的 V 头从 grouped 重排成 tiled 顺序, V 头 h 对应 QK 头 `h % 16`

## 性能 (AMD EPYC 9115 16 核, 实测内存带宽约 117GB/s)

- prefill 约 20 tok/s, decode 约 3.6 tok/s (理论上限约 7 tok/s)
