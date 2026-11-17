#!/usr/bin/env python3
# doorbell 改动正确性/速度对比: 固定随机 token prefill 300 + 贪心 decode 49, 打印 id 序列和速度
# 用法: QWEN_NPU_NODB=1 运行一次 (sync 基线), 不设再运行一次 (doorbell), 对比输出逐 token 一致
import os, sys, time, hashlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qwen

st = "/root/local_llm/models/qwen38_27b_redhat_int4/model.safetensors"
qwen.load(st, "npu", device=0, max_ctx=4096)
rng = np.random.default_rng(0)
qwen.reset()
toks = rng.integers(0, 248000, 300).tolist()
t0 = time.time()
qwen.eval_tokens(toks)
t1 = time.time()
md5 = hashlib.md5()
md5.update(np.array(qwen.logits(), dtype=np.float32).tobytes())
out = []
for _ in range(49):
    t = qwen.sample(temperature=0, think_ok=False)
    out.append(t)
    qwen.eval_tokens([t])
    md5.update(np.array(qwen.logits(), dtype=np.float32).tobytes())
t2 = time.time()
qwen.log(f"ℹ️ prefill 300: {300 / (t1 - t0):.1f} tok/s, decode 49: {49 / (t2 - t1):.2f} tok/s (计时含 logits 取回)")
print("OUT", out, flush=True)
print("LOGITS_MD5", md5.hexdigest(), flush=True)
