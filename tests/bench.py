#!/usr/bin/env python3
# 性能测试: 不同长度 prefill 的速度, 以及不同上下文长度下的 decode 速度 (随机 token, 只看速度)
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qwen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="/root/local_llm/models/qwen38_27b_q4km/Qwen3.8-27B-Q4_K_M.gguf")
    p.add_argument("--type", default="cuda", choices=list(qwen.LIBS), help="推理后端")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--lens", default="256,1024,4096,16384")
    p.add_argument("--decode", type=int, default=64)
    args = p.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    qwen.load(args.gguf, args.type, args.threads, args.device, max(lens) + args.decode + 16)
    rng = np.random.default_rng(0)
    qwen.reset()
    qwen.eval_tokens(rng.integers(0, 248000, 64).tolist()) # 预热
    for n in lens:
        qwen.reset()
        toks = rng.integers(0, 248000, n).tolist()
        t0 = time.time()
        qwen.eval_tokens(toks)
        t1 = time.time()
        for _ in range(args.decode):
            qwen.eval_tokens([qwen.sample(temperature=0)])
        t2 = time.time()
        qwen.log(f"ℹ️ [{args.type}] 上下文 {n}: prefill {n / (t1 - t0):.1f} tok/s ({t1 - t0:.2f}s), decode {args.decode / (t2 - t1):.2f} tok/s")


if __name__ == "__main__":
    main()
