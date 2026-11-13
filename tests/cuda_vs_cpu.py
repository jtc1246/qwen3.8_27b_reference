#!/usr/bin/env python3
# CUDA 与 CPU 实现的数值对比: 同一进程加载两个库, 相同输入下比较每层 hidden、logits 和贪心生成结果
import os, sys, argparse, importlib.util
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def compare(cpu, gpu, tag):
    hc = np.ctypeslib.as_array(cpu.G_lib.qwen_hidden(), (65, 5120)).copy() # [65, 5120]
    hg = np.ctypeslib.as_array(gpu.G_lib.qwen_hidden(), (65, 5120)).copy() # [65, 5120]
    errs = [rel(hg[i], hc[i]) for i in range(65)]
    lc = np.ctypeslib.as_array(cpu.G_lib.qwen_logits(), (248320,)).copy() # [248320]
    lg = np.ctypeslib.as_array(gpu.G_lib.qwen_logits(), (248320,)).copy() # [248320]
    top_c, top_g = np.argsort(-lc)[:10], np.argsort(-lg)[:10] # [10]
    overlap = len(set(top_c.tolist()) & set(top_g.tolist()))
    ok = top_c[0] == top_g[0] and max(errs) < 5e-3 and rel(lg, lc) < 5e-3 # CUDA 的 matmul 输入和 KV 是 fp16, 实测误差 ~1e-3
    print(f"{'✅' if ok else '❌'} {tag}: hidden 最大相对误差 {max(errs):.2e} (第 {int(np.argmax(errs))} 层), logits 相对误差 {rel(lg, lc):.2e} 最大差 {np.abs(lg - lc).max():.3e}, top1 {top_c[0]}/{top_g[0]}, top10 重合 {overlap}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="/root/local_llm/models/qwen38_27b_q4km/Qwen3.8-27B-Q4_K_M.gguf")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--n", type=int, default=1100, help="随机 token 序列长度 (超过 1024 会跨 CUDA chunk)")
    args = p.parse_args()
    cpu = load_mod("qwen_cpu_mod", os.path.join(ROOT, "qwen.py")) # 同一个 qwen.py 加载两次, 两个模块实例各自持有一个后端
    gpu = load_mod("qwen_cuda_mod", os.path.join(ROOT, "qwen.py"))
    cpu.load(args.gguf, "cpu", n_threads=args.threads, max_ctx=4096)
    gpu.load(args.gguf, "cuda", device=0, max_ctx=4096)
    rng = np.random.default_rng(0)
    fails = 0
    # 1. 随机 token 长序列 prefill + 逐 token decode
    toks = rng.integers(0, 248000, args.n).tolist()
    for m in (cpu, gpu):
        m.reset()
        m.eval_tokens(toks)
    fails += not compare(cpu, gpu, f"prefill {args.n} 个随机 token")
    for i in range(4):
        t = int(rng.integers(0, 248000))
        for m in (cpu, gpu):
            m.eval_tokens([t])
        fails += not compare(cpu, gpu, f"decode 第 {i + 1} 步")
    # 2. 真实 prompt 的贪心生成 (smoke_test 默认问题)
    prompt = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 11316, 13, 248046, 198, 248045, 74455, 198, 248068, 198, 198, 248069, 271]
    outs = []
    for m in (cpu, gpu):
        m.reset()
        outs.append(list(m.generate(prompt, 32, thinking=False, temperature=0))) # prompt 结尾已关闭思考
    same = outs[0] == outs[1]
    fails += not same
    print(f"{'✅' if same else '❌'} 贪心生成 32 token: CPU {outs[0]}\n   GPU {outs[1]}", flush=True)
    # 3. 图片路径: 部分位置用外部 embedding + 自定义 3D 位置
    n = 80
    toks = rng.integers(0, 248000, n).tolist()
    mask = np.zeros(n, np.uint8) # [80]
    mask[10:50] = 1
    embd = np.ascontiguousarray((rng.standard_normal((n, 5120)) * 0.02).astype(np.float32)[mask == 1]) # [40, 5120], 只传 mask 为 1 的位置 (先生成 [80, 5120], 随机数与之前一致)
    pos3 = np.zeros((n, 3), np.int32) # [80, 3]
    pos3[:10] = np.arange(10)[:, None]
    pos3[10:50, 0] = 10
    pos3[10:50, 1] = 10 + np.arange(40) // 8
    pos3[10:50, 2] = 10 + np.arange(40) % 8
    pos3[50:] = (18 + np.arange(30))[:, None]
    for m in (cpu, gpu):
        m.reset()
        m.eval_tokens(toks, embd, mask, pos3)
    fails += not compare(cpu, gpu, "图片 embedding + MRoPE 位置")
    t = int(rng.integers(0, 248000))
    for m in (cpu, gpu):
        m.eval_tokens([t])
    fails += not compare(cpu, gpu, "图片之后 decode 1 步 (默认位置)")
    print("✅ 全部一致" if fails == 0 else f"❌ {fails} 项不一致", flush=True)


if __name__ == "__main__":
    main()
