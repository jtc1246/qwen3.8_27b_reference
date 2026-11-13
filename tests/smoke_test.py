#!/usr/bin/env python3
# 冒烟测试: 只用标准库, 从 GGUF 读词表做贪心最长匹配分词 (不是真正的 BPE, 仅用于在没装 tokenizers 时验证模型能跑)
import os, sys, time, struct, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qwen

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_vocab(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    rs = lambda: f.read(rd("<Q")).decode("utf-8", errors="replace")
    def rv(t):
        if t in SCALAR:
            return rd(SCALAR[t])
        if t == 8:
            return rs()
        it, n = rd("<I"), rd("<Q")
        return [rv(it) for _ in range(n)]
    assert f.read(4) == b"GGUF"
    rd("<I"), rd("<Q")
    n_kv = rd("<Q")
    for _ in range(n_kv):
        k = rs()
        v = rv(rd("<I"))
        if k == "tokenizer.ggml.tokens":
            return v
    raise RuntimeError("❌ GGUF 中没有词表")


def bytes_to_unicode(): # GPT-2 byte-level BPE 的字节到可见字符映射
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="/root/local_llm/models/qwen38_27b_q4km/Qwen3.8-27B-Q4_K_M.gguf")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--think", action="store_true", help="开启思考模式 (不预填空的 think 块)")
    p.add_argument("--type", default="cpu", choices=list(qwen.LIBS), help="推理后端")
    p.add_argument("--device", type=int, default=0, help="cuda 后端的 GPU 编号")
    args = p.parse_args()
    t0 = time.time()
    vocab = read_vocab(args.gguf)
    tok2id = {t: i for i, t in enumerate(vocab)}
    max_len = max(len(t) for t in vocab)
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    qwen.log(f"ℹ️ 词表 {len(vocab)}, 最长 token {max_len} 字符, 读取耗时 {time.time() - t0:.1f}s")
    def encode(text):
        s = "".join(b2u[b] for b in text.encode("utf-8"))
        ids, i = [], 0
        while i < len(s):
            for L in range(min(max_len, len(s) - i), 0, -1):
                if s[i:i + L] in tok2id:
                    ids.append(tok2id[s[i:i + L]])
                    i += L
                    break
            else:
                raise RuntimeError(f"❌ 无法编码 {s[i]!r}")
        return ids
    def decode(ids):
        return bytes(u2b.get(c, ord("?")) for i in ids for c in (vocab[i] if not vocab[i].startswith("<|") else "")).decode("utf-8", errors="replace")
    sp = lambda t: [tok2id[t]]
    prompt = sp("<|im_start|>") + encode("user\n" + args.prompt) + sp("<|im_end|>") + encode("\n") + sp("<|im_start|>") + encode("assistant\n") + sp("<think>") + encode("\n")
    if not args.think:
        prompt += encode("\n") + sp("</think>") + encode("\n\n")
    qwen.log(f"ℹ️ prompt token 数 {len(prompt)}: {prompt}")
    lib = qwen.load(args.gguf, args.type, args.threads, args.device, 4096)
    fails = lib.qwen_selftest()
    qwen.log(("✅ 自测全部通过" if fails == 0 else f"❌ 自测失败 {fails} 项"))
    out = []
    for tok in qwen.generate(prompt, args.max_new, thinking=args.think, temperature=args.temp):
        out.append(tok)
        sys.stderr.write(decode([tok]))
        sys.stderr.flush()
    sys.stderr.write("\n")
    qwen.log(f"ℹ️ 输出 token: {out}")
    qwen.log(f"ℹ️ 输出文本: {decode(out)!r}")


if __name__ == "__main__":
    main()
