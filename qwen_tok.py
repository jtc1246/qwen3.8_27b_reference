#!/usr/bin/env python3
# Qwen3.8 tokenizer 自己实现, 只用标准库: 读 tokenizer.json 的 vocab / merges / added_tokens
# encode (正文): NFC -> 预分词 (手写, 等价于官方 Split 正则) -> byte-level 映射 -> BPE; 只查 BPE 词表, 任何文本都不会产生控制 token
# 控制 token (33 个 added token) 只能由调用方按 id 插入 (CTL); decode: id -> 字节 -> utf-8 (不完整 / 非法字节换成 U+FFFD)
import json, heapq, unicodedata

G_vocab = {} # byte-level 字符串 -> id
G_ranks = {} # (左, 右) -> (merge 序号, 合并后的字符串)
G_tok = [] # id -> 字节
CTL = {} # 控制 token 内容 -> id
G_cache = {} # 预分词片段 -> id 列表
CACHE_MAX = 1 << 20
WS = set("\t\n\x0b\x0c\r \x85\xa0     　") | {chr(c) for c in range(0x2000, 0x200b)} # Unicode White_Space (正则的 \s)


def byte_map(): # GPT-2 的 byte -> 可见字符映射
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


G_b2u = byte_map()
G_u2b = {v: k for k, v in G_b2u.items()}


def load(path): # tokenizer.json; 只支持本模型用到的配置, 其余直接报错
    j = json.load(open(path, encoding="utf-8"))
    m = j["model"]
    want = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False}
    if {k: m.get(k) for k in want} != want or j["normalizer"] != {"type": "NFC"} or j["pre_tokenizer"]["pretokenizers"][1]["type"] != "ByteLevel":
        raise RuntimeError(f"❌ {path} 的 tokenizer 配置与本实现不符")
    if j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] != r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+":
        raise RuntimeError("❌ 预分词正则与本实现 (pretokenize) 不符")
    G_vocab.update(m["vocab"])
    for r, mg in enumerate(m["merges"]):
        a, b = mg.split(" ") if isinstance(mg, str) else mg
        G_ranks[(a, b)] = (r, a + b)
    n = max(G_vocab.values()) + 1
    for a in j["added_tokens"]:
        CTL[a["content"]] = a["id"]
        n = max(n, a["id"] + 1)
    G_tok.extend([b""] * n)
    for t, i in G_vocab.items():
        G_tok[i] = bytes(G_u2b[c] for c in t)
    for t, i in CTL.items():
        G_tok[i] = t.encode()


def cls(c): # L 字母 / M 组合符号 / N 数字 / W 空白 / O 其他
    if c in WS:
        return "W"
    k = unicodedata.category(c)[0]
    return k if k in "LMN" else "O"


def pretokenize(s): # 等价于 load 里检查过的正则 (Isolated, 从左到右, 每处取第一个能匹配的分支)
    n, cl, out, i = len(s), [cls(c) for c in s], [], 0
    while i < n:
        c, e = s[i], -1
        if c == "'" and i + 1 < n: # (?i:'s|'t|'re|'ve|'m|'ll|'d)
            for suf in ("s", "t", "re", "ve", "m", "ll", "d"):
                if i + len(suf) < n and all(s[i + 1 + k].casefold() == suf[k] for k in range(len(suf))):
                    e = i + 1 + len(suf)
                    break
        if e < 0: # [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            j = i + 1 if cl[i] not in "LN" and c not in "\r\n" and i + 1 < n and cl[i + 1] in "LM" else (i if cl[i] in "LM" else -1)
            if j >= 0:
                e = j
                while e < n and cl[e] in "LM":
                    e += 1
        if e < 0 and cl[i] == "N": # \p{N}
            e = i + 1
        if e < 0: # ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
            j = i + 1 if c == " " and i + 1 < n and cl[i + 1] == "O" else (i if cl[i] == "O" else -1)
            if j >= 0:
                e = j
                while e < n and cl[e] == "O":
                    e += 1
                while e < n and s[e] in "\r\n":
                    e += 1
        if e < 0: # 空白开头: \s*[\r\n]+ | \s+(?!\S) | \s+
            w = i
            while w < n and cl[w] == "W":
                w += 1
            nl = max((k for k in range(i, w) if s[k] in "\r\n"), default=-1) # 空白段内最后一个换行
            e = nl + 1 if nl >= 0 else (w if w == n or w - 1 == i else w - 1)
        out.append(s[i:e])
        i = e
    return out


def bpe(word): # 与 HF BPE 相同: 每次合并 (merge 序号, 位置) 最小的一对
    sym = list(word)
    nxt, prv = list(range(1, len(sym) + 1)), list(range(-1, len(sym) - 1))
    heap = [(G_ranks[(sym[k], sym[k + 1])][0], k) for k in range(len(sym) - 1) if (sym[k], sym[k + 1]) in G_ranks]
    heapq.heapify(heap)
    while heap:
        r, k = heapq.heappop(heap)
        if not sym[k] or nxt[k] >= len(sym):
            continue
        q = nxt[k]
        m = G_ranks.get((sym[k], sym[q]))
        if not m or m[0] != r: # 这一对已经变了
            continue
        sym[k], sym[q] = m[1], ""
        nxt[k] = nxt[q]
        if nxt[k] < len(sym):
            prv[nxt[k]] = k
            m2 = G_ranks.get((sym[k], sym[nxt[k]]))
            if m2:
                heapq.heappush(heap, (m2[0], k))
        if prv[k] >= 0:
            m2 = G_ranks.get((sym[prv[k]], sym[k]))
            if m2:
                heapq.heappush(heap, (m2[0], prv[k]))
    return [G_vocab[t] for t in sym if t]


def encode(text): # 正文 -> id (不含任何控制 token)
    ids = []
    for p in pretokenize(unicodedata.normalize("NFC", text)):
        r = G_cache.get(p)
        if r is None:
            r = bpe("".join(G_b2u[b] for b in p.encode()))
            if len(G_cache) >= CACHE_MAX:
                G_cache.clear()
            G_cache[p] = r
        ids += r
    return ids


def decode(ids): # 控制 token 输出其内容字符串
    return b"".join(G_tok[i] for i in ids).decode("utf-8", errors="replace")
