#!/usr/bin/env python3
# libqwen.so 的 ctypes 封装: 输入 token 序列, C++ 内部完成推理和采样, 返回 token id
import os, sys, time, ctypes

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmake_build", "libqwen.so")
EOS_IDS = (248046, 248044) # <|im_end|>, <|endoftext|>
G_lib = None


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load(gguf_path, n_threads=16, max_ctx=32768):
    global G_lib
    if G_lib is not None:
        return G_lib
    lib = ctypes.CDLL(LIB_PATH)
    lib.qwen_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.qwen_eval.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.qwen_sample.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.qwen_seed.argtypes = [ctypes.c_uint64]
    lib.qwen_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.qwen_hidden.restype = ctypes.POINTER(ctypes.c_float)
    if lib.qwen_init(gguf_path.encode(), n_threads, max_ctx) != 0:
        raise RuntimeError(f"❌ 模型加载失败: {gguf_path}")
    G_lib = lib
    return lib


def eval_tokens(tokens, embd=None, embd_mask=None, pos3=None):
    # embd: 可选, [n, 5120] float32 连续内存 (例如 numpy 数组), 对应 embd_mask 为 1 的位置替换 token embedding (图片)
    # pos3: 可选, [n, 3] int32 的 MRoPE 位置 (t, h, w); 为空时默认接着上次的位置递增
    n = len(tokens)
    arr = (ctypes.c_int * n)(*tokens)
    p_embd = embd.ctypes.data if embd is not None else None
    p_mask = embd_mask.ctypes.data if embd_mask is not None else None
    p_pos = pos3.ctypes.data if pos3 is not None else None
    ret = G_lib.qwen_eval(arr, n, p_embd, p_mask, p_pos)
    if ret != 0:
        raise RuntimeError(f"❌ qwen_eval 失败, 返回 {ret}")


def sample(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, presence_penalty=0.0, hist=None):
    hist = hist or []
    arr = (ctypes.c_int * max(1, len(hist)))(*hist)
    return G_lib.qwen_sample(temperature, top_k, top_p, min_p, presence_penalty, arr, len(hist))


def logits():
    return G_lib.qwen_logits()[:248320]


def hidden(layer):
    p = G_lib.qwen_hidden()
    return p[layer * 5120:(layer + 1) * 5120]


def reset():
    G_lib.qwen_reset()


def seed(s):
    G_lib.qwen_seed(s)


def n_past():
    return G_lib.qwen_n_past()


def generate(prompt_tokens, max_new_tokens=512, stop_ids=EOS_IDS, embd=None, embd_mask=None, pos3=None, **sample_kwargs):
    # 生成器: 逐个返回新 token id; 调用方负责在新对话前 reset(); embd/embd_mask/pos3 含义同 eval_tokens
    t0 = time.time()
    eval_tokens(prompt_tokens, embd, embd_mask, pos3)
    t1 = time.time()
    log(f"ℹ️ prefill {len(prompt_tokens)} tokens, {t1 - t0:.2f}s, {len(prompt_tokens) / (t1 - t0):.2f} tok/s")
    out = []
    for _ in range(max_new_tokens):
        tok = sample(hist=out, **sample_kwargs)
        out.append(tok)
        yield tok
        if tok in stop_ids:
            break
        eval_tokens([tok])
    t2 = time.time()
    log(f"ℹ️ decode {len(out)} tokens, {t2 - t1:.2f}s, {len(out) / max(t2 - t1, 1e-9):.2f} tok/s")
