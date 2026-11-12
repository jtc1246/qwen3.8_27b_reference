#!/usr/bin/env python3
# C++ 推理库的 ctypes 封装: 输入 token 序列, C++ 内部完成推理和采样, 返回 token id
# 各后端的 C 接口相同, load 时用 backend 选择 (见 LIBS), 一个进程 (模块实例) 只加载一个后端
# cuda 另有引擎模式 (engine_*): 最多 qwen_nseq() 个请求同时进行, prefill 按 FIFO 逐个算, decode 所有序列每步一起算; 与 eval_tokens 等 (直接模式) 不能混用
import os, sys, time, ctypes, hashlib, threading
from array import array

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBS = {"cpu": "qwen_cpu/cmake_build/libqwen.so", "cuda": "qwen_cuda/cmake_build/libqwen_cuda.so"}
CACHE_BACKENDS = ("cuda",) # 支持前缀缓存 (qwen_cache_init) 的后端
EOS_IDS = (248046, 248044) # <|im_end|>, <|endoftext|>
IM_START, IM_END, IMAGE_TOKEN = 248045, 248046, 248056
G_lib = None
G_backend = None


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load(gguf_path, backend="cpu", n_threads=16, device=0, max_ctx=32768, dec_sm=None, cache_dir=None, cache_threads=8):
    # backend: LIBS 的 key; n_threads 只对 cpu 有效, device (GPU 编号) 只对 cuda 有效
    # cuda: max_ctx 是每个槽的上下文 (qwen_nseq() 个槽的 KV 全部预先分配); dec_sm: prefill 和 decode 同时运行时 decode 分区的 SM 数, None 用 C++ 默认值
    # cache_dir: 前缀缓存目录 (只有 cuda, 调用方已检查并加锁, 其中 tmp/ 已清空), cache_threads: 后台量化 state 的线程数
    global G_lib, G_backend
    if G_lib is not None:
        return G_lib
    if backend not in LIBS:
        raise ValueError(f"❌ 未知后端 {backend}, 可选 {list(LIBS)}")
    lib = ctypes.CDLL(os.path.join(ROOT, LIBS[backend]))
    lib.qwen_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.qwen_eval.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.qwen_sample.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.qwen_seed.argtypes = [ctypes.c_uint64]
    lib.qwen_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.qwen_hidden.restype = ctypes.POINTER(ctypes.c_float)
    if backend == "cuda":
        lib.qwen_prefill.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.qwen_cache_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.qwen_next.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.qwen_cancel.argtypes = [ctypes.c_int]
        if dec_sm:
            lib.qwen_set_dec_sm(dec_sm)
    if lib.qwen_init(gguf_path.encode(), n_threads if backend == "cpu" else device, max_ctx) != 0: # 第二个参数: cpu 是线程数, cuda 是 GPU 编号
        raise RuntimeError(f"❌ 模型加载失败: {gguf_path}")
    if cache_dir and lib.qwen_cache_init(cache_dir.encode(), cache_threads) != 0:
        raise RuntimeError(f"❌ 前缀缓存初始化失败: {cache_dir}")
    G_lib = lib
    G_backend = backend
    return lib


def eval_tokens(tokens, embd=None, embd_mask=None, pos3=None):
    # embd: 可选, [embd_mask 中 1 的个数, 5120] float32 连续内存 (例如 numpy 数组), 按位置顺序依次替换 embd_mask 为 1 的位置的 token embedding (图片)
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


# ---------------- 引擎模式 (只有 cuda) ----------------

def engine_start(): # 启动 decode 循环线程 (C++ 里不返回)
    threading.Thread(target=G_lib.qwen_decode_loop, daemon=True).start()


def nseq():
    return G_lib.qwen_nseq()


def cache_plan(ids, images=()):
    # 前缀缓存的缓存点: 每个 <|im_end|> 之后 (消息边界); 第 i 段 = token [pos[i-1], pos[i]), hash[i] = sha256(第 i 段的序列化), key[i] = sha256(hash[0] || ... || hash[i])
    # 序列化: token 按 int32 小端; 每张图片 (连续 t*h*w/4 个 IMAGE_TOKEN) 换成 int32 [-1, t, h, w] + 图片原始字节的 sha256 (32 字节)
    # images: 按出现顺序 [(图片 sha256 hex, (t, h, w))]; 返回 (pos, hash, key, role, tail, tail_role), hash/key 是各 32 字节拼接的 bytes, role 是每段 <|im_start|> 后的 token (没有为 -1)
    # tail: 最后一个缓存点之后 prompt 的序列化, C++ 在输出以 <|im_end|> 结束时接上输出 token 算出输出缓存点的 key
    n, runs, i, k = len(ids), [], 0, 0
    while i < n:
        if ids[i] != IMAGE_TOKEN:
            i += 1
            continue
        if k >= len(images):
            raise ValueError("❌ 图片 token 段数多于图片数")
        sha, (t, h, w) = images[k]
        cnt = t * h * w // 4
        if ids[i:i + cnt] != [IMAGE_TOKEN] * cnt or (i + cnt < n and ids[i + cnt] == IMAGE_TOKEN):
            raise ValueError(f"❌ 第 {k} 张图片的 token 数与 grid {(t, h, w)} 不一致")
        runs.append((i, i + cnt, array("i", [-1, t, h, w]).tobytes() + bytes.fromhex(sha)))
        i, k = i + cnt, k + 1
    if k != len(images):
        raise ValueError(f"❌ 图片数 {len(images)} 与图片 token 段数 {k} 不一致")
    def ser(s, e):
        out, p = [], s
        for a, b, mark in runs:
            if a >= s and b <= e:
                out += [array("i", ids[p:a]).tobytes(), mark]
                p = b
        out.append(array("i", ids[p:e]).tobytes())
        return b"".join(out)
    def role(s, e):
        try:
            j = ids.index(IM_START, s, e)
        except ValueError:
            return -1
        return ids[j + 1] if j + 1 < e else -1
    pos = [j + 1 for j in range(n) if ids[j] == IM_END]
    hs, ks, roles, acc, s = [], [], [], hashlib.sha256(), 0
    for e in pos:
        h = hashlib.sha256(ser(s, e)).digest()
        acc.update(h)
        hs.append(h)
        ks.append(acc.copy().digest())
        roles.append(role(s, e))
        s = e
    return pos, b"".join(hs), b"".join(ks), roles, ser(s, n), role(s, n)


def engine_args(tokens, embd=None, embd_mask=None, pos3=None, temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, presence_penalty=0.0, seed=0, max_new=512, stop_ids=EOS_IDS, cache=None):
    # 预先准备好 qwen_prefill 的参数 (拿到 ticket 之后不能再出错, 否则后面的 prefill 都会卡住); embd/embd_mask/pos3 同 eval_tokens, 调用方要保持它们存活
    # cache: cache_plan 的返回值, None 表示本请求不用前缀缓存
    n = len(tokens)
    p_embd = embd.ctypes.data if embd is not None else None
    p_mask = embd_mask.ctypes.data if embd_mask is not None else None
    p_pos = pos3.ctypes.data if pos3 is not None else None
    if cache is None:
        c = (-1, None, None, None, None, None, 0, -1)
    else:
        pos, hs, ks, roles, tail, tail_role = cache
        m = max(1, len(pos))
        c = (len(pos), (ctypes.c_int * m)(*pos), hs, ks, (ctypes.c_int * m)(*roles), tail, len(tail), tail_role)
    return ((ctypes.c_int * n)(*tokens), n, p_embd, p_mask, p_pos, temperature, top_k, top_p, min_p, presence_penalty, seed & 0xffffffffffffffff, max_new, (ctypes.c_int * len(stop_ids))(*stop_ids), len(stop_ids)) + c


def engine_prefill(slot, ticket, args): # 阻塞: 在 C++ 里按 ticket 顺序等前面的 prefill, 算完并采样出第一个 token 后返回
    ret = G_lib.qwen_prefill(slot, ticket, *args)
    if ret != 0:
        raise RuntimeError(f"❌ qwen_prefill 失败, 返回 {ret}")


def engine_next(slot, buf): # 阻塞等新 token, 返回 list; 空 list 表示已结束 (之后槽才能复用); buf: ctypes int 数组
    n = G_lib.qwen_next(slot, buf, len(buf))
    return buf[:n]


def engine_cancel(slot): # 让 decode 循环停止该槽, 之后仍要 engine_next 读到空 list
    G_lib.qwen_cancel(slot)
