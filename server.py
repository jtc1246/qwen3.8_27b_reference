#!/usr/bin/env python3
# OpenAI 兼容 chat completions 服务: 支持图片、system、temperature、seed、thinking 控制, 其余参数忽略
# cpu: C++ 推理是单序列全局状态, 请求用全局锁串行处理, 并发请求排队等待而不报错
# cuda: 引擎模式, 最多 qwen.nseq() (4) 个请求同时进行: 超出的在 python 按到达顺序排队; prefill 在 C++ 里按顺序逐个算, decode 所有序列每步一起算, 两者同时跑时按 SM 分区
# --type 选择后端 (qwen.LIBS), cuda 时视觉塔也在 GPU 上; 视觉塔输出按图片原始字节的 sha256 缓存在内存 (LRU)
import os, re, sys, io, json, time, uuid, codecs, fcntl, shlex, base64, random, ctypes, shutil, hashlib, argparse, threading, contextlib
from collections import OrderedDict, deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor, AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
import qwen
import qwen_tok

IMAGE_TOKEN = 248056
THINK_END = 248069
SAMPLING = { # 官方推荐参数 (README Best Practices), 客户端只能覆盖 temperature
    True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
    False: dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5),
}
EFFORT_MAP = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"} # OpenAI reasoning_effort -> 模板支持的取值

G_args = None
G_imgproc = None # 图片预处理 (transformers 的 image processor); 分词全部用自己的 qwen_tok
G_vis = None
G_dev = "cpu" # 视觉塔所在的 torch device
G_img_cache = OrderedDict() # sha256 -> 该图的视觉输出 (n_tok, 5120) float32, 只在 G_lock 内访问
G_img_cache_bytes = 0
G_lock = threading.Lock() # cpu: 整个推理; cuda: 只用于视觉塔和图片缓存
G_queue_lock = threading.Lock()
G_waiting = 0
G_pool = None # cuda 的 SlotPool
G_cache_dir = None # 前缀缓存目录 (--cache-dir), None 表示不启用
G_cache_lock = None # 缓存目录锁文件的 fd, 进程生命周期内一直持有


class BadRequest(Exception):
    pass


class SlotPool: # cuda 引擎的槽位: 最多 n 个请求同时进入 C++, 其余按到达顺序 (FIFO) 在这里等; 分配时按顺序发 ticket, C++ 按 ticket 顺序逐个 prefill
    def __init__(self, n):
        self.lock = threading.Lock()
        self.free = list(range(n))
        self.waiters = deque() # [Event, (槽, ticket)]
        self.ticket = 0

    def take(self, slot): # 锁内调用
        self.ticket += 1
        return slot, self.ticket - 1

    def acquire(self, rid): # 返回 (槽, ticket)
        with self.lock:
            if self.free and not self.waiters:
                return self.take(self.free.pop(0))
            w = [threading.Event(), None]
            self.waiters.append(w)
            log(f"⏳ {rid} 槽位已满, 排队等待, 前面 {len(self.waiters) - 1} 个请求")
        w[0].wait()
        return w[1]

    def release(self, slot):
        with self.lock:
            if not self.waiters:
                self.free.append(slot)
                return
            w = self.waiters.popleft()
            w[1] = self.take(slot)
            w[0].set()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_vision(hf_dir):
    cfg = AutoConfig.from_pretrained(hf_dir)
    vis = Qwen3_5VisionModel._from_config(cfg.vision_config, dtype=torch.float32)
    idx = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for shard in sorted({v for k, v in idx.items() if k.startswith("model.visual.")}):
        with safe_open(os.path.join(hf_dir, shard), "pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k).float()
    missing, unexpected = vis.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"❌ 视觉塔权重不匹配, missing={missing}, unexpected={unexpected}")
    return vis.eval().to(G_dev)


def load_image(url): # 只支持 base64 的 data URL (data:<类型>;base64,<数据>), 不支持链接, 不会因为图片产生网络请求; 返回 (RGB 图片, 原始字节的 sha256)
    head, sep, b64 = url.partition(",")
    if not head.startswith("data:") or not head.endswith(";base64") or not sep:
        raise BadRequest(f"图片只支持 base64 的 data URL (data:<类型>;base64,<数据>), 收到的开头是 {url[:64]!r}")
    try:
        data = base64.b64decode(b64, validate=True) # 严格解码: 有非 base64 字符直接报错, 不悄悄跳过
    except ValueError as e:
        raise BadRequest(f"图片的 base64 数据不合法: {e}")
    return Image.open(io.BytesIO(data)).convert("RGB"), hashlib.sha256(data).hexdigest()


REASONING = { # 与 chat_template.jinja 的 reasoning_instructions 相同 (medium 没有指令)
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "medium": None,
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
}


def render_content(content, is_system): # 同模板的 render_content, 但不做 trim (正文原样): 返回片段列表, str 是正文, int 是控制 token id
    if isinstance(content, str):
        parts = [content]
    elif content is None:
        parts = []
    elif isinstance(content, list):
        parts = []
        for it in content:
            if it.get("type") == "image":
                if is_system:
                    raise BadRequest("System message cannot contain images.")
                parts += [qwen_tok.CTL["<|vision_start|>"], qwen_tok.CTL["<|image_pad|>"], qwen_tok.CTL["<|vision_end|>"]]
            else:
                parts.append(it["text"]) # convert_messages 已保证是字符串
    else:
        raise BadRequest("Unexpected content type.")
    return parts


def build_tokens(msgs, enable_thinking, effort, grids): # 按 chat_template.jinja 的结构 (不含 tools) 直接拼 token, 但正文不 trim, 也不补模板为 trim 加的换行: 正文只用 qwen_tok.encode 按文本分词 (不可能产生控制 token), 控制 token 由这里按 id 插入; <|image_pad|> 按 grid 展开
    C = qwen_tok.CTL
    seq = [] # str 或 int
    ri = REASONING[effort if effort is not None else "xhigh"] if enable_thinking else None # None: 没有 reasoning 指令
    if msgs[0]["role"] == "system": # 有 system 消息就有 system 块, 内容原样 (不判断是否为空)
        seq += [C["<|im_start|>"], "system\n" + (ri + "\n\n" if ri is not None else "")] + render_content(msgs[0]["content"], True) + [C["<|im_end|>"], "\n"]
    elif ri is not None:
        seq += [C["<|im_start|>"], "system\n" + ri, C["<|im_end|>"], "\n"]
    if not any(m["role"] == "user" for m in msgs): # 模板要求至少有一条 user 消息; 不按正文内容判断 (正文一律是普通文本)
        raise BadRequest("No user query found in messages.")
    for i, m in enumerate(msgs):
        c = render_content(m["content"], False)
        if m["role"] == "system":
            if i:
                raise BadRequest("System message must be at the beginning.")
        elif m["role"] == "user":
            seq += [C["<|im_start|>"], "user\n"] + c + [C["<|im_end|>"], "\n"]
        elif m["role"] == "assistant": # 还原成当时的生成提示 + 模型原样生成的内容 (本服务返回的 reasoning_content / content 都是原样的)
            rc = m.get("reasoning_content")
            if isinstance(rc, str): # 有 reasoning_content (不管是否为空) 就是开 thinking 的轮次: 生成提示到思考开始 token + "\n", 之后 (含思考结束 token 前后的换行) 都是模型生成的
                seq += [C["<|im_start|>"], "assistant\n", C["<think>"], "\n" + rc, C["</think>"]] + c + [C["<|im_end|>"], "\n"]
            else: # 没有 reasoning_content 字段: 不开 thinking 的轮次, 生成提示是 思考开始 "\n\n" 思考结束 "\n\n", 之后是正文
                seq += [C["<|im_start|>"], "assistant\n", C["<think>"], "\n\n", C["</think>"], "\n\n"] + c + [C["<|im_end|>"], "\n"]
        elif m["role"] == "tool":
            if i and msgs[i - 1]["role"] != "tool":
                seq += [C["<|im_start|>"], "user"]
            seq += ["\n", C["<tool_response>"], "\n"] + c + ["\n", C["</tool_response>"]]
            if i == len(msgs) - 1 or msgs[i + 1]["role"] != "tool":
                seq += [C["<|im_end|>"], "\n"]
        else:
            raise BadRequest("Unexpected message role.")
    seq += [C["<|im_start|>"], "assistant\n", C["<think>"]] + (["\n\n", C["</think>"], "\n\n"] if not enable_thinking else ["\n"])
    ids, buf, gi, merge = [], [], 0, G_imgproc.merge_size ** 2
    for x in seq + [None]:
        if isinstance(x, str):
            buf.append(x)
            continue
        ids += qwen_tok.encode("".join(buf)) # 两个控制 token 之间的正文整段分词 (与官方在 added token 处切开后各段分词相同)
        buf = []
        if x is None:
            break
        if x != IMAGE_TOKEN:
            ids.append(x)
            continue
        if gi >= len(grids):
            raise BadRequest("图片占位多于图片数")
        ids += [IMAGE_TOKEN] * (int(np.prod(grids[gi])) // merge)
        gi += 1
    if gi != len(grids):
        raise BadRequest(f"图片数 {len(grids)} 与图片占位数 {gi} 不一致")
    return ids


def convert_messages(messages): # OpenAI 格式 -> chat 模板格式, 图片和 sha256 单独取出
    out, images, hashes = [], [], []
    for m in messages:
        role = "system" if m.get("role") == "developer" else m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "text":
                    if not isinstance(p.get("text"), str):
                        raise BadRequest("text 类型的 content 必须有字符串 text 字段")
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url":
                    iu = p.get("image_url")
                    url = iu if isinstance(iu, str) else (iu.get("url") if isinstance(iu, dict) else None)
                    if not isinstance(url, str):
                        raise BadRequest("image_url 必须是字符串或带字符串 url 字段的对象")
                    img, h = load_image(url)
                    images.append(img)
                    hashes.append(h)
                    parts.append({"type": "image"})
                else:
                    raise BadRequest(f"不支持的 content 类型: {p.get('type')}")
            content = parts
        msg = {"role": role, "content": content}
        if isinstance(m.get("reasoning_content"), str):
            msg["reasoning_content"] = m["reasoning_content"]
        out.append(msg)
    return out, images, hashes


def lock_holder(path): # 从 /proc/locks 按 inode 找持有 flock 的进程, 返回 "pid N, 命令 ..." 或找不到的说明
    st = os.stat(path)
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    pids = [l.split()[4] for l in open("/proc/locks") if " FLOCK " in l and l.split()[5] == dev] # 格式: 序号: FLOCK ADVISORY WRITE pid 主:次:inode 起 止
    if not pids:
        return "在 /proc/locks 里找不到持有者"
    out = []
    for pid in pids:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            out.append(f"pid {pid}, 命令 {shlex.join(x.decode(errors='replace') for x in cmd if x)}")
        except OSError:
            out.append(f"pid {pid} (找不到该进程, 可能在其他 pid 命名空间)")
    return "; ".join(out)


def open_cache_dir(d): # 前缀缓存目录: 必须是 (不存在且上级存在) / 空目录 / 有 cache.lock (空文件, 只用来 flock) 的目录之一, 否则退出; 加文件锁, 已被占用时打印占用者并退出
    global G_cache_lock
    d = os.path.abspath(d)
    lock = os.path.join(d, "cache.lock")
    if not os.path.exists(d):
        if not os.path.isdir(os.path.dirname(d)):
            log(f"❌ 缓存目录 {d} 的上级目录不存在")
            os._exit(1)
        os.mkdir(d)
    elif not os.path.isdir(d) or (os.listdir(d) and not os.path.isfile(lock)):
        log(f"❌ 缓存目录 {d} 不是空目录, 也没有 cache.lock, 不是本程序的缓存目录, 拒绝使用")
        os._exit(1)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"❌ 缓存目录 {d} 已被其他进程占用: {lock_holder(lock)}")
        os._exit(1)
    G_cache_lock = fd
    shutil.rmtree(os.path.join(d, "tmp"), ignore_errors=True) # 上次进程没写完的输出缓存
    os.mkdir(os.path.join(d, "tmp"))
    return d


def thinking_options(body): # 返回 (enable_thinking, reasoning_effort)
    ctk = body.get("chat_template_kwargs") or {}
    enable = body.get("enable_thinking", ctk.get("enable_thinking", True))
    effort = body.get("reasoning_effort", ctk.get("reasoning_effort"))
    if effort in ("none", "minimal"):
        return False, None
    if effort is not None and effort not in EFFORT_MAP:
        raise BadRequest(f"不支持的 reasoning_effort: {effort}")
    if not isinstance(enable, bool):
        raise BadRequest("enable_thinking 必须是 true / false")
    return enable, (EFFORT_MAP[effort] if effort is not None else None)


def rope_positions(ids, grids): # 与 HF Qwen3_5Model.get_rope_index 一致 (只处理图片)
    n = len(ids)
    pos = np.zeros((n, 3), dtype=np.int32) # (n, 3) t/h/w
    cur, i, gi = 0, 0, 0
    while i < n:
        if ids[i] != IMAGE_TOKEN:
            pos[i] = cur
            cur += 1
            i += 1
            continue
        t, h, w = grids[gi]
        gi += 1
        lh, lw = h // 2, w // 2 # spatial_merge_size = 2
        cnt = t * lh * lw
        tt, hh, ww = np.meshgrid(np.arange(t), np.arange(lh), np.arange(lw), indexing="ij") # 各 (t, lh, lw)
        pos[i:i + cnt, 0] = tt.ravel() + cur
        pos[i:i + cnt, 1] = hh.ravel() + cur
        pos[i:i + cnt, 2] = ww.ravel() + cur
        cur += max(lh, lw)
        i += cnt
    return pos


def build_prompt(body): # 锁外执行: 解析消息、下载图片、拼 token (build_tokens)、图片预处理
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages 不能为空")
    msgs, images, hashes = convert_messages(messages)
    enable_thinking, effort = thinking_options(body)
    out = None
    if images:
        size = {"longest_edge": G_args.max_image_pixels, "shortest_edge": 65536} if G_args.max_image_pixels > 0 else None
        out = G_imgproc(images=images, return_tensors="pt", **({"size": size} if size else {})) # pixel_values (patch 总数, 1536), image_grid_thw (n_img, 3)
    ids = build_tokens(msgs, enable_thinking, effort, out["image_grid_thw"].tolist() if images else [])
    temp = body.get("temperature")
    sp = dict(SAMPLING[enable_thinking])
    if temp is not None:
        if not isinstance(temp, (int, float)) or temp < 0:
            raise BadRequest("temperature 必须是非负数")
        sp["temperature"] = float(temp)
    seed = body.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise BadRequest("seed 必须是整数")
    cache = qwen.cache_plan(ids, list(zip(hashes, out["image_grid_thw"].tolist() if images else []))) if G_cache_dir is not None else None
    so = body.get("stream_options")
    return {"ids": ids, "out": out, "hashes": hashes, "thinking": enable_thinking, "sampling": sp, "seed": seed, "n_images": len(images), "cache": cache, "cache_stats": (0, 0, 0, 0),
            "include_usage": bool(isinstance(so, dict) and so.get("include_usage"))}


def make_usage(p, n_out): # OpenAI 格式的 usage; 有的字段用 OpenAI 的名字 (cached_tokens), 其余是前缀缓存的统计, 所有字段始终存在
    hit, kv_in, st_in, st_out = p["cache_stats"]
    n_in = len(p["ids"])
    return {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out,
            "prompt_tokens_details": {"cached_tokens": hit, "uncached_tokens": n_in - hit, "cache_written_tokens": kv_in, "cache_written_states": st_in},
            "completion_tokens_details": {"cache_written_states": st_out}}


def vision_embeds(p): # 锁内执行: 命中缓存的图片直接取, 其余图片一起过视觉塔, 返回 (所有图片 token 数, 5120) 和命中数
    global G_img_cache_bytes
    grids = p["out"]["image_grid_thw"] # (n_img, 3) t/h/w (patch 数)
    pv = p["out"]["pixel_values"] # (所有图片 patch 总数, 1536), 按图片顺序拼接
    offs = np.cumsum([0] + grids.prod(-1).tolist()) # 每张图 patch 的起止位置
    embs = [G_img_cache.get(h) for h in p["hashes"]]
    for h, e in zip(p["hashes"], embs):
        if e is not None:
            G_img_cache.move_to_end(h)
    miss = [i for i, e in enumerate(embs) if e is None]
    if miss:
        with torch.no_grad():
            pv_m = torch.cat([pv[offs[i]:offs[i + 1]] for i in miss]).to(G_dev) # (未命中 patch 数, 1536)
            out = G_vis(pv_m, grid_thw=grids[miss].to(G_dev)).pooler_output.float().cpu().numpy() # (未命中图片 token 数, 5120), spatial_merge 后每 4 个 patch 一个 token
        cnt = np.cumsum([0] + [int(grids[i].prod()) // 4 for i in miss])
        for j, i in enumerate(miss):
            e = out[cnt[j]:cnt[j + 1]].copy() # (该图 token 数, 5120)
            embs[i] = e
            if p["hashes"][i] in G_img_cache or e.nbytes > G_args.image_cache_mb << 20:
                continue
            G_img_cache[p["hashes"][i]] = e
            G_img_cache_bytes += e.nbytes
            while G_img_cache_bytes > G_args.image_cache_mb << 20:
                G_img_cache_bytes -= G_img_cache.popitem(last=False)[1].nbytes
    return np.concatenate(embs), len(p["hashes"]) - len(miss)


def run_vision(p): # 锁内执行: 视觉塔前向 (带缓存), 生成图片 token 的 embedding (只含图片位置, 按顺序) / mask / 3D 位置
    if p["out"] is None:
        return None, None, None
    ids = p["ids"]
    t0 = time.time()
    emb, hit = vision_embeds(p)
    mask = (np.array(ids) == IMAGE_TOKEN).astype(np.uint8) # (n,)
    log(f"ℹ️ 视觉塔 {p['n_images']} 张图 (缓存命中 {hit}), {int(mask.sum())} 个图片 token, {time.time() - t0:.2f}s, 缓存 {len(G_img_cache)} 张 {G_img_cache_bytes / 1048576:.0f}MB")
    if int(mask.sum()) != emb.shape[0]:
        raise RuntimeError(f"❌ 图片 token 数 {int(mask.sum())} 与视觉输出 {emb.shape[0]} 不一致")
    pos3 = rope_positions(ids, p["out"]["image_grid_thw"].tolist()) # (n, 3)
    return np.ascontiguousarray(emb, dtype=np.float32), mask, pos3 # emb: (图片 token 数, 5120)


def generate_engine(p): # cuda 引擎模式的生成器: 视觉塔 (G_lock 内) -> 等槽位 (python FIFO) -> prefill (C++ 里 FIFO) -> 逐个返回 token; 中途关闭 (客户端断开) 时取消
    with G_lock:
        embd, mask, pos3 = run_vision(p)
    seed = p["seed"] if p["seed"] is not None else random.getrandbits(63)
    args = qwen.engine_args(p["ids"], embd, mask, pos3, seed=seed, max_new=G_args.ctx - len(p["ids"]), cache=p["cache"], **p["sampling"]) # 拿到槽位前准备好, 之后不能再出错
    slot, ticket = G_pool.acquire(p["rid"])
    buf = (ctypes.c_int * 256)()
    started, done = False, False
    try:
        t0 = time.time()
        qwen.engine_prefill(slot, ticket, args)
        started = True
        t1, n = time.time(), 0
        log(f"ℹ️ {p['rid']} 槽 {slot}: 拿到槽位到第一个 token {t1 - t0:.2f}s (含等待前面的 prefill)")
        while True:
            toks = qwen.engine_next(slot, buf)
            if not toks:
                break
            n += len(toks)
            yield from toks
        done = True
        p["cache_stats"] = qwen.engine_cache_stats(slot) # 此时缓存已全部写完, 槽还没释放
        log(f"ℹ️ {p['rid']} 槽 {slot}: 生成 {n} tokens, decode {time.time() - t1:.2f}s, {(n - 1) / max(time.time() - t1, 1e-9):.2f} tok/s, 缓存 (命中 / 写入 KV / 输入 state / 输出 state) {p['cache_stats']}")
    finally:
        if started and not done:
            qwen.engine_cancel(slot)
            while qwen.engine_next(slot, buf): # 读到结束, 之后槽才能复用
                pass
        G_pool.release(slot)


def generate_tokens(p): # 生成器, 逐个返回 token id; cpu 在锁内执行, cuda 走引擎模式
    if len(p["ids"]) >= G_args.ctx:
        raise BadRequest(f"prompt 长度 {len(p['ids'])} 超出上下文 {G_args.ctx}")
    if G_args.type == "cuda":
        yield from generate_engine(p)
        return
    embd, mask, pos3 = run_vision(p)
    qwen.reset()
    qwen.seed(p["seed"] if p["seed"] is not None else random.getrandbits(63))
    yield from qwen.generate(p["ids"], G_args.ctx - len(p["ids"]), embd=embd, embd_mask=mask, pos3=pos3, cache=p["cache"], **p["sampling"])
    if p["cache"] is not None:
        p["cache_stats"] = qwen.cache_stats() # 缓存已全部写完


class Decoder: # 增量解码 (按字节, 标准库 utf-8 增量解码器), 遇到 </think> 切换到正文; 文本原样, 不去掉任何字符
    def __init__(self, thinking):
        self.in_reasoning = thinking
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace") # 多字节字符不完整时暂缓, 非法字节换成 U+FFFD (与 HF 解码相同)
        self.reasoning = ""
        self.content = ""

    def add(self, field, new):
        if field == "reasoning_content":
            self.reasoning += new
        else:
            self.content += new
        return field, new

    def push(self, tok): # 返回 (字段名, 新增文本): 每个 token 都返回一次, 没有新增文本 (</think>、停止 token、字符还不完整) 时是空串
        field = "reasoning_content" if self.in_reasoning else "content"
        if tok in qwen.EOS_IDS or (self.in_reasoning and tok == THINK_END): # 停止 token / </think> 本身不进文本, 之前还不完整的字节吐出来 (U+FFFD), 不丢
            new = self.dec.decode(b"", final=True)
            self.dec.reset()
            self.in_reasoning = self.in_reasoning and tok != THINK_END
            return self.add(field, new)
        return self.add(field, self.dec.decode(qwen_tok.token_bytes(tok)))

    def finish(self): # 输出没有以停止 token 结束 (长度上限) 时, 剩下不完整的字节; 返回 (字段名, 文本)
        return self.add("reasoning_content" if self.in_reasoning else "content", self.dec.decode(b"", final=True))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, code, msg):
        self.send_json(code, {"error": {"message": msg, "type": "invalid_request_error" if code == 400 else "server_error"}})

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_json(200, {"object": "list", "data": [{"id": G_args.model_name, "object": "model", "created": int(time.time()), "owned_by": "local"}]})
            return
        if self.path.rstrip("/") == "/health":
            self.send_json(200, {"status": "ok"})
            return
        self.send_error_json(404, f"未知路径 {self.path}")

    def do_POST(self):
        global G_waiting
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error_json(404, f"未知路径 {self.path}")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            p = build_prompt(body)
        except (BadRequest, json.JSONDecodeError) as e:
            self.send_error_json(400, str(e))
            return
        except Exception as e:
            log(f"❌ 预处理失败: {e!r}")
            self.send_error_json(400, f"请求处理失败: {e}")
            return
        stream = bool(body.get("stream"))
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        p["rid"] = rid[-8:]
        with G_queue_lock:
            G_waiting += 1
            ahead = G_waiting - 1
        if ahead > 0 and G_args.type != "cuda": # cuda 的排队在 SlotPool 里打印
            log(f"⏳ {rid[-8:]} 排队等待, 前面 {ahead} 个请求")
        if stream: # 先回响应头, 客户端排队期间也不会超时报错
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
        try:
            with G_lock if G_args.type != "cuda" else contextlib.nullcontext():
                log(f"ℹ️ {rid[-8:]} 开始: prompt {len(p['ids'])} tokens, 图片 {p['n_images']}, thinking={p['thinking']}, {p['sampling']}, seed={p['seed']}")
                if stream:
                    self.handle_stream(p, rid)
                else:
                    self.handle_normal(p, rid)
        except BadRequest as e:
            if stream:
                self.write_sse({"error": {"message": str(e), "type": "invalid_request_error"}})
            else:
                self.send_error_json(400, str(e))
        except (BrokenPipeError, ConnectionResetError):
            log(f"⚠️ {rid[-8:]} 客户端断开, 停止生成")
        except Exception as e:
            log(f"❌ {rid[-8:]} 推理失败: {e!r}")
            if not stream:
                self.send_error_json(500, f"推理失败: {e}")
        finally:
            with G_queue_lock:
                G_waiting -= 1

    def write_sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()

    def handle_normal(self, p, rid):
        dec = Decoder(p["thinking"])
        n_out, last = 0, None
        for tok in generate_tokens(p):
            n_out += 1
            last = tok
            dec.push(tok)
        dec.finish()
        msg = {"role": "assistant", "content": dec.content}
        if p["thinking"]:
            msg["reasoning_content"] = dec.reasoning
        self.send_json(200, {
            "id": rid, "object": "chat.completion", "created": int(time.time()), "model": G_args.model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop" if last in qwen.EOS_IDS else "length"}],
            "usage": make_usage(p, n_out),
        })

    def handle_stream(self, p, rid):
        base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": G_args.model_name}
        self.write_sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        dec = Decoder(p["thinking"])
        last, n_out = None, 0
        gen = generate_tokens(p)
        try:
            for tok in gen: # 每个输出 token (含停止 token) 发一次, 没有新增文本时发空串
                last = tok
                n_out += 1
                field, new = dec.push(tok)
                self.write_sse({**base, "choices": [{"index": 0, "delta": {field: new}, "finish_reason": None}]})
        finally:
            gen.close() # 客户端断开时也会走到这里, 结束生成并释放锁
        field, rest = dec.finish()
        fin = {**base, "choices": [{"index": 0, "delta": {field: rest}, "finish_reason": "stop" if last in qwen.EOS_IDS else "length"}]} # rest: 截断时剩下的不完整字节 (U+FFFD), 一般为空串
        if p["include_usage"]: # OpenAI: stream_options.include_usage 时在 [DONE] 前单独发一个 choices 为空、带 usage 的 chunk
            self.write_sse(fin)
            self.write_sse({**base, "choices": [], "usage": make_usage(p, n_out)})
        else: # 没要求时附在最后一个 chunk 上 (不会让只读 choices[0] 的客户端出错)
            self.write_sse({**fin, "usage": make_usage(p, n_out)})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global G_args, G_imgproc, G_vis, G_dev, G_pool, G_cache_dir
    p = argparse.ArgumentParser(description="Qwen3.8-27B OpenAI 兼容服务")
    p.add_argument("--gguf", required=True, help="Q4_K_M GGUF 路径")
    p.add_argument("--hf-dir", required=True, help="官方 HF 仓库目录: tokenizer/processor 配置 + 含 model.visual.* 的 safetensors 分片")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000, help="0 表示由系统自动分配, 实际端口见启动完成的日志")
    p.add_argument("--type", default="cpu", choices=list(qwen.LIBS), help="推理后端; cuda 时文本主干和视觉塔都在 GPU 上")
    p.add_argument("--threads", type=int, default=16, help="cpu 后端的推理线程数, 以及 torch (CPU 上的视觉塔、预处理) 的线程数")
    p.add_argument("--device", type=int, default=0, help="cuda 后端使用的 GPU 编号")
    p.add_argument("--ctx", type=int, default=None, help="最大上下文 (prompt + 输出), 默认 cpu 32768, cuda 262144; cuda 是每个槽的上下文, 启动时分配全部槽的 KV")
    p.add_argument("--dec-ratio", type=float, default=0.31, help="cuda: prefill 和 decode 同时运行时 decode 分区占的 SM 比例 (其余给 prefill), 取最近的可用值 (偶数个, 两边都至少 8 个), 本卡 156 SM 时 0.31 为 48 个")
    p.add_argument("--max-image-pixels", type=int, default=16777216, help="单张图片最大像素数 (约每 1024 像素 1 个 token), 默认与官方 preprocessor_config 相同 (16777216), 0 也表示用官方配置")
    p.add_argument("--model-name", default="qwen3.8-27b", help="模型名前缀, 对外返回的名称是 {model-name}-{type}; 请求里的 model 不校验")
    p.add_argument("--image-cache-mb", type=int, default=4096, help="视觉塔输出缓存上限 (MB, 只在内存, LRU 淘汰), 0 表示不缓存")
    p.add_argument("--cache-dir", default=None, help=f"前缀缓存 (磁盘) 目录, 不指定则不启用; 支持的后端 {list(qwen.CACHE_BACKENDS)}; 通过 router 启动时由 router 传入 <router 的 --cache-dir>/<type>")
    p.add_argument("--cache-threads", type=int, default=8, help="前缀缓存后台量化 state 的线程数")
    G_args = p.parse_args()
    G_args.model_name = f"{G_args.model_name}-{G_args.type}"
    if G_args.ctx is None:
        G_args.ctx = 262144 if G_args.type == "cuda" else 32768
    t0 = time.time()
    torch.set_num_threads(G_args.threads)
    if G_args.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"❌ --type cuda 需要 CUDA 版 torch, 当前 torch {torch.__version__} 不支持 CUDA")
        G_dev = f"cuda:{G_args.device}"
        n_sm = torch.cuda.get_device_properties(G_args.device).multi_processor_count
        dec_sm = min(max(round(n_sm * G_args.dec_ratio / 2) * 2, 8), n_sm - 8) # green context 按 2 个 SM 分, 每个分区至少 8 个
    if G_args.cache_dir is not None and G_args.type not in qwen.CACHE_BACKENDS:
        log(f"⚠️ {G_args.type} 后端暂不支持前缀缓存, 忽略 --cache-dir")
    elif G_args.cache_dir is not None:
        G_cache_dir = open_cache_dir(G_args.cache_dir)
    qwen.load(G_args.gguf, G_args.type, G_args.threads, G_args.device, G_args.ctx, dec_sm if G_args.type == "cuda" else None, G_cache_dir, G_args.cache_threads)
    if G_args.type == "cuda":
        G_pool = SlotPool(qwen.nseq())
        qwen.engine_start()
    G_imgproc = AutoImageProcessor.from_pretrained(G_args.hf_dir)
    qwen_tok.load(os.path.join(G_args.hf_dir, "tokenizer.json"))
    log("✅ 图片预处理 / tokenizer (qwen_tok) 加载完成")
    G_vis = load_vision(G_args.hf_dir)
    log(f"✅ 视觉塔加载完成 ({G_dev}), 参数 {sum(x.numel() for x in G_vis.parameters()) / 1e6:.1f}M")
    server = ThreadingHTTPServer((G_args.host, G_args.port), Handler)
    server.daemon_threads = True
    log(f"✅ 全部加载完成 ({time.time() - t0:.1f}s), 模型 {G_args.model_name}, 服务已启动 http://{G_args.host}:{server.server_address[1]}/v1") # router.py 靠这一行获取实际端口
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("⚠️ 收到中断, 退出")
        os._exit(0)


if __name__ == "__main__":
    main()
