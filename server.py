#!/usr/bin/env python3
# OpenAI 兼容 chat completions 服务: 支持图片、system、temperature、seed、thinking 控制, 其余参数忽略
# C++ 推理是单序列全局状态, 请求用全局锁串行处理, 并发请求排队等待而不报错
# --type 选择后端 (qwen.LIBS), cuda 时视觉塔也在 GPU 上; 视觉塔输出按图片原始字节的 sha256 缓存在内存 (LRU)
import os, sys, io, json, time, uuid, base64, random, hashlib, argparse, threading, urllib.request
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
import qwen

IMAGE_TOKEN = 248056
THINK_END = 248069
SAMPLING = { # 官方推荐参数 (README Best Practices), 客户端只能覆盖 temperature
    True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
    False: dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5),
}
EFFORT_MAP = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"} # OpenAI reasoning_effort -> 模板支持的取值

G_args = None
G_proc = None
G_vis = None
G_dev = "cpu" # 视觉塔所在的 torch device
G_img_cache = OrderedDict() # sha256 -> 该图的视觉输出 (n_tok, 5120) float32, 只在 G_lock 内访问
G_img_cache_bytes = 0
G_lock = threading.Lock()
G_queue_lock = threading.Lock()
G_waiting = 0


class BadRequest(Exception):
    pass


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


def load_image(url): # 返回 (RGB 图片, 原始字节的 sha256)
    if url.startswith("data:"):
        data = base64.b64decode(url.split(",", 1)[1])
    elif url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    else:
        raise BadRequest("图片只支持 data URL (base64) 或 http(s) URL")
    return Image.open(io.BytesIO(data)).convert("RGB"), hashlib.sha256(data).hexdigest()


def convert_messages(messages): # OpenAI 格式 -> chat 模板格式, 图片和 sha256 单独取出
    out, images, hashes = [], [], []
    for m in messages:
        role = "system" if m.get("role") == "developer" else m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    iu = p.get("image_url")
                    img, h = load_image(iu if isinstance(iu, str) else iu.get("url", ""))
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


def thinking_options(body): # 返回 (enable_thinking, reasoning_effort)
    ctk = body.get("chat_template_kwargs") or {}
    enable = body.get("enable_thinking", ctk.get("enable_thinking", True))
    effort = body.get("reasoning_effort", ctk.get("reasoning_effort"))
    if effort in ("none", "minimal"):
        return False, None
    if effort is not None and effort not in EFFORT_MAP:
        raise BadRequest(f"不支持的 reasoning_effort: {effort}")
    return bool(enable), (EFFORT_MAP[effort] if effort else None)


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


def build_prompt(body): # 锁外执行: 解析消息、下载图片、模板渲染、分词、图片预处理
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages 不能为空")
    msgs, images, hashes = convert_messages(messages)
    enable_thinking, effort = thinking_options(body)
    kw = {"enable_thinking": enable_thinking}
    if effort:
        kw["reasoning_effort"] = effort
    try:
        text = G_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception as e: # 模板里的 raise_exception, 比如 system 不在开头
        raise BadRequest(f"chat 模板渲染失败: {e}")
    if images:
        size = {"longest_edge": G_args.max_image_pixels, "shortest_edge": 65536} if G_args.max_image_pixels > 0 else None
        out = G_proc(text=[text], images=images, return_tensors="pt", **({"size": size} if size else {}))
    else:
        out = G_proc(text=[text], return_tensors="pt")
    ids = out["input_ids"][0].tolist()
    temp = body.get("temperature")
    sp = dict(SAMPLING[enable_thinking])
    if temp is not None:
        if not isinstance(temp, (int, float)) or temp < 0:
            raise BadRequest("temperature 必须是非负数")
        sp["temperature"] = float(temp)
    seed = body.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise BadRequest("seed 必须是整数")
    return {"ids": ids, "out": out if images else None, "hashes": hashes, "thinking": enable_thinking, "sampling": sp, "seed": seed, "n_images": len(images)}


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


def run_vision(p): # 锁内执行: 视觉塔前向 (带缓存), 生成 embedding / mask / 3D 位置
    if p["out"] is None:
        return None, None, None
    ids = p["ids"]
    t0 = time.time()
    emb, hit = vision_embeds(p)
    mask = (np.array(ids) == IMAGE_TOKEN).astype(np.uint8) # (n,)
    log(f"ℹ️ 视觉塔 {p['n_images']} 张图 (缓存命中 {hit}), {int(mask.sum())} 个图片 token, {time.time() - t0:.2f}s, 缓存 {len(G_img_cache)} 张 {G_img_cache_bytes / 1048576:.0f}MB")
    if int(mask.sum()) != emb.shape[0]:
        raise RuntimeError(f"❌ 图片 token 数 {int(mask.sum())} 与视觉输出 {emb.shape[0]} 不一致")
    embd = np.zeros((len(ids), 5120), dtype=np.float32) # (n, 5120)
    embd[mask.astype(bool)] = emb
    pos3 = rope_positions(ids, p["out"]["image_grid_thw"].tolist()) # (n, 3)
    return embd, mask, pos3


def generate_tokens(p): # 锁内执行的生成器, 逐个返回 token id
    if len(p["ids"]) >= G_args.ctx:
        raise BadRequest(f"prompt 长度 {len(p['ids'])} 超出上下文 {G_args.ctx}")
    embd, mask, pos3 = run_vision(p)
    qwen.reset()
    qwen.seed(p["seed"] if p["seed"] is not None else random.getrandbits(63))
    yield from qwen.generate(p["ids"], G_args.ctx - len(p["ids"]), embd=embd, embd_mask=mask, pos3=pos3, **p["sampling"])


class Decoder: # 增量解码, 遇到 </think> 切换到正文; 多字节字符未完整时暂缓输出
    def __init__(self, thinking):
        self.in_reasoning = thinking
        self.seg = []
        self.emitted = ""
        self.reasoning = ""
        self.content = ""

    def push(self, tok): # 返回 (字段名, 新增文本) 或 None
        if self.in_reasoning and tok == THINK_END:
            self.in_reasoning = False
            self.seg, self.emitted = [], ""
            return None
        self.seg.append(tok)
        text = G_proc.tokenizer.decode(self.seg, skip_special_tokens=False)
        if text.endswith("�"):
            return None
        if not self.in_reasoning: # 正文段每次都去掉开头的换行 (</think> 后的 \n\n), 保证增量偏移一致
            text = text.lstrip("\n")
        new = text[len(self.emitted):]
        self.emitted = text
        if not new:
            return None
        if self.in_reasoning:
            self.reasoning += new
            return "reasoning_content", new
        self.content += new
        return "content", new


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
        with G_queue_lock:
            G_waiting += 1
            ahead = G_waiting - 1
        if ahead > 0:
            log(f"⏳ {rid[-8:]} 排队等待, 前面 {ahead} 个请求")
        if stream: # 先回响应头, 客户端排队期间也不会超时报错
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
        try:
            with G_lock:
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
            if tok not in qwen.EOS_IDS:
                dec.push(tok)
        msg = {"role": "assistant", "content": dec.content}
        if p["thinking"]:
            msg["reasoning_content"] = dec.reasoning.strip()
        self.send_json(200, {
            "id": rid, "object": "chat.completion", "created": int(time.time()), "model": G_args.model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop" if last in qwen.EOS_IDS else "length"}],
            "usage": {"prompt_tokens": len(p["ids"]), "completion_tokens": n_out, "total_tokens": len(p["ids"]) + n_out},
        })

    def handle_stream(self, p, rid):
        base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": G_args.model_name}
        self.write_sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        dec = Decoder(p["thinking"])
        last = None
        gen = generate_tokens(p)
        try:
            for tok in gen:
                last = tok
                if tok in qwen.EOS_IDS:
                    continue
                r = dec.push(tok)
                if r:
                    self.write_sse({**base, "choices": [{"index": 0, "delta": {r[0]: r[1]}, "finish_reason": None}]})
        finally:
            gen.close() # 客户端断开时也会走到这里, 结束生成并释放锁
        self.write_sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if last in qwen.EOS_IDS else "length"}]})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global G_args, G_proc, G_vis, G_dev
    p = argparse.ArgumentParser(description="Qwen3.8-27B OpenAI 兼容服务")
    p.add_argument("--gguf", required=True, help="Q4_K_M GGUF 路径")
    p.add_argument("--hf-dir", required=True, help="官方 HF 仓库目录: tokenizer/processor 配置 + 含 model.visual.* 的 safetensors 分片")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000, help="0 表示由系统自动分配, 实际端口见启动完成的日志")
    p.add_argument("--type", default="cpu", choices=list(qwen.LIBS), help="推理后端; cuda 时文本主干和视觉塔都在 GPU 上")
    p.add_argument("--threads", type=int, default=16, help="cpu 后端的推理线程数, 以及 torch (CPU 上的视觉塔、预处理) 的线程数")
    p.add_argument("--device", type=int, default=0, help="cuda 后端使用的 GPU 编号")
    p.add_argument("--ctx", type=int, default=32768, help="最大上下文 (prompt + 输出)")
    p.add_argument("--max-image-pixels", type=int, default=1048576, help="单张图片最大像素数 (约每 1024 像素 1 个 token), 0 表示用官方默认 16777216")
    p.add_argument("--model-name", default="qwen3.8-27b", help="模型名前缀, 对外返回的名称是 {model-name}-{type}; 请求里的 model 不校验")
    p.add_argument("--image-cache-mb", type=int, default=4096, help="视觉塔输出缓存上限 (MB, 只在内存, LRU 淘汰), 0 表示不缓存")
    G_args = p.parse_args()
    G_args.model_name = f"{G_args.model_name}-{G_args.type}"
    t0 = time.time()
    torch.set_num_threads(G_args.threads)
    if G_args.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"❌ --type cuda 需要 CUDA 版 torch, 当前 torch {torch.__version__} 不支持 CUDA")
        G_dev = f"cuda:{G_args.device}"
    qwen.load(G_args.gguf, G_args.type, G_args.threads, G_args.device, G_args.ctx)
    G_proc = AutoProcessor.from_pretrained(G_args.hf_dir)
    log("✅ processor / tokenizer 加载完成")
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
