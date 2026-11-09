#!/usr/bin/env python3
# OpenAI 兼容 chat completions 服务: 支持图片、system、temperature、seed、thinking 控制, 其余参数忽略
# C++ 推理是单序列全局状态, 请求用全局锁串行处理, 并发请求排队等待而不报错
import os, sys, io, json, time, uuid, base64, random, argparse, threading, urllib.request
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
    return vis.eval()


def load_image(url):
    if url.startswith("data:"):
        data = base64.b64decode(url.split(",", 1)[1])
    elif url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    else:
        raise BadRequest("图片只支持 data URL (base64) 或 http(s) URL")
    return Image.open(io.BytesIO(data)).convert("RGB")


def convert_messages(messages): # OpenAI 格式 -> chat 模板格式, 图片单独取出
    out, images = [], []
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
                    images.append(load_image(iu if isinstance(iu, str) else iu.get("url", "")))
                    parts.append({"type": "image"})
                else:
                    raise BadRequest(f"不支持的 content 类型: {p.get('type')}")
            content = parts
        msg = {"role": role, "content": content}
        if isinstance(m.get("reasoning_content"), str):
            msg["reasoning_content"] = m["reasoning_content"]
        out.append(msg)
    return out, images


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
    msgs, images = convert_messages(messages)
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
    return {"ids": ids, "out": out if images else None, "thinking": enable_thinking, "sampling": sp, "seed": seed, "n_images": len(images)}


def run_vision(p): # 锁内执行: 视觉塔前向, 生成 embedding / mask / 3D 位置
    if p["out"] is None:
        return None, None, None
    ids = p["ids"]
    with torch.no_grad():
        emb = G_vis(p["out"]["pixel_values"], grid_thw=p["out"]["image_grid_thw"]).pooler_output # (n_img_tok, 5120) float32
    mask = (np.array(ids) == IMAGE_TOKEN).astype(np.uint8) # (n,)
    if int(mask.sum()) != emb.shape[0]:
        raise RuntimeError(f"❌ 图片 token 数 {int(mask.sum())} 与视觉输出 {emb.shape[0]} 不一致")
    embd = np.zeros((len(ids), 5120), dtype=np.float32) # (n, 5120)
    embd[mask.astype(bool)] = emb.numpy()
    pos3 = rope_positions(ids, p["out"]["image_grid_thw"].tolist()) # (n, 3)
    return embd, mask, pos3


def generate_tokens(p): # 锁内执行的生成器, 逐个返回 token id
    if len(p["ids"]) >= G_args.ctx:
        raise BadRequest(f"prompt 长度 {len(p['ids'])} 超出上下文 {G_args.ctx}")
    t0 = time.time()
    embd, mask, pos3 = run_vision(p)
    if p["out"] is not None:
        log(f"ℹ️ 视觉塔 {p['n_images']} 张图, {int(mask.sum())} 个图片 token, {time.time() - t0:.2f}s")
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
    global G_args, G_proc, G_vis
    p = argparse.ArgumentParser(description="Qwen3.8-27B OpenAI 兼容服务 (CPU)")
    p.add_argument("--gguf", required=True, help="Q4_K_M GGUF 路径")
    p.add_argument("--hf-dir", required=True, help="官方 HF 仓库目录: tokenizer/processor 配置 + 含 model.visual.* 的 safetensors 分片")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--ctx", type=int, default=32768, help="最大上下文 (prompt + 输出)")
    p.add_argument("--max-image-pixels", type=int, default=1048576, help="单张图片最大像素数 (约每 1024 像素 1 个 token), 0 表示用官方默认 16777216")
    p.add_argument("--model-name", default="qwen3.8-27b")
    G_args = p.parse_args()
    t0 = time.time()
    torch.set_num_threads(G_args.threads)
    qwen.load(G_args.gguf, G_args.threads, G_args.ctx)
    G_proc = AutoProcessor.from_pretrained(G_args.hf_dir)
    log("✅ processor / tokenizer 加载完成")
    G_vis = load_vision(G_args.hf_dir)
    log(f"✅ 视觉塔加载完成, 参数 {sum(x.numel() for x in G_vis.parameters()) / 1e6:.1f}M")
    server = ThreadingHTTPServer((G_args.host, G_args.port), Handler)
    server.daemon_threads = True
    log(f"✅ 全部加载完成 ({time.time() - t0:.1f}s), 服务已启动 http://{G_args.host}:{G_args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("⚠️ 收到中断, 退出")
        os._exit(0)


if __name__ == "__main__":
    main()
