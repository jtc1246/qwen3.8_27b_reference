#!/usr/bin/env python3
# OpenAI 兼容服务的端到端测试 (标准库 urllib + PIL 生成测试图片), 需要先启动 server.py
import io, sys, json, time, base64, argparse, threading, urllib.request
from PIL import Image, ImageDraw

G_base = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def post(body, stream=False):
    req = urllib.request.Request(G_base + "/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=3600)
    if not stream:
        return json.load(resp)
    reasoning, content, finish = "", "", None
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])["choices"][0]
        reasoning += ch["delta"].get("reasoning_content", "")
        content += ch["delta"].get("content", "")
        finish = ch.get("finish_reason") or finish
    return {"reasoning": reasoning, "content": content, "finish": finish}


def make_image():
    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 260, 260), fill="red")
    d.rectangle((380, 100, 580, 300), fill="blue")
    d.text((200, 380), "QWEN 2026", fill="black", font_size=60)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    global G_base
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    args = p.parse_args()
    G_base = args.base
    log(f"ℹ️ models: {json.load(urllib.request.urlopen(G_base + '/models'))}")
    r = post({"messages": [{"role": "system", "content": "你是一个只用一句话回答的助手。"}, {"role": "user", "content": "1+1等于几?"}], "temperature": 0, "enable_thinking": False})
    log(f"✅ [system + 关闭思考 + 非流式] {r['choices'][0]['message']} finish={r['choices'][0]['finish_reason']} usage={r['usage']}")
    r = post({"messages": [{"role": "user", "content": "2+3等于几? 只回答数字。"}], "reasoning_effort": "low", "stream": True}, stream=True)
    log(f"✅ [思考 low + 流式] reasoning={r['reasoning']!r} content={r['content']!r} finish={r['finish']}")
    r = post({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": make_image()}}, {"type": "text", "text": "图里有哪些形状、分别什么颜色? 图里写了什么字? 简短回答。"}]}], "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0})
    log(f"✅ [图片 + 关闭思考] {r['choices'][0]['message']['content']!r} usage={r['usage']}")
    body = {"messages": [{"role": "user", "content": "随便说一个水果的名字, 只说名字。"}], "enable_thinking": False, "temperature": 1.5, "seed": 1234}
    a, b = post(body), post(body)
    log(f"{'✅' if a['choices'][0]['message'] == b['choices'][0]['message'] else '⚠️'} [seed 一致性] {a['choices'][0]['message']['content']!r} vs {b['choices'][0]['message']['content']!r}")
    res = [None, None]
    def worker(i):
        res[i] = post({"messages": [{"role": "user", "content": f"用一句话介绍数字 {i + 7}。"}], "enable_thinking": False})
    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for i in range(2):
        log(f"✅ [并发 {i}] {res[i]['choices'][0]['message']['content']!r}")
    log(f"ℹ️ 2 个并发请求总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
