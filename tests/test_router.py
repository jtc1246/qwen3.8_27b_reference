#!/usr/bin/env python3
# router.py 的路由逻辑测试 (模型列表、按 model 选后端、错误情况), 需要先启动 router.py; 各后端的完整功能用 test_server.py --model 测
import sys, json, time, argparse, urllib.request, urllib.error

G_base = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def post(body): # 返回 (状态码, json)
    req = urllib.request.Request(G_base + "/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=3600))
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def main():
    global G_base
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    args = p.parse_args()
    G_base = args.base
    fails = 0
    models = [m["id"] for m in json.load(urllib.request.urlopen(G_base + "/models"))["data"]]
    log(f"ℹ️ 模型列表: {models}")
    for m in models: # 每个模型名的大小写变体都应该路由到自己
        code, r = post({"model": m.upper(), "messages": [{"role": "user", "content": "1+1等于几? 只回答数字。"}], "enable_thinking": False, "temperature": 0})
        ok = code == 200 and r["model"] == m
        fails += not ok
        log(f"{'✅' if ok else '❌'} [model={m.upper()!r}] {code} 响应模型 {r.get('model')} 内容 {r['choices'][0]['message']['content'] if code == 200 else r}")
    for m in ["qwen3.8-27b", "", None, "qwen-cpu-cuda"]: # 一个都不匹配 / 匹配多个
        code, r = post({"model": m, "messages": [{"role": "user", "content": "hi"}]})
        ok = code == 400
        fails += not ok
        log(f"{'✅' if ok else '❌'} [model={m!r} 应报错] {code} {r['error']['message'] if 'error' in r else r}")
    print("✅ 全部通过" if fails == 0 else f"❌ {fails} 项失败", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
