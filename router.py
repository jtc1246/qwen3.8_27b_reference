#!/usr/bin/env python3
# 多后端路由: 按 --mask 为每个选中的后端 (qwen.LIBS) 启动一个 server.py 子进程 (只监听 127.0.0.1, 端口自动分配), 按请求的 model 名转发
# model 名 (转小写) 必须恰好包含一个后端名, 否则报错; 任一子进程启动失败、超时或中途退出, 整个路由退出 (其余子进程一起结束)
import os, sys, re, json, time, shlex, signal, ctypes, argparse, threading, subprocess, http.client, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import qwen

ROOT = os.path.dirname(os.path.abspath(__file__))
READY_RE = re.compile(r"服务已启动 http://\S+:(\d+)/v1") # 对应 server.py 启动完成的那行日志
G_children = {} # 后端名 -> {"proc", "port", "models", "dead"}, 按 qwen.LIBS 的顺序
G_started = False


def log(msg): # 整行一次写入, 避免多线程日志接在一起
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] [router] {msg}\n")
    sys.stderr.flush()


def kill_all_and_exit(code):
    for c in G_children.values():
        c["proc"].kill()
    os._exit(code)


def set_pdeathsig(): # 子进程里执行: 路由进程退出 (包括被 kill -9) 时子进程收到 SIGKILL
    ctypes.CDLL(None).prctl(1, signal.SIGKILL) # PR_SET_PDEATHSIG = 1


def pump(name): # 转发子进程日志 (加前缀), 从中解析实际端口; 子进程退出时: 启动阶段由主线程发现, 运行阶段整个路由退出
    c = G_children[name]
    for raw in c["proc"].stderr:
        line = raw.decode(errors="replace").rstrip("\n")
        sys.stderr.write(f"[{name}] {line}\n")
        sys.stderr.flush()
        m = READY_RE.search(line)
        if m and c["port"] is None:
            c["port"] = int(m.group(1))
    c["proc"].wait()
    log(f"❌ 后端 {name} 的子进程退出, 返回码 {c['proc'].returncode}")
    c["dead"] = True
    if G_started:
        kill_all_and_exit(1)


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
            now = int(time.time())
            self.send_json(200, {"object": "list", "data": [{**m, "created": now} for c in G_children.values() for m in c["models"]]})
            return
        if self.path.rstrip("/") == "/health":
            self.send_json(200, {"status": "ok"})
            return
        self.send_error_json(404, f"未知路径 {self.path}")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error_json(404, f"未知路径 {self.path}")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            model = json.loads(body or b"{}").get("model")
        except (json.JSONDecodeError, AttributeError) as e:
            self.send_error_json(400, f"请求体不是合法的 JSON 对象: {e}")
            return
        model = model if isinstance(model, str) else ""
        hit = [n for n in qwen.LIBS if n in model.lower()]
        if len(hit) != 1:
            self.send_error_json(400, f"model 必须恰好包含一个后端名 {list(qwen.LIBS)}, 当前 {model!r} 匹配到 {hit}; 可用模型 {[m['id'] for c in G_children.values() for m in c['models']]}")
            return
        if hit[0] not in G_children:
            self.send_error_json(400, f"后端 {hit[0]} 没有启动 (--mask), 已启动 {list(G_children)}")
            return
        self.forward(hit[0], body)

    def forward(self, name, body): # 原样转发给子进程, 响应边收边回 (流式 SSE 也一样); 客户端断开时关闭上游连接, 子进程随即停止生成
        conn = http.client.HTTPConnection("127.0.0.1", G_children[name]["port"])
        sent = False
        try:
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("server", "date", "connection", "transfer-encoding"):
                    self.send_header(k, v)
            self.end_headers()
            sent = True
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log(f"⚠️ {name} 请求的客户端断开")
        except OSError as e:
            log(f"❌ 转发到 {name} 失败: {e!r}")
            if not sent:
                self.send_error_json(502, f"转发到后端 {name} 失败: {e}")
        finally:
            conn.close()


def main():
    global G_started
    names = list(qwen.LIBS)
    p = argparse.ArgumentParser(description="Qwen3.8-27B 多后端路由 (OpenAI 兼容)", allow_abbrev=False, epilog="其余参数 (--gguf、--hf-dir、--ctx、--threads、--device 等) 原样传给每个 server.py 子进程")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000, help="0 表示由系统自动分配")
    p.add_argument("--mask", default="1" * len(names), help=f"每一位对应一个后端, 顺序 {names}, 1 启动 0 不启动")
    p.add_argument("--start-timeout", type=float, default=600, help="等待每个子进程启动完成的秒数")
    p.add_argument("--cache-dir", required=True, help=f"前缀缓存根目录 (必填): 不存在则创建 (上级目录必须存在); 支持缓存的后端 {list(qwen.CACHE_BACKENDS)} 收到 <cache-dir>/<后端名>, 其余后端不传")
    args, extra = p.parse_known_args()
    if len(args.mask) != len(names) or set(args.mask) - {"0", "1"}:
        p.error(f"❌ --mask 必须是 {len(names)} 位的 0/1 串, 依次对应 {names}, 当前 {args.mask!r}")
    sel = [n for n, b in zip(names, args.mask) if b == "1"]
    if not sel:
        p.error(f"❌ --mask {args.mask} 没有选择任何后端")
    for a in extra:
        if a.split("=")[0] in ("--type", "--host", "--port"):
            p.error(f"❌ {a.split('=')[0]} 由路由决定, 不能传给子进程")
    root = os.path.abspath(args.cache_dir)
    if not os.path.isdir(root):
        if os.path.exists(root) or not os.path.isdir(os.path.dirname(root)):
            p.error(f"❌ 缓存根目录 {root} 不是目录, 或者它的上级目录不存在")
        os.mkdir(root)
        log(f"ℹ️ 创建缓存根目录 {root}")
    for name in sel: # 先全部启动, 再开日志线程 (preexec_fn 在没有其他线程时执行更安全)
        cache = ["--cache-dir", os.path.join(root, name)] if name in qwen.CACHE_BACKENDS else []
        cmd = "exec " + shlex.join([sys.executable, "-u", os.path.join(ROOT, "server.py"), "--type", name, "--host", "127.0.0.1", "--port", "0"] + cache + extra) # exec: pdeathsig 作用在 python 进程而不是 sh 上
        log(f"ℹ️ 启动后端 {name}: {cmd}")
        G_children[name] = {"proc": subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, preexec_fn=set_pdeathsig), "port": None, "models": [], "dead": False}
    for name in sel:
        threading.Thread(target=pump, args=(name,), daemon=True).start()
    t0 = time.time()
    while not all(G_children[n]["port"] for n in sel): # 同时等所有子进程, 任何一个退出或超时立即全部结束
        dead = [n for n in sel if G_children[n]["dead"]]
        if dead or time.time() - t0 > args.start_timeout:
            log(f"❌ 后端 {dead} 启动失败 (子进程已退出), 全部退出" if dead else f"❌ 超过 {args.start_timeout:.0f}s 仍有后端未就绪 {[n for n in sel if not G_children[n]['port']]}, 全部退出")
            kill_all_and_exit(1)
        time.sleep(0.2)
    for name in sel:
        c = G_children[name]
        c["models"] = json.load(urllib.request.urlopen(f"http://127.0.0.1:{c['port']}/v1/models", timeout=30))["data"]
        log(f"✅ 后端 {name} 就绪: 端口 {c['port']}, 模型 {[m['id'] for m in c['models']]}")
    G_started = True
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log(f"✅ 路由已启动 http://{args.host}:{server.server_address[1]}/v1, 后端 {sel}, 启动耗时 {time.time() - t0:.1f}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("⚠️ 收到中断, 退出")
        kill_all_and_exit(0)


if __name__ == "__main__":
    main()
