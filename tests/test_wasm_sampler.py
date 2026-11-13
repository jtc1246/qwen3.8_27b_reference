#!/usr/bin/env python3
# 自定义采样 (sampler_cpp + sampler_data, wasm) 的端到端测试 (标准库 + 仓库的 qwen_tok), 需要先启动带 --sampler-rootfs / --cache-dir 的 cuda server.py
# 检查: custom_data 回放 (按给定 token 序列生成, 文本逐字节相同)、fp16 贪心与内置 temperature=0 对比、编译隔离 (读不到宿主文件)、运行出错 / 返回值不合法的报错、槽复用、并发
import os, sys, json, time, base64, struct, argparse, threading, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qwen_tok

G_base = None
G_model = None
MSG = [{"role": "user", "content": "用一句话介绍一下北京。"}]
THINK_END, IM_END, END_OF_TEXT = 248069, 248046, 248044 # endoftext 不允许返回
REPLAY = r"""
// 回放: custom_data 是 int32 小端的 token 序列, 依次返回; 用完后返回最后一个 (由调用方放结束 token)
const int* G_tok = nullptr;
uint64_t G_n = 0, G_i = 0;
void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) {
    G_tok = (const int*)custom_data;
    G_n = custom_data_len / 4;
}
int sample(void* data) { return G_i < G_n ? G_tok[G_i++] : G_tok[G_n - 1]; }
"""
GREEDY = r"""
// 贪心: 可采样的 token 里取 fp16 logit 最大 (相同取 id 小的), 已生成的 token 先减 PP (与内置采样的 presence penalty 相同, 只减一次)
const float PP = %s;
bool G_seen[QS_VOCAB], G_think = false;
void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) { G_think = is_thinking_mode; }
int sample(void* data) {
    const uint16_t* h = (const uint16_t*)data;
    int best = -1;
    float bv = 0;
    for (int i = 0; i < QS_VOCAB; i++) {
        if (!qs_allowed(i, G_think)) continue;
        float v = qs_half_to_float(h[i]) - (G_seen[i] ? PP : 0.0f);
        if (best < 0 || v > bv) best = i, bv = v;
    }
    G_seen[best] = true;
    if (best == QS_THINK_END) G_think = false;
    return best;
}
"""
INIT0 = "void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) {}\n"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def post(body, stream=False): # 返回 (HTTP 状态码, 结果); 流式结果: {"reasoning", "content", "finish", "n_chunks", "error"}
    if G_model:
        body = {**body, "model": G_model}
    req = urllib.request.Request(G_base + "/chat/completions", data=json.dumps({**body, "stream": stream}, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)
    if not stream:
        return resp.status, json.load(resp)
    out = {"reasoning": "", "content": "", "finish": None, "n_chunks": 0, "error": None}
    for line in resp:
        line = line.decode()
        if not line.startswith("data: ") or line.startswith("data: [DONE]"):
            continue
        obj = json.loads(line[6:])
        if "error" in obj:
            out["error"] = obj["error"]["message"]
            continue
        if not obj["choices"]:
            continue
        ch = obj["choices"][0]
        out["n_chunks"] += 1
        out["reasoning"] += ch["delta"].get("reasoning_content", "")
        out["content"] += ch["delta"].get("content", "")
        out["finish"] = ch.get("finish_reason") or out["finish"]
    return resp.status, out


def check(cond, msg):
    if not cond:
        log(f"❌ {msg}")
        sys.exit(1)
    log(f"✅ {msg}")


def b64(b):
    return base64.b64encode(b).decode()


def toks(ids):
    return b"".join(struct.pack("<i", t) for t in ids)


def expect_400(body, keyword, name):
    t0 = time.time()
    code, r = post(body)
    msg = r.get("error", {}).get("message", "") if code != 200 else ""
    check(code == 400 and keyword in msg, f"{name}: HTTP {code}, {time.time() - t0:.1f}s, {msg[:200]!r}")


def test_replay(): # custom_data 里的 token 序列原样生成, 解码后与原文逐字节相同
    text = "你好，世界！Hello, 🌍 123。"
    _, r = post({"messages": MSG, "enable_thinking": False, "sampler_cpp": REPLAY, "sampler_data": b64(toks(qwen_tok.encode(text) + [IM_END]))})
    m = r["choices"][0]["message"]
    check(m["content"] == text and r["choices"][0]["finish_reason"] == "stop", f"回放 (关 thinking, im_end 结束): {m['content']!r}")
    rs, ct = "让我想想。\n", "\n\n答案是 42。"
    ids = qwen_tok.encode(rs) + [THINK_END] + qwen_tok.encode(ct) + [IM_END]
    code, r = post({"messages": MSG, "enable_thinking": True, "sampler_cpp": REPLAY, "sampler_data": b64(toks(ids))}, stream=True)
    check(code == 200 and r["error"] is None and r["reasoning"] == rs and r["content"] == ct and r["finish"] == "stop", f"回放 (开 thinking, 流式): {r['reasoning']!r} / {r['content']!r}")
    src = "bool G_ok = false;\nvoid sampler_init(bool t, void* p, uint64_t n) { G_ok = p == nullptr && n == 0; }\nint sample(void* data) { return G_ok ? QS_IM_END : -1; }\n" # 不满足时返回 -1, 请求报错
    code, r = post({"messages": MSG, "enable_thinking": False, "sampler_cpp": src})
    check(code == 200 and r["choices"][0]["message"]["content"] == "", "不带 sampler_data 时 sampler_init 收到 nullptr / 0")


def test_greedy(): # fp16 贪心 vs 内置 temperature=0 (fp32): 只有 fp16 舍入后出现相同或倒序的 top logit 时才会不同
    for think, pp in ((False, "1.5f"), (True, "0.0f")):
        base = {"messages": MSG, "enable_thinking": think}
        _, a = post({**base, "temperature": 0})
        _, b = post({**base, "sampler_cpp": GREEDY % pp})
        ma, mb = a["choices"][0]["message"], b["choices"][0]["message"]
        log(f"ℹ️ thinking={think}: 内置 {a['usage']['completion_tokens']} tokens, wasm {b['usage']['completion_tokens']} tokens, 正文 {mb['content'][-60:]!r}")
        check(ma == mb, f"thinking={think}: wasm fp16 贪心与内置 temperature=0 输出相同")


def test_top_k():
    body = {"messages": MSG, "enable_thinking": False, "sampler_cpp": open(G_args.example).read(), "sampler_data": b64(struct.pack("<Q", 12345))}
    _, a = post(body)
    code, b = post(body, stream=True)
    ca = a["choices"][0]["message"]["content"]
    log(f"ℹ️ top_k 示例: {ca!r}")
    check(code == 200 and b["error"] is None and ca == b["content"] and b["finish"] == "stop", "top_k 示例: 同一 sampler_data (种子) 非流式 / 流式输出相同")


def test_errors():
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "int sample(void* data) { return x; }"}, "编译失败", "编译错误")
    expect_400({"messages": MSG, "sampler_cpp": f'#include "{os.path.abspath(__file__)}"\n' + INIT0 + "int sample(void* data) { return 0; }"}, "file not found", "include 宿主机文件 (chroot 里不存在)")
    expect_400({"messages": MSG, "sampler_cpp": "#include <cstdio>\n" + INIT0 + 'int sample(void* data) { printf("x"); return 0; }'}, "导入", "调用 WASI 函数")
    expect_400({"messages": MSG, "sampler_cpp": "//" + "x" * (1 << 20)}, "超过上限", "源码超过 1MB")
    expect_400({"messages": MSG, "sampler_cpp": INIT0}, "缺少 int sample(void* data)", "没有 sample")
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "int sample(float* data) { return 0; }"}, "缺少 int sample(void* data)", "sample 参数类型不对")
    expect_400({"messages": MSG, "sampler_cpp": "int sample(void* data) { return 0; }"}, "缺少 void sampler_init", "没有 sampler_init")
    expect_400({"messages": MSG, "sampler_cpp": "void sampler_init(bool t, void* p, int n) {}\nint sample(void* data) { return 0; }"}, "缺少 void sampler_init", "sampler_init 参数类型不对")
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "float sample(void* data) { return 0; }"}, "返回值类型不对", "sample 返回值类型不对")
    expect_400({"messages": MSG, "sampler_cpp": REPLAY, "sampler_data": b64(b"\0" * ((1 << 20) + 1))}, "超过上限", "sampler_data 超过 1MB")
    expect_400({"messages": MSG, "sampler_cpp": REPLAY, "sampler_data": "!!!"}, "base64", "sampler_data 不是 base64")
    expect_400({"messages": MSG, "sampler_data": "AAAA"}, "只能和 sampler_cpp 一起用", "只有 sampler_data")
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "int sample(void* data) { volatile int x = 0; while (true) x = x + 1; }"}, "instruction limit exceeded", "sample 死循环")
    expect_400({"messages": MSG, "sampler_cpp": "void sampler_init(bool t, void* p, uint64_t n) { while (true) {} }\nint sample(void* data) { return 0; }"}, "sampler_init 执行出错", "sampler_init 死循环")
    expect_400({"messages": MSG, "sampler_cpp": "struct A { A() { while (true) {} } } G_a;\n" + INIT0 + "int sample(void* data) { return 0; }"}, "全局构造", "全局构造函数死循环")
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "int sample(void* data) { return *(int*)0x7ffffff0; }"}, "out of bounds memory access", "越界访问")
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "int sample(void* data) { return 248045; }"}, "不允许的 token 248045", "返回控制 token")
    expect_400({"messages": MSG, "sampler_cpp": INIT0 + "int sample(void* data) { return QS_VOCAB; }"}, "不允许的 token 248320", "返回超出词表的 token")
    expect_400({"messages": MSG, "sampler_cpp": REPLAY, "sampler_data": b64(toks([100, 101, END_OF_TEXT]))}, "已生成 2 个 token): sample 返回了不允许的 token 248044", "返回 endoftext")
    expect_400({"messages": MSG, "enable_thinking": False, "sampler_cpp": INIT0 + "int sample(void* data) { return QS_THINK_END; }"}, "不是 thinking 模式", "非 thinking 模式返回思考结束 token")
    expect_400({"messages": MSG, "enable_thinking": True, "sampler_cpp": REPLAY, "sampler_data": b64(toks([100, THINK_END, 101, THINK_END]))}, "思考已经结束", "思考结束后再次返回思考结束 token")
    trap5 = "int G_n = 0;\n" + INIT0 + "int sample(void* data) { if (G_n == 5) __builtin_trap(); return 100 + G_n++; }"
    expect_400({"messages": MSG, "sampler_cpp": trap5}, "已生成 5 个 token", "第 6 次 sample trap (非流式)")
    code, r = post({"messages": MSG, "sampler_cpp": trap5}, stream=True)
    check(code == 200 and r["n_chunks"] == 6 and r["error"] and "unreachable" in r["error"], f"第 6 次 sample trap (流式): role + 5 个 token chunk 后收到错误 {r['error']!r}")


def test_concurrent(): # 4 个请求同时: 2 个 wasm 回放 + 2 个内置 temperature=0, 结果都与单独跑的相同
    base = {"messages": MSG, "enable_thinking": False}
    _, ref = post({**base, "temperature": 0})
    ref = ref["choices"][0]["message"]["content"]
    texts = ["第一个并发回放。", "第二个并发回放, 内容不同。"]
    res = [None] * 4
    def run(i):
        body = {**base, "sampler_cpp": REPLAY, "sampler_data": b64(toks(qwen_tok.encode(texts[i // 2]) + [IM_END]))} if i % 2 else {**base, "temperature": 0}
        _, r = post(body)
        res[i] = r["choices"][0]["message"]["content"]
    ths = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    check(res == [ref, texts[0], ref, texts[1]], "并发 4 个请求 (wasm 回放 / 内置交替) 输出都正确")


def test_parallel(): # 2 个慢的自定义采样同时跑: decode 时各槽的 wasm 并行执行, 总耗时应接近单个请求 (串行会接近 2 倍)
    slow = REPLAY.replace("int sample(void* data) {", "int sample(void* data) {\n    for (volatile int i = 0; i < 10000000; i = i + 1) {}")
    body = {"messages": MSG, "enable_thinking": False, "sampler_cpp": slow, "sampler_data": b64(toks(qwen_tok.encode("并行" * 20) + [IM_END]))}
    post(body) # 先编译进缓存
    t0 = time.time()
    post(body)
    t1 = time.time()
    ths = [threading.Thread(target=post, args=(body,)) for _ in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    t2 = time.time()
    check(t2 - t1 < 1.5 * (t1 - t0), f"2 个慢的自定义采样并发: 单个 {t1 - t0:.2f}s, 并发 {t2 - t1:.2f}s (< 1.5 倍)")


def test_long(): # 一直不返回结束 token: 生成到上下文用完, finish_reason 为 length
    t0 = time.time()
    _, r = post({"messages": MSG, "enable_thinking": False, "sampler_cpp": INIT0 + "int sample(void* data) { return 100; }"})
    u = r["usage"]
    check(r["choices"][0]["finish_reason"] == "length", f"不返回结束 token: 生成 {u['completion_tokens']} tokens (prompt {u['prompt_tokens']}, 合计 {u['total_tokens']}) 后 length 结束, {time.time() - t0:.1f}s")


def main():
    global G_base, G_model, G_args
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="请求里的模型名 (经 router 测试时用)")
    p.add_argument("--hf-dir", required=True, help="tokenizer.json 所在目录 (回放测试用 qwen_tok 分词)")
    p.add_argument("--example", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wasm_sampler/examples/top_k.cpp"))
    p.add_argument("--long", action="store_true", help="另测一直生成到上下文用完 (上下文大时很慢)")
    G_args = p.parse_args()
    G_base, G_model = G_args.base.rstrip("/"), G_args.model
    qwen_tok.load(os.path.join(G_args.hf_dir, "tokenizer.json"))
    test_errors()
    test_replay()
    test_greedy()
    test_top_k()
    test_concurrent()
    test_parallel()
    if G_args.long:
        test_long()
    log("✅ 全部通过")


if __name__ == "__main__":
    main()
