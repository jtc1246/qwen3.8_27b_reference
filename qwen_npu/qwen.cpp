// Qwen3.8-27B (qwen35 架构) NPU (Ascend 310P) 推理, 输入 GGUF (Q4_K_M), 通过 extern "C" 给 python ctypes 调用
// matmul 权重重打包成 fractal 量化布局走 cube (k_gemm, fp16 反量化 + mmad), decode 和 prefill 同一 kernel
// alpha/beta (f32) 走 k_gemv_f32t; 其余计算 fp32 向量
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "acl.h"
#include "gguf.h"
#include "quants_ref.h"
#include "repack.h"
#include "aclrtlaunch_k_gemm_i8.h"
#include "aclrtlaunch_k_gemm_i8b.h"
#include "aclrtlaunch_k_gemm_i4.h"
#include "aclrtlaunch_k_pack_x.h"
#include "aclrtlaunch_k_gemv_f32t.h"
#include "aclrtlaunch_k_dq_f32.h"
#include "aclrtlaunch_k_dq_i8.h"
#include "aclrtlaunch_k_dq_i8b.h"
#include "aclrtlaunch_k_dq_i4.h"
#include "aclrtlaunch_k_add_norm.h"
#include "aclrtlaunch_k_silu_mul.h"
#include "aclrtlaunch_k_conv.h"
#include "aclrtlaunch_k_deltanet.h"
#include "aclrtlaunch_k_qk_rope.h"
#include "aclrtlaunch_k_attn.h"

using std::string;
using std::vector;

namespace {

const int N_LAYER = 64, D = 5120, FF = 17408, VOCAB = 248320;
const int N_HEAD = 24, N_KV = 4, HD = 256, ROT = 64;
const int LK = 16, LV = 48, LHD = 128;
const int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM;
const int Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD, ATT_DIM = N_HEAD * HD;
const int CHUNK = 256; // prefill 每批处理的 token 数
const float THETA = 1e7f;
const int PAD = 256; // 每个 matmul 输出缓冲末尾的安全余量 (字节)

struct Weight {
    void* data = nullptr;
    int fmt = 0;
    int k = 0, n = 0;
    int stride = 0;
};

struct Layer {
    bool recr = false;
    void* attn_norm = nullptr;
    void* post_norm = nullptr;
    Weight ffn_gate, ffn_up, ffn_down;
    Weight qkv, gate_z, alpha, beta, ssm_out;
    void* ssm_a = nullptr; // 补零到 64
    void* dt_bias = nullptr;
    void* conv_w = nullptr; // 已重排为 [4][QKV_DIM]
    void* ssm_norm = nullptr;
    void* conv_state = nullptr; // [3][QKV_DIM]
    void* ssm_state = nullptr; // [48][128][128]
    Weight wq, wk, wv, wo;
    void* q_norm = nullptr;
    void* k_norm = nullptr;
    void* k_cache = nullptr; // [max_ctx][1024] f32
    void* v_cache = nullptr;
};

struct SelfRow { string name; uint32_t type; int k, n; int rows[8]; vector<uint8_t> raw; }; // 自测保存的原始行

Gguf::File G_file;
Layer G_layers[N_LAYER];
Weight G_tok_embd, G_output;
void* G_output_norm = nullptr;
int G_max_ctx = 0, G_n_past = 0, G_next_pos = 0;
bool G_inited = false;
uint64_t G_rng = 0x853c49e6748fea9bULL;
void *G_d_x = nullptr, *G_d_xn = nullptr, *G_d_out = nullptr, *G_d_qkv = nullptr, *G_d_z = nullptr, *G_d_a = nullptr, *G_d_b = nullptr, *G_d_o = nullptr;
void *G_d_qg = nullptr, *G_d_k = nullptr, *G_d_v = nullptr, *G_d_att = nullptr, *G_d_g = nullptr, *G_d_u = nullptr;
void *G_d_logits = nullptr, *G_d_hidden = nullptr, *G_d_tok = nullptr, *G_d_cos = nullptr, *G_d_sin = nullptr;
void *G_d_xp = nullptr; // k_pack_x 的 fp16 zZ 布局 x
float G_inv_freq[ROT / 2];
float* G_logits = nullptr;
vector<float> G_hidden;
int G_pos[CHUNK * 3];
vector<float> G_work;
vector<int> G_idx;
vector<uint8_t> G_seen;
vector<SelfRow> G_selfrows;

inline void* fp(void* base, size_t n_floats) { return (uint8_t*)base + n_floats * sizeof(float); }

const Gguf::Tensor& find_tensor(const string& name) {
    auto it = G_file.tensors.find(name);
    if (it == G_file.tensors.end()) {
        fprintf(stderr, "❌ 缺少 tensor %s\n", name.c_str());
        _Exit(1);
    }
    return it->second;
}

Weight get_w(const string& name, int k, int n) { // matmul 权重: fractal 重打包 + 上传 (f32 原样)
    const Gguf::Tensor& t = find_tensor(name);
    if (t.dims.size() != 2 || t.dims[0] != k || t.dims[1] != n) {
        fprintf(stderr, "❌ %s 形状不对, 期望 [%d, %d]\n", name.c_str(), k, n);
        _Exit(1);
    }
    Weight w;
    w.k = k;
    w.n = n;
    if (t.type == Gguf::T_F32) {
        w.fmt = Repack::FMT_F32;
        w.stride = k * 4;
        w.data = Acl::alloc((uint64_t)n * k * 4 + PAD);
        Acl::h2d(w.data, t.data, (uint64_t)n * k * 4);
        return w;
    }
    if (k % 128 || n % 128) {
        fprintf(stderr, "❌ %s 形状 [%d, %d] 不满足 fractal 布局要求\n", name.c_str(), k, n);
        _Exit(1);
    }
    w.fmt = Repack::fmt2_for(t.type);
    int SB = Repack::sb2_ni(w.fmt), KS = k / 128, nr = n / 8;
    w.data = Acl::alloc((uint64_t)(n / 16) * KS * SB);
    vector<uint8_t> staging;
    uint64_t off = 0;
    for (int c = 0; c < 8; c++) { // 每核的行连续存放, 核内按 256 行分块
        for (int r0 = 0; r0 < nr; r0 += 256) {
            int nb = std::min(256, nr - r0);
            uint64_t bb = (uint64_t)KS * (nb / 16) * SB;
            staging.resize(bb);
            Repack::repack2_block(t.type, (const uint8_t*)t.data, k, c * nr + r0, nb, staging.data());
            Acl::h2d((uint8_t*)w.data + off, staging.data(), bb);
            off += bb;
        }
    }
    return w;
}

Weight get_w_embd(const string& name, int k, int n) { // tok_embd: 旧 wave 布局 (只做反量化查行)
    const Gguf::Tensor& t = find_tensor(name);
    if (t.dims.size() != 2 || t.dims[0] != k || t.dims[1] != n) {
        fprintf(stderr, "❌ %s 形状不对, 期望 [%d, %d]\n", name.c_str(), k, n);
        _Exit(1);
    }
    Weight w;
    w.fmt = Repack::fmt_for(t.type);
    w.k = k;
    w.n = n;
    w.stride = (int)Repack::row_stride(w.fmt, k);
    w.data = Acl::alloc((uint64_t)n * w.stride + PAD);
    const int SLICE = 2048; // 每批重打包的行数
    vector<uint8_t> staging((uint64_t)SLICE * w.stride);
    for (int r0 = 0; r0 < n; r0 += SLICE) {
        int nr = std::min(SLICE, n - r0);
        Repack::repack_rows(t.type, (const uint8_t*)t.data, k, r0, nr, staging.data());
        Acl::h2d((uint8_t*)w.data + (uint64_t)r0 * w.stride, staging.data(), (uint64_t)nr * w.stride);
    }
    return w;
}

void* get_f(const string& name, int64_t numel, int64_t alloc_numel = 0) { // f32 原样上传, 可指定更大的分配 (补零)
    const Gguf::Tensor& t = find_tensor(name);
    int64_t cnt = 1;
    for (int64_t d : t.dims) cnt *= d;
    if (t.type != Gguf::T_F32 || cnt != numel) {
        fprintf(stderr, "❌ %s 需要 f32 且元素数 %ld, 实际 type=%u numel=%ld\n", name.c_str(), (long)numel, t.type, (long)cnt);
        _Exit(1);
    }
    if (!alloc_numel) alloc_numel = numel;
    void* p = Acl::alloc((uint64_t)alloc_numel * 4);
    Acl::h2d(p, t.data, (uint64_t)numel * 4);
    return p;
}

void check_kv(const char* key, double expect) {
    auto it = G_file.kv_num.find(key);
    if (it == G_file.kv_num.end() || std::fabs(it->second - expect) > 1e-6 * std::fabs(expect)) {
        fprintf(stderr, "❌ GGUF 参数 %s 不符合预期 %g\n", key, expect);
        _Exit(1);
    }
}

void save_selfrow(const string& name) { // 自测用: 保存首末各 4 行的原始量化数据
    const Gguf::Tensor& t = find_tensor(name);
    SelfRow s;
    s.name = name;
    s.type = t.type;
    s.k = (int)t.dims[0];
    s.n = (int)t.dims[1];
    uint64_t rb = Quants::row_bytes(t.type, s.k);
    int rows[8] = {0, 1, 2, 3, s.n - 4, s.n - 3, s.n - 2, s.n - 1};
    s.raw.resize(rb * 8);
    for (int i = 0; i < 8; i++) {
        s.rows[i] = rows[i];
        memcpy(s.raw.data() + rb * i, (const uint8_t*)t.data + rb * rows[i], rb);
    }
    G_selfrows.push_back(std::move(s));
}

const void* G_pack_src = nullptr; // k_pack_x 结果缓存 (同一 x 的多个 matmul 只打包一次)
int G_pack_t = 0, G_pack_k = 0;
void pack_invalidate() { G_pack_src = nullptr; }

// QWEN_NPU_PROF=1 时逐 kernel 同步计时 (调试用, 会拖慢)
const char* G_prof_names[] = {"gemm", "pack_x", "gemv_f32t", "conv", "deltanet", "qk_rope", "attn", "add_norm", "silu", "misc"};
double G_prof_ms[10];
bool G_prof_on = false;
std::chrono::steady_clock::time_point G_prof_t0;
void prof_mark(int id) { // 把距上个标记的耗时记到 id 上
    if (!G_prof_on) return;
    Acl::sync();
    auto now = std::chrono::steady_clock::now();
    G_prof_ms[id] += std::chrono::duration<double>(now - G_prof_t0).count() * 1e3;
    G_prof_t0 = now;
}

void matmul(const Weight& w, void* x, int T, void* y) { // x: [T][k] f32, y: [T][n] f32
    if (w.fmt == Repack::FMT_F32) { // alpha/beta: 行常驻向量 GEMV, 每核 8 行
        prof_mark(9);
        Acl::ck(ACLRT_LAUNCH_KERNEL(k_gemv_f32t)((uint32_t)(w.n / 8), Acl::G_stream, w.data, x, y, w.n, w.k, T), "launch gemv_f32t");
        prof_mark(2);
        return;
    }
    void* xp = nullptr;
    if (T > 1) {
        if (x != G_pack_src || T != G_pack_t || w.k != G_pack_k) {
            prof_mark(9);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_pack_x)(8, Acl::G_stream, x, G_d_xp, T, w.k), "launch pack_x");
            prof_mark(1);
            G_pack_src = x;
            G_pack_t = T;
            G_pack_k = w.k;
        }
        xp = G_d_xp;
    }
    prof_mark(9);
    aclError e;
    if (w.fmt == Repack::FMT_I8F) e = ACLRT_LAUNCH_KERNEL(k_gemm_i8)(8, Acl::G_stream, w.data, x, xp, y, T, w.n, w.k);
    else if (w.fmt == Repack::FMT_I8BF) e = ACLRT_LAUNCH_KERNEL(k_gemm_i8b)(8, Acl::G_stream, w.data, x, xp, y, T, w.n, w.k);
    else e = ACLRT_LAUNCH_KERNEL(k_gemm_i4)(8, Acl::G_stream, w.data, x, xp, y, T, w.n, w.k);
    Acl::ck(e, "launch gemm");
    prof_mark(0);
}

void matmul_tok(const Weight& w, void* x, void* y) { matmul(w, x, 1, y); }

void dq_rows(const Weight& w, void* d_rows, void* d_out, int nr) { // 反量化 nr 行到自然顺序 f32
    aclError e;
    if (w.fmt == Repack::FMT_F32) e = ACLRT_LAUNCH_KERNEL(k_dq_f32)(8, Acl::G_stream, w.data, d_rows, d_out, nr, w.k, w.stride);
    else if (w.fmt == Repack::FMT_I8) e = ACLRT_LAUNCH_KERNEL(k_dq_i8)(8, Acl::G_stream, w.data, d_rows, d_out, nr, w.k, w.stride);
    else if (w.fmt == Repack::FMT_I8B) e = ACLRT_LAUNCH_KERNEL(k_dq_i8b)(8, Acl::G_stream, w.data, d_rows, d_out, nr, w.k, w.stride);
    else e = ACLRT_LAUNCH_KERNEL(k_dq_i4)(8, Acl::G_stream, w.data, d_rows, d_out, nr, w.k, w.stride);
    Acl::ck(e, "launch dq");
}

void add_norm(void* add, void* w, void* hidden, int T) {
    prof_mark(9);
    Acl::ck(ACLRT_LAUNCH_KERNEL(k_add_norm)(8, Acl::G_stream, G_d_x, add, w, G_d_xn, hidden, T), "launch add_norm");
    prof_mark(7);
    pack_invalidate(); // xn 内容更新
}

void eval_chunk(const int* tokens, int T, const float* embd, const uint8_t* mask, const int* pos3, bool need_logits) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 3; j++) G_pos[t * 3 + j] = pos3 ? pos3[t * 3 + j] : G_next_pos + t;
    float csr[CHUNK * 32], snr[CHUNK * 32];
    for (int t = 0; t < T; t++) { // 与 CUDA 版 k_qk_rope 相同的 MRoPE 分段规则, cos/sin 在 host 算好
        for (int f = 0; f < 32; f++) {
            int sec = (f % 3 == 1 && f < 33) ? 1 : ((f % 3 == 2 && f < 30) ? 2 : 0);
            float ang = (float)G_pos[t * 3 + sec] * G_inv_freq[f];
            csr[t * 32 + f] = cosf(ang);
            snr[t * 32 + f] = sinf(ang);
        }
    }
    Acl::h2d(G_d_tok, tokens, (size_t)T * 4);
    Acl::h2d(G_d_cos, csr, (size_t)T * 128);
    Acl::h2d(G_d_sin, snr, (size_t)T * 128);
    dq_rows(G_tok_embd, G_d_tok, G_d_x, T); // token embedding
    if (mask) { // 图片 embedding 行直接覆盖
        Acl::sync();
        for (int t = 0; t < T; t++)
            if (mask[t]) Acl::h2d(fp(G_d_x, (size_t)t * D), embd + (size_t)t * D, (size_t)D * 4);
    }
    const char* stop_env = getenv("QWEN_NPU_STOP"); // 调试: 跑完第 N 层后停
    int stop_l = stop_env ? atoi(stop_env) : -1;
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        add_norm(l ? G_d_out : nullptr, L.attn_norm, fp(G_d_hidden, (size_t)l * D), T);
        if (L.recr) {
            matmul(L.qkv, G_d_xn, T, G_d_qkv);
            matmul(L.gate_z, G_d_xn, T, G_d_z);
            matmul(L.alpha, G_d_xn, T, G_d_a);
            matmul(L.beta, G_d_xn, T, G_d_b);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_conv)(8, Acl::G_stream, G_d_qkv, L.conv_state, L.conv_w, T), "launch conv");
            prof_mark(3);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_deltanet)(8, Acl::G_stream, G_d_qkv, G_d_z, G_d_a, G_d_b, L.ssm_a, L.dt_bias, L.ssm_norm, L.ssm_state, G_d_o, T), "launch deltanet");
            prof_mark(4);
            pack_invalidate();
            matmul(L.ssm_out, G_d_o, T, G_d_out);
        } else {
            matmul(L.wq, G_d_xn, T, G_d_qg);
            matmul(L.wk, G_d_xn, T, G_d_k);
            matmul(L.wv, G_d_xn, T, G_d_v);
            const char* ph2 = getenv("QWEN_NPU_PHASE");
            if (ph2 && l == stop_l && atoi(ph2) == 10) { Acl::sync(); return; }
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_qk_rope)(8, Acl::G_stream, G_d_qg, G_d_k, G_d_v, L.q_norm, L.k_norm, G_d_cos, G_d_sin, L.k_cache, L.v_cache, G_n_past, T), "launch qk_rope");
            prof_mark(5);
            if (ph2 && l == stop_l && atoi(ph2) == 11) { Acl::sync(); return; }
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_attn)(8, Acl::G_stream, G_d_qg, L.k_cache, L.v_cache, G_d_att, G_n_past, T), "launch attn");
            prof_mark(6);
            if (ph2 && l == stop_l && atoi(ph2) == 12) { Acl::sync(); return; }
            pack_invalidate();
            matmul(L.wo, G_d_att, T, G_d_out);
        }
        add_norm(G_d_out, L.post_norm, nullptr, T);
        matmul(L.ffn_gate, G_d_xn, T, G_d_g);
        const char* ph = getenv("QWEN_NPU_PHASE"); // 调试: 在 FFN 的某一步后直接停
        if (ph && l == stop_l && atoi(ph) == 0) { Acl::sync(); return; }
        matmul(L.ffn_up, G_d_xn, T, G_d_u);
        if (ph && l == stop_l && atoi(ph) == 1) { Acl::sync(); return; }
        Acl::ck(ACLRT_LAUNCH_KERNEL(k_silu_mul)(8, Acl::G_stream, G_d_g, G_d_u, T * FF), "launch silu_mul");
        prof_mark(8);
        if (ph && l == stop_l && atoi(ph) == 2) { Acl::sync(); return; }
        pack_invalidate();
        matmul(L.ffn_down, G_d_g, T, G_d_out);
        if (l == stop_l) {
            Acl::sync();
            return;
        }
    }
    add_norm(G_d_out, G_output_norm, fp(G_d_hidden, (size_t)N_LAYER * D), T);
    int mx = G_next_pos - 1;
    for (int i = 0; i < T * 3; i++) mx = std::max(mx, G_pos[i]);
    G_next_pos = mx + 1;
    G_n_past += T;
    if (!need_logits) return;
    matmul_tok(G_output, fp(G_d_xn, (size_t)(T - 1) * D), G_d_logits);
    Acl::sync();
    Acl::d2h(G_logits, G_d_logits, (size_t)VOCAB * 4);
}

uint64_t rng_next() { // splitmix64
    uint64_t z = (G_rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

} // namespace

extern "C" {

int qwen_init(const char* path, int device, int max_ctx) {
    if (G_inited) {
        fprintf(stderr, "⚠️ 模型已经加载过\n");
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    Acl::init(device);
    if (!Gguf::open(G_file, path)) return -1;
    if (G_file.kv_str["general.architecture"] != "qwen35") {
        fprintf(stderr, "❌ 架构不是 qwen35: %s\n", G_file.kv_str["general.architecture"].c_str());
        return -1;
    }
    check_kv("qwen35.embedding_length", D);
    check_kv("qwen35.feed_forward_length", FF);
    check_kv("qwen35.attention.head_count", N_HEAD);
    check_kv("qwen35.attention.head_count_kv", N_KV);
    check_kv("qwen35.attention.key_length", HD);
    check_kv("qwen35.rope.dimension_count", ROT);
    check_kv("qwen35.rope.freq_base", THETA);
    check_kv("qwen35.ssm.state_size", LHD);
    check_kv("qwen35.ssm.group_count", LK);
    check_kv("qwen35.ssm.time_step_rank", LV);
    check_kv("qwen35.ssm.conv_kernel", 4);
    check_kv("qwen35.full_attention_interval", 4);
    G_max_ctx = max_ctx;
    const char* selfnames[] = {"token_embd.weight", "blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight", "blk.51.ffn_down.weight", "blk.0.ffn_up.weight", "blk.0.ssm_alpha.weight", "output.weight"};
    for (const char* nm : selfnames) save_selfrow(nm);
    auto t1 = std::chrono::steady_clock::now();
    G_tok_embd = get_w_embd("token_embd.weight", D, VOCAB);
    G_output = get_w("output.weight", D, VOCAB);
    G_output_norm = get_f("output_norm.weight", D);
    void* conv_all = Acl::alloc((size_t)N_LAYER * 3 * QKV_DIM * 4);
    void* ssm_all = Acl::alloc((size_t)N_LAYER * LV * LHD * LHD * 4);
    vector<float> cwt(4 * QKV_DIM);
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        string p = "blk." + std::to_string(l) + ".";
        L.recr = (l + 1) % 4 != 0;
        L.attn_norm = get_f(p + "attn_norm.weight", D);
        L.post_norm = get_f(p + "post_attention_norm.weight", D);
        L.ffn_gate = get_w(p + "ffn_gate.weight", D, FF);
        L.ffn_up = get_w(p + "ffn_up.weight", D, FF);
        L.ffn_down = get_w(p + "ffn_down.weight", FF, D);
        if (L.recr) {
            L.qkv = get_w(p + "attn_qkv.weight", D, QKV_DIM);
            L.gate_z = get_w(p + "attn_gate.weight", D, V_DIM);
            L.alpha = get_w(p + "ssm_alpha.weight", D, LV);
            L.beta = get_w(p + "ssm_beta.weight", D, LV);
            L.ssm_out = get_w(p + "ssm_out.weight", V_DIM, D);
            L.ssm_a = get_f(p + "ssm_a", LV, 64);
            L.dt_bias = get_f(p + "ssm_dt.bias", LV, 64);
            L.ssm_norm = get_f(p + "ssm_norm.weight", LHD);
            const Gguf::Tensor& ct = find_tensor(p + "ssm_conv1d.weight");
            int64_t cnt = 1;
            for (int64_t dd : ct.dims) cnt *= dd;
            if (ct.type != Gguf::T_F32 || cnt != (int64_t)QKV_DIM * 4) {
                fprintf(stderr, "❌ %s 形状/类型不对\n", (p + "ssm_conv1d.weight").c_str());
                _Exit(1);
            }
            const float* cw = (const float*)ct.data; // GGUF [通道][4] -> NPU [4][通道]
            for (int c = 0; c < QKV_DIM; c++)
                for (int j = 0; j < 4; j++) cwt[j * QKV_DIM + c] = cw[c * 4 + j];
            L.conv_w = Acl::alloc((size_t)4 * QKV_DIM * 4);
            Acl::h2d(L.conv_w, cwt.data(), (size_t)4 * QKV_DIM * 4);
            L.conv_state = fp(conv_all, (size_t)l * 3 * QKV_DIM);
            L.ssm_state = fp(ssm_all, (size_t)l * LV * LHD * LHD);
            continue;
        }
        L.wq = get_w(p + "attn_q.weight", D, Q_DIM);
        L.wk = get_w(p + "attn_k.weight", D, KV_DIM);
        L.wv = get_w(p + "attn_v.weight", D, KV_DIM);
        L.wo = get_w(p + "attn_output.weight", ATT_DIM, D);
        L.q_norm = get_f(p + "attn_q_norm.weight", HD);
        L.k_norm = get_f(p + "attn_k_norm.weight", HD);
        L.k_cache = Acl::alloc((size_t)max_ctx * KV_DIM * 4);
        L.v_cache = Acl::alloc((size_t)max_ctx * KV_DIM * 4);
    }
    double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "ℹ️ 权重重打包并上传完成, %.1fs\n", up);
    free((void*)G_file.base);
    G_file.base = nullptr;
    for (int i = 0; i < ROT / 2; i++) G_inv_freq[i] = 1.0f / powf(THETA, (float)(2 * i) / ROT);
    G_d_x = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
    G_d_xn = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
    G_d_out = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
    G_d_qkv = Acl::alloc((size_t)CHUNK * QKV_DIM * 4 + PAD);
    G_d_z = Acl::alloc((size_t)CHUNK * V_DIM * 4 + PAD);
    G_d_a = Acl::alloc((size_t)CHUNK * LV * 4 + PAD);
    G_d_b = Acl::alloc((size_t)CHUNK * LV * 4 + PAD);
    G_d_o = Acl::alloc((size_t)CHUNK * V_DIM * 4 + PAD);
    G_d_qg = Acl::alloc((size_t)CHUNK * Q_DIM * 4 + PAD);
    G_d_k = Acl::alloc((size_t)CHUNK * KV_DIM * 4 + PAD);
    G_d_v = Acl::alloc((size_t)CHUNK * KV_DIM * 4 + PAD);
    G_d_att = Acl::alloc((size_t)CHUNK * ATT_DIM * 4 + PAD);
    G_d_g = Acl::alloc((size_t)CHUNK * FF * 4 + PAD);
    G_d_u = Acl::alloc((size_t)CHUNK * FF * 4 + PAD);
    G_d_logits = Acl::alloc((size_t)VOCAB * 4 + PAD);
    G_d_hidden = Acl::alloc((size_t)(N_LAYER + 1) * D * 4);
    G_d_tok = Acl::alloc((size_t)CHUNK * 4 + 64);
    G_d_xp = Acl::alloc((size_t)(FF / 128) * (CHUNK / 16) * 2048 * 2); // 最大 k=FF, m=CHUNK
    G_d_cos = Acl::alloc((size_t)CHUNK * 128);
    G_d_sin = Acl::alloc((size_t)CHUNK * 128);
    Acl::ck(aclrtMallocHost((void**)&G_logits, (size_t)VOCAB * 4), "aclrtMallocHost");
    memset(G_logits, 0, (size_t)VOCAB * 4);
    G_hidden.resize((size_t)(N_LAYER + 1) * D);
    G_work.resize(VOCAB);
    G_idx.resize(VOCAB);
    G_seen.resize(VOCAB);
    G_inited = true;
    size_t free_b = 0, total_b = 0;
    Acl::ck(aclrtGetMemInfo(ACL_HBM_MEM, &free_b, &total_b), "aclrtGetMemInfo");
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "✅ 模型初始化完成, 最大上下文 %d, 显存已用 %.2f / %.2f GB, 耗时 %.1fs\n", max_ctx, (total_b - free_b) / 1073741824.0, total_b / 1073741824.0, sec);
    return 0;
}

int qwen_eval(const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3) {
    if (!G_inited || n <= 0) return -1;
    if (G_n_past + n > G_max_ctx) {
        fprintf(stderr, "❌ 超出最大上下文 %d (已有 %d, 新增 %d)\n", G_max_ctx, G_n_past, n);
        return -2;
    }
    for (int i = 0; i < n; i++) {
        if ((!embd_mask || !embd_mask[i]) && (tokens[i] < 0 || tokens[i] >= VOCAB)) {
            fprintf(stderr, "❌ token id 越界: %d\n", tokens[i]);
            return -3;
        }
    }
    Acl::bind();
    G_prof_on = getenv("QWEN_NPU_PROF") != nullptr;
    if (G_prof_on) {
        Acl::sync();
        G_prof_t0 = std::chrono::steady_clock::now();
    }
    for (int i = 0; i < n; i += CHUNK) {
        int T = std::min(CHUNK, n - i);
        eval_chunk(tokens + i, T, embd ? embd + (size_t)i * D : nullptr, embd_mask ? embd_mask + i : nullptr, pos3 ? pos3 + i * 3 : nullptr, i + T == n);
    }
    return 0;
}

void qwen_prof_dump() { // 打印并清零 QWEN_NPU_PROF 的逐 kernel 累计耗时
    for (int i = 0; i < 10; i++) {
        if (G_prof_ms[i] > 0) fprintf(stderr, "⏱️ %s: %.1f ms\n", G_prof_names[i], G_prof_ms[i]);
        G_prof_ms[i] = 0;
    }
}

int qwen_sample(float temperature, int top_k, float top_p, float min_p, float presence_penalty, const int* hist, int n_hist) {
    float* l = G_work.data();
    memcpy(l, G_logits, VOCAB * sizeof(float));
    if (presence_penalty != 0 && hist) {
        for (int i = 0; i < n_hist; i++) {
            int t = hist[i];
            if (t < 0 || t >= VOCAB || G_seen[t]) continue;
            G_seen[t] = 1;
            l[t] -= presence_penalty;
        }
        for (int i = 0; i < n_hist; i++)
            if (hist[i] >= 0 && hist[i] < VOCAB) G_seen[hist[i]] = 0;
    }
    if (temperature <= 0) return (int)(std::max_element(l, l + VOCAB) - l);
    int* idx = G_idx.data();
    for (int i = 0; i < VOCAB; i++) idx[i] = i;
    int n = (top_k > 0 && top_k < VOCAB) ? top_k : VOCAB;
    auto cmp = [&](int a, int b) { return l[a] > l[b]; };
    if (n < VOCAB) std::nth_element(idx, idx + n, idx + VOCAB, cmp);
    std::sort(idx, idx + n, cmp);
    vector<double> p(n);
    double sum = 0;
    for (int i = 0; i < n; i++) {
        p[i] = std::exp((double)(l[idx[i]] - l[idx[0]]) / temperature);
        sum += p[i];
    }
    int keep = n;
    double cum = 0;
    for (int i = 0; i < n; i++) {
        if (min_p > 0 && p[i] / sum < min_p * p[0] / sum) {
            keep = std::max(i, 1);
            break;
        }
        cum += p[i] / sum;
        if (top_p < 1.0f && cum >= top_p) {
            keep = i + 1;
            break;
        }
    }
    double tot = 0;
    for (int i = 0; i < keep; i++) tot += p[i];
    double r = (rng_next() >> 11) * (1.0 / 9007199254740992.0) * tot;
    for (int i = 0; i < keep; i++) {
        r -= p[i];
        if (r <= 0) return idx[i];
    }
    return idx[keep - 1];
}

void qwen_seed(uint64_t seed) { G_rng = seed; }

void qwen_reset() {
    Acl::bind();
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        if (!L.recr) continue;
        Acl::ck(aclrtMemset(L.conv_state, (size_t)3 * QKV_DIM * 4, 0, (size_t)3 * QKV_DIM * 4), "memset conv");
        Acl::ck(aclrtMemset(L.ssm_state, (size_t)LV * LHD * LHD * 4, 0, (size_t)LV * LHD * LHD * 4), "memset ssm");
    }
    G_n_past = 0;
    G_next_pos = 0;
}

int qwen_n_past() { return G_n_past; }
float* qwen_logits() { return G_logits; }

float* qwen_hidden() {
    Acl::bind();
    Acl::sync();
    Acl::d2h(G_hidden.data(), G_d_hidden, G_hidden.size() * 4);
    return G_hidden.data();
}

int qwen_selftest() { // tok_embd: 反量化逐位对照; matmul 权重: m=1 GEMM vs 保存行的 f32 反量化点积 (fp16 计算, 阈值 5e-3)
    Acl::bind();
    int fails = 0;
    struct Item { const char* name; Weight* w; };
    Item items[] = {{"token_embd.weight", &G_tok_embd}, {"blk.0.attn_qkv.weight", &G_layers[0].qkv},
                    {"blk.0.ffn_gate.weight", &G_layers[0].ffn_gate}, {"blk.51.ffn_down.weight", &G_layers[51].ffn_down},
                    {"blk.0.ffn_up.weight", &G_layers[0].ffn_up}, {"blk.0.ssm_alpha.weight", &G_layers[0].alpha},
                    {"output.weight", &G_output}};
    for (SelfRow& s : G_selfrows) {
        Weight* w = nullptr;
        for (Item& it : items)
            if (s.name == it.name) w = it.w;
        if (!w) continue;
        int k = s.k;
        uint64_t rb = Quants::row_bytes(s.type, k);
        vector<float> ref(k);
        if (s.name == "token_embd.weight") {
            Acl::h2d(G_d_tok, s.rows, 8 * 4);
            dq_rows(*w, G_d_tok, G_d_out, 8);
            Acl::sync();
            vector<float> got((size_t)8 * k);
            Acl::d2h(got.data(), G_d_out, got.size() * 4);
            double max_diff = 0, max_abs = 0;
            for (int r = 0; r < 8; r++) {
                Quants::Ref::dequant_row(s.type, s.raw.data() + rb * r, ref.data(), k);
                for (int i = 0; i < k; i++) {
                    max_diff = std::max(max_diff, (double)std::fabs(got[(size_t)r * k + i] - ref[i]));
                    max_abs = std::max(max_abs, (double)std::fabs(ref[i]));
                }
            }
            bool ok = max_diff <= 1e-6 * max_abs;
            fails += !ok;
            fprintf(stderr, "%s %s fmt=%d [%d, %d] 反量化最大差 %.3g (最大绝对值 %.3g)\n", ok ? "✅" : "❌", s.name.c_str(), w->fmt, k, w->n, max_diff, max_abs);
            continue;
        }
        vector<float> x(k), y((size_t)w->n);
        for (auto& v : x) v = (float)((rng_next() >> 11) * (1.0 / 9007199254740992.0) * 2 - 1);
        Acl::h2d(G_d_g, x.data(), (size_t)k * 4);
        matmul_tok(*w, G_d_g, G_d_u);
        Acl::sync();
        Acl::d2h(y.data(), G_d_u, (size_t)w->n * 4);
        double max_rel = 0;
        for (int r = 0; r < 8; r++) {
            Quants::Ref::dequant_row(s.type, s.raw.data() + rb * r, ref.data(), k);
            double refd = 0, mag = 0;
            for (int j = 0; j < k; j++) {
                refd += (double)ref[j] * x[j];
                mag += std::fabs((double)ref[j] * x[j]);
            }
            double rel = std::fabs(refd - y[s.rows[r]]) / (mag + 1e-30);
            if (!(rel == rel)) rel = 1e30; // NaN 直接视为失败
            max_rel = std::max(max_rel, rel);
        }
        bool ok = max_rel < 5e-3;
        fails += !ok;
        fprintf(stderr, "%s %s fmt=%d [%d, %d] GEMM 最大相对误差 %.3g\n", ok ? "✅" : "❌", s.name.c_str(), w->fmt, k, w->n, max_rel);
    }
    return fails;
}

int qwen_dbg(const char* name, float* out, long long n) { // 调试: 读取中间激活缓冲
    void* p = nullptr;
    string s = name;
    if (s == "x") p = G_d_x;
    else if (s == "xn") p = G_d_xn;
    else if (s == "out") p = G_d_out;
    else if (s == "qkv") p = G_d_qkv;
    else if (s == "z") p = G_d_z;
    else if (s == "a") p = G_d_a;
    else if (s == "b") p = G_d_b;
    else if (s == "o") p = G_d_o;
    else if (s == "qg") p = G_d_qg;
    else if (s == "k") p = G_d_k;
    else if (s == "v") p = G_d_v;
    else if (s == "att") p = G_d_att;
    else if (s == "g") p = G_d_g;
    else if (s == "u") p = G_d_u;
    else return -1;
    Acl::bind();
    Acl::sync();
    Acl::d2h(out, p, (size_t)n * 4);
    return 0;
}

} // extern "C"
