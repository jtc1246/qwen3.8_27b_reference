// Qwen3.8-27B (qwen35 架构) 纯 CPU 推理, 输入 GGUF (Q4_K_M), 通过 extern "C" 给 python ctypes 调用
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include "gguf.h"
#include "ops.h"
#include "pool.h"
#include "quants.h"
#include "cache.h"

using std::string;
using std::vector;

namespace {

const int N_LAYER = 64, D = 5120, FF = 17408, VOCAB = 248320;
const int N_HEAD = 24, N_KV = 4, HD = 256, ROT = 64; // full attention: q 头 24, kv 头 4, 每头 256 维, 前 64 维做 RoPE
const int LK = 16, LV = 48, LHD = 128; // DeltaNet: qk 头 16, v 头 48, 每头 128 维
const int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM; // 2048, 6144, 10240
const int Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD, ATT_DIM = N_HEAD * HD; // q+gate 12288, 1024, 6144
const int CHUNK = 64; // prefill 每批处理的 token 数
const float EPS = 1e-6f, THETA = 1e7f;

struct Layer {
    bool recr = false; // true: Gated DeltaNet, false: full attention
    const float* attn_norm = nullptr;
    const float* post_norm = nullptr;
    Ops::Weight ffn_gate, ffn_up, ffn_down;
    Ops::Weight qkv, gate_z, alpha, beta, ssm_out; // DeltaNet
    const float* ssm_a = nullptr; // -exp(A_log), 转换时已算好
    const float* dt_bias = nullptr;
    const float* conv_w = nullptr; // [10240][4]
    const float* ssm_norm = nullptr; // [128], 不含 +1
    float* conv_state = nullptr; // [10240][3], 最近 3 个输入, 旧的在前
    float* ssm_state = nullptr; // [48][128 k][128 v]
    Ops::Weight wq, wk, wv, wo; // full attention
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    float* k_cache = nullptr; // [max_ctx][1024]
    float* v_cache = nullptr;
};

Gguf::File G_file;
Layer G_layers[N_LAYER];
Ops::Weight G_tok_embd, G_output;
const float* G_output_norm = nullptr;
int G_max_ctx = 0, G_n_past = 0, G_next_pos = 0, G_n_threads = 1;
bool G_inited = false;
float G_inv_freq[ROT / 2];
uint64_t G_rng = 0x853c49e6748fea9bULL;
float *G_logits, *G_hidden, *G_x, *G_xn, *G_out, *G_qkv, *G_z, *G_a, *G_b, *G_o, *G_qg, *G_k, *G_v, *G_att, *G_g, *G_u;
int* G_pos; // [CHUNK][3]
float* G_scores[Ops::MAX_THREADS];
vector<float> G_work;
vector<int> G_idx;
vector<uint8_t> G_seen;

float* alloc_f(size_t n) { // 直接分配并写零, 物理内存在初始化时就全部占用, 不释放
    float* p = (float*)malloc(n * sizeof(float));
    if (!p) {
        fprintf(stderr, "❌ 内存分配失败 %zu floats\n", n);
        _Exit(1);
    }
    memset(p, 0, n * sizeof(float));
    for (size_t i = 0; i < n * sizeof(float); i += 4096) ((volatile char*)p)[i] = 0; // gcc 会把 malloc+memset(0) 合并成 calloc (延迟分配), 用 volatile 逐页写强制占用
    return p;
}

const Gguf::Tensor& find_tensor(const string& name) {
    auto it = G_file.tensors.find(name);
    if (it == G_file.tensors.end()) {
        fprintf(stderr, "❌ 缺少 tensor %s\n", name.c_str());
        _Exit(1);
    }
    return it->second;
}

Ops::Weight get_w(const string& name, int k, int n) {
    const Gguf::Tensor& t = find_tensor(name);
    if (t.dims.size() != 2 || t.dims[0] != k || t.dims[1] != n) {
        fprintf(stderr, "❌ %s 形状不对, 期望 [%d, %d]\n", name.c_str(), k, n);
        _Exit(1);
    }
    Ops::Weight w;
    w.data = t.data;
    w.type = t.type;
    w.k = k;
    w.n = n;
    w.rb = Quants::row_bytes(t.type, k);
    return w;
}

const float* get_f(const string& name, int64_t numel) {
    const Gguf::Tensor& t = find_tensor(name);
    int64_t cnt = 1;
    for (int64_t d : t.dims) cnt *= d;
    if (t.type != Gguf::T_F32 || cnt != numel) {
        fprintf(stderr, "❌ %s 需要 f32 且元素数 %ld, 实际 type=%u numel=%ld\n", name.c_str(), (long)numel, t.type, (long)cnt);
        _Exit(1);
    }
    return (const float*)t.data;
}

void check_kv(const char* key, double expect) {
    auto it = G_file.kv_num.find(key);
    if (it == G_file.kv_num.end() || std::fabs(it->second - expect) > 1e-6 * std::fabs(expect)) {
        fprintf(stderr, "❌ GGUF 参数 %s 不符合预期 %g\n", key, expect);
        _Exit(1);
    }
}

void rope(float* x, const float* cs) { // cs: [32][cos, sin], 对前 64 维做 rotate_half 形式的旋转, 第 i 维与第 i+32 维配对
    for (int i = 0; i < ROT / 2; i++) {
        float c = cs[2 * i], s = cs[2 * i + 1];
        float x0 = x[i], x1 = x[i + ROT / 2];
        x[i] = x0 * c - x1 * s;
        x[i + ROT / 2] = x1 * c + x0 * s;
    }
}

void rope_cs(const int* p3, float* cs) { // 交错 MRoPE: 频率 i%3==1 且 i<33 用 h 位置, i%3==2 且 i<30 用 w 位置, 其余用 t 位置
    for (int i = 0; i < ROT / 2; i++) {
        int sec = (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0);
        float ang = (float)p3[sec] * G_inv_freq[i];
        cs[2 * i] = cosf(ang);
        cs[2 * i + 1] = sinf(ang);
    }
}

void linear_attn(Layer& L, int T) {
    Ops::matmul(L.qkv, G_xn, T, G_qkv);
    Ops::matmul(L.gate_z, G_xn, T, G_z);
    Ops::matmul(L.alpha, G_xn, T, G_a);
    Ops::matmul(L.beta, G_xn, T, G_b);
    Pool::run(QKV_DIM / 256, [&](int item, int) { // 因果 depthwise conv1d (k=4) + SiLU, 按通道并行
        for (int c = item * 256; c < item * 256 + 256; c++) {
            float* st = L.conv_state + c * 3;
            const float* w = L.conv_w + c * 4;
            for (int t = 0; t < T; t++) {
                float xv = G_qkv[(size_t)t * QKV_DIM + c];
                float o = w[0] * st[0] + w[1] * st[1] + w[2] * st[2] + w[3] * xv;
                st[0] = st[1];
                st[1] = st[2];
                st[2] = xv;
                G_qkv[(size_t)t * QKV_DIM + c] = Ops::silu(o);
            }
        }
    });
    const float qscale = 1.0f / sqrtf((float)LHD);
    for (int t = 0; t < T; t++) {
        float* row = G_qkv + (size_t)t * QKV_DIM;
        for (int h = 0; h < 2 * LK; h++) { // 前 16 个是 q 头, 后 16 个是 k 头, 都做 L2 归一化
            float* p = row + h * LHD;
            float r = 1.0f / sqrtf(Ops::dot(p, p, LHD) + 1e-6f);
            if (h < LK) r *= qscale;
            for (int i = 0; i < LHD; i++) p[i] *= r;
        }
        for (int h = 0; h < LV; h++) {
            G_b[t * LV + h] = Ops::sigmoid(G_b[t * LV + h]); // beta
            G_a[t * LV + h] = L.ssm_a[h] * Ops::softplus(G_a[t * LV + h] + L.dt_bias[h]); // g (log 空间衰减)
        }
    }
    Pool::run(LV, [&](int h, int) { // 每个 v 头独立递推; GGUF 已把 v 头转成 tiled 顺序, v 头 h 对应 qk 头 h % 16
        int kh = h % LK;
        float* S = L.ssm_state + (size_t)h * LHD * LHD;
        for (int t = 0; t < T; t++) {
            const float* row = G_qkv + (size_t)t * QKV_DIM;
            const float* q = row + kh * LHD;
            const float* k = row + KEY_DIM + kh * LHD;
            const float* v = row + 2 * KEY_DIM + h * LHD;
            __m512 decay = _mm512_set1_ps(expf(G_a[t * LV + h]));
            __m512 acc[8];
            for (int c = 0; c < 8; c++) acc[c] = _mm512_setzero_ps();
            for (int i = 0; i < LHD; i++) { // S *= exp(g); kv_mem = S^T k
                float* sr = S + i * LHD;
                __m512 ki = _mm512_set1_ps(k[i]);
                for (int c = 0; c < 8; c++) {
                    __m512 s = _mm512_mul_ps(_mm512_loadu_ps(sr + 16 * c), decay);
                    _mm512_storeu_ps(sr + 16 * c, s);
                    acc[c] = _mm512_fmadd_ps(ki, s, acc[c]);
                }
            }
            __m512 bt = _mm512_set1_ps(G_b[t * LV + h]);
            for (int c = 0; c < 8; c++) acc[c] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v + 16 * c), acc[c]), bt); // delta
            __m512 o[8];
            for (int c = 0; c < 8; c++) o[c] = _mm512_setzero_ps();
            for (int i = 0; i < LHD; i++) { // S += k delta^T; o = S^T q
                float* sr = S + i * LHD;
                __m512 ki = _mm512_set1_ps(k[i]), qi = _mm512_set1_ps(q[i]);
                for (int c = 0; c < 8; c++) {
                    __m512 s = _mm512_fmadd_ps(ki, acc[c], _mm512_loadu_ps(sr + 16 * c));
                    _mm512_storeu_ps(sr + 16 * c, s);
                    o[c] = _mm512_fmadd_ps(qi, s, o[c]);
                }
            }
            float* dst = G_o + (size_t)t * V_DIM + h * LHD;
            for (int c = 0; c < 8; c++) _mm512_storeu_ps(dst + 16 * c, o[c]);
            const float* z = G_z + (size_t)t * V_DIM + h * LHD; // gated RMSNorm: norm(o) * w * silu(z)
            float r = 1.0f / sqrtf(Ops::dot(dst, dst, LHD) / LHD + EPS);
            for (int i = 0; i < LHD; i++) dst[i] = dst[i] * r * L.ssm_norm[i] * Ops::silu(z[i]);
        }
    });
    Ops::matmul(L.ssm_out, G_o, T, G_out);
}

void full_attn(Layer& L, int T) {
    Ops::matmul(L.wq, G_xn, T, G_qg);
    Ops::matmul(L.wk, G_xn, T, G_k);
    Ops::matmul(L.wv, G_xn, T, G_v);
    for (int t = 0; t < T; t++) {
        float cs[ROT];
        rope_cs(G_pos + t * 3, cs);
        for (int h = 0; h < N_HEAD; h++) { // q_proj 输出每个头是 [q(256), gate(256)] 交错
            float* q = G_qg + (size_t)t * Q_DIM + h * HD * 2;
            Ops::rmsnorm(q, L.q_norm, q, HD, EPS);
            rope(q, cs);
        }
        for (int h = 0; h < N_KV; h++) {
            float* k = G_k + (size_t)t * KV_DIM + h * HD;
            Ops::rmsnorm(k, L.k_norm, k, HD, EPS);
            rope(k, cs);
        }
        int s = G_n_past + t;
        memcpy(L.k_cache + (size_t)s * KV_DIM, G_k + (size_t)t * KV_DIM, KV_DIM * sizeof(float));
        memcpy(L.v_cache + (size_t)s * KV_DIM, G_v + (size_t)t * KV_DIM, KV_DIM * sizeof(float));
    }
    const float scale = 1.0f / sqrtf((float)HD);
    Pool::run(T * N_HEAD, [&](int item, int tid) {
        int t = item / N_HEAD, h = item % N_HEAD, kvh = h / (N_HEAD / N_KV); // GQA: 连续 6 个 q 头共用一个 kv 头
        int n = G_n_past + t + 1; // 因果: 只看到自己及之前
        const float* q = G_qg + (size_t)t * Q_DIM + h * HD * 2;
        const float* gate = q + HD;
        float* sc = G_scores[tid];
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) {
            sc[j] = Ops::dot(q, L.k_cache + (size_t)j * KV_DIM + kvh * HD, HD) * scale;
            mx = std::max(mx, sc[j]);
        }
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sc[j] = expf(sc[j] - mx);
            sum += sc[j];
        }
        __m512 acc[HD / 16];
        for (int c = 0; c < HD / 16; c++) acc[c] = _mm512_setzero_ps();
        for (int j = 0; j < n; j++) {
            __m512 p = _mm512_set1_ps(sc[j] / sum);
            const float* vr = L.v_cache + (size_t)j * KV_DIM + kvh * HD;
            for (int c = 0; c < HD / 16; c++) acc[c] = _mm512_fmadd_ps(p, _mm512_loadu_ps(vr + 16 * c), acc[c]);
        }
        float* dst = G_att + (size_t)t * ATT_DIM + h * HD;
        for (int c = 0; c < HD / 16; c++) _mm512_storeu_ps(dst + 16 * c, acc[c]);
        for (int i = 0; i < HD; i++) dst[i] *= Ops::sigmoid(gate[i]); // 输出门控
    });
    Ops::matmul(L.wo, G_att, T, G_out);
}

void eval_chunk(const int* tokens, int T, const float* embd, const uint8_t* mask, const int* pos3, bool need_logits) { // embd: 本 chunk 第一个图片行 (只含 mask 为 1 的行)
    for (int t = 0, j = 0; t < T; t++) {
        float* x = G_x + (size_t)t * D;
        if (mask && mask[t])
            memcpy(x, embd + (size_t)j++ * D, D * sizeof(float));
        else
            Quants::dequant_row(G_tok_embd.type, G_tok_embd.data + (size_t)tokens[t] * G_tok_embd.rb, x, D);
        for (int j = 0; j < 3; j++) {
            int p = pos3 ? pos3[t * 3 + j] : G_next_pos + t;
            G_pos[t * 3 + j] = p;
        }
    }
    memcpy(G_hidden, G_x + (size_t)(T - 1) * D, D * sizeof(float));
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        for (int t = 0; t < T; t++) Ops::rmsnorm(G_x + (size_t)t * D, L.attn_norm, G_xn + (size_t)t * D, D, EPS);
        if (L.recr)
            linear_attn(L, T);
        else
            full_attn(L, T);
        for (size_t i = 0; i < (size_t)T * D; i++) G_x[i] += G_out[i];
        for (int t = 0; t < T; t++) Ops::rmsnorm(G_x + (size_t)t * D, L.post_norm, G_xn + (size_t)t * D, D, EPS);
        Ops::matmul(L.ffn_gate, G_xn, T, G_g);
        Ops::matmul(L.ffn_up, G_xn, T, G_u);
        for (size_t i = 0; i < (size_t)T * FF; i++) G_g[i] = Ops::silu(G_g[i]) * G_u[i];
        Ops::matmul(L.ffn_down, G_g, T, G_out);
        for (size_t i = 0; i < (size_t)T * D; i++) G_x[i] += G_out[i];
        memcpy(G_hidden + (size_t)(l + 1) * D, G_x + (size_t)(T - 1) * D, D * sizeof(float));
    }
    int mx = G_next_pos - 1;
    for (int i = 0; i < T * 3; i++) mx = std::max(mx, G_pos[i]);
    G_next_pos = mx + 1; // 后续默认位置接在最大位置之后, 与 HF get_rope_index 一致
    G_n_past += T;
    if (!need_logits) return;
    Ops::rmsnorm(G_x + (size_t)(T - 1) * D, G_output_norm, G_xn, D, EPS);
    Ops::matmul(G_output, G_xn, 1, G_logits);
}

void ref_dequant(uint32_t type, const uint8_t* src, float* y, int k) { // 自测用的标量参考
    switch (type) {
        case Gguf::T_F32: memcpy(y, src, k * 4); return;
        case Gguf::T_Q8_0: Quants::Ref::q8_0(src, y, k); return;
        case Gguf::T_Q4_K: Quants::Ref::q4_k(src, y, k); return;
        case Gguf::T_Q5_K: Quants::Ref::q5_k(src, y, k); return;
        case Gguf::T_Q6_K: Quants::Ref::q6_k(src, y, k); return;
    }
}

uint64_t rng_next() { // splitmix64
    uint64_t z = (G_rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ---------------- 前缀缓存 (文件格式和线程见 cache.h) ----------------
// 与 cuda 版不同: 每个缓存点都写 kvcache 和 state (没有 2048/512 的规则), prefill 在每个要写的缓存点处切开; KV 是 fp32
// 调用都在 python 线程 (单序列): qwen_prefill_cached 查找加载、prefill、等 prompt 的缓存写完; decode 时 qwen_eval 每 FLUSH 个 token 交出一段 KV; qwen_cache_finish 结束

const int IM_END = 248046;
const int N_REC = 48; // DeltaNet 层数
const size_t SS_FLOATS = (size_t)LV * LHD * LHD, CS_FLOATS = (size_t)QKV_DIM * 3; // 每层 DeltaNet 状态 / conv 状态
const size_t STATE_FLOATS = N_REC * (SS_FLOATS + CS_FLOATS); // 写盘时拼成 [48 层 DeltaNet][48 层 conv], 与 cuda 的槽状态布局相同
const size_t KV_PARTS[2] = {KV_DIM * 4, KV_DIM * 4}; // 每 token 每层: K fp32, V fp32
static_assert(N_LAYER / 4 == Cache::N_ATT, "全注意力层数与 cache.h 不一致");
char G_model_id[600] = {}; // GGUF 文件名 + 字节数 + 后端格式, 写入缓存 metadata (与 cuda 的不同, 两者的缓存点互相视为无效)
Cache::Plan* G_plan = nullptr; // qwen_cache_init 之后才有
std::atomic<int> G_cpend{0}, G_cpf{0}; // 未完成的写入数 / 其中 prompt 的
std::atomic<uint32_t> G_cver{0}; // 写盘线程完成一项时 +1 并唤醒
char* G_mbuf = nullptr; // 读 metadata 的缓冲
int* G_segbuf = nullptr;
uint8_t* G_lbuf = nullptr; // 读缓存文件的缓冲 (按需扩大)
size_t G_lcap = 0;
float* G_sbuf = nullptr; // 压缩 state 解码后的 [STATE_FLOATS]

void state_get(float* dst) { // 各层状态拼成 [48 层 DeltaNet][48 层 conv]
    for (int l = 0, r = 0; l < N_LAYER; l++) {
        if (!G_layers[l].recr) continue;
        memcpy(dst + r * SS_FLOATS, G_layers[l].ssm_state, SS_FLOATS * 4);
        memcpy(dst + N_REC * SS_FLOATS + r * CS_FLOATS, G_layers[l].conv_state, CS_FLOATS * 4);
        r++;
    }
}

void state_set(const float* src) {
    for (int l = 0, r = 0; l < N_LAYER; l++) {
        if (!G_layers[l].recr) continue;
        memcpy(G_layers[l].ssm_state, src + r * SS_FLOATS, SS_FLOATS * 4);
        memcpy(G_layers[l].conv_state, src + N_REC * SS_FLOATS + r * CS_FLOATS, CS_FLOATS * 4);
        r++;
    }
}

void kv_get(int a, int b, uint8_t* dst) { // KV 行 [a, b) 拷到 dst (布局见 Cache::gather_seg)
    size_t n = b - a;
    for (int l = 3; l < N_LAYER; l += 4) { // 全注意力层
        memcpy(dst, G_layers[l].k_cache + (size_t)a * KV_DIM, n * KV_DIM * 4);
        memcpy(dst + n * KV_DIM * 4, G_layers[l].v_cache + (size_t)a * KV_DIM, n * KV_DIM * 4);
        dst += n * KV_DIM * 8;
    }
}

void kv_set(int a, int n, const uint8_t* src) { // kvcache.bin 的一段 (n 个 token) 写回 KV 行 [a, a + n)
    for (int l = 3; l < N_LAYER; l += 4) {
        memcpy(G_layers[l].k_cache + (size_t)a * KV_DIM, src, (size_t)n * KV_DIM * 4);
        memcpy(G_layers[l].v_cache + (size_t)a * KV_DIM, src + (size_t)n * KV_DIM * 4, (size_t)n * KV_DIM * 4);
        src += (size_t)n * KV_DIM * 8;
    }
}

uint8_t* lbuf(size_t n) {
    if (n > G_lcap) {
        Sys::unmap(G_lbuf, G_lcap);
        G_lcap = std::max(n, (size_t)256 << 20);
        G_lbuf = (uint8_t*)Sys::amap(G_lcap);
    }
    return G_lbuf;
}

int64_t read_file(const char* path) { // 整个文件 (direct IO) 读进 G_lbuf, 返回字节数, 失败 -1
    int fd = Sys::open(path, Sys::O_RD | Sys::O_DIRECT_);
    if (fd < 0) return -1;
    int64_t sz = Sys::fsize(fd);
    bool ok = sz > 0 && sz % Cache::AL == 0 && Sys::read_all(fd, lbuf(sz), sz, 0);
    Sys::close(fd);
    return ok ? sz : -1;
}

bool load_state(const char* dir, size_t& bytes) { // 先看 state.bin.compressed (多线程解码), 再看 state.bin; 量化线程可能刚好在替换, 最后再试一次压缩版
    char p[2400];
    for (int k = 0; k < 3; k++) {
        bool comp = k != 1;
        snprintf(p, sizeof(p), comp ? "%s/state.bin.compressed" : "%s/state.bin", dir);
        int64_t sz = read_file(p);
        if (sz < 0) continue;
        bytes += sz;
        if (!comp) {
            if (sz != (int64_t)(STATE_FLOATS * 4)) return false;
            state_set((const float*)G_lbuf);
            return true;
        }
        uint64_t hdr[2];
        memcpy(hdr, G_lbuf + 8, 16);
        if (memcmp(G_lbuf, "QSC1", 4) || hdr[0] != STATE_FLOATS || (int64_t)Codec::comp_bytes(hdr[0], hdr[1]) != sz) return false;
        const uint8_t* c = G_lbuf;
        size_t n = hdr[0], nb = n / Codec::QK, step = 4096;
        Pool::run((int)((nb + step - 1) / step), [&](int item, int) { // 与 Codec::decode_cpu 相同, 按块分给线程池
            const float* amax = (const float*)(c + Codec::off_amax());
            const uint8_t* idx = c + Codec::off_idx(n);
            const int8_t* code = (const int8_t*)(c + Codec::off_code(n));
            for (size_t i = item * step; i < std::min(nb, (item + 1) * step); i++) Codec::decode_block(amax[i], Codec::oct(idx[i]) / 127.0f, code + i * Codec::QK, G_sbuf + i * Codec::QK);
        });
        const uint32_t* zi = (const uint32_t*)(c + Codec::off_zero(n));
        for (size_t i = 0; i < hdr[1]; i++) G_sbuf[zi[i]] = 0;
        state_set(G_sbuf);
        return true;
    }
    return false;
}

Cache::Item* new_item(int kind) { // 新建一项, 未完成写入数 +1
    Cache::Item* it = (Cache::Item*)Sys::amap(sizeof(Cache::Item)); // 已清零
    it->kind = kind, it->slot = 0, it->P = G_plan;
    it->pend = &G_cpend, it->ver = &G_cver;
    G_cpend.fetch_add(1);
    return it;
}

void wait_zero(std::atomic<int>& c) { // 等写盘线程把计数清零
    while (true) {
        uint32_t v = G_cver.load();
        if (!c.load()) return;
        Sys::wait(&G_cver, v);
    }
}

void qwen_reset_impl();

int cache_load(int n) { // 状态已清空: 查找 + 加载命中的缓存点 (从后往前找第一个自身有 state、之前所有点都有效的; 不假设 state 一定存在), 决定之后哪些缓存点写入; 返回命中位置 (token 数), 未命中 0
    Cache::Plan& P = *G_plan;
    auto t0 = std::chrono::steady_clock::now();
    char dir[2300], p[2400];
    int bad = P.n_pts; // 第一个无效的缓存点, 命中只能在它之前
    for (int i = 0; i < P.n_pts; i++) {
        Cache::key_dir(P.key[i], dir);
        Cache::Meta m;
        P.exist[i] = Cache::meta_read(dir, m, G_segbuf, G_mbuf) && m.start == (i ? P.pos[i - 1] : 0) && m.end == P.pos[i];
        if (!P.exist[i]) {
            bad = std::min(bad, i);
            P.st[i] = 0;
            continue;
        }
        P.npos[i] = m.next_pos;
        snprintf(p, sizeof(p), "%s/state.bin.compressed", dir);
        P.st[i] = !Sys::access(p);
        snprintf(p, sizeof(p), "%s/state.bin", dir);
        P.st[i] |= !Sys::access(p); // 这里 st 暂时表示 "磁盘上有 state"
    }
    int hit = -1;
    for (int i = bad - 1; i >= 0 && hit < 0; i--) // 最后一个 token 必须重算 (要它的 logits)
        if (P.st[i] && P.pos[i] < n) hit = i;
    size_t bytes = 0;
    for (int j = 0; j <= hit; j++) { // 依次读各缓存点的 kvcache.bin
        Cache::key_dir(P.key[j], dir);
        Cache::Meta m;
        snprintf(p, sizeof(p), "%s/kvcache.bin", dir);
        int64_t sz = Cache::meta_read(dir, m, G_segbuf, G_mbuf) ? read_file(p) : -1;
        size_t need = 0;
        for (int k = 0; k < m.n_segs; k++) need += Cache::seg_bytes(G_segbuf[k]);
        if (sz < 0 || (size_t)sz != need) {
            fprintf(stderr, "⚠️ 缓存点 %s 读取失败, 不使用缓存\n", dir);
            hit = -1;
            break;
        }
        bytes += sz;
        for (int k = 0, a = m.start, off = 0; k < m.n_segs; k++) {
            kv_set(a, G_segbuf[k], G_lbuf + off);
            a += G_segbuf[k];
            off += Cache::seg_bytes(G_segbuf[k]);
        }
    }
    if (hit >= 0) {
        Cache::key_dir(P.key[hit], dir);
        if (!load_state(dir, bytes)) {
            fprintf(stderr, "⚠️ 缓存点 %s 的 state 读取失败, 不使用缓存\n", dir);
            hit = -1;
        }
    }
    if (hit < 0) {
        qwen_reset_impl(); // 可能已经写了一部分
        bytes = 0;
    } else {
        for (int j = 0; j <= hit; j++) { // 命中: 更新所用缓存点 metadata 的 mtime (清理按它判断)
            Cache::key_dir(P.key[j], dir);
            snprintf(p, sizeof(p), "%s/metadata.txt", dir);
            Sys::touch(p);
        }
        G_n_past = P.pos[hit];
        G_next_pos = P.npos[hit];
    }
    for (int i = 0; i < P.n_pts; i++) P.st[i] = i > hit && !P.exist[i]; // 命中之后的缓存点全部写 (kvcache + state), 已有效存在的不重写
    P.hit = hit, P.pf_next = hit + 1;
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (hit >= 0) fprintf(stderr, "ℹ️ 前缀缓存命中 %d / %d tokens (第 %d / %d 个缓存点), 读取 %.1fMB, %.3fs\n", G_n_past, n, hit + 1, P.n_pts, bytes / 1048576.0, sec);
    else fprintf(stderr, "ℹ️ 前缀缓存未命中 (%d 个缓存点), %.3fs\n", P.n_pts, sec);
    return G_n_past;
}

void pf_flush(int p1, bool state) { // 缓存点 [pf_next, p1) 的 KV (以及 p1 - 1 的 state) 拷出, 交给写盘线程
    Cache::Plan& P = *G_plan;
    int p0 = P.pf_next;
    bool any = state;
    for (int j = p0; j < p1; j++) any |= !P.exist[j];
    P.pf_next = std::max(p0, p1);
    if (p1 <= p0 || !any) return;
    Cache::Item* it = new_item(Cache::K_PF);
    it->p0 = p0, it->p1 = p1, it->a = p0 ? P.pos[p0 - 1] : 0, it->b = P.pos[p1 - 1];
    it->kv_cap = Cache::al((size_t)(it->b - it->a) * Cache::G_tok_bytes);
    it->kv = (uint8_t*)Sys::amap(it->kv_cap);
    kv_get(it->a, it->b, it->kv);
    if (state) {
        it->state = (uint8_t*)Sys::amap(STATE_FLOATS * 4);
        state_get((float*)it->state);
    }
    it->pend_pf = &G_cpf;
    G_cpf.fetch_add(1);
    Cache::push(it);
}

void out_flush(bool fin, const int* out, int n_out) { // 输出的 KV 每 FLUSH 个 token 交给写盘线程一段; fin: 最后一段 + 输出缓存点 (key、state 一定写); out: 全部输出 token (含 im_end), 只在 fin 时用
    Cache::Plan& P = *G_plan;
    int a = P.out_flushed, b = G_n_past;
    if (!fin && b - a < Cache::FLUSH) return;
    Cache::Item* it = new_item(fin ? Cache::K_FIN : Cache::K_SEG);
    it->a = a, it->b = b, it->off = P.out_off;
    if (b > a) {
        it->kv_cap = Cache::al((size_t)(b - a) * Cache::G_tok_bytes);
        it->kv = (uint8_t*)Sys::amap(it->kv_cap);
        kv_get(a, b, it->kv);
        P.segs[P.n_segs++] = b - a;
        P.out_off += Cache::seg_bytes(b - a);
    }
    P.out_flushed = b;
    if (fin) {
        uint8_t h[32];
        Sha::Ctx c = P.out_sha; // 输出段 = prompt 末尾 (已输入) + 全部输出 token (含 im_end)
        Sha::update(c, out, (size_t)n_out * 4);
        Sha::final(c, h);
        Sha::init(c);
        for (int i = 0; i < P.n_pts; i++) Sha::update(c, P.hash[i], 32);
        Sha::update(c, h, 32);
        Sha::final(c, it->key);
        it->npos = G_next_pos;
        it->state = (uint8_t*)Sys::amap(STATE_FLOATS * 4);
        state_get((float*)it->state);
        P.out_on = 0;
    }
    Cache::push(it);
}

void out_abort() { // 输出没有以 im_end 结束 (长度上限 / 其他停止 token / 客户端断开): 不生成输出缓存点, 删除已写的临时文件
    if (!G_plan || !G_plan->out_on) return;
    G_plan->out_on = 0;
    if (G_plan->n_segs) Cache::push(new_item(Cache::K_ABORT));
    wait_zero(G_cpend); // 写盘线程处理时要读 G_plan (临时目录编号), 下一个请求会立刻覆盖它
}

} // namespace

extern "C" {

int qwen_init(const char* path, int n_threads, int max_ctx) {
    if (G_inited) {
        fprintf(stderr, "⚠️ 模型已经加载过\n");
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    if (!Gguf::open(G_file, path)) return -1;
    snprintf(G_model_id, sizeof(G_model_id), "%s %zu cpu_f32", strrchr(path, '/') ? strrchr(path, '/') + 1 : path, G_file.size);
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
    G_tok_embd = get_w("token_embd.weight", D, VOCAB);
    G_output = get_w("output.weight", D, VOCAB);
    G_output_norm = get_f("output_norm.weight", D);
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
            L.ssm_a = get_f(p + "ssm_a", LV);
            L.dt_bias = get_f(p + "ssm_dt.bias", LV);
            L.conv_w = get_f(p + "ssm_conv1d.weight", (int64_t)QKV_DIM * 4);
            L.ssm_norm = get_f(p + "ssm_norm.weight", LHD);
            L.conv_state = alloc_f((size_t)QKV_DIM * 3);
            L.ssm_state = alloc_f((size_t)LV * LHD * LHD);
            continue;
        }
        L.wq = get_w(p + "attn_q.weight", D, Q_DIM);
        L.wk = get_w(p + "attn_k.weight", D, KV_DIM);
        L.wv = get_w(p + "attn_v.weight", D, KV_DIM);
        L.wo = get_w(p + "attn_output.weight", ATT_DIM, D);
        L.q_norm = get_f(p + "attn_q_norm.weight", HD);
        L.k_norm = get_f(p + "attn_k_norm.weight", HD);
        L.k_cache = alloc_f((size_t)max_ctx * KV_DIM);
        L.v_cache = alloc_f((size_t)max_ctx * KV_DIM);
    }
    for (int i = 0; i < ROT / 2; i++) G_inv_freq[i] = 1.0f / powf(THETA, (float)(2 * i) / ROT); // 与 HF 一致用 fp32 计算
    G_n_threads = std::min(n_threads, Ops::MAX_THREADS);
    Pool::init(G_n_threads);
    Ops::init_buffers(G_n_threads);
    for (int i = 0; i < G_n_threads; i++) G_scores[i] = alloc_f(max_ctx + 16);
    G_logits = alloc_f(VOCAB);
    G_hidden = alloc_f((size_t)(N_LAYER + 1) * D);
    G_x = alloc_f((size_t)CHUNK * D);
    G_xn = alloc_f((size_t)CHUNK * D);
    G_out = alloc_f((size_t)CHUNK * D);
    G_qkv = alloc_f((size_t)CHUNK * QKV_DIM);
    G_z = alloc_f((size_t)CHUNK * V_DIM);
    G_a = alloc_f((size_t)CHUNK * LV);
    G_b = alloc_f((size_t)CHUNK * LV);
    G_o = alloc_f((size_t)CHUNK * V_DIM);
    G_qg = alloc_f((size_t)CHUNK * Q_DIM);
    G_k = alloc_f((size_t)CHUNK * KV_DIM);
    G_v = alloc_f((size_t)CHUNK * KV_DIM);
    G_att = alloc_f((size_t)CHUNK * ATT_DIM);
    G_g = alloc_f((size_t)CHUNK * FF);
    G_u = alloc_f((size_t)CHUNK * FF);
    G_pos = (int*)calloc((size_t)CHUNK * 3, sizeof(int));
    G_work.resize(VOCAB);
    G_idx.resize(VOCAB);
    G_seen.resize(VOCAB);
    G_inited = true;
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "✅ 模型初始化完成, 线程 %d, 最大上下文 %d, 耗时 %.1fs\n", G_n_threads, max_ctx, sec);
    return 0;
}

int qwen_eval(const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3) { // embd/embd_mask/pos3 可为空, embd: [embd_mask 中 1 的个数][5120] (按位置顺序), pos3: [n][3] (t,h,w)
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
    for (int i = 0, r = 0; i < n; i += CHUNK) { // r: 之前各 chunk 的图片行数
        int T = std::min(CHUNK, n - i);
        eval_chunk(tokens + i, T, embd ? embd + (size_t)r * D : nullptr, embd_mask ? embd_mask + i : nullptr, pos3 ? pos3 + i * 3 : nullptr, i + T == n);
        for (int t = i; embd_mask && t < i + T; t++) r += embd_mask[t];
    }
    if (G_plan && G_plan->out_on) out_flush(false, nullptr, 0); // qwen_prefill_cached 之后的 decode: 每 FLUSH 个 token 交出一段 KV
    return 0;
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
    out_abort(); // 上一次 qwen_prefill_cached 没有 qwen_cache_finish (异常中断) 时, 丢掉它的输出缓存
    qwen_reset_impl();
}

} // extern "C"

namespace {

void qwen_reset_impl() {
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        if (!L.recr) continue;
        memset(L.conv_state, 0, sizeof(float) * QKV_DIM * 3);
        memset(L.ssm_state, 0, sizeof(float) * LV * LHD * LHD);
    }
    G_n_past = 0;
    G_next_pos = 0;
}

} // namespace

extern "C" {

int qwen_cache_init(const char* dir, int quant_threads) { // 在 qwen_init 之后调用一次: 启用前缀缓存, dir 由 python 检查并加锁, 其中 tmp/ 已清空; 0 成功
    if (!G_inited || !Cache::init(dir, G_model_id, STATE_FLOATS * 4, G_max_ctx, quant_threads, KV_PARTS, 2)) return -1;
    G_mbuf = (char*)Sys::amap(Cache::MBUF + 1);
    G_segbuf = (int*)Sys::amap((size_t)Cache::G_max_segs * 4);
    G_sbuf = (float*)Sys::amap(STATE_FLOATS * 4);
    Cache::Plan* P = new (Sys::amap(sizeof(Cache::Plan))) Cache::Plan();
    const int M = Cache::MAXP;
    P->pos = (int*)Sys::amap((size_t)M * 4);
    P->hash = (uint8_t (*)[32])Sys::amap((size_t)M * 32);
    P->key = (uint8_t (*)[32])Sys::amap((size_t)M * 32);
    P->role = (int*)Sys::amap((size_t)M * 4);
    P->npos = (int*)Sys::amap((size_t)M * 4);
    P->exist = (uint8_t*)Sys::amap(M);
    P->st = (uint8_t*)Sys::amap(M);
    P->segs = (int*)Sys::amap((size_t)Cache::G_max_segs * 4);
    G_plan = P;
    fprintf(stderr, "✅ 前缀缓存已启用: %s, 量化线程 %d\n", dir, quant_threads);
    return 0;
}

int qwen_prefill_cached(const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3, int n_pts, const int* pts_pos, const uint8_t* pts_hash, const uint8_t* pts_key, const int* pts_role, const uint8_t* tail, int n_tail, int tail_role) {
    // 带前缀缓存的 prefill (先清空状态): 参数同 qwen_eval, 缓存点参数同 cuda 版 qwen_prefill; 返回命中的 token 数 (>= 0), 失败 < 0
    // 返回时 prompt 的缓存已全部写完; 之后用 qwen_sample / qwen_eval 逐个生成, 结束时必须调 qwen_cache_finish
    if (!G_inited || !G_plan || n <= 0 || n > G_max_ctx || n_pts < 0 || n_pts > Cache::MAXP) {
        fprintf(stderr, "❌ qwen_prefill_cached 参数错误: prompt %d (最大上下文 %d), 缓存点 %d 个 (上限 %d), 缓存%s启用\n", n, G_max_ctx, n_pts, Cache::MAXP, G_plan ? "已" : "未");
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if ((!embd_mask || !embd_mask[i]) && (tokens[i] < 0 || tokens[i] >= VOCAB)) {
            fprintf(stderr, "❌ token id 越界: %d\n", tokens[i]);
            return -3;
        }
    }
    for (int i = 0; i < n_pts; i++) {
        if (pts_pos[i] <= (i ? pts_pos[i - 1] : 0) || pts_pos[i] > n) {
            fprintf(stderr, "❌ 缓存点位置不合法: 第 %d 个 = %d\n", i, pts_pos[i]);
            return -1;
        }
    }
    qwen_reset();
    Cache::Plan& P = *G_plan;
    P.n_pts = n_pts;
    P.hit = -1, P.w_kv_in = 0, P.w_st_in = 0, P.w_st_out = 0;
    memcpy(P.pos, pts_pos, (size_t)n_pts * 4);
    memcpy(P.hash, pts_hash, (size_t)n_pts * 32);
    memcpy(P.key, pts_key, (size_t)n_pts * 32);
    memcpy(P.role, pts_role, (size_t)n_pts * 4);
    Sha::init(P.out_sha);
    Sha::update(P.out_sha, tail, n_tail);
    P.out_role = tail_role;
    auto t0 = std::chrono::steady_clock::now();
    int i = cache_load(n), hit = i, r = 0, ck = P.hit + 1; // r: 之前的图片行数; ck: 下一个要写的缓存点
    for (int t = 0; embd_mask && t < i; t++) r += embd_mask[t];
    while (i < n) {
        int T = std::min(CHUNK, n - i);
        while (ck < P.n_pts && !P.st[ck]) ck++;
        if (ck < P.n_pts) T = std::min(T, P.pos[ck] - i); // chunk 在要写的缓存点处切开
        eval_chunk(tokens + i, T, embd ? embd + (size_t)r * D : nullptr, embd_mask ? embd_mask + i : nullptr, pos3 ? pos3 + i * 3 : nullptr, i + T == n);
        for (int t = i; embd_mask && t < i + T; t++) r += embd_mask[t];
        i += T;
        if (ck < P.n_pts && P.pos[ck] == i) {
            P.npos[ck] = G_next_pos;
            pf_flush(++ck, true);
        }
    }
    pf_flush(P.n_pts, false);
    P.out_on = 1; // 输出缓存点的段从最后一个缓存点开始 (含 prompt 末尾)
    P.out_start = P.out_flushed = P.n_pts ? P.pos[P.n_pts - 1] : 0;
    P.out_off = 0, P.n_segs = 0, P.out_id = Cache::G_out_seq.fetch_add(1);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto t1 = std::chrono::steady_clock::now();
    wait_zero(G_cpf); // prompt 的缓存写完才返回 (之后才出第一个 token)
    fprintf(stderr, "ℹ️ prefill %d tokens (缓存命中 %d, 计算 %d), %.2fs, %.2f tok/s; 等缓存写完 %.2fs\n", n, hit, n - hit, sec, (n - hit) / sec, std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    return hit;
}

int qwen_cache_finish(const int* out, int n_out, int ended) { // qwen_prefill_cached 之后生成结束时调用: out 是全部输出 token; ended: 是否因停止 token 结束
    // 以 im_end 结束时先把 im_end 算进 KV / 状态, 再生成输出缓存点 (kvcache + state); 否则丢掉输出缓存; 返回时输出的缓存已全部写完
    if (!G_plan || !G_plan->out_on) return 0;
    if (ended && n_out > 0 && out[n_out - 1] == IM_END && G_n_past < G_max_ctx) {
        int r = qwen_eval(&IM_END, 1, nullptr, nullptr, nullptr);
        if (r) return r;
        out_flush(true, out, n_out);
    } else {
        out_abort();
    }
    wait_zero(G_cpend);
    return 0;
}

int qwen_cache_stats(int sq, int* out) { // 与 cuda 接口相同, cpu 只有一个序列, sq 必须是 0: [命中的输入 token 数, 输入新写入的 KV token 数, 输入新写入的 state 数, 输出写入的 state 数]
    if (sq != 0) return -1;
    bool on = G_plan && G_plan->n_pts >= 0;
    out[0] = on && G_plan->hit >= 0 ? G_plan->pos[G_plan->hit] : 0;
    out[1] = on ? G_plan->w_kv_in.load() : 0;
    out[2] = on ? G_plan->w_st_in.load() : 0;
    out[3] = on ? G_plan->w_st_out.load() : 0;
    return 0;
}

int qwen_n_past() { return G_n_past; }
float* qwen_logits() { return G_logits; }
float* qwen_hidden() { return G_hidden; } // [65][5120], 0 是 embedding, i 是第 i 层输出, 都只保存最后一个 token

int qwen_selftest() { // AVX-512 反量化 vs 标量参考, matmul vs 逐行点积, 返回失败项数
    int fails = 0;
    const char* names[] = {"blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight", "blk.51.ffn_down.weight", "blk.0.ffn_up.weight", "blk.0.ssm_alpha.weight"};
    for (const char* name : names) {
        const Gguf::Tensor& t = find_tensor(name);
        int k = (int)t.dims[0], n = (int)t.dims[1];
        Ops::Weight w = get_w(name, k, n);
        vector<float> a(k), b(k), x(5 * k), y(5 * n);
        double max_diff = 0;
        for (int r = 0; r < 8; r++) {
            const uint8_t* src = w.data + (size_t)r * w.rb;
            Quants::dequant_row(w.type, src, a.data(), k);
            ref_dequant(w.type, src, b.data(), k);
            for (int i = 0; i < k; i++) max_diff = std::max(max_diff, (double)std::fabs(a[i] - b[i]));
        }
        for (auto& v : x) v = (float)((rng_next() >> 11) * (1.0 / 9007199254740992.0) * 2 - 1);
        Ops::matmul(w, x.data(), 5, y.data()); // T=5 覆盖 4 + 1 两种 kernel
        double max_rel = 0;
        for (int t = 0; t < 5; t++) {
            for (int r = 0; r < n; r += std::max(1, n / 37)) {
                ref_dequant(w.type, w.data + (size_t)r * w.rb, b.data(), k);
                double ref = 0, mag = 0;
                for (int i = 0; i < k; i++) {
                    ref += (double)b[i] * x[(size_t)t * k + i];
                    mag += std::fabs((double)b[i] * x[(size_t)t * k + i]);
                }
                max_rel = std::max(max_rel, std::fabs(ref - y[(size_t)t * n + r]) / (mag + 1e-30));
            }
        }
        bool ok = max_diff == 0 && max_rel < 1e-5;
        fails += !ok;
        fprintf(stderr, "%s %s type=%u [%d, %d] 反量化最大差 %.3g, matmul 最大相对误差 %.3g\n", ok ? "✅" : "❌", name, w.type, k, n, max_diff, max_rel);
    }
    return fails;
}

} // extern "C"
