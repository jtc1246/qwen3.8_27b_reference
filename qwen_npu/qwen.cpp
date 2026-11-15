// Qwen3.8-27B NPU (Ascend 310P) 推理, 输入 RedHatAI/Qwen3.8-27B-INT4 的 model.safetensors
// (compressed-tensors int4 对称 group128, AWQ 平滑 + GPTQ), 通过 extern "C" 给 python ctypes 调用
// 主干 int4 重打包成 W4A8 布局: decode 走 u8xu8 整数 mmad (x 由 k_qx8 动态量化 int8), prefill 五平面 And 反量化 fp16 走 mmad
// lm_head/tok_embd 由 bf16 在 host 按 per-16 RTN 转 int8 (lm_head 走 I8F fp16 路径); alpha/beta (f32) 走 k_gemv_f32t
// HF -> 本实现的权重变换 (与 GGUF 转换一致, 见 docs/model.md 第 11 节): norm +1 (ssm_norm 除外)、
// ssm_a = -exp(A_log)、DeltaNet v 头 grouped -> tiled 重排 (第 j 个 v 头用第 j%16 个 qk 头)
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
#include "st.h"
#include "repack.h"
#include "aclrtlaunch_k_gemm_i8.h"
#include "aclrtlaunch_k_gemm_w8d.h"
#include "aclrtlaunch_k_gemm_w8p.h"
#include "aclrtlaunch_k_qx8.h"
#include "aclrtlaunch_k_pack_x.h"
#include "aclrtlaunch_k_gemv_f32t.h"
#include "aclrtlaunch_k_dq_i8.h"
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
const int QKVZ = QKV_DIM + V_DIM, QKVC = Q_DIM + 2 * KV_DIM; // qkv+z / q+k+v 合并矩阵的每 token 输出宽度
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
    Weight qkv, ab, ssm_out; // qkv = in_proj_qkv + in_proj_z 行拼接 [QKVZ][D]; ab = α+β 行拼接 f32 [96][D]
    void* ssm_a = nullptr; // 补零到 64
    void* dt_bias = nullptr;
    void* conv_w = nullptr; // 已重排为 [4][QKV_DIM]
    void* ssm_norm = nullptr;
    void* conv_state = nullptr; // [3][QKV_DIM]
    void* ssm_state = nullptr; // [48][128][128]
    Weight wqkv, wo; // wqkv = q+k+v 行拼接 [QKVC][D]
    void* q_norm = nullptr;
    void* k_norm = nullptr;
    void* k_cache = nullptr; // [max_ctx][1024] f32
    void* v_cache = nullptr;
};

struct SelfRef { string name; const Weight* w; int rows[8]; vector<float> ref; }; // 自测: 8 行反量化参考值 (重排后语义, [8][k])

St::File G_st;
// DeltaNet v 头重排 (HF grouped -> tiled, 同 GGUF 转换): tiled 头 j 对应 HF 头 (j%16)*3 + j/16
int G_perm48[LV];
int G_rows_qkv[QKV_DIM]; // in_proj_qkv 行重排 (q/k 恒等, v 段按头重排); conv1d 通道同此
int G_rows_z[V_DIM]; // in_proj_z 行重排
int G_kblk_out[V_DIM / 128]; // out_proj k 方向 128 块重排 (一块恰好一个 v 头, 与量化组对齐)
Layer G_layers[N_LAYER];
Weight G_tok_embd, G_output;
void* G_output_norm = nullptr;
int G_max_ctx = 0, G_n_past = 0, G_next_pos = 0;
bool G_inited = false;
uint64_t G_rng = 0x853c49e6748fea9bULL;
void *G_d_x = nullptr, *G_d_xn = nullptr, *G_d_out = nullptr, *G_d_qkv = nullptr, *G_d_ab = nullptr, *G_d_o = nullptr;
void *G_d_qg = nullptr, *G_d_att = nullptr, *G_d_g = nullptr, *G_d_u = nullptr;
void *G_d_logits = nullptr, *G_d_hidden = nullptr, *G_d_tok = nullptr, *G_d_cos = nullptr, *G_d_sin = nullptr;
void *G_d_xp = nullptr; // k_pack_x 的 fp16 zZ 布局 x
void *G_d_xrep = nullptr, *G_d_xaux = nullptr; // k_qx8 输出 (decode 的 u8 x 旋转副本 + 组 scale/修正量)
float G_inv_freq[ROT / 2];
float* G_logits = nullptr;
vector<float> G_hidden;
int G_pos[CHUNK * 3];
vector<float> G_work;
vector<int> G_idx;
vector<uint8_t> G_seen;
vector<SelfRef> G_selfrefs;

inline void* fp(void* base, size_t n_floats) { return (uint8_t*)base + n_floats * sizeof(float); }

const St::Tensor& find_t(const string& name) {
    auto it = G_st.tensors.find(name);
    if (it == G_st.tensors.end()) {
        fprintf(stderr, "❌ 缺少 tensor %s\n", name.c_str());
        _Exit(1);
    }
    return it->second;
}

void check_shape(const St::Tensor& t, const string& name, int dtype, std::initializer_list<int64_t> shape) {
    if (t.dtype != dtype || t.shape != vector<int64_t>(shape)) {
        fprintf(stderr, "❌ %s dtype/形状不对 (dtype=%d, ndim=%zu)\n", name.c_str(), t.dtype, t.shape.size());
        _Exit(1);
    }
}

struct QSrc { // int4 量化矩阵三元组 + 可选重排 (行 / k 方向 128 块, 128 块与量化组对齐, 重排无损)
    const uint32_t* pk;
    const uint16_t* sc;
    int k, n;
    const int* row_map;
    const int* kblk_map;
};

QSrc get_qsrc(const string& base, int k, int n, const int* row_map, const int* kblk_map) {
    const St::Tensor& tp = find_t(base + ".weight_packed");
    const St::Tensor& ts = find_t(base + ".weight_scale");
    const St::Tensor& th = find_t(base + ".weight_shape");
    check_shape(tp, base + ".weight_packed", St::DT_I32, {n, k / 8});
    check_shape(ts, base + ".weight_scale", St::DT_BF16, {n, k / 128});
    const int64_t* sh = (const int64_t*)th.data;
    if (th.dtype != St::DT_I64 || St::numel(th) != 2 || sh[0] != n || sh[1] != k) {
        fprintf(stderr, "❌ %s weight_shape 不符 (期望 [%d, %d])\n", base.c_str(), n, k);
        _Exit(1);
    }
    return {(const uint32_t*)tp.data, (const uint16_t*)ts.data, k, n, row_map, kblk_map};
}

// 解一行 int4 (按重排后的行号 r): nat 为 0..15 的 nibble 值 (q+8), s16 为 per-16 f32 scale (per-128 组复制)
void decode_q4_row(const QSrc& q, int r, int8_t* nat, float* s16) {
    int rr = q.row_map ? q.row_map[r] : r;
    const uint32_t* w = q.pk + (uint64_t)rr * (q.k / 8);
    const uint16_t* sp = q.sc + (uint64_t)rr * (q.k / 128);
    for (int kb = 0; kb < q.k / 128; kb++) {
        int sb = q.kblk_map ? q.kblk_map[kb] : kb;
        const uint32_t* ww = w + sb * 16;
        int8_t* o = nat + kb * 128;
        for (int i = 0; i < 16; i++) {
            uint32_t v = ww[i];
            for (int j = 0; j < 8; j++) o[i * 8 + j] = (int8_t)((v >> (4 * j)) & 15);
        }
        float s = Repack::bf2f(sp[sb]);
        for (int g = 0; g < 8; g++) s16[kb * 8 + g] = s;
    }
}

struct QSeg { string base; int n; const int* row_map; }; // 行方向拼接的一段 (共享 k)

// 多段 int4 矩阵行拼接后重打包 (段内可带行重排): 合并后一次 matmul, 省启动和小矩阵的流水浪费
Weight get_w_q4_cat(const vector<QSeg>& segs, int k, const int* kblk_map = nullptr) {
    vector<QSrc> qs;
    vector<int> soff{0};
    for (const QSeg& g : segs) {
        qs.push_back(get_qsrc(g.base, k, g.n, g.row_map, kblk_map));
        soff.push_back(soff.back() + g.n);
    }
    int n = soff.back();
    if (k % 256 || n % 128) { // W4A8: k 按 128 分组且组对打包, 需为 256 倍数
        fprintf(stderr, "❌ %s 形状 [%d, %d] 不满足 W4A8 布局要求\n", segs[0].base.c_str(), k, n);
        _Exit(1);
    }
    Weight w;
    w.fmt = Repack::FMT_I4W8;
    w.k = k;
    w.n = n;
    int KS = k / 128, SP = KS / 2, nr = n / 8;
    w.data = Acl::alloc((uint64_t)(n / 16) * SP * 2240);
    vector<uint8_t> staging;
    vector<int8_t> nat((size_t)16 * k);
    vector<float> s16((size_t)16 * (k / 16));
    uint64_t off = 0;
    for (int c = 0; c < 8; c++) { // 每核的行连续存放, 核内按 256 行分块
        for (int r0 = 0; r0 < nr; r0 += 256) {
            int nb = std::min(256, nr - r0), NI = nb / 16;
            uint64_t bb = (uint64_t)SP * NI * 2240;
            staging.resize(bb);
            for (int ni = 0; ni < NI; ni++) {
                for (int rr = 0; rr < 16; rr++) {
                    int r = c * nr + r0 + ni * 16 + rr, si = 0;
                    while (r >= soff[si + 1]) si++;
                    decode_q4_row(qs[si], r - soff[si], &nat[(size_t)rr * k], &s16[(size_t)rr * (k / 16)]);
                }
                Repack::pack_w8_16rows(k, NI, ni, nat.data(), s16.data(), staging.data());
            }
            Acl::h2d((uint8_t*)w.data + off, staging.data(), bb);
            off += bb;
        }
    }
    return w;
}

Weight get_w_q4(const string& base, int k, int n, const int* row_map = nullptr, const int* kblk_map = nullptr) {
    return get_w_q4_cat({{base, n, row_map}}, k, kblk_map);
}

inline void bf16_row_f32(const St::Tensor& t, int64_t r, int k, float* out) {
    const uint16_t* p = (const uint16_t*)t.data + (uint64_t)r * k;
    for (int i = 0; i < k; i++) out[i] = Repack::bf2f(p[i]);
}

Weight get_w_i8_bf16(const string& name, int k, int n) { // lm_head: bf16 -> per-16 RTN int8 -> I8F fractal
    const St::Tensor& t = find_t(name);
    check_shape(t, name, St::DT_BF16, {n, k});
    Weight w;
    w.fmt = Repack::FMT_I8F;
    w.k = k;
    w.n = n;
    int SB = Repack::sb2_ni(w.fmt), KS = k / 128, nr = n / 8;
    w.data = Acl::alloc((uint64_t)(n / 16) * KS * SB);
    vector<uint8_t> staging;
    vector<float> row(k);
    vector<int8_t> nat((size_t)16 * k);
    vector<float> s16((size_t)16 * (k / 16));
    uint64_t off = 0;
    for (int c = 0; c < 8; c++) {
        for (int r0 = 0; r0 < nr; r0 += 256) {
            int nb = std::min(256, nr - r0), NI = nb / 16;
            uint64_t bb = (uint64_t)KS * NI * SB;
            staging.resize(bb);
            for (int ni = 0; ni < NI; ni++) {
                for (int rr = 0; rr < 16; rr++) {
                    bf16_row_f32(t, c * nr + r0 + ni * 16 + rr, k, row.data());
                    Repack::quant_row_i8(row.data(), k, &nat[(size_t)rr * k], &s16[(size_t)rr * (k / 16)]);
                }
                Repack::pack2_16rows(w.fmt, k, NI, ni, nat.data(), s16.data(), staging.data());
            }
            Acl::h2d((uint8_t*)w.data + off, staging.data(), bb);
            off += bb;
        }
    }
    return w;
}

Weight get_embd_i8(const string& name, int k, int n) { // tok_embd: bf16 -> per-16 RTN int8 -> wave I8 (k_dq_i8 查行)
    const St::Tensor& t = find_t(name);
    check_shape(t, name, St::DT_BF16, {n, k});
    Weight w;
    w.fmt = Repack::FMT_I8;
    w.k = k;
    w.n = n;
    w.stride = (int)Repack::row_stride(w.fmt, k);
    w.data = Acl::alloc((uint64_t)n * w.stride + PAD);
    const int SLICE = 2048; // 每批重打包的行数
    vector<uint8_t> staging((uint64_t)SLICE * w.stride);
    vector<float> row(k);
    vector<int8_t> nat(k);
    vector<float> s(k / 16);
    for (int r0 = 0; r0 < n; r0 += SLICE) {
        int nr = std::min(SLICE, n - r0);
        for (int r = 0; r < nr; r++) {
            bf16_row_f32(t, r0 + r, k, row.data());
            Repack::quant_row_i8(row.data(), k, nat.data(), s.data());
            Repack::pack_row_i8(k, nat.data(), s.data(), staging.data() + (uint64_t)r * w.stride);
        }
        Acl::h2d((uint8_t*)w.data + (uint64_t)r0 * w.stride, staging.data(), (uint64_t)nr * w.stride);
    }
    return w;
}

vector<float> load_f32(const string& name, int64_t numel) { // bf16/f32 小张量 -> f32
    const St::Tensor& t = find_t(name);
    if (St::numel(t) != numel || (t.dtype != St::DT_BF16 && t.dtype != St::DT_F32)) {
        fprintf(stderr, "❌ %s 需要 bf16/f32 且元素数 %ld, 实际 dtype=%d numel=%ld\n", name.c_str(), (long)numel, t.dtype, (long)St::numel(t));
        _Exit(1);
    }
    vector<float> v(numel);
    if (t.dtype == St::DT_F32) memcpy(v.data(), t.data, numel * 4);
    else
        for (int64_t i = 0; i < numel; i++) v[i] = Repack::bf2f(((const uint16_t*)t.data)[i]);
    return v;
}

void* up_f32(const float* v, int64_t numel, int64_t alloc_numel = 0) { // f32 上传, 可指定更大的分配 (补零)
    if (!alloc_numel) alloc_numel = numel;
    void* p = Acl::alloc((uint64_t)alloc_numel * 4);
    Acl::h2d(p, v, (uint64_t)numel * 4);
    return p;
}

void* get_norm(const string& name, int n, bool add1) { // HF norm 权重为零中心 w, 计算用 1+w (ssm_norm 除外)
    vector<float> v = load_f32(name, n);
    if (add1)
        for (float& x : v) x += 1.0f;
    return up_f32(v.data(), n);
}

Weight get_w_ab(const string& p) { // α+β 合并 [96][D] f32 (行按 tiled 头序), 一次 GEMV 算完
    const St::Tensor& ta = find_t(p + "linear_attn.in_proj_a.weight");
    const St::Tensor& tb = find_t(p + "linear_attn.in_proj_b.weight");
    check_shape(ta, p + "in_proj_a", St::DT_BF16, {LV, D});
    check_shape(tb, p + "in_proj_b", St::DT_BF16, {LV, D});
    vector<float> o((size_t)2 * LV * D);
    for (int j = 0; j < LV; j++) {
        bf16_row_f32(ta, G_perm48[j], D, &o[(size_t)j * D]);
        bf16_row_f32(tb, G_perm48[j], D, &o[(size_t)(LV + j) * D]);
    }
    Weight w;
    w.fmt = Repack::FMT_F32;
    w.k = D;
    w.n = 2 * LV;
    w.stride = D * 4;
    w.data = Acl::alloc((uint64_t)2 * LV * D * 4 + PAD);
    Acl::h2d(w.data, o.data(), (uint64_t)2 * LV * D * 4);
    return w;
}

void save_ref(const string& name, const Weight* w, const vector<float>& ref) { // 自测参考: 首末各 4 行的反量化 f32
    SelfRef s;
    s.name = name;
    s.w = w;
    for (int i = 0; i < 8; i++) s.rows[i] = i < 4 ? i : w->n - 8 + i;
    s.ref = ref;
    G_selfrefs.push_back(std::move(s));
}

void save_ref_q4(const string& name, const Weight* w, const vector<QSeg>& segs, const int* kblk_map) {
    vector<QSrc> qs;
    vector<int> soff{0};
    for (const QSeg& g : segs) {
        qs.push_back(get_qsrc(g.base, w->k, g.n, g.row_map, kblk_map));
        soff.push_back(soff.back() + g.n);
    }
    vector<float> ref((size_t)8 * w->k);
    vector<int8_t> nat(w->k);
    vector<float> s16(w->k / 16);
    for (int i = 0; i < 8; i++) {
        int r = i < 4 ? i : w->n - 8 + i, si = 0;
        while (r >= soff[si + 1]) si++;
        decode_q4_row(qs[si], r - soff[si], nat.data(), s16.data());
        for (int c = 0; c < w->k; c++) ref[(size_t)i * w->k + c] = (nat[c] - 8) * s16[c / 16];
    }
    save_ref(name, w, ref);
}

void save_ref_i8(const string& name, const Weight* w) { // lm_head / tok_embd: 参考 = RTN 后的 s*q
    const St::Tensor& t = find_t(name);
    vector<float> ref((size_t)8 * w->k);
    vector<float> row(w->k);
    vector<int8_t> nat(w->k);
    vector<float> s(w->k / 16);
    for (int i = 0; i < 8; i++) {
        bf16_row_f32(t, i < 4 ? i : w->n - 8 + i, w->k, row.data());
        Repack::quant_row_i8(row.data(), w->k, nat.data(), s.data());
        for (int c = 0; c < w->k; c++) ref[(size_t)i * w->k + c] = s[c / 16] * nat[c];
    }
    save_ref(name, w, ref);
}

const void* G_pack_src = nullptr; // k_pack_x 结果缓存 (同一 x 的多个 matmul 只打包一次)
int G_pack_t = 0, G_pack_k = 0;
const void* G_qx_src = nullptr; // k_qx8 结果缓存 (decode 时同一 x 只量化一次)
int G_qx_k = 0;
void pack_invalidate() {
    G_pack_src = nullptr;
    G_qx_src = nullptr;
}

// QWEN_NPU_PROF=1 时逐 kernel 同步计时 (调试用, 会拖慢)
const char* G_prof_names[] = {"gemm", "pack_x/qx8", "gemv_f32t", "conv", "deltanet", "qk_rope", "attn", "add_norm", "silu", "misc"};
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
    } else if (w.fmt == Repack::FMT_I4W8 && (x != G_qx_src || w.k != G_qx_k)) { // decode: x 量化 int8, 同一 x 只做一次
        prof_mark(9);
        Acl::ck(ACLRT_LAUNCH_KERNEL(k_qx8)(1, Acl::G_stream, x, G_d_xrep, G_d_xaux, w.k), "launch qx8");
        prof_mark(1);
        G_qx_src = x;
        G_qx_k = w.k;
    }
    prof_mark(9);
    aclError e;
    if (w.fmt == Repack::FMT_I8F) e = ACLRT_LAUNCH_KERNEL(k_gemm_i8)(8, Acl::G_stream, w.data, x, xp, y, T, w.n, w.k);
    else if (T == 1) e = ACLRT_LAUNCH_KERNEL(k_gemm_w8d)(8, Acl::G_stream, w.data, G_d_xrep, G_d_xaux, y, w.n, w.k);
    else e = ACLRT_LAUNCH_KERNEL(k_gemm_w8p)(8, Acl::G_stream, w.data, xp, y, T, w.n, w.k);
    Acl::ck(e, "launch gemm");
    prof_mark(0);
}

void matmul_tok(const Weight& w, void* x, void* y) { matmul(w, x, 1, y); }

void dq_rows(const Weight& w, void* d_rows, void* d_out, int nr) { // 反量化 nr 行到自然顺序 f32 (tok_embd, wave I8)
    Acl::ck(ACLRT_LAUNCH_KERNEL(k_dq_i8)(8, Acl::G_stream, w.data, d_rows, d_out, nr, w.k, w.stride), "launch dq");
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
            matmul(L.ab, G_d_xn, T, G_d_ab);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_conv)(8, Acl::G_stream, G_d_qkv, L.conv_state, L.conv_w, T), "launch conv");
            prof_mark(3);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_deltanet)(8, Acl::G_stream, G_d_qkv, G_d_ab, L.ssm_a, L.dt_bias, L.ssm_norm, L.ssm_state, G_d_o, T), "launch deltanet");
            prof_mark(4);
            pack_invalidate();
            matmul(L.ssm_out, G_d_o, T, G_d_out);
        } else {
            matmul(L.wqkv, G_d_xn, T, G_d_qg);
            const char* ph2 = getenv("QWEN_NPU_PHASE");
            if (ph2 && l == stop_l && atoi(ph2) == 10) { Acl::sync(); return; }
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_qk_rope)(8, Acl::G_stream, G_d_qg, L.q_norm, L.k_norm, G_d_cos, G_d_sin, L.k_cache, L.v_cache, G_n_past, T), "launch qk_rope");
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
    if (!St::open(G_st, path)) return -1;
    G_max_ctx = max_ctx;
    for (int j = 0; j < LV; j++) G_perm48[j] = (j % LK) * 3 + j / LK; // tiled 头 j <- HF 头
    for (int r = 0; r < QKV_DIM; r++) G_rows_qkv[r] = r < 2 * KEY_DIM ? r : 2 * KEY_DIM + G_perm48[(r - 2 * KEY_DIM) / LHD] * LHD + r % LHD;
    for (int r = 0; r < V_DIM; r++) G_rows_z[r] = G_perm48[r / LHD] * LHD + r % LHD;
    for (int j = 0; j < LV; j++) G_kblk_out[j] = G_perm48[j];
    auto t1 = std::chrono::steady_clock::now();
    G_tok_embd = get_embd_i8("model.language_model.embed_tokens.weight", D, VOCAB);
    save_ref_i8("model.language_model.embed_tokens.weight", &G_tok_embd);
    G_output = get_w_i8_bf16("lm_head.weight", D, VOCAB);
    save_ref_i8("lm_head.weight", &G_output);
    G_output_norm = get_norm("model.language_model.norm.weight", D, true);
    void* conv_all = Acl::alloc((size_t)N_LAYER * 3 * QKV_DIM * 4);
    void* ssm_all = Acl::alloc((size_t)N_LAYER * LV * LHD * LHD * 4);
    vector<float> cwt(4 * QKV_DIM);
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        string p = "model.language_model.layers." + std::to_string(l) + ".";
        L.recr = (l + 1) % 4 != 0;
        L.attn_norm = get_norm(p + "input_layernorm.weight", D, true);
        L.post_norm = get_norm(p + "post_attention_layernorm.weight", D, true);
        L.ffn_gate = get_w_q4(p + "mlp.gate_proj", D, FF);
        L.ffn_up = get_w_q4(p + "mlp.up_proj", D, FF);
        L.ffn_down = get_w_q4(p + "mlp.down_proj", FF, D);
        if (l == 0) save_ref_q4(p + "mlp.gate_proj", &L.ffn_gate, {{p + "mlp.gate_proj", FF, nullptr}}, nullptr);
        if (l == 51) save_ref_q4(p + "mlp.down_proj", &L.ffn_down, {{p + "mlp.down_proj", D, nullptr}}, nullptr);
        if (L.recr) {
            vector<QSeg> qkv_segs = {{p + "linear_attn.in_proj_qkv", QKV_DIM, G_rows_qkv}, {p + "linear_attn.in_proj_z", V_DIM, G_rows_z}};
            L.qkv = get_w_q4_cat(qkv_segs, D);
            L.ab = get_w_ab(p);
            L.ssm_out = get_w_q4(p + "linear_attn.out_proj", V_DIM, D, nullptr, G_kblk_out);
            if (l == 0) {
                save_ref_q4(p + "linear_attn.in_proj_qkv+z", &L.qkv, qkv_segs, nullptr);
                save_ref_q4(p + "linear_attn.out_proj", &L.ssm_out, {{p + "linear_attn.out_proj", D, nullptr}}, G_kblk_out);
                const St::Tensor& ta = find_t(p + "linear_attn.in_proj_a.weight");
                const St::Tensor& tb = find_t(p + "linear_attn.in_proj_b.weight");
                vector<float> aref((size_t)8 * D);
                for (int i = 0; i < 4; i++) bf16_row_f32(ta, G_perm48[i], D, &aref[(size_t)i * D]);
                for (int i = 4; i < 8; i++) bf16_row_f32(tb, G_perm48[LV - 8 + i], D, &aref[(size_t)i * D]); // 合并矩阵尾 4 行 = β 的 44..47
                save_ref(p + "linear_attn.in_proj_a+b", &L.ab, aref);
            }
            vector<float> alog = load_f32(p + "linear_attn.A_log", LV);
            vector<float> a64(64, 0.0f), b64(64, 0.0f);
            for (int j = 0; j < LV; j++) a64[j] = -expf(alog[G_perm48[j]]); // ssm_a = -exp(A_log), 按 tiled 头序
            L.ssm_a = up_f32(a64.data(), 64);
            vector<float> dtb = load_f32(p + "linear_attn.dt_bias", LV);
            for (int j = 0; j < LV; j++) b64[j] = dtb[G_perm48[j]];
            L.dt_bias = up_f32(b64.data(), 64);
            L.ssm_norm = get_norm(p + "linear_attn.norm.weight", LHD, false);
            vector<float> cw = load_f32(p + "linear_attn.conv1d.weight", (int64_t)QKV_DIM * 4); // [通道][1][4] -> [4][通道], v 通道重排
            for (int c = 0; c < QKV_DIM; c++)
                for (int j = 0; j < 4; j++) cwt[j * QKV_DIM + c] = cw[G_rows_qkv[c] * 4 + j];
            L.conv_w = Acl::alloc((size_t)4 * QKV_DIM * 4);
            Acl::h2d(L.conv_w, cwt.data(), (size_t)4 * QKV_DIM * 4);
            L.conv_state = fp(conv_all, (size_t)l * 3 * QKV_DIM);
            L.ssm_state = fp(ssm_all, (size_t)l * LV * LHD * LHD);
            continue;
        }
        L.wqkv = get_w_q4_cat({{p + "self_attn.q_proj", Q_DIM, nullptr}, {p + "self_attn.k_proj", KV_DIM, nullptr}, {p + "self_attn.v_proj", KV_DIM, nullptr}}, D);
        L.wo = get_w_q4(p + "self_attn.o_proj", ATT_DIM, D);
        L.q_norm = get_norm(p + "self_attn.q_norm.weight", HD, true);
        L.k_norm = get_norm(p + "self_attn.k_norm.weight", HD, true);
        L.k_cache = Acl::alloc((size_t)max_ctx * KV_DIM * 4);
        L.v_cache = Acl::alloc((size_t)max_ctx * KV_DIM * 4);
    }
    double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "ℹ️ 权重重打包并上传完成, %.1fs\n", up);
    St::close(G_st);
    for (int i = 0; i < ROT / 2; i++) G_inv_freq[i] = 1.0f / powf(THETA, (float)(2 * i) / ROT);
    G_d_x = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
    G_d_xn = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
    G_d_out = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
    G_d_qkv = Acl::alloc((size_t)CHUNK * QKVZ * 4 + PAD);
    G_d_ab = Acl::alloc((size_t)CHUNK * 2 * LV * 4 + PAD);
    G_d_o = Acl::alloc((size_t)CHUNK * V_DIM * 4 + PAD);
    G_d_qg = Acl::alloc((size_t)CHUNK * QKVC * 4 + PAD);
    G_d_att = Acl::alloc((size_t)CHUNK * ATT_DIM * 4 + PAD);
    G_d_g = Acl::alloc((size_t)CHUNK * FF * 4 + PAD);
    G_d_u = Acl::alloc((size_t)CHUNK * FF * 4 + PAD);
    G_d_logits = Acl::alloc((size_t)VOCAB * 4 + PAD);
    G_d_hidden = Acl::alloc((size_t)(N_LAYER + 1) * D * 4);
    G_d_tok = Acl::alloc((size_t)CHUNK * 4 + 64);
    G_d_xp = Acl::alloc((size_t)(FF / 128) * (CHUNK / 16) * 2048 * 2); // 最大 k=FF, m=CHUNK
    G_d_xrep = Acl::alloc((size_t)2 * 4 * 8704 + PAD); // 最大 k=17408 (两半)
    G_d_xaux = Acl::alloc((size_t)(FF / 128) * 8 * 3 * 4 + PAD);
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

int qwen_selftest() { // tok_embd: 反量化 vs host 参考逐位; matmul 权重: m=1 GEMM vs 参考行的 f32 点积 (fp16 计算, 阈值 5e-3)
    Acl::bind();
    int fails = 0;
    for (SelfRef& s : G_selfrefs) {
        const Weight* w = s.w;
        int k = w->k;
        if (w->fmt == Repack::FMT_I8) { // tok_embd 反量化查行, 与 host 参考逐位一致
            Acl::h2d(G_d_tok, s.rows, 8 * 4);
            dq_rows(*w, G_d_tok, G_d_out, 8);
            Acl::sync();
            vector<float> got((size_t)8 * k);
            Acl::d2h(got.data(), G_d_out, got.size() * 4);
            double max_diff = 0, max_abs = 0;
            for (size_t i = 0; i < got.size(); i++) {
                max_diff = std::max(max_diff, (double)std::fabs(got[i] - s.ref[i]));
                max_abs = std::max(max_abs, (double)std::fabs(s.ref[i]));
            }
            bool ok = max_diff <= 1e-6 * max_abs;
            fails += !ok;
            fprintf(stderr, "%s %s fmt=%d [%d, %d] 反量化最大差 %.3g (最大绝对值 %.3g)\n", ok ? "✅" : "❌", s.name.c_str(), w->fmt, k, w->n, max_diff, max_abs);
            continue;
        }
        vector<float> x(k), y((size_t)w->n);
        for (auto& v : x) v = (float)((rng_next() >> 11) * (1.0 / 9007199254740992.0) * 2 - 1);
        Acl::h2d(G_d_g, x.data(), (size_t)k * 4);
        pack_invalidate(); // x 内容换了, 指针没换, 需失效 qx8 缓存
        matmul_tok(*w, G_d_g, G_d_u);
        Acl::sync();
        Acl::d2h(y.data(), G_d_u, (size_t)w->n * 4);
        if (w->fmt == Repack::FMT_I4W8) { // W4A8 路径 x 会被量化, 参考也按同样规则量化 (与 k_qx8 逐字节一致)
            for (int g = 0; g < k / 128; g++) {
                float amax = 1e-30f;
                for (int i = 0; i < 128; i++) amax = std::max(amax, std::fabs(x[g * 128 + i]));
                float sx = amax / 127.0f, inv = 127.0f / amax;
                for (int i = 0; i < 128; i++) x[g * 128 + i] = (float)lrintf(x[g * 128 + i] * inv) * sx;
            }
        }
        double max_rel = 0;
        for (int r = 0; r < 8; r++) {
            const float* ref = &s.ref[(size_t)r * k];
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
    else if (s == "qkv") p = G_d_qkv; // 含 z 段 ([T][QKVZ])
    else if (s == "ab") p = G_d_ab;
    else if (s == "o") p = G_d_o;
    else if (s == "qg") p = G_d_qg; // 含 k/v 段 ([T][QKVC])
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
