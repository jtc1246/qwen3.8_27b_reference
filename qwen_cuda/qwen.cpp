// Qwen3.8-27B (qwen35 架构) CUDA 推理, 输入 GGUF (Q4_K_M), 通过 extern "C" 给 python ctypes 调用
// host 端只用 CUDA driver API (libcuda.so.1), kernel 编译成 fatbin 嵌在 so 里; 接口与 qwen_cpu 一致 (qwen_init 第二个参数改为 GPU 编号)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "cu.h"
#include "gguf.h"
#include "quants_ref.h"

using std::string;
using std::vector;

__asm__(".pushsection .rodata\n.balign 64\n.globl qwen_fatbin\n.hidden qwen_fatbin\nqwen_fatbin:\n.incbin \"" FATBIN_PATH "\"\n.byte 0\n.popsection\n");
extern "C" const unsigned char qwen_fatbin[];

namespace {

const int N_LAYER = 64, D = 5120, FF = 17408, VOCAB = 248320;
const int N_HEAD = 24, N_KV = 4, HD = 256, ROT = 64;
const int LK = 16, LV = 48, LHD = 128;
const int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM;
const int Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD, ATT_DIM = N_HEAD * HD;
const int CHUNK = 1024; // prefill 每批处理的 token 数
const int ATT_SPLIT = 78; // decode 注意力沿序列最多切的段数, 与 kernels.cu 一致
const int FA_SMEM = (64 + 4 * 32) * (HD + 8) * 2; // k_attn_pf 的动态 smem: sq [64][264] + sk/sv [2][32][264] (fp16), 101376 字节
const float THETA = 1e7f;

struct Weight {
    CUdeviceptr data = 0;
    uint32_t type = 0;
    int k = 0; // 输入维度, 每行长度
    int n = 0; // 输出维度, 行数
    uint64_t rb = 0; // 每行字节数
};

struct Layer {
    bool recr = false; // true: Gated DeltaNet, false: full attention
    CUdeviceptr attn_norm = 0, post_norm = 0;
    Weight ffn_gate, ffn_up, ffn_down;
    Weight qkv, gate_z, alpha, beta, ssm_out;
    CUdeviceptr ssm_a = 0, dt_bias = 0, conv_w = 0, ssm_norm = 0;
    CUdeviceptr conv_state = 0, ssm_state = 0; // [10240][3], [48][128 k][128 v]
    Weight wq, wk, wv, wo;
    CUdeviceptr q_norm = 0, k_norm = 0;
    CUdeviceptr k_cache = 0, v_cache = 0, k_scale = 0, v_scale = 0; // q8_0: [max_ctx][1024] int8, scale [max_ctx][32] fp16 (每 32 维一个)
};

Gguf::File G_file;
uintptr_t G_hbase = 0; // GGUF 在 host 内存的起始地址 (上传后 host 内存已释放, 只用来算偏移)
CUdeviceptr G_dbase = 0;
Layer G_layers[N_LAYER];
Weight G_tok_embd, G_output;
CUdeviceptr G_output_norm = 0;
int G_max_ctx = 0, G_n_past = 0, G_next_pos = 0;
bool G_inited = false;
uint64_t G_rng = 0x853c49e6748fea9bULL;
struct MmGroup { // 与 kernels.cu 的 MmGroup 布局一致
    uint64_t w[4], rb[4], y[4];
    int n[4], type[4];
    int cnt, glu;
    int xp; // x 是 xperm 布局 (decode, 见 kernels.cu)
};

CUfunction G_k_gemv, G_k_gemv_glu, G_k_gemv1, G_k_gemv1_glu, G_k_gemm[16], G_k_dq[16], G_k_embed[16], G_k_add_norm, G_k_silu_mul, G_k_dn_prep, G_k_dn_rec, G_k_dn_post, G_k_qk_rope, G_k_attn_pf, G_k_attn_dec, G_k_attn_merge, G_k_repack;
bool G_xperm = false; // 当前 forward 是 T = 1: GEMV 的输入 (xn/o/att/h) 是 xperm 布局
CUdeviceptr G_d_x, G_d_xn, G_d_out, G_d_qkv, G_d_z, G_d_a, G_d_b, G_d_o, G_d_qg, G_d_k, G_d_v, G_d_att, G_d_g, G_d_u, G_d_h; // matmul 的输入 xn/o/att/h 是 fp16, 其余 fp32
CUdeviceptr G_d_logits, G_d_hidden, G_d_tok, G_d_mask, G_d_embd, G_d_pos, G_d_inv_freq, G_d_conv_all, G_d_ssm_all, G_d_po, G_d_pml, G_d_cv, G_d_qk, G_d_of, G_d_np; // cv/qk/of: DeltaNet 预处理后的 qkv、q·k、未归一化输出 (fp32); np: n_past (int)
CUgraphExec G_graph = nullptr; // decode (T = 1) 的 CUDA Graph
float* G_logits = nullptr; // pinned host 内存
vector<float> G_hidden;
int G_pos[CHUNK * 3];
vector<float> G_work;
vector<int> G_idx;
vector<uint8_t> G_seen;

inline CUdeviceptr fp(CUdeviceptr base, size_t n_floats) { return base + n_floats * sizeof(float); } // 设备指针按 float 偏移

const Gguf::Tensor& find_tensor(const string& name) {
    auto it = G_file.tensors.find(name);
    if (it == G_file.tensors.end()) {
        fprintf(stderr, "❌ 缺少 tensor %s\n", name.c_str());
        _Exit(1);
    }
    return it->second;
}

CUdeviceptr dev_ptr(const Gguf::Tensor& t) { return G_dbase + ((uintptr_t)t.data - G_hbase); }

Weight get_w(const string& name, int k, int n) {
    const Gguf::Tensor& t = find_tensor(name);
    if (t.dims.size() != 2 || t.dims[0] != k || t.dims[1] != n) {
        fprintf(stderr, "❌ %s 形状不对, 期望 [%d, %d]\n", name.c_str(), k, n);
        _Exit(1);
    }
    if (t.type >= 16 || !G_k_gemm[t.type]) {
        fprintf(stderr, "❌ %s 的量化类型 %u 没有对应 kernel\n", name.c_str(), t.type);
        _Exit(1);
    }
    if ((t.type == Gguf::T_Q8_0 || t.type == Gguf::T_Q6_K) && k % 1024) { // 没有被 repack_weights 重排
        fprintf(stderr, "❌ %s: q8_0/q6_k 要求 k 是 1024 的倍数, 实际 %d\n", name.c_str(), k);
        _Exit(1);
    }
    Weight w;
    w.data = dev_ptr(t);
    w.type = t.type;
    w.k = k;
    w.n = n;
    w.rb = Quants::row_bytes(t.type, k);
    return w;
}

// q8_0 / q6_k 的 k 字节映射, 与 kernels.cu 的 k_repack 一致: 重排后第 p 字节 = 原第 repack_src(p) 字节
uint64_t repack_src(uint32_t type, uint64_t p) {
    if (type == Gguf::T_Q8_0) {
        uint64_t g = p / 272, r = p % 272;
        return r < 256 ? (g * 8 + (r & 127) / 16) * 34 + 2 + (r / 128) * 16 + (r & 15) : (g * 8 + (r - 256) / 2) * 34 + (r & 1);
    }
    uint64_t g = p / 840, r = p % 840;
    return r < 832 ? (g * 4 + r / 208) * 210 + r % 208 : (g * 4 + (r - 832) / 2) * 210 + 208 + (r & 1);
}

void repack_weights() { // 上传后把所有 q8_0 / q6_k 矩阵 (k 是 1024 的倍数) 按行原地重排, 使 decode GEMV 对齐读取; 字节数不变
    for (auto& [name, t] : G_file.tensors) {
        if ((t.type != Gguf::T_Q8_0 && t.type != Gguf::T_Q6_K) || t.dims.size() < 2 || t.dims[0] % 1024) continue;
        int64_t rows = 1;
        for (size_t i = 1; i < t.dims.size(); i++) rows *= t.dims[i];
        uint64_t rb = Quants::row_bytes(t.type, (int)t.dims[0]);
        Cu::launch_s(G_k_repack, (unsigned)rows, 1, 256, (unsigned)rb, dev_ptr(t), rb, (int)t.type);
    }
}

void unrepack_row(uint32_t type, uint8_t* row, uint64_t rb) { // 自测用: 还原成 GGUF 布局再交给标量参考
    if (type != Gguf::T_Q8_0 && type != Gguf::T_Q6_K) return;
    vector<uint8_t> o(row, row + rb);
    for (uint64_t p = 0; p < rb; p++) row[repack_src(type, p)] = o[p];
}

CUdeviceptr get_f(const string& name, int64_t numel) {
    const Gguf::Tensor& t = find_tensor(name);
    int64_t cnt = 1;
    for (int64_t d : t.dims) cnt *= d;
    if (t.type != Gguf::T_F32 || cnt != numel) {
        fprintf(stderr, "❌ %s 需要 f32 且元素数 %ld, 实际 type=%u numel=%ld\n", name.c_str(), (long)numel, t.type, (long)cnt);
        _Exit(1);
    }
    return dev_ptr(t);
}

void check_kv(const char* key, double expect) {
    auto it = G_file.kv_num.find(key);
    if (it == G_file.kv_num.end() || std::fabs(it->second - expect) > 1e-6 * std::fabs(expect)) {
        fprintf(stderr, "❌ GGUF 参数 %s 不符合预期 %g\n", key, expect);
        _Exit(1);
    }
}

void gemv(const Weight* const* ws, const CUdeviceptr* ys, int cnt, bool glu, CUdeviceptr x, int T) { // decode (T < 16): 共享输入 x 的 cnt 个矩阵一次 launch; glu: ws[0]/ws[1] 是 gate/up, ys[0] = silu(gate)·up (fp16)
    if (ws[0]->k % 1024) { // gemv_row 每轮 4 个超块
        fprintf(stderr, "❌ decode GEMV 要求 k 是 1024 的倍数, 实际 %d\n", ws[0]->k);
        _Exit(1);
    }
    MmGroup g = {};
    int rows = 0;
    for (int i = 0; i < cnt; i++) {
        g.w[i] = ws[i]->data, g.rb[i] = ws[i]->rb, g.n[i] = ws[i]->n, g.type[i] = (int)ws[i]->type;
        g.y[i] = glu ? ys[0] : ys[i];
        rows += ws[i]->n;
    }
    g.cnt = glu ? 1 : cnt;
    g.glu = glu;
    g.xp = T == 1 && G_xperm;
    if (glu) rows = ws[0]->n;
    CUfunction f = T == 1 ? (glu ? G_k_gemv1_glu : G_k_gemv1) : (glu ? G_k_gemv_glu : G_k_gemv);
    Cu::launch(f, (rows + 1) / 2, 1, 64, g, ws[0]->k, x, T);
}

void matmul(const Weight& w, CUdeviceptr x, int T, CUdeviceptr y) { // token 数少 (decode) 用 GEMV, 否则用 tensor core GEMM
    if (T < 16) {
        const Weight* ws[1] = {&w};
        gemv(ws, &y, 1, false, x, T);
        return;
    }
    Cu::launch(G_k_gemm[w.type], (T + 63) / 64, (w.n + 63) / 64, 128, w.data, w.rb, w.k, w.n, x, T, y);
}

void add_norm(CUdeviceptr add, CUdeviceptr w, CUdeviceptr hidden, int T) { Cu::launch(G_k_add_norm, T, 1, 1024, G_d_x, add, w, G_d_xn, hidden, T); }

void forward(int T, CUdeviceptr d_mask, CUdeviceptr d_embd, bool need_logits) { // 一个 chunk 的全部 kernel (输入已在显存: token、位置、n_past); 只 launch, 不同步, 可被 CUDA Graph 捕获
    G_xperm = T == 1; // T = 1 时各 kernel 把 GEMV 输入写成 xperm 布局; T > 1 时只有末尾 lm_head 走 GEMV, 输入是正常布局
    Cu::launch(G_k_embed[G_tok_embd.type], T, 1, 256, G_tok_embd.data, G_tok_embd.rb, G_d_tok, d_mask, d_embd, G_d_x);
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l];
        add_norm(l ? G_d_out : 0, L.attn_norm, fp(G_d_hidden, (size_t)l * D), T); // 先加上一层 FFN 的残差, 同时保存上一层输出
        if (L.recr) {
            if (T < 16) {
                const Weight* ws[4] = {&L.qkv, &L.gate_z, &L.alpha, &L.beta};
                CUdeviceptr ys[4] = {G_d_qkv, G_d_z, G_d_a, G_d_b};
                gemv(ws, ys, 4, false, G_d_xn, T);
            } else {
                matmul(L.qkv, G_d_xn, T, G_d_qkv);
                matmul(L.gate_z, G_d_xn, T, G_d_z);
                matmul(L.alpha, G_d_xn, T, G_d_a);
                matmul(L.beta, G_d_xn, T, G_d_b);
            }
            Cu::launch(G_k_dn_prep, T, LK, 128, G_d_qkv, L.conv_state, L.conv_w, G_d_cv, G_d_a, G_d_b, L.ssm_a, L.dt_bias, G_d_qk, T);
            Cu::launch(G_k_dn_rec, LV, 4, 128, G_d_qkv, L.conv_state, G_d_cv, G_d_a, G_d_b, G_d_qk, L.ssm_state, G_d_of, T);
            Cu::launch(G_k_dn_post, T, LV, 128, G_d_of, G_d_z, L.ssm_norm, G_d_o);
            matmul(L.ssm_out, G_d_o, T, G_d_out);
        } else {
            if (T < 16) {
                const Weight* ws[3] = {&L.wq, &L.wk, &L.wv};
                CUdeviceptr ys[3] = {G_d_qg, G_d_k, G_d_v};
                gemv(ws, ys, 3, false, G_d_xn, T);
            } else {
                matmul(L.wq, G_d_xn, T, G_d_qg);
                matmul(L.wk, G_d_xn, T, G_d_k);
                matmul(L.wv, G_d_xn, T, G_d_v);
            }
            Cu::launch(G_k_qk_rope, T, N_HEAD + N_KV, 256, G_d_qg, G_d_k, G_d_v, L.q_norm, L.k_norm, G_d_pos, G_d_inv_freq, L.k_cache, L.v_cache, L.k_scale, L.v_scale, G_d_np);
            if (T == 1) { // decode: 沿序列切段 (段数在 kernel 里由 n_past 算出, 多余的 block 直接退出)
                Cu::launch(G_k_attn_dec, N_KV, ATT_SPLIT, 128, G_d_qg, L.k_cache, L.v_cache, L.k_scale, L.v_scale, G_d_po, G_d_pml, G_d_np);
                Cu::launch(G_k_attn_merge, N_HEAD, 1, 256, G_d_qg, G_d_po, G_d_pml, G_d_np, G_d_att);
            } else {
                Cu::launch_s(G_k_attn_pf, (T + 63) / 64, N_HEAD, 128, FA_SMEM, G_d_qg, L.k_cache, L.v_cache, L.k_scale, L.v_scale, G_d_att, G_n_past, T);
            }
            matmul(L.wo, G_d_att, T, G_d_out);
        }
        add_norm(G_d_out, L.post_norm, 0, T);
        if (T < 16) { // gate/up 一次 launch, 直接输出 silu(gate)·up
            const Weight* ws[2] = {&L.ffn_gate, &L.ffn_up};
            gemv(ws, &G_d_h, 2, true, G_d_xn, T);
        } else {
            matmul(L.ffn_gate, G_d_xn, T, G_d_g);
            matmul(L.ffn_up, G_d_xn, T, G_d_u);
            int n = T * FF;
            Cu::launch(G_k_silu_mul, (n + 255) / 256, 1, 256, G_d_g, G_d_u, G_d_h, n);
        }
        matmul(L.ffn_down, G_d_h, T, G_d_out);
    }
    add_norm(G_d_out, G_output_norm, fp(G_d_hidden, (size_t)N_LAYER * D), T);
    if (need_logits) matmul(G_output, G_d_xn + (size_t)(T - 1) * D * 2, 1, G_d_logits);
}

void eval_chunk(const int* tokens, int T, const float* embd, const uint8_t* mask, const int* pos3, bool need_logits) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 3; j++) G_pos[t * 3 + j] = pos3 ? pos3[t * 3 + j] : G_next_pos + t;
    Cu::h2d(G_d_tok, tokens, (size_t)T * 4);
    Cu::h2d(G_d_pos, G_pos, (size_t)T * 3 * 4);
    Cu::h2d(G_d_np, &G_n_past, 4);
    CUdeviceptr d_mask = 0, d_embd = 0;
    if (mask) {
        Cu::h2d(G_d_mask, mask, T);
        Cu::h2d(G_d_embd, embd, (size_t)T * D * 4);
        d_mask = G_d_mask;
        d_embd = G_d_embd;
    }
    if (T == 1 && !mask && need_logits) { // 普通 decode: 第一次捕获成 CUDA Graph, 之后直接重放 (kernel 参数都不随 token 变化)
        if (!G_graph) {
            CUgraph g;
            Cu::ck(cuStreamBeginCapture(Cu::G_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture");
            forward(T, 0, 0, true);
            Cu::ck(cuStreamEndCapture(Cu::G_stream, &g), "cuStreamEndCapture");
            Cu::ck(cuGraphInstantiate(&G_graph, g, 0), "cuGraphInstantiate");
            Cu::ck(cuGraphDestroy(g), "cuGraphDestroy");
        }
        Cu::ck(cuGraphLaunch(G_graph, Cu::G_stream), "cuGraphLaunch");
    } else {
        forward(T, d_mask, d_embd, need_logits);
    }
    int mx = G_next_pos - 1;
    for (int i = 0; i < T * 3; i++) mx = std::max(mx, G_pos[i]);
    G_next_pos = mx + 1; // 后续默认位置接在最大位置之后, 与 HF get_rope_index 一致
    G_n_past += T;
    if (need_logits) Cu::d2h(G_logits, G_d_logits, (size_t)VOCAB * 4);
}

uint64_t rng_next() { // splitmix64
    uint64_t z = (G_rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint16_t f2h_exact(float f) { // 只用于自测: f 必须能被 fp16 精确表示且不是非规格化数
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) == 0) return (uint16_t)(u >> 16);
    return (uint16_t)(((u >> 16) & 0x8000u) | ((((u >> 23) & 0xffu) - 112) << 10) | ((u >> 13) & 0x3ffu));
}

} // namespace

extern "C" {

int qwen_init(const char* path, int device, int max_ctx) {
    if (G_inited) {
        fprintf(stderr, "⚠️ 模型已经加载过\n");
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    Cu::init(device, qwen_fatbin);
    const char* tnames[] = {"f32", "", "", "", "", "", "", "", "q8_0", "", "", "", "q4_k", "q5_k", "q6_k"};
    for (int t = 0; t < 15; t++) {
        if (!tnames[t][0]) continue;
        G_k_gemm[t] = Cu::func(("k_gemm_" + string(tnames[t])).c_str());
        G_k_dq[t] = Cu::func(("k_dq_" + string(tnames[t])).c_str());
        G_k_embed[t] = Cu::func(("k_embed_" + string(tnames[t])).c_str());
    }
    G_k_gemv = Cu::func("k_gemv");
    G_k_gemv_glu = Cu::func("k_gemv_glu");
    G_k_gemv1 = Cu::func("k_gemv1");
    G_k_gemv1_glu = Cu::func("k_gemv1_glu");
    G_k_add_norm = Cu::func("k_add_norm");
    G_k_silu_mul = Cu::func("k_silu_mul");
    G_k_dn_prep = Cu::func("k_dn_prep");
    G_k_dn_rec = Cu::func("k_dn_rec");
    G_k_dn_post = Cu::func("k_dn_post");
    G_k_qk_rope = Cu::func("k_qk_rope");
    G_k_attn_pf = Cu::func("k_attn_pf");
    Cu::ck(cuFuncSetAttribute(G_k_attn_pf, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, FA_SMEM), "cuFuncSetAttribute");
    G_k_attn_dec = Cu::func("k_attn_dec");
    G_k_attn_merge = Cu::func("k_attn_merge");
    G_k_repack = Cu::func("k_repack");
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
    auto t1 = std::chrono::steady_clock::now();
    Cu::ck(cuMemAlloc(&G_dbase, G_file.size), "cuMemAlloc 权重");
    Cu::h2d(G_dbase, G_file.base, G_file.size); // 整个文件原样上传, 之后 repack_weights 只重排 q8_0 / q6_k, 其余按 GGUF 格式直接使用
    G_hbase = (uintptr_t)G_file.base;
    free((void*)G_file.base);
    G_file.base = nullptr;
    repack_weights();
    double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "ℹ️ 权重上传 %.2f GB, %.1fs\n", G_file.size / 1073741824.0, up);
    G_max_ctx = max_ctx;
    G_tok_embd = get_w("token_embd.weight", D, VOCAB);
    G_output = get_w("output.weight", D, VOCAB);
    G_output_norm = get_f("output_norm.weight", D);
    G_d_conv_all = Cu::alloc((size_t)N_LAYER * QKV_DIM * 3 * 4);
    G_d_ssm_all = Cu::alloc((size_t)N_LAYER * LV * LHD * LHD * 4);
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
            L.conv_state = fp(G_d_conv_all, (size_t)l * QKV_DIM * 3);
            L.ssm_state = fp(G_d_ssm_all, (size_t)l * LV * LHD * LHD);
            continue;
        }
        L.wq = get_w(p + "attn_q.weight", D, Q_DIM);
        L.wk = get_w(p + "attn_k.weight", D, KV_DIM);
        L.wv = get_w(p + "attn_v.weight", D, KV_DIM);
        L.wo = get_w(p + "attn_output.weight", ATT_DIM, D);
        L.q_norm = get_f(p + "attn_q_norm.weight", HD);
        L.k_norm = get_f(p + "attn_k_norm.weight", HD);
        L.k_cache = Cu::alloc((size_t)max_ctx * KV_DIM);
        L.v_cache = Cu::alloc((size_t)max_ctx * KV_DIM);
        L.k_scale = Cu::alloc((size_t)max_ctx * 32 * 2);
        L.v_scale = Cu::alloc((size_t)max_ctx * 32 * 2);
    }
    float inv_freq[ROT / 2];
    for (int i = 0; i < ROT / 2; i++) inv_freq[i] = 1.0f / powf(THETA, (float)(2 * i) / ROT); // 与 HF 一致用 fp32 计算
    G_d_inv_freq = Cu::alloc(sizeof(inv_freq));
    Cu::h2d(G_d_inv_freq, inv_freq, sizeof(inv_freq));
    G_d_x = Cu::alloc((size_t)CHUNK * D * 4);
    G_d_xn = Cu::alloc((size_t)CHUNK * D * 2);
    G_d_out = Cu::alloc((size_t)CHUNK * D * 4);
    G_d_qkv = Cu::alloc((size_t)CHUNK * QKV_DIM * 4);
    G_d_z = Cu::alloc((size_t)CHUNK * V_DIM * 4);
    G_d_a = Cu::alloc((size_t)CHUNK * LV * 4);
    G_d_b = Cu::alloc((size_t)CHUNK * LV * 4);
    G_d_o = Cu::alloc((size_t)CHUNK * V_DIM * 2);
    G_d_qg = Cu::alloc((size_t)CHUNK * Q_DIM * 4);
    G_d_k = Cu::alloc((size_t)CHUNK * KV_DIM * 4);
    G_d_v = Cu::alloc((size_t)CHUNK * KV_DIM * 4);
    G_d_att = Cu::alloc((size_t)CHUNK * ATT_DIM * 2);
    G_d_g = Cu::alloc((size_t)CHUNK * FF * 4);
    G_d_u = Cu::alloc((size_t)CHUNK * FF * 4);
    G_d_h = Cu::alloc((size_t)CHUNK * FF * 2);
    G_d_logits = Cu::alloc((size_t)VOCAB * 4);
    G_d_hidden = Cu::alloc((size_t)(N_LAYER + 1) * D * 4);
    G_d_tok = Cu::alloc((size_t)CHUNK * 4);
    G_d_mask = Cu::alloc(CHUNK);
    G_d_embd = Cu::alloc((size_t)CHUNK * D * 4);
    G_d_pos = Cu::alloc((size_t)CHUNK * 3 * 4);
    G_d_cv = Cu::alloc((size_t)CHUNK * QKV_DIM * 4);
    G_d_qk = Cu::alloc((size_t)CHUNK * LK * 4);
    G_d_of = Cu::alloc((size_t)CHUNK * V_DIM * 4);
    G_d_np = Cu::alloc(4);
    G_d_po = Cu::alloc((size_t)ATT_SPLIT * ATT_DIM * 4);
    G_d_pml = Cu::alloc((size_t)ATT_SPLIT * N_HEAD * 8);
    Cu::ck(cuMemAllocHost((void**)&G_logits, (size_t)VOCAB * 4), "cuMemAllocHost");
    memset(G_logits, 0, (size_t)VOCAB * 4);
    G_hidden.resize((size_t)(N_LAYER + 1) * D);
    G_work.resize(VOCAB);
    G_idx.resize(VOCAB);
    G_seen.resize(VOCAB);
    G_inited = true;
    size_t free_b = 0, total_b = 0;
    Cu::ck(cuStreamSynchronize(Cu::G_stream), "cuStreamSynchronize");
    Cu::ck(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo");
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "✅ 模型初始化完成, 最大上下文 %d, 显存已用 %.2f / %.2f GB, 耗时 %.1fs\n", max_ctx, (total_b - free_b) / 1073741824.0, total_b / 1073741824.0, sec);
    return 0;
}

int qwen_eval(const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3) { // embd/embd_mask/pos3 可为空, embd: [n][5120], pos3: [n][3] (t,h,w)
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
    Cu::bind();
    for (int i = 0; i < n; i += CHUNK) {
        int T = std::min(CHUNK, n - i);
        eval_chunk(tokens + i, T, embd ? embd + (size_t)i * D : nullptr, embd_mask ? embd_mask + i : nullptr, pos3 ? pos3 + i * 3 : nullptr, i + T == n);
    }
    return 0;
}

int qwen_sample(float temperature, int top_k, float top_p, float min_p, float presence_penalty, const int* hist, int n_hist) {
    float* l = G_logits; // 只读; 有 presence penalty 时才拷贝一份再改
    if (presence_penalty != 0 && hist) {
        l = G_work.data();
        memcpy(l, G_logits, VOCAB * sizeof(float));
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
    int n = (top_k > 0 && top_k < VOCAB) ? top_k : VOCAB;
    if (n <= 256) { // top_k 较小: 单遍扫描, 维护按 logit 降序的前 n 个 (多数元素只和当前第 n 大比较一次)
        int cnt = 0;
        for (int i = 0; i < VOCAB; i++) {
            float v = l[i];
            if (cnt == n && v <= l[idx[n - 1]]) continue;
            int j = cnt < n ? cnt++ : n - 1;
            for (; j > 0 && l[idx[j - 1]] < v; j--) idx[j] = idx[j - 1];
            idx[j] = i;
        }
    } else {
        for (int i = 0; i < VOCAB; i++) idx[i] = i;
        auto cmp = [&](int a, int b) { return l[a] > l[b]; };
        if (n < VOCAB) std::nth_element(idx, idx + n, idx + VOCAB, cmp);
        std::sort(idx, idx + n, cmp);
    }
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
    Cu::bind();
    Cu::ck(cuMemsetD8Async(G_d_conv_all, 0, (size_t)N_LAYER * QKV_DIM * 3 * 4, Cu::G_stream), "cuMemsetD8Async");
    Cu::ck(cuMemsetD8Async(G_d_ssm_all, 0, (size_t)N_LAYER * LV * LHD * LHD * 4, Cu::G_stream), "cuMemsetD8Async");
    G_n_past = 0;
    G_next_pos = 0;
}

int qwen_n_past() { return G_n_past; }
float* qwen_logits() { return G_logits; }

float* qwen_hidden() { // [65][5120], 0 是 embedding, i 是第 i 层输出, 都只保存最后一个 token
    Cu::bind();
    Cu::d2h(G_hidden.data(), G_d_hidden, G_hidden.size() * 4);
    return G_hidden.data();
}

int qwen_selftest() { // GPU 反量化 vs 标量参考, GPU matmul (T=13 走 GEMV, 覆盖 8 + 5 两组; T=100 走 tensor core GEMM, 覆盖不满 64 的 tile) vs double 逐行点积, 返回失败项数; 会覆盖激活缓冲
    Cu::bind();
    int fails = 0;
    const char* names[] = {"blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight", "blk.51.ffn_down.weight", "blk.0.ffn_up.weight", "blk.0.ssm_alpha.weight", "output.weight"};
    for (const char* name : names) {
        const Gguf::Tensor& t = find_tensor(name);
        int k = (int)t.dims[0], n = (int)t.dims[1];
        Weight w = get_w(name, k, n);
        vector<uint8_t> raw(w.rb * 8);
        vector<float> a((size_t)8 * k), b(k);
        int rows[8] = {0, 1, 2, 3, n - 4, n - 3, n - 2, n - 1};
        Cu::h2d(G_d_tok, rows, sizeof(rows));
        Cu::launch(G_k_dq[w.type], 8, 1, 256, w.data, w.rb, k, G_d_tok, G_d_out);
        Cu::d2h(a.data(), G_d_out, a.size() * 4);
        double max_diff = 0, max_abs = 0;
        for (int r = 0; r < 8; r++) {
            Cu::d2h(raw.data(), w.data + (size_t)rows[r] * w.rb, w.rb);
            unrepack_row(w.type, raw.data(), w.rb);
            Quants::Ref::dequant_row(w.type, raw.data(), b.data(), k);
            for (int i = 0; i < k; i++) {
                max_diff = std::max(max_diff, (double)std::fabs(a[(size_t)r * k + i] - b[i]));
                max_abs = std::max(max_abs, (double)std::fabs(b[i]));
            }
        }
        double max_rel[2] = {0, 0}; // T = 13, 100
        for (int ti = 0; ti < 2; ti++) {
            int T = ti ? 100 : 13;
            if ((size_t)T * n > (size_t)CHUNK * FF) continue; // 输出缓冲放不下 (lm_head 只测 T=13)
            vector<float> x((size_t)T * k), y((size_t)T * n);
            vector<uint16_t> xh(x.size());
            for (size_t i = 0; i < x.size(); i++) { // matmul 输入是 fp16, 取 [-1, 1] 内 1/1024 的整数倍, fp16 可精确表示
                x[i] = (float)((int)(rng_next() % 2049) - 1024) / 1024.0f;
                xh[i] = f2h_exact(x[i]);
            }
            Cu::h2d(G_d_g, xh.data(), xh.size() * 2);
            matmul(w, G_d_g, T, G_d_u);
            Cu::d2h(y.data(), G_d_u, y.size() * 4);
            for (int r = 0; r < n; r += std::max(1, n / 37)) {
                Cu::d2h(raw.data(), w.data + (size_t)r * w.rb, w.rb);
                unrepack_row(w.type, raw.data(), w.rb);
                Quants::Ref::dequant_row(w.type, raw.data(), b.data(), k);
                for (int tt = 0; tt < T; tt++) {
                    double ref = 0, mag = 0;
                    for (int i = 0; i < k; i++) {
                        ref += (double)b[i] * x[(size_t)tt * k + i];
                        mag += std::fabs((double)b[i] * x[(size_t)tt * k + i]);
                    }
                    max_rel[ti] = std::max(max_rel[ti], std::fabs(ref - y[(size_t)tt * n + r]) / (mag + 1e-30));
                }
            }
        }
        bool ok = max_diff <= 1e-6 * max_abs && max_rel[0] < 2e-4 && max_rel[1] < 2e-4; // q4_k/q5_k 在 GPU 上用 fma, 与标量参考差最后一位; GEMV 权重解成 fp16 + half2 部分和, 实测 2~5e-5
        fails += !ok;
        fprintf(stderr, "%s %s type=%u [%d, %d] 反量化最大差 %.3g (最大绝对值 %.3g), matmul 最大相对误差 GEMV %.3g / GEMM %.3g\n", ok ? "✅" : "❌", name, w.type, k, n, max_diff, max_abs, max_rel[0], max_rel[1]);
    }
    return fails;
}

} // extern "C"
