// Qwen3.8-27B (qwen35 架构) CUDA 推理, 输入 GGUF (Q4_K_M), 通过 extern "C" 给 python ctypes 调用
// host 端只用 CUDA driver API (libcuda.so.1), kernel 编译成 fatbin 嵌在 so 里; 接口与 qwen_cpu 一致 (qwen_init 第二个参数改为 GPU 编号)
// 另有引擎模式 (qwen_prefill / qwen_next / qwen_decode_loop): 4 个序列槽, prefill 按 FIFO 逐个算, decode 所有序列每步一起算, 两者同时运行时用两个 green context 分 SM
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
#include "cu.h"
#include "gguf.h"
#include "quants_ref.h"
#include "cache.h"

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
const int NSEQ = 4; // 序列槽数: 最多 4 个请求同时进行 (prefill + decode), 启动时全部分配
const int N_REC = 48; // DeltaNet 层数
const size_t SS_FLOATS = (size_t)LV * LHD * LHD, CS_FLOATS = (size_t)QKV_DIM * 3; // 每层 DeltaNet 状态 [48][128][128] / conv 状态 [10240][3]
const size_t SLOT_FLOATS = N_REC * (SS_FLOATS + CS_FLOATS); // 每槽状态连续存放: [48 层 DeltaNet][48 层 conv], 共 149.6MiB
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
int G_max_ctx = 0, G_dec_sm = 78; // G_max_ctx: 每槽上下文; G_dec_sm: prefill 和 decode 同时运行时 decode 分区的 SM 数
bool G_inited = false;
uint64_t G_rng = 0x853c49e6748fea9bULL;
struct MmGroup { // 与 kernels.cu 的 MmGroup 布局一致
    uint64_t w[4], rb[4], y[4];
    int n[4], type[4];
    int cnt, glu;
    int xp; // x 是 xperm 布局 (decode, 见 kernels.cu)
};

CUfunction G_k_gemv, G_k_gemv_glu, G_k_gemv1, G_k_gemv1_glu, G_k_gemm[16], G_k_dq[16], G_k_embed[16], G_k_add_norm, G_k_silu_mul, G_k_dn_prep, G_k_dn_rec, G_k_dn_post, G_k_qk_rope, G_k_attn_pf, G_k_attn_dec, G_k_attn_merge, G_k_repack;
CUfunction G_k_dn_prep_b, G_k_dn_rec_b, G_k_qk_rope_b, G_k_attn_dec_b, G_k_attn_merge_b;
CUfunction G_k_state_dec, G_k_state_zero;
bool G_xperm = false; // 当前 forward 是 T < 16: GEMV 的输入 (xn/o/att/h) 是 xperm 布局; 只在 prefill 线程和初始化时用
CUdeviceptr G_d_x, G_d_xn, G_d_out, G_d_qkv, G_d_z, G_d_a, G_d_b, G_d_o, G_d_qg, G_d_k, G_d_v, G_d_att, G_d_g, G_d_u, G_d_h; // matmul 的输入 xn/o/att/h 是 fp16, 其余 fp32
CUdeviceptr G_d_logits, G_d_hidden, G_d_tok, G_d_mask, G_d_embd, G_d_pos, G_d_inv_freq, G_d_state, G_d_po, G_d_pml, G_d_cv, G_d_qk, G_d_of, G_d_np; // cv/qk/of: DeltaNet 预处理后的 qkv、q·k、未归一化输出 (fp32); np: n_past (int)
// G_d_*: prefill 的激活缓冲 (CHUNK 个 token); G_d_state: 各槽状态 [NSEQ][SLOT_FLOATS]
struct DecBuf { // decode (每步 B <= NSEQ 个 token) 的激活缓冲, 与 prefill 分开 (两者同时运行)
    CUdeviceptr x, xn, out, qkv, z, a, b, o, qg, k, v, att, h, cv, qk, of, po, pml, logits, hidden;
    CUdeviceptr in; // int: tok [NSEQ], pos [NSEQ][3], np [NSEQ], slot [NSEQ]
} G_e;
CUgraphExec G_bgraph[2][NSEQ + 1] = {}; // 多序列 decode 的 CUDA Graph, [整卡 / 分区][B], 初始化时全部捕获
float* G_logits = nullptr; // pinned host 内存, prefill 的 logits [VOCAB]
float* G_dlogits = nullptr; // pinned, decode 的 logits [NSEQ][VOCAB]
float* G_cur_logits = nullptr; // 直接模式 (qwen_eval) 最近一次的 logits
CUdeviceptr G_cur_hidden = 0;
vector<float> G_hidden;
int G_pos[CHUNK * 3];

// 引擎模式: python 每个请求一个线程调用 qwen_prefill (按 ticket FIFO, 同时只有一个在算) + qwen_next, 另一个 python 线程跑 qwen_decode_loop
// 等待都用 futex 直接 syscall; 并发路径上不分配内存 (musl 没有自己创建的线程时 malloc 不加锁, 这里的线程都是 python 的)
struct Slot {
    int n_past = 0, next_pos = 0; // 同一时刻只有 prefill 线程或 decode 线程在用
    std::atomic<int> state{0}; // 1: decode 中, 由 decode 循环推进
    std::atomic<int> cancel{0};
    std::atomic<int> n_out{0}, done{0}; // 已生成 token 数; done: 0 进行中, 1 遇到停止 token, 2 达到长度上限, 3 取消
    std::atomic<uint32_t> ver{0}; // 产生 token 或结束时 +1, qwen_next 在上面等
    int rd = 0; // qwen_next 已读到的位置
    int* out = nullptr; // 生成的 token [max_ctx], 也是 presence penalty 的历史
    int max_new = 0, n_stop = 0, stop[8] = {}, top_k = 0;
    float temp = 0, top_p = 1, min_p = 0, pp = 0;
    uint64_t rng = 0;
    float* work = nullptr; // 采样临时空间, 各 VOCAB 个
    int* idx = nullptr;
    uint8_t* seen = nullptr;
    double* p = nullptr;
    std::atomic<int> cpend{0}, cpf{0}; // 前缀缓存: 未完成的写入数 / 其中 prompt 的; qwen_next 在 cpf 为 0 之前不返回 token, cpend 为 0 之前不返回结束
    int c_final = 0; // 1: 已采样到 <|im_end|>, decode 循环再算一步把它写进 KV/状态, 然后生成输出缓存点
    Cache::Plan* cp = nullptr; // qwen_cache_init 之后才有
};
Slot G_slots[NSEQ]; // 直接模式 (qwen_eval / qwen_sample / qwen_reset) 用槽 0
std::atomic<uint32_t> G_pf_turn{0}; // 当前可以 prefill 的 ticket
std::atomic<uint32_t> G_dec_wake{0}; // 有槽进入 decode 时 +1
std::atomic<int> G_n_dec{0}, G_pf_busy{0}; // decode 中的槽数, 是否有 prefill 在算: 两者都有时各用自己的分区, 否则用整卡
int G_pf_slot = 0, G_pf_sid = 0; // 当前 prefill 的槽和 stream
CUstream G_dec_st = nullptr; // decode 线程最近一步用的 stream
char G_model_id[512] = {}; // GGUF 文件名 + 字节数, 写入缓存 metadata

long futex(void* addr, int op, uint32_t val) { // 直接 syscall, 不经过 libc (musl 的 errno 在 glibc 线程里不可用)
    long r;
    register void* tmo __asm__("r10") = nullptr;
    __asm__ volatile("syscall" : "=a"(r) : "0"(202L), "D"(addr), "S"((long)op), "d"((long)val), "r"(tmo) : "rcx", "r11", "memory");
    return r;
}
void fwait(std::atomic<uint32_t>& a, uint32_t old) { futex(&a, 128, old); } // FUTEX_WAIT_PRIVATE: 值仍是 old 才睡眠, 调用方循环检查
void fwake(std::atomic<uint32_t>& a) { futex(&a, 129, 0x7fffffff); } // FUTEX_WAKE_PRIVATE

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
    g.xp = T < 16 && G_xperm;
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

Layer seq_layer(int l, int sq) { // 第 l 层, 状态和 KV 指向槽 sq
    Layer L = G_layers[l];
    if (L.recr) {
        L.conv_state = fp(L.conv_state, sq * SLOT_FLOATS);
        L.ssm_state = fp(L.ssm_state, sq * SLOT_FLOATS);
    } else {
        size_t r = (size_t)sq * G_max_ctx;
        L.k_cache += r * KV_DIM, L.v_cache += r * KV_DIM, L.k_scale += r * 64, L.v_scale += r * 64;
    }
    return L;
}

void add_norm(CUdeviceptr add, CUdeviceptr w, CUdeviceptr hidden, int T) { Cu::launch(G_k_add_norm, T, 1, 1024, G_d_x, add, w, G_d_xn, hidden, T); }

void forward(int T, CUdeviceptr d_mask, CUdeviceptr d_embd, bool need_logits) { // prefill 槽 G_pf_slot 的一个 chunk 的全部 kernel (输入已在显存: token、位置、n_past); 只 launch, 不同步
    G_xperm = T < 16; // T < 16 时各 kernel 把 GEMV 输入写成 xperm 布局; 否则只有末尾 lm_head 走 GEMV, 输入是正常布局
    Cu::launch(G_k_embed[G_tok_embd.type], T, 1, 256, G_tok_embd.data, G_tok_embd.rb, G_d_tok, d_mask, d_embd, G_d_x);
    for (int l = 0; l < N_LAYER; l++) {
        Layer L = seq_layer(l, G_pf_slot);
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
                Cu::launch_s(G_k_attn_pf, (T + 63) / 64, N_HEAD, 128, FA_SMEM, G_d_qg, L.k_cache, L.v_cache, L.k_scale, L.v_scale, G_d_att, G_slots[G_pf_slot].n_past, T);
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

void forward_batch(int B) { // 多序列 decode: B 个槽各 1 个 token (输入在 G_e.in), logits [B][VOCAB]; 只在初始化时被捕获成 CUDA Graph
    G_xperm = true;
    uint64_t ctx = (uint64_t)G_max_ctx, ss = SLOT_FLOATS;
    CUdeviceptr tok = G_e.in, pos = G_e.in + NSEQ * 4, np = G_e.in + NSEQ * 16, slot = G_e.in + NSEQ * 20;
    Cu::launch(G_k_embed[G_tok_embd.type], B, 1, 256, G_tok_embd.data, G_tok_embd.rb, tok, (CUdeviceptr)0, (CUdeviceptr)0, G_e.x);
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = G_layers[l]; // 槽 0 的指针, 批量 kernel 按 slot 偏移
        Cu::launch(G_k_add_norm, B, 1, 1024, G_e.x, l ? G_e.out : (CUdeviceptr)0, L.attn_norm, G_e.xn, fp(G_e.hidden, (size_t)l * D), B);
        if (L.recr) {
            const Weight* ws[4] = {&L.qkv, &L.gate_z, &L.alpha, &L.beta};
            CUdeviceptr ys[4] = {G_e.qkv, G_e.z, G_e.a, G_e.b};
            gemv(ws, ys, 4, false, G_e.xn, B);
            Cu::launch(G_k_dn_prep_b, B, LK, 128, G_e.qkv, L.conv_state, L.conv_w, G_e.cv, G_e.a, G_e.b, L.ssm_a, L.dt_bias, G_e.qk, slot, ss);
            Cu::launch3(G_k_dn_rec_b, LV, 4, B, 128, G_e.qkv, L.conv_state, G_e.cv, G_e.a, G_e.b, G_e.qk, L.ssm_state, G_e.of, slot, ss);
            Cu::launch(G_k_dn_post, B, LV, 128, G_e.of, G_e.z, L.ssm_norm, G_e.o);
            matmul(L.ssm_out, G_e.o, B, G_e.out);
        } else {
            const Weight* ws[3] = {&L.wq, &L.wk, &L.wv};
            CUdeviceptr ys[3] = {G_e.qg, G_e.k, G_e.v};
            gemv(ws, ys, 3, false, G_e.xn, B);
            Cu::launch(G_k_qk_rope_b, B, N_HEAD + N_KV, 256, G_e.qg, G_e.k, G_e.v, L.q_norm, L.k_norm, pos, G_d_inv_freq, L.k_cache, L.v_cache, L.k_scale, L.v_scale, np, slot, ctx);
            Cu::launch3(G_k_attn_dec_b, N_KV, ATT_SPLIT, B, 128, G_e.qg, L.k_cache, L.v_cache, L.k_scale, L.v_scale, G_e.po, G_e.pml, np, slot, ctx);
            Cu::launch(G_k_attn_merge_b, N_HEAD, B, 256, G_e.qg, G_e.po, G_e.pml, np, G_e.att);
            matmul(L.wo, G_e.att, B, G_e.out);
        }
        Cu::launch(G_k_add_norm, B, 1, 1024, G_e.x, G_e.out, L.post_norm, G_e.xn, (CUdeviceptr)0, B);
        const Weight* wg[2] = {&L.ffn_gate, &L.ffn_up};
        gemv(wg, &G_e.h, 2, true, G_e.xn, B);
        matmul(L.ffn_down, G_e.h, B, G_e.out);
    }
    Cu::launch(G_k_add_norm, B, 1, 1024, G_e.x, G_e.out, G_output_norm, G_e.xn, fp(G_e.hidden, (size_t)N_LAYER * D), B);
    matmul(G_output, G_e.xn, B, G_e.logits);
}

void decode_step(const int* slots, const int* toks, int B) { // 槽 slots[b] 输入 toks[b], 一起算一步 (重放 CUDA Graph), logits 到 G_dlogits [B][VOCAB]; 有 prefill 在算时用 decode 分区, 否则整卡
    int part = G_pf_busy.load() ? 1 : 0, sid = part ? Cu::ST_DEC_PART : Cu::ST_DEC;
    CUstream st = Cu::G_sts[sid];
    G_dec_st = st;
    Cu::ck(cuCtxSetCurrent(Cu::G_ctxs[sid]), "cuCtxSetCurrent");
    int in[NSEQ * 6] = {};
    for (int b = 0; b < B; b++) {
        Slot& S = G_slots[slots[b]];
        in[b] = toks[b];
        in[NSEQ + b * 3] = in[NSEQ + b * 3 + 1] = in[NSEQ + b * 3 + 2] = S.next_pos;
        in[NSEQ * 4 + b] = S.n_past;
        in[NSEQ * 5 + b] = slots[b];
    }
    Cu::h2d(G_e.in, in, sizeof(in), st);
    Cu::ck(cuGraphLaunch(G_bgraph[part][B], st), "cuGraphLaunch");
    for (int b = 0; b < B; b++) G_slots[slots[b]].n_past++, G_slots[slots[b]].next_pos++;
    Cu::d2h(G_dlogits, G_e.logits, (size_t)B * VOCAB * 4, st);
}

void eval_chunk(const int* tokens, int T, const float* embd, const uint8_t* mask, const int* pos3, bool need_logits) { // prefill 槽 G_pf_slot 的一个 chunk, 走当前 stream; embd: 本 chunk 第一个图片行 (只含 mask 为 1 的行)
    Slot& S = G_slots[G_pf_slot];
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 3; j++) G_pos[t * 3 + j] = pos3 ? pos3[t * 3 + j] : S.next_pos + t;
    Cu::h2d(G_d_tok, tokens, (size_t)T * 4);
    Cu::h2d(G_d_pos, G_pos, (size_t)T * 3 * 4);
    Cu::h2d(G_d_np, &S.n_past, 4);
    CUdeviceptr d_mask = 0, d_embd = 0;
    if (mask) {
        Cu::h2d(G_d_mask, mask, T);
        for (int t = 0, j = 0; t < T;) { // 连续的图片行一次拷到 G_d_embd ([T][5120]) 的对应行, k_embed 按 mask 读第 t 行
            if (!mask[t]) {
                t++;
                continue;
            }
            int t1 = t;
            while (t1 < T && mask[t1]) t1++;
            Cu::h2d(G_d_embd + (size_t)t * D * 4, embd + (size_t)j * D, (size_t)(t1 - t) * D * 4);
            j += t1 - t;
            t = t1;
        }
        d_mask = G_d_mask;
        d_embd = G_d_embd;
    }
    forward(T, d_mask, d_embd, need_logits);
    int mx = S.next_pos - 1;
    for (int i = 0; i < T * 3; i++) mx = std::max(mx, G_pos[i]);
    S.next_pos = mx + 1; // 后续默认位置接在最大位置之后, 与 HF get_rope_index 一致
    S.n_past += T;
    if (need_logits) Cu::d2h(G_logits, G_d_logits, (size_t)VOCAB * 4);
}

void reset_slot(int sq) { // 清空槽 sq 的 DeltaNet / conv 状态 (当前 stream, 异步); KV 不用清
    Cu::ck(cuMemsetD8Async(fp(G_d_state, sq * SLOT_FLOATS), 0, SLOT_FLOATS * 4, Cu::G_stream), "cuMemsetD8Async");
    G_slots[sq].n_past = G_slots[sq].next_pos = 0;
}

// ---------------- 前缀缓存 (文件格式和线程见 cache.h) ----------------

const int IM_END = 248046;
static_assert(KV_DIM == Cache::KV_ROW && N_LAYER / 4 == Cache::N_ATT, "KV 布局与 cache.h 不一致");
char* G_mbuf = nullptr; // prefill 线程读 metadata 的缓冲
int* G_segbuf = nullptr;
uint8_t* G_lbuf = nullptr; // prefill 线程读缓存文件的缓冲 (按需扩大)
size_t G_lcap = 0;

void kv_d2h(int sq, int a, int b, uint8_t* dst, CUstream st) { // 槽 sq 的 KV 行 [a, b) 拷到 dst (布局见 Cache::gather_seg), 返回时已完成
    size_t n = b - a, r = (size_t)sq * G_max_ctx + a, w[4] = {KV_DIM, KV_DIM, 64, 64};
    for (int l = 3; l < N_LAYER; l += 4) { // 全注意力层
        const Layer& L = G_layers[l];
        CUdeviceptr src[4] = {L.k_cache + r * KV_DIM, L.v_cache + r * KV_DIM, L.k_scale + r * 64, L.v_scale + r * 64};
        for (int k = 0; k < 4; k++) {
            Cu::ck(cuMemcpyDtoHAsync(dst, src[k], n * w[k], st), "cuMemcpyDtoHAsync");
            dst += n * w[k];
        }
    }
    Cu::ck(cuStreamSynchronize(st), "cuStreamSynchronize");
}

void kv_h2d(int sq, int a, int n, const uint8_t* src) { // kvcache.bin 的一段 (n 个 token) 上传到槽 sq 的 KV 行 [a, a + n), 当前 stream
    size_t r = (size_t)sq * G_max_ctx + a, w[4] = {KV_DIM, KV_DIM, 64, 64};
    for (int l = 3; l < N_LAYER; l += 4) {
        const Layer& L = G_layers[l];
        CUdeviceptr dst[4] = {L.k_cache + r * KV_DIM, L.v_cache + r * KV_DIM, L.k_scale + r * 64, L.v_scale + r * 64};
        for (int k = 0; k < 4; k++) {
            Cu::h2d(dst[k], src, n * w[k]);
            src += n * w[k];
        }
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

int64_t read_file(const char* path, bool direct) { // 整个文件读进 G_lbuf, 返回字节数, 失败 -1
    int fd = Sys::open(path, Sys::O_RD | (direct ? Sys::O_DIRECT_ : 0));
    if (fd < 0) return -1;
    int64_t sz = Sys::fsize(fd);
    bool ok = sz > 0 && (!direct || sz % Cache::AL == 0) && Sys::read_all(fd, lbuf(sz), sz, 0);
    Sys::close(fd);
    return ok ? sz : -1;
}

bool load_state(int sq, const char* dir, size_t& bytes) { // 先看 state.bin.compressed (GPU 解码), 再看 state.bin; 量化线程可能刚好在替换, 最后再试一次压缩版
    char p[2400];
    CUdeviceptr out = fp(G_d_state, sq * SLOT_FLOATS);
    for (int k = 0; k < 3; k++) {
        bool comp = k != 1;
        snprintf(p, sizeof(p), comp ? "%s/state.bin.compressed" : "%s/state.bin", dir);
        int64_t sz = read_file(p, true);
        if (sz < 0) continue;
        bytes += sz;
        if (!comp) {
            if (sz != (int64_t)(SLOT_FLOATS * 4)) return false;
            Cu::h2d(out, G_lbuf, SLOT_FLOATS * 4);
            return true;
        }
        uint64_t hdr[2];
        memcpy(hdr, G_lbuf + 8, 16);
        if (memcmp(G_lbuf, "QSC1", 4) || hdr[0] != SLOT_FLOATS || (int64_t)Codec::comp_bytes(hdr[0], hdr[1]) != sz || (size_t)sz > (size_t)CHUNK * FF * 4) return false;
        Cu::h2d(G_d_g, G_lbuf, sz); // 借用 prefill 的激活缓冲 (之后的 chunk 才会用到)
        uint64_t n = hdr[0], nz = hdr[1];
        Cu::launch(G_k_state_dec, (unsigned)((n + 255) / 256), 1, 256, G_d_g + Codec::off_amax(), G_d_g + Codec::off_idx(n), G_d_g + Codec::off_code(n), n, out);
        if (nz) Cu::launch(G_k_state_zero, (unsigned)((nz + 255) / 256), 1, 256, G_d_g + Codec::off_zero(n), nz, out);
        return true;
    }
    return false;
}

Cache::Item* new_item(int kind, int sq) { // 计算线程: 新建一项, 该槽未完成写入数 +1
    Slot& S = G_slots[sq];
    Cache::Item* it = (Cache::Item*)Sys::amap(sizeof(Cache::Item)); // 已清零
    it->kind = kind, it->slot = sq, it->P = S.cp;
    it->pend = &S.cpend, it->ver = &S.ver;
    S.cpend.fetch_add(1);
    return it;
}

int cache_load(int sq, int n, const int* pos3) { // prefill 线程, 槽已清空: 查找 + 加载命中的缓存点, 决定之后哪些缓存点写 state; 返回命中位置 (token 数), 未命中 0
    Slot& S = G_slots[sq];
    Cache::Plan& P = *S.cp;
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
    for (int i = bad - 1; i >= 0 && hit < 0; i--) // 从后往前, 最后一个 token 必须重算 (要它的 logits)
        if (P.st[i] && P.pos[i] < n) hit = i;
    size_t bytes = 0;
    for (int j = 0; j <= hit; j++) { // 依次读各缓存点的 kvcache.bin
        Cache::key_dir(P.key[j], dir);
        Cache::Meta m;
        snprintf(p, sizeof(p), "%s/kvcache.bin", dir);
        int64_t sz = Cache::meta_read(dir, m, G_segbuf, G_mbuf) ? read_file(p, true) : -1;
        size_t need = 0;
        for (int k = 0; k < m.n_segs; k++) need += Cache::seg_bytes(G_segbuf[k]);
        if (sz < 0 || (size_t)sz != need) {
            fprintf(stderr, "⚠️ 缓存点 %s 读取失败, 不使用缓存\n", dir);
            hit = -1;
            break;
        }
        bytes += sz;
        for (int k = 0, a = m.start, off = 0; k < m.n_segs; k++) {
            kv_h2d(sq, a, G_segbuf[k], G_lbuf + off);
            a += G_segbuf[k];
            off += Cache::seg_bytes(G_segbuf[k]);
        }
    }
    if (hit >= 0) {
        Cache::key_dir(P.key[hit], dir);
        if (!load_state(sq, dir, bytes)) {
            fprintf(stderr, "⚠️ 缓存点 %s 的 state 读取失败, 不使用缓存\n", dir);
            hit = -1;
        }
    }
    if (hit < 0) {
        reset_slot(sq); // 可能已经写了一部分
        bytes = 0;
    } else {
        for (int j = 0; j <= hit; j++) { // 命中: 更新所用缓存点 metadata 的 mtime (清理按它判断)
            Cache::key_dir(P.key[j], dir);
            snprintf(p, sizeof(p), "%s/metadata.txt", dir);
            Sys::touch(p);
        }
        S.n_past = P.pos[hit];
        S.next_pos = P.npos[hit];
    }
    int start = S.n_past, last = start; // last: 最后一个有 state 的位置
    for (int i = hit + 1; i < P.n_pts; i++) { // state 写入规则 (从前往后贪心): 所在 2048 块 (按最后一个 token) 内的第一个缓存点写; prompt 最后一个缓存点另外在距上一个有 state 的点 >= 512 时也写
        bool on_disk = P.exist[i] && P.st[i];
        bool first = !i || (P.pos[i - 1] - 1) / Cache::BLK != (P.pos[i] - 1) / Cache::BLK;
        P.st[i] = !P.exist[i] && (first || (i == P.n_pts - 1 && P.pos[i] - last >= Cache::STATE_MIN));
        if (P.st[i] || on_disk) last = P.pos[i];
    }
    for (int i = 0; i <= hit; i++) P.st[i] = 0;
    P.hit = hit, P.last_state = last, P.pf_next = hit + 1;
    for (int t = start, k = hit + 1, mx = S.next_pos - 1; t < n && k < P.n_pts; t++) { // 各缓存点之后的 next_pos, 与 eval_chunk 的算法一致
        mx = std::max(mx, pos3 ? std::max({pos3[t * 3], pos3[t * 3 + 1], pos3[t * 3 + 2]}) : S.next_pos + (t - start));
        while (k < P.n_pts && P.pos[k] == t + 1) P.npos[k++] = mx + 1;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (hit >= 0) fprintf(stderr, "ℹ️ 槽 %d 前缀缓存命中 %d / %d tokens (第 %d / %d 个缓存点), 读取 %.1fMB, %.3fs\n", sq, start, n, hit + 1, P.n_pts, bytes / 1048576.0, sec);
    else fprintf(stderr, "ℹ️ 槽 %d 前缀缓存未命中 (%d 个缓存点), %.3fs\n", sq, P.n_pts, sec);
    return start;
}

void pf_flush(int sq, int p1, bool state) { // prefill 线程: 缓存点 [pf_next, p1) 的 KV (以及 p1 - 1 的 state) 拷出, 交给写盘线程
    Slot& S = G_slots[sq];
    Cache::Plan& P = *S.cp;
    int p0 = P.pf_next;
    bool any = state;
    for (int j = p0; j < p1; j++) any |= !P.exist[j];
    P.pf_next = std::max(p0, p1);
    if (p1 <= p0 || !any) return;
    Cache::Item* it = new_item(Cache::K_PF, sq);
    it->p0 = p0, it->p1 = p1, it->a = p0 ? P.pos[p0 - 1] : 0, it->b = P.pos[p1 - 1];
    it->kv_cap = Cache::al((size_t)(it->b - it->a) * Cache::TOK_BYTES);
    it->kv = (uint8_t*)Sys::amap(it->kv_cap);
    kv_d2h(sq, it->a, it->b, it->kv, Cu::G_stream);
    if (state) {
        it->state = (uint8_t*)Sys::amap(SLOT_FLOATS * 4);
        Cu::d2h(it->state, fp(G_d_state, sq * SLOT_FLOATS), SLOT_FLOATS * 4);
    }
    it->pend_pf = &S.cpf;
    S.cpf.fetch_add(1);
    Cache::push(it);
}

void out_flush(int sq, bool fin) { // decode 线程: 输出的 KV 每 FLUSH 个 token 交给写盘线程一段; fin: 最后一段 + 输出缓存点 (key、state)
    Slot& S = G_slots[sq];
    Cache::Plan& P = *S.cp;
    int a = P.out_flushed, b = S.n_past;
    if (!fin && b - a < Cache::FLUSH) return;
    Cache::Item* it = new_item(fin ? Cache::K_FIN : Cache::K_SEG, sq);
    it->a = a, it->b = b, it->off = P.out_off;
    if (b > a) {
        it->kv_cap = Cache::al((size_t)(b - a) * Cache::TOK_BYTES);
        it->kv = (uint8_t*)Sys::amap(it->kv_cap);
        kv_d2h(sq, a, b, it->kv, G_dec_st);
        P.segs[P.n_segs++] = b - a;
        P.out_off += Cache::seg_bytes(b - a);
    }
    P.out_flushed = b;
    if (fin) {
        uint8_t h[32];
        Sha::Ctx c = P.out_sha; // 输出段 = prompt 末尾 (已输入) + 全部输出 token (含 <|im_end|>)
        Sha::update(c, S.out, (size_t)S.n_out.load() * 4);
        Sha::final(c, h);
        Sha::init(c);
        for (int i = 0; i < P.n_pts; i++) Sha::update(c, P.hash[i], 32);
        Sha::update(c, h, 32);
        Sha::final(c, it->key);
        it->npos = S.next_pos;
        bool first = !P.n_pts || (P.pos[P.n_pts - 1] - 1) / Cache::BLK != (b - 1) / Cache::BLK;
        if (first || b - P.last_state >= Cache::STATE_MIN) {
            it->state = (uint8_t*)Sys::amap(SLOT_FLOATS * 4);
            Cu::d2h(it->state, fp(G_d_state, sq * SLOT_FLOATS), SLOT_FLOATS * 4, G_dec_st);
        }
        P.out_on = 0;
    }
    Cache::push(it);
}

void out_abort(int sq) { // 输出没有以 <|im_end|> 结束 (长度上限 / 其他停止 token / 取消): 不生成输出缓存点, 删除已写的临时文件
    Cache::Plan* P = G_slots[sq].cp;
    if (!P || !P->out_on) return;
    P->out_on = 0;
    if (P->n_segs) Cache::push(new_item(Cache::K_ABORT, sq));
}

void prefill(int sq, const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3, bool fresh) { // 在调用线程上逐 chunk prefill 槽 sq, 返回时 G_logits 已就绪; fresh: 先清空槽 (此时若槽有缓存计划, 先查找加载)
    G_pf_busy.store(1);
    G_pf_slot = sq;
    Cache::Plan* P = fresh && G_slots[sq].cp && G_slots[sq].cp->n_pts >= 0 ? G_slots[sq].cp : nullptr;
    G_pf_sid = G_n_dec.load() ? Cu::ST_PF_PART : Cu::ST_PF;
    Cu::use(G_pf_sid);
    if (fresh) reset_slot(sq);
    int i = P ? cache_load(sq, n, pos3) : 0, r = 0, ck = P ? P->hit + 1 : 0; // r: 之前各 chunk 的图片行数; ck: 下一个要写 state 的缓存点
    for (int t = 0; embd_mask && t < i; t++) r += embd_mask[t];
    while (i < n) {
        int sid = G_n_dec.load() ? Cu::ST_PF_PART : Cu::ST_PF; // 每个 chunk 重新选: 有 decode 在跑时只用 prefill 分区
        if (sid != G_pf_sid) { // 换 stream 前等上一个 chunk 算完
            Cu::ck(cuStreamSynchronize(Cu::G_sts[G_pf_sid]), "cuStreamSynchronize");
            Cu::use(sid);
            G_pf_sid = sid;
        }
        int T = std::min(CHUNK, n - i);
        while (P && ck < P->n_pts && !P->st[ck]) ck++;
        if (P && ck < P->n_pts) T = std::min(T, P->pos[ck] - i); // chunk 在要写 state 的缓存点处切开
        eval_chunk(tokens + i, T, embd ? embd + (size_t)r * D : nullptr, embd_mask ? embd_mask + i : nullptr, pos3 ? pos3 + i * 3 : nullptr, i + T == n);
        for (int t = i; embd_mask && t < i + T; t++) r += embd_mask[t];
        i += T;
        if (P && ck < P->n_pts && P->pos[ck] == i) pf_flush(sq, ++ck, true);
    }
    if (P) pf_flush(sq, P->n_pts, false);
    G_pf_busy.store(0);
}

int check_tokens(const int* tokens, int n, const uint8_t* embd_mask) {
    for (int i = 0; i < n; i++) {
        if ((!embd_mask || !embd_mask[i]) && (tokens[i] < 0 || tokens[i] >= VOCAB)) {
            fprintf(stderr, "❌ token id 越界: %d\n", tokens[i]);
            return -3;
        }
    }
    return 0;
}

uint64_t rng_next(uint64_t& st) { // splitmix64
    uint64_t z = (st += 0x9e3779b97f4a7c15ULL);
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

void qwen_set_dec_sm(int n) { G_dec_sm = n; } // 在 qwen_init 之前调用: prefill 和 decode 同时运行时 decode 分区的 SM 数 (其余给 prefill)
int qwen_nseq() { return NSEQ; }

int qwen_init(const char* path, int device, int max_ctx) { // max_ctx: 每个槽的上下文, NSEQ 个槽的 KV 全部预先分配
    if (G_inited) {
        fprintf(stderr, "⚠️ 模型已经加载过\n");
        return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    Cu::init(device, qwen_fatbin, G_dec_sm);
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
    Cu::ck(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, FA_SMEM, (CUkernel)G_k_attn_pf, Cu::G_dev), "cuKernelSetAttribute");
    G_k_attn_dec = Cu::func("k_attn_dec");
    G_k_attn_merge = Cu::func("k_attn_merge");
    G_k_dn_prep_b = Cu::func("k_dn_prep_b");
    G_k_dn_rec_b = Cu::func("k_dn_rec_b");
    G_k_qk_rope_b = Cu::func("k_qk_rope_b");
    G_k_attn_dec_b = Cu::func("k_attn_dec_b");
    G_k_attn_merge_b = Cu::func("k_attn_merge_b");
    G_k_repack = Cu::func("k_repack");
    G_k_state_dec = Cu::func("k_state_dec");
    G_k_state_zero = Cu::func("k_state_zero");
    if (!Gguf::open(G_file, path)) return -1;
    snprintf(G_model_id, sizeof(G_model_id), "%s %zu", strrchr(path, '/') ? strrchr(path, '/') + 1 : path, G_file.size);
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
    G_d_state = Cu::alloc(NSEQ * SLOT_FLOATS * 4);
    for (int l = 0, rl = 0; l < N_LAYER; l++) { // rl: 第几个 DeltaNet 层
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
            L.ssm_state = fp(G_d_state, rl * SS_FLOATS); // 槽 0, 槽 s 再加 s * SLOT_FLOATS
            L.conv_state = fp(G_d_state, N_REC * SS_FLOATS + rl * CS_FLOATS);
            rl++;
            continue;
        }
        L.wq = get_w(p + "attn_q.weight", D, Q_DIM);
        L.wk = get_w(p + "attn_k.weight", D, KV_DIM);
        L.wv = get_w(p + "attn_v.weight", D, KV_DIM);
        L.wo = get_w(p + "attn_output.weight", ATT_DIM, D);
        L.q_norm = get_f(p + "attn_q_norm.weight", HD);
        L.k_norm = get_f(p + "attn_k_norm.weight", HD);
        L.k_cache = Cu::alloc((size_t)NSEQ * max_ctx * KV_DIM); // [槽][max_ctx][1024]
        L.v_cache = Cu::alloc((size_t)NSEQ * max_ctx * KV_DIM);
        L.k_scale = Cu::alloc((size_t)NSEQ * max_ctx * 32 * 2);
        L.v_scale = Cu::alloc((size_t)NSEQ * max_ctx * 32 * 2);
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
    G_e.x = Cu::alloc((size_t)NSEQ * D * 4);
    G_e.xn = Cu::alloc((size_t)NSEQ * D * 2);
    G_e.out = Cu::alloc((size_t)NSEQ * D * 4);
    G_e.qkv = Cu::alloc((size_t)NSEQ * QKV_DIM * 4);
    G_e.z = Cu::alloc((size_t)NSEQ * V_DIM * 4);
    G_e.a = Cu::alloc((size_t)NSEQ * LV * 4);
    G_e.b = Cu::alloc((size_t)NSEQ * LV * 4);
    G_e.o = Cu::alloc((size_t)NSEQ * V_DIM * 2);
    G_e.qg = Cu::alloc((size_t)NSEQ * Q_DIM * 4);
    G_e.k = Cu::alloc((size_t)NSEQ * KV_DIM * 4);
    G_e.v = Cu::alloc((size_t)NSEQ * KV_DIM * 4);
    G_e.att = Cu::alloc((size_t)NSEQ * ATT_DIM * 2);
    G_e.h = Cu::alloc((size_t)NSEQ * FF * 2);
    G_e.cv = Cu::alloc((size_t)NSEQ * QKV_DIM * 4);
    G_e.qk = Cu::alloc((size_t)NSEQ * LK * 4);
    G_e.of = Cu::alloc((size_t)NSEQ * V_DIM * 4);
    G_e.po = Cu::alloc((size_t)NSEQ * ATT_SPLIT * ATT_DIM * 4);
    G_e.pml = Cu::alloc((size_t)NSEQ * ATT_SPLIT * N_HEAD * 8);
    G_e.logits = Cu::alloc((size_t)NSEQ * VOCAB * 4);
    G_e.hidden = Cu::alloc((size_t)(N_LAYER + 1) * D * 4);
    G_e.in = Cu::alloc(NSEQ * 6 * 4);
    Cu::ck(cuMemHostAlloc((void**)&G_logits, (size_t)VOCAB * 4, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc"); // PORTABLE: 在 green context 的 stream 上也按 pinned 处理
    Cu::ck(cuMemHostAlloc((void**)&G_dlogits, (size_t)NSEQ * VOCAB * 4, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
    memset(G_logits, 0, (size_t)VOCAB * 4);
    memset(G_dlogits, 0, (size_t)NSEQ * VOCAB * 4);
    G_cur_logits = G_logits;
    G_cur_hidden = G_d_hidden;
    G_hidden.resize((size_t)(N_LAYER + 1) * D);
    for (Slot& S : G_slots) {
        S.out = new int[max_ctx];
        S.work = new float[VOCAB];
        S.idx = new int[VOCAB];
        S.seen = new uint8_t[VOCAB]();
        S.p = new double[VOCAB];
    }
    Cu::ck(cuStreamSynchronize(Cu::G_stream), "cuStreamSynchronize");
    for (int part = 0; part < 2; part++) { // decode 的 CUDA Graph: 整卡 / 分区 x B = 1..NSEQ, 全部在这里捕获 (之后 decode 线程只重放, 不 launch)
        Cu::use(part ? Cu::ST_DEC_PART : Cu::ST_DEC);
        for (int B = 1; B <= NSEQ; B++) {
            CUgraph g;
            Cu::ck(cuStreamBeginCapture(Cu::G_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture");
            forward_batch(B);
            Cu::ck(cuStreamEndCapture(Cu::G_stream, &g), "cuStreamEndCapture");
            Cu::ck(cuGraphInstantiate(&G_bgraph[part][B], g, 0), "cuGraphInstantiate");
            Cu::ck(cuGraphDestroy(g), "cuGraphDestroy");
        }
    }
    Cu::use(Cu::ST_PF);
    G_inited = true;
    size_t free_b = 0, total_b = 0;
    Cu::ck(cuStreamSynchronize(Cu::G_stream), "cuStreamSynchronize");
    Cu::ck(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo");
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "✅ 模型初始化完成, %d 个槽, 每槽最大上下文 %d, 显存已用 %.2f / %.2f GB, 耗时 %.1fs\n", NSEQ, max_ctx, (total_b - free_b) / 1073741824.0, total_b / 1073741824.0, sec);
    return 0;
}

int qwen_eval(const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3) { // 直接模式 (槽 0, 不能与引擎模式同时用); embd/embd_mask/pos3 可为空, embd: [embd_mask 中 1 的个数][5120] (按位置顺序), pos3: [n][3] (t,h,w)
    if (!G_inited || n <= 0) return -1;
    Slot& S = G_slots[0];
    if (S.n_past + n > G_max_ctx) {
        fprintf(stderr, "❌ 超出最大上下文 %d (已有 %d, 新增 %d)\n", G_max_ctx, S.n_past, n);
        return -2;
    }
    if (check_tokens(tokens, n, embd_mask)) return -3;
    if (S.cp) S.cp->n_pts = -1; // 直接模式不用前缀缓存
    if (n == 1 && !embd_mask && !pos3) { // 普通 decode: 走多序列 decode 的 CUDA Graph (B = 1)
        int sq = 0;
        decode_step(&sq, tokens, 1);
        G_cur_logits = G_dlogits;
        G_cur_hidden = G_e.hidden;
        return 0;
    }
    prefill(0, tokens, n, embd, embd_mask, pos3, false);
    G_cur_logits = G_logits;
    G_cur_hidden = G_d_hidden;
    return 0;
}

} // extern "C"

namespace {

int sample_impl(const float* logits, Slot& S, float temperature, int top_k, float top_p, float min_p, float presence_penalty, const int* hist, int n_hist, uint64_t& rng) { // 临时空间用槽 S 的
    const float* l = logits; // 只读; 有 presence penalty 时才拷贝一份再改
    if (presence_penalty != 0 && hist) {
        float* w = S.work;
        memcpy(w, logits, VOCAB * sizeof(float));
        l = w;
        for (int i = 0; i < n_hist; i++) {
            int t = hist[i];
            if (t < 0 || t >= VOCAB || S.seen[t]) continue;
            S.seen[t] = 1;
            w[t] -= presence_penalty;
        }
        for (int i = 0; i < n_hist; i++)
            if (hist[i] >= 0 && hist[i] < VOCAB) S.seen[hist[i]] = 0;
    }
    if (temperature <= 0) return (int)(std::max_element(l, l + VOCAB) - l);
    int* idx = S.idx;
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
    double* p = S.p;
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
    double r = (rng_next(rng) >> 11) * (1.0 / 9007199254740992.0) * tot;
    for (int i = 0; i < keep; i++) {
        r -= p[i];
        if (r <= 0) return idx[i];
    }
    return idx[keep - 1];
}

} // namespace

extern "C" {

int qwen_sample(float temperature, int top_k, float top_p, float min_p, float presence_penalty, const int* hist, int n_hist) { // 直接模式: 从最近一次 qwen_eval 的 logits 采样
    return sample_impl(G_cur_logits, G_slots[0], temperature, top_k, top_p, min_p, presence_penalty, hist, n_hist, G_rng);
}

void qwen_seed(uint64_t seed) { G_rng = seed; } // 直接模式

void qwen_reset() { // 直接模式: 清空槽 0
    Cu::use(Cu::ST_PF);
    reset_slot(0);
    Cu::ck(cuStreamSynchronize(Cu::G_stream), "cuStreamSynchronize"); // 之后的 decode 走另一个 stream
}

int qwen_n_past() { return G_slots[0].n_past; }
float* qwen_logits() { return G_cur_logits; }

float* qwen_hidden() { // [65][5120], 0 是 embedding, i 是第 i 层输出, 都只保存最后一个 token
    Cu::use(Cu::ST_PF);
    Cu::d2h(G_hidden.data(), G_cur_hidden, G_hidden.size() * 4);
    return G_hidden.data();
}

int qwen_prefill(int sq, int ticket, const int* tokens, int n, const float* embd, const uint8_t* embd_mask, const int* pos3, float temperature, int top_k, float top_p, float min_p, float presence_penalty, uint64_t seed, int max_new, const int* stop, int n_stop,
                 int n_pts, const int* pts_pos, const uint8_t* pts_hash, const uint8_t* pts_key, const int* pts_role, const uint8_t* tail, int n_tail, int tail_role) {
    // 引擎模式: 按 ticket 顺序 (FIFO, 从 0 开始连续) 等前面的输入算完, 在调用线程上 prefill 槽 sq, 采样出第一个 token 后交给 decode 循环; 0 成功
    // 失败也会让出 prefill (ticket 不能跳过); 成功后用 qwen_next 取 token
    // 前缀缓存 (n_pts < 0 或没有 qwen_cache_init 则不用): n_pts 个缓存点, 第 i 个 = 前 pts_pos[i] 个 token (递增), 该段 sha256 为 pts_hash[i], key 为 pts_key[i] (各 32 字节), 角色 token pts_role[i];
    // tail: 最后一个缓存点之后 prompt 的序列化 (n_tail 字节, 格式同 python 的 cache_plan), 输出以 <|im_end|> 结束时接上输出 token 算出输出缓存点的 key
    for (uint32_t t; (t = G_pf_turn.load()) != (uint32_t)ticket;) fwait(G_pf_turn, t);
    auto t0 = std::chrono::steady_clock::now();
    int ret = 0;
    if (!G_inited || sq < 0 || sq >= NSEQ || G_slots[sq].state.load() || n <= 0 || n >= G_max_ctx || max_new < 1 || n_stop < 0 || n_stop > 8) {
        fprintf(stderr, "❌ qwen_prefill 参数错误: 槽 %d, prompt %d (每槽上下文 %d), max_new %d, 停止 token %d 个\n", sq, n, G_max_ctx, max_new, n_stop);
        ret = -1;
    } else if (check_tokens(tokens, n, embd_mask)) {
        ret = -3;
    } else {
        Slot& S = G_slots[sq];
        Cache::Plan* P = S.cp;
        if (P) {
            P->n_pts = Cache::G_dir[0] ? n_pts : -1;
            if (n_pts > Cache::MAXP) {
                fprintf(stderr, "⚠️ 缓存点 %d 个超过上限 %d, 本请求不用前缀缓存\n", n_pts, Cache::MAXP);
                P->n_pts = -1;
            }
            for (int i = 0; i < P->n_pts; i++) {
                if (pts_pos[i] <= (i ? pts_pos[i - 1] : 0) || pts_pos[i] > n) {
                    fprintf(stderr, "⚠️ 缓存点位置不合法, 本请求不用前缀缓存\n");
                    P->n_pts = -1;
                    break;
                }
            }
            if (P->n_pts >= 0) {
                memcpy(P->pos, pts_pos, (size_t)P->n_pts * 4);
                memcpy(P->hash, pts_hash, (size_t)P->n_pts * 32);
                memcpy(P->key, pts_key, (size_t)P->n_pts * 32);
                memcpy(P->role, pts_role, (size_t)P->n_pts * 4);
                Sha::init(P->out_sha);
                Sha::update(P->out_sha, tail, n_tail);
                P->out_role = tail_role;
            }
            P->out_on = 0;
        }
        S.temp = temperature, S.top_k = top_k, S.top_p = top_p, S.min_p = min_p, S.pp = presence_penalty, S.rng = seed;
        S.max_new = max_new, S.n_stop = n_stop;
        for (int i = 0; i < n_stop; i++) S.stop[i] = stop[i];
        S.rd = 0;
        S.n_out.store(0);
        S.done.store(0);
        S.cancel.store(0);
        bool part = G_n_dec.load() > 0;
        prefill(sq, tokens, n, embd, embd_mask, pos3, true);
        if (P && P->n_pts >= 0) { // 输出缓存点的段从最后一个缓存点开始 (含 prompt 末尾)
            P->out_on = 1;
            P->out_start = P->out_flushed = P->n_pts ? P->pos[P->n_pts - 1] : 0;
            P->out_off = 0, P->n_segs = 0, P->out_id = Cache::G_out_seq.fetch_add(1);
        }
        int tok = sample_impl(G_logits, S, S.temp, S.top_k, S.top_p, S.min_p, S.pp, nullptr, 0, S.rng);
        S.out[0] = tok;
        S.n_out.store(1);
        int fin = 0;
        for (int i = 0; i < n_stop; i++) fin |= tok == stop[i];
        if (!fin && (max_new <= 1 || S.n_past >= G_max_ctx)) fin = 2;
        S.c_final = fin == 1 && tok == IM_END && P && P->out_on && S.n_past < G_max_ctx; // 第一个 token 就是 <|im_end|>: 交给 decode 循环算完它再结束
        if (S.c_final) fin = 0;
        if (fin) out_abort(sq);
        if (fin) {
            S.done.store(fin);
        } else {
            G_n_dec.fetch_add(1);
            S.state.store(1);
            G_dec_wake.fetch_add(1);
            fwake(G_dec_wake);
        }
        S.ver.fetch_add(1);
        fwake(S.ver);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int hit = P && P->n_pts >= 0 && P->hit >= 0 ? P->pos[P->hit] : 0;
        fprintf(stderr, "ℹ️ 槽 %d prefill %d tokens (缓存命中 %d, 计算 %d), %.2fs, %.1f tok/s (开始时%s)\n", sq, n, hit, n - hit, sec, (n - hit) / sec, part ? "有 decode 在跑, 用 prefill 分区" : "整卡");
    }
    G_pf_turn.fetch_add(1);
    fwake(G_pf_turn);
    return ret;
}

int qwen_next(int sq, int* buf, int cap) { // 引擎模式: 等槽 sq 的新 token, 最多拷 cap 个到 buf, 返回个数; 已结束且全部读完返回 0
    Slot& S = G_slots[sq];
    while (true) {
        uint32_t v = S.ver.load();
        int d = S.done.load(), n = S.n_out.load(); // 先读 done 再读 n_out: done 是在最后一个 token 之后写的
        if (n > S.rd && !S.cpf.load()) { // prompt 的缓存写完之后才给出 token
            int c = std::min(cap, n - S.rd);
            memcpy(buf, S.out + S.rd, (size_t)c * 4);
            S.rd += c;
            return c;
        }
        if (d && !S.cpend.load()) return 0; // 输出的缓存也写完才算结束
        fwait(S.ver, v);
    }
}

void qwen_cancel(int sq) { G_slots[sq].cancel.store(1); } // 引擎模式: 让 decode 循环结束槽 sq, 之后仍要用 qwen_next 读到 0 才能复用该槽

void qwen_decode_loop() { // 引擎模式: decode 循环, 由一个 python 线程调用, 不返回; 每步把所有 decode 中的槽各 1 个 token 一起算 (进度对齐)
    auto finish = [](Slot& S, int why) {
        out_abort((int)(&S - G_slots)); // 正常生成输出缓存点时已经清掉 out_on, 这里什么都不做
        S.state.store(0);
        G_n_dec.fetch_sub(1);
        S.done.store(why);
        S.ver.fetch_add(1);
        fwake(S.ver);
    };
    while (true) {
        uint32_t w = G_dec_wake.load();
        int slots[NSEQ], toks[NSEQ], B = 0;
        for (int s = 0; s < NSEQ; s++) {
            Slot& S = G_slots[s];
            if (!S.state.load()) continue;
            if (S.cancel.load()) {
                finish(S, 3);
                continue;
            }
            if (S.n_past >= G_max_ctx) {
                finish(S, 2);
                continue;
            }
            slots[B] = s;
            toks[B++] = S.out[S.n_out.load() - 1];
        }
        if (!B) {
            fwait(G_dec_wake, w);
            continue;
        }
        decode_step(slots, toks, B);
        for (int b = 0; b < B; b++) {
            Slot& S = G_slots[slots[b]];
            if (S.c_final) { // 刚把 <|im_end|> 算进 KV/状态
                S.c_final = 0;
                out_flush(slots[b], true);
                finish(S, 1);
                continue;
            }
            int n = S.n_out.load();
            int tok = sample_impl(G_dlogits + (size_t)b * VOCAB, S, S.temp, S.top_k, S.top_p, S.min_p, S.pp, S.out, n, S.rng);
            S.out[n] = tok;
            S.n_out.store(n + 1);
            bool stop = false;
            for (int i = 0; i < S.n_stop; i++) stop |= tok == S.stop[i];
            if (stop && tok == IM_END && S.cp && S.cp->out_on && S.n_past < G_max_ctx) S.c_final = 1; // 下一步算完 <|im_end|> 再结束
            else if (stop || n + 1 >= S.max_new) {
                finish(S, stop ? 1 : 2);
                continue;
            }
            if (S.cp && S.cp->out_on) out_flush(slots[b], false);
            S.ver.fetch_add(1);
            fwake(S.ver);
        }
    }
}

int qwen_cache_init(const char* dir, int quant_threads) { // 在 qwen_init 之后、引擎开始前调用一次: 启用前缀缓存, dir 由 python 检查并加锁, 其中 tmp/ 已清空; 0 成功
    if (!G_inited || !Cache::init(dir, G_model_id, SLOT_FLOATS * 4, G_max_ctx, quant_threads)) return -1;
    G_mbuf = (char*)Sys::amap(Cache::MBUF + 1);
    G_segbuf = (int*)Sys::amap((size_t)Cache::G_max_segs * 4);
    for (Slot& S : G_slots) {
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
        S.cp = P;
    }
    fprintf(stderr, "✅ 前缀缓存已启用: %s, 量化线程 %d\n", dir, quant_threads);
    return 0;
}

void qwen_sha256(const uint8_t* data, uint64_t n, uint8_t* out) { Sha::hash(data, n, out); } // 自测用

int64_t qwen_state_codec_test(const float* x, float* out_cpu, float* out_gpu, int n_threads) { // 自测用 (在 python 线程调用, 会建 musl 线程): 槽状态大小的 x 编码, 分别在 CPU / GPU 上解码; 返回压缩字节数
    size_t n = SLOT_FLOATS, cap = Codec::comp_bytes(n, n / 64);
    uint8_t* c = (uint8_t*)Sys::amap(cap);
    size_t bytes = Codec::encode(x, n, c, cap, n_threads);
    if (!bytes) return -1;
    Codec::decode_cpu(c, out_cpu);
    uint64_t hdr[2];
    memcpy(hdr, c + 8, 16);
    Cu::use(Cu::ST_PF);
    CUdeviceptr d_out = fp(G_d_state, 3 * SLOT_FLOATS); // 借用槽 3 的状态
    Cu::h2d(G_d_g, c, bytes);
    Cu::launch(G_k_state_dec, (unsigned)((n + 255) / 256), 1, 256, G_d_g + Codec::off_amax(), G_d_g + Codec::off_idx(n), G_d_g + Codec::off_code(n), (uint64_t)n, d_out);
    if (hdr[1]) Cu::launch(G_k_state_zero, (unsigned)((hdr[1] + 255) / 256), 1, 256, G_d_g + Codec::off_zero(n), (uint64_t)hdr[1], d_out);
    Cu::d2h(out_gpu, d_out, n * 4);
    Sys::unmap(c, cap);
    return (int64_t)bytes;
}

int qwen_selftest() { // GPU 反量化 vs 标量参考, GPU matmul (T=13 走 GEMV, 覆盖 8 + 5 两组; T=100 走 tensor core GEMM, 覆盖不满 64 的 tile) vs double 逐行点积, 返回失败项数; 会覆盖激活缓冲
    Cu::use(Cu::ST_PF);
    G_xperm = false; // 测试输入是正常布局
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
                x[i] = (float)((int)(rng_next(G_rng) % 2049) - 1024) / 1024.0f;
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
