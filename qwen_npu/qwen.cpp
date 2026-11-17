// Qwen3.8-27B NPU (Ascend 310P) 推理, 输入 RedHatAI/Qwen3.8-27B-INT4 的 model.safetensors
// (compressed-tensors int4 对称 group128, AWQ 平滑 + GPTQ), 通过 extern "C" 给 python ctypes 调用
// 主干 int4 重打包成 W4A8 布局: decode 走 u8xu8 整数 mmad (x 由 k_qx8 动态量化 int8), prefill 五平面 And 反量化 fp16 走 mmad
// lm_head/tok_embd 由 bf16 在 host 按 per-16 RTN 转 int8 (lm_head 走 I8F fp16 路径); alpha/beta (f32) 走 k_gemv_f32t
// HF -> 本实现的权重变换 (与 GGUF 转换一致, 见 docs/model.md 第 11 节): norm +1 (ssm_norm 除外)、
// ssm_a = -exp(A_log)、DeltaNet v 头 grouped -> tiled 重排 (第 j 个 v 头用第 j%16 个 qk 头)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
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
#include "aclrtlaunch_k_add.h"
#include "aclrtlaunch_k_silu_mul.h"
#include "aclrtlaunch_k_conv.h"
#include "aclrtlaunch_k_dn_pre.h"
#include "aclrtlaunch_k_deltanet.h"
#include "aclrtlaunch_k_dnc.h"
#include "aclrtlaunch_k_qk_rope.h"
#include "aclrtlaunch_k_attn.h"
#include "aclrtlaunch_k_attn_merge.h"

using std::string;
using std::vector;

namespace {

#ifndef QWEN_TP
#define QWEN_TP 1 // 张量并行芯片数, 须与 kernel 编译宏一致
#endif
const int NDEV = QWEN_TP;
const int N_LAYER = 64, D = 5120, FF = 17408, VOCAB = 248320;
const int N_HEAD = 24, N_KV = 4, HD = 256, ROT = 64;
const int LK = 16, LV = 48, LHD = 128;
const int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM;
const int Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD, ATT_DIM = N_HEAD * HD;
const int QKVZ = QKV_DIM + V_DIM, QKVC = Q_DIM + 2 * KV_DIM; // qkv+z / q+k+v 合并矩阵的每 token 输出宽度
// 每芯分片维度 (按 head 切分): DeltaNet 按 qk 头奇偶半切 (每芯 LK_TP 个 qk 头 + 对应 v 头, 局部 j%LK_TP 关系不变)
const int N_KV_TP = N_KV / NDEV, LK_TP = LK / NDEV, LV_TP = LV / NDEV;
const int KEY_TP = LK_TP * LHD, V_TP = LV_TP * LHD, QKV_TP = 2 * KEY_TP + V_TP, QKVZ_TP = QKV_TP + V_TP;
const int Q_TP = (N_HEAD / NDEV) * HD * 2, KV_TP = N_KV_TP * HD, QKVC_TP = Q_TP + 2 * KV_TP, ATT_TP = (N_HEAD / NDEV) * HD;
const int VOCAB_TP = VOCAB / NDEV;
const int FF_N[2] = {NDEV == 1 ? FF : 8704, 8704}; // FF 列切分对半 (k_qx8 已支持 KS 非 8 倍数, k 只需 %512)
const int FF_OFF[2] = {0, 8704};
const int CHUNK = 256; // prefill 每批处理的 token 数
const float THETA = 1e7f;
const int PAD = 256; // 每个 matmul 输出缓冲末尾的安全余量 (字节)
// decode host 推送的合并上传布局 (一次 h2d): [xrep 旋转副本 | xaux | xn f32], 偏移单位字节
const int XQ_AUX = 4 * D, XQ_XN = XQ_AUX + 3 * (D / 128) * 8 * 4, XQ_TOTAL = XQ_XN + D * 4;

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
    Weight ffn_gu, ffn_down; // ffn_gu = gate+up 行拼接 [2*FF_N][D] (省一次 gemm 启动), 输出行内 [g|u]
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
    void* kq = nullptr; // KV cache q8: u8 [N_KV][max_ctx][256] + scale half [N_KV][max_ctx][2][16]
    void* ksc = nullptr;
    void* vq = nullptr;
    void* vsc = nullptr;
};

struct SelfRef { string name; const Weight* w; int rows[8]; vector<float> ref; }; // 自测: 8 行反量化参考值 (重排后语义, [8][k])

St::File G_st;
// DeltaNet v 头重排 (HF grouped -> tiled, 同 GGUF 转换): tiled 头 j 对应 HF 头 (j%16)*3 + j/16
int G_perm48[LV];
struct Dev { // 每芯一份的全部状态 (权重分片 + 激活缓冲 + 重排映射)
    int idx = 0;
    Layer layers[N_LAYER];
    Weight tok_embd, output;
    void* output_norm = nullptr;
    void *d_x = nullptr, *d_xn = nullptr, *d_out = nullptr, *d_qkv = nullptr, *d_ab = nullptr, *d_bd = nullptr, *d_o = nullptr;
    void *d_qg = nullptr, *d_att = nullptr, *d_g = nullptr, *d_u = nullptr;
    void *d_logits = nullptr, *d_hidden = nullptr, *d_tok = nullptr, *d_cos = nullptr; // d_cos 前半 cos 后半 sin
    void* d_xp = nullptr; // k_pack_x 的 fp16 zZ 布局 x
    void *d_xrep = nullptr, *d_xaux = nullptr; // k_qx8 输出 (decode 的 u8 x 旋转副本 + 组 scale/修正量)
    void* d_xq = nullptr; // host 推送的合并缓冲 (XQ_* 布局)
    void *qx_rep = nullptr, *qx_aux = nullptr; // 当前已就绪的量化 x 所在 (d_xrep/d_xaux 或 d_xq 内)
    vector<float> x_res; // [D] decode 残差流 (两线程各算一份相同的, 免去等对方的 barrier)
    uint8_t* qx_pin = nullptr; // pinned, XQ_* 布局 (xrep + xaux + xn), 一次 h2d 上传
    float* xn_pin = nullptr; // = qx_pin + XQ_XN
    uint32_t ar_seq = 0; // all-reduce 奇偶计数 (两线程调用序列相同, 天然同步)
    void* d_attp = nullptr; // decode attention 两段部分结果 (O [2*N_KV_TP][16][256] + m/l 各 [..][16])
    void* d_peer = nullptr; // all-reduce 时对方芯片 partial 的落地缓冲
    const void* pack_src = nullptr; // k_pack_x 结果缓存 (同一 x 的多个 matmul 只打包一次)
    int pack_t = 0, pack_k = 0, pack_xs = 0;
    const void* qx_src = nullptr; // k_qx8 结果缓存 (decode 时同一 x 只量化一次)
    int qx_k = 0;
    void* d_flag = nullptr; // doorbell 标志位 (int32 槽, 各占 64B): [0] 门槛 seq, [16+16b] block b 完成 seq
    int32_t dn_seq = 0, db_seq = 0; // 完成 / 门槛序号 (单调递增, qwen_reset 清零)
    vector<int> vheads; // 本芯 v 头 -> 全局 tiled v 头
    vector<int> rows_qkv, rows_z; // in_proj_qkv / in_proj_z 的本地行 -> HF 行 (conv1d 通道同 rows_qkv)
    vector<int> kblk_out; // out_proj k 方向 128 块 -> HF k 块 (一块恰好一个 v 头, 与量化组对齐)
    vector<int> rows_q, rows_kv; // 全注意力 q_proj / k,v_proj 的本地行 -> HF 行
    vector<int> kblk_wo; // o_proj k 块映射
    vector<int> rows_ff, kblk_ff; // FFN gate/up 行 / down k 块映射
    vector<SelfRef> selfrefs;
};
Dev G_dev[QWEN_TP];
thread_local Dev* DV = &G_dev[0]; // 当前线程操作的芯片
int G_max_ctx = 0, G_n_past = 0, G_next_pos = 0;
bool G_inited = false;
uint64_t G_rng = 0x853c49e6748fea9bULL;
float G_inv_freq[ROT / 2];
float* G_logits = nullptr; // pinned host, 完整 VOCAB (各芯写各自的一半)
// NDEV>1 时 decode (T==1) 的公共计算搬到 CPU: 残差流 x 常驻 host, 片上不跑 k_add/add_norm/qx8/embedding 查行
// (prefill 不搬: host 每次要过 [256][5120] 几遍内存, 比片上慢, 而通信字节数两种方案相同)
float* G_ar_host[QWEN_TP][2]; // all-reduce 的 host 中转缓冲 (pinned, 每芯 × 奇偶双缓冲, 免去"对方读完才能覆盖"的 barrier)
vector<float> G_hnorm; // host 侧 norm 权重 [N_LAYER][2][D] (attn, post) + [D] (final)
const St::Tensor* G_embd_t = nullptr; // embedding bf16 tensor (NDEV>1 时 St 保持 mmap 打开)
bool G_use_db = false; // decode 汇合点用 doorbell 标志位轮询替代 sync (实测省 ~14us/往返); QWEN_NPU_NODB=1 退回 sync
bool G_hid_host = false; // 最近一个 chunk 的 hidden 在 host (G_hidden) 还是设备
vector<float> G_hidden;
vector<float> G_work;
vector<int> G_idx;
vector<uint8_t> G_seen;

void dev_enter(int di) {
    Acl::bind(di);
    G_dev[di].idx = di;
    DV = &G_dev[di];
}

template <typename F> void run_devs(F f) { // 每芯一个线程并发执行 f (芯 0 在当前线程)
    if (NDEV == 1) {
        dev_enter(0);
        f();
        return;
    }
    std::thread th([&] {
        dev_enter(1);
        f();
    });
    dev_enter(0);
    f();
    th.join();
}

std::atomic<int> G_bar_cnt{0}, G_bar_gen{0};
void dev_barrier() { // 两线程自旋 barrier (等待都在 us~ms 级)
    if (NDEV == 1) return;
    int g = G_bar_gen.load();
    if (G_bar_cnt.fetch_add(1) == NDEV - 1) {
        G_bar_cnt.store(0);
        G_bar_gen.fetch_add(1);
    } else
        while (G_bar_gen.load() == g) {}
}

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
    int k, kfull, n; // k = 本分片使用的 k 宽度, kfull = checkpoint 矩阵的完整 k (行跨步)
    const int* row_map;
    const int* kblk_map;
};

QSrc get_qsrc(const string& base, int k, int kfull, int n, int nfull, const int* row_map, const int* kblk_map) {
    const St::Tensor& tp = find_t(base + ".weight_packed");
    const St::Tensor& ts = find_t(base + ".weight_scale");
    const St::Tensor& th = find_t(base + ".weight_shape");
    check_shape(tp, base + ".weight_packed", St::DT_I32, {nfull, kfull / 8});
    check_shape(ts, base + ".weight_scale", St::DT_BF16, {nfull, kfull / 128});
    const int64_t* sh = (const int64_t*)th.data;
    if (th.dtype != St::DT_I64 || St::numel(th) != 2 || sh[0] != nfull || sh[1] != kfull) {
        fprintf(stderr, "❌ %s weight_shape 不符 (期望 [%d, %d])\n", base.c_str(), nfull, kfull);
        _Exit(1);
    }
    return {(const uint32_t*)tp.data, (const uint16_t*)ts.data, k, kfull, n, row_map, kblk_map};
}

// 解一行 int4 (按重排后的行号 r): nat 为 0..15 的 nibble 值 (q+8), s16 为 per-16 f32 scale (per-128 组复制)
void decode_q4_row(const QSrc& q, int r, int8_t* nat, float* s16) {
    int rr = q.row_map ? q.row_map[r] : r;
    const uint32_t* w = q.pk + (uint64_t)rr * (q.kfull / 8);
    const uint16_t* sp = q.sc + (uint64_t)rr * (q.kfull / 128);
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

struct QSeg { string base; int n; const int* row_map; int nfull = 0; }; // 行方向拼接的一段 (共享 k); nfull = checkpoint 完整行数 (0 表示 = n)

// 多段 int4 矩阵行拼接后重打包 (段内可带行重排): 合并后一次 matmul, 省启动和小矩阵的流水浪费
Weight get_w_q4_cat(const vector<QSeg>& segs, int k, const int* kblk_map = nullptr, int kfull = 0) {
    if (!kfull) kfull = k;
    vector<QSrc> qs;
    vector<int> soff{0};
    for (const QSeg& g : segs) {
        qs.push_back(get_qsrc(g.base, k, kfull, g.n, g.nfull ? g.nfull : g.n, g.row_map, kblk_map));
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

Weight get_w_q4(const string& base, int k, int n, const int* row_map = nullptr, const int* kblk_map = nullptr, int kfull = 0, int nfull = 0) {
    return get_w_q4_cat({{base, n, row_map, nfull}}, k, kblk_map, kfull);
}

inline void bf16_row_f32(const St::Tensor& t, int64_t r, int k, float* out) {
    const uint16_t* p = (const uint16_t*)t.data + (uint64_t)r * k;
    for (int i = 0; i < k; i++) out[i] = Repack::bf2f(p[i]);
}

Weight get_w_i8_bf16(const string& name, int k, int n, int nfull, int row_off) { // lm_head: bf16 -> per-16 RTN int8 -> I8F fractal (取 [row_off, row_off+n) 行)
    const St::Tensor& t = find_t(name);
    check_shape(t, name, St::DT_BF16, {nfull, k});
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
                    bf16_row_f32(t, (int64_t)row_off + c * nr + r0 + ni * 16 + rr, k, row.data());
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

void* get_norm(const string& name, int n, bool add1, float* keep = nullptr) { // HF norm 权重为零中心 w, 计算用 1+w (ssm_norm 除外); keep 非空时留一份 host 副本
    vector<float> v = load_f32(name, n);
    if (add1)
        for (float& x : v) x += 1.0f;
    if (keep) memcpy(keep, v.data(), (size_t)n * 4);
    return up_f32(v.data(), n);
}

Weight get_w_ab(const string& p) { // α+β 合并 [2*LV_TP][D] f32 (行按本芯 tiled 头序), 一次 GEMV 算完
    const St::Tensor& ta = find_t(p + "linear_attn.in_proj_a.weight");
    const St::Tensor& tb = find_t(p + "linear_attn.in_proj_b.weight");
    check_shape(ta, p + "in_proj_a", St::DT_BF16, {LV, D});
    check_shape(tb, p + "in_proj_b", St::DT_BF16, {LV, D});
    vector<float> o((size_t)2 * LV_TP * D);
    for (int j = 0; j < LV_TP; j++) {
        bf16_row_f32(ta, G_perm48[DV->vheads[j]], D, &o[(size_t)j * D]);
        bf16_row_f32(tb, G_perm48[DV->vheads[j]], D, &o[(size_t)(LV_TP + j) * D]);
    }
    Weight w;
    w.fmt = Repack::FMT_F32;
    w.k = D;
    w.n = 2 * LV_TP;
    w.stride = D * 4;
    w.data = Acl::alloc((uint64_t)2 * LV_TP * D * 4 + PAD);
    Acl::h2d(w.data, o.data(), (uint64_t)2 * LV_TP * D * 4);
    return w;
}

void save_ref(const string& name, const Weight* w, const vector<float>& ref) { // 自测参考: 首末各 4 行的反量化 f32
    SelfRef s;
    s.name = name;
    s.w = w;
    for (int i = 0; i < 8; i++) s.rows[i] = i < 4 ? i : w->n - 8 + i;
    s.ref = ref;
    DV->selfrefs.push_back(std::move(s));
}

void save_ref_q4(const string& name, const Weight* w, const vector<QSeg>& segs, const int* kblk_map, int kfull = 0) {
    vector<QSrc> qs;
    vector<int> soff{0};
    for (const QSeg& g : segs) {
        qs.push_back(get_qsrc(g.base, w->k, kfull ? kfull : w->k, g.n, g.nfull ? g.nfull : g.n, g.row_map, kblk_map));
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

void save_ref_i8(const string& name, const Weight* w, int row_off = 0) { // lm_head / tok_embd: 参考 = RTN 后的 s*q
    const St::Tensor& t = find_t(name);
    vector<float> ref((size_t)8 * w->k);
    vector<float> row(w->k);
    vector<int8_t> nat(w->k);
    vector<float> s(w->k / 16);
    for (int i = 0; i < 8; i++) {
        bf16_row_f32(t, (int64_t)row_off + (i < 4 ? i : w->n - 8 + i), w->k, row.data());
        Repack::quant_row_i8(row.data(), w->k, nat.data(), s.data());
        for (int c = 0; c < w->k; c++) ref[(size_t)i * w->k + c] = s[c / 16] * nat[c];
    }
    save_ref(name, w, ref);
}

void pack_invalidate() {
    DV->pack_src = nullptr;
    DV->qx_src = nullptr;
}

void seed_qx() { // hostx 预下发: d_xn 的量化版由 host 算好放 d_xq, 先于 host_norm_push 下发的 matmul 不得再启动 k_qx8
    DV->qx_src = DV->d_xn;
    DV->qx_k = D;
    DV->qx_rep = DV->d_xq;
    DV->qx_aux = (uint8_t*)DV->d_xq + XQ_AUX;
}

// QWEN_NPU_PROF=1 时逐 kernel 同步计时 (调试用, 会拖慢; 多芯时只统计芯 0)
const char* G_prof_names[] = {"gemm", "pack_x/qx8", "gemv_f32t", "conv", "deltanet", "qk_rope", "attn", "add_norm", "silu", "misc", "allreduce"};
double G_prof_ms[11];
bool G_prof_on = false;
std::chrono::steady_clock::time_point G_prof_t0;
void prof_mark(int id) { // 把距上个标记的耗时记到 id 上
    if (!G_prof_on || DV->idx != 0) return;
    Acl::sync();
    auto now = std::chrono::steady_clock::now();
    G_prof_ms[id] += std::chrono::duration<double>(now - G_prof_t0).count() * 1e3;
    G_prof_t0 = now;
}

void wait_done(int32_t seq) { // 轮询段尾 kernel 8 个 block 的完成标志 (一次 512B d2h), 绕开 sync 的完成上报链路
    int32_t s[128];
    while (true) {
        Acl::d2h(s, (uint8_t*)DV->d_flag + 64, 512);
        int ok = 1;
        for (int b = 0; b < 8; b++) ok &= s[b * 16] >= seq;
        if (ok) return;
    }
}

void allreduce_out(int T) { // 两芯 partial 求和 (P2P 不可用, 走 host 中转): d_out += 对方 d_out
    if (NDEV == 1) return;
    prof_mark(9);
    size_t bytes = (size_t)T * D * 4;
    int par = ++DV->ar_seq & 1; // 奇偶双缓冲: 对方最多落后一个汇合点, 免去"对方读完才能覆盖"的第二个 barrier
    Acl::sync(); // 本芯 partial 算完 (h2d/d2h 不进 stream 队列, 必须显式同步)
    Acl::d2h(G_ar_host[DV->idx][par], DV->d_out, bytes);
    dev_barrier(); // 两边 partial 都已在 host
    Acl::h2d(DV->d_peer, G_ar_host[1 - DV->idx][par], bytes);
    Acl::ck(ACLRT_LAUNCH_KERNEL(k_add)(8, Acl::stream(), DV->d_out, DV->d_peer, T * D), "launch add");
    prof_mark(10);
}

// 与 k_qx8 同算法/同布局的 CPU 版 (k < 16384, 单 half): rep = 4 份旋转副本, aux = [sx|tt|nc2] 各 [KS][8] 复制
void qx8_cpu(const float* x, int k, uint8_t* rep, float* aux) {
    int KS = k / 128;
    uint8_t xu[8704];
    for (int g = 0; g < KS; g++) {
        float amax = 1e-30f;
        for (int i = 0; i < 128; i++) amax = std::max(amax, std::fabs(x[g * 128 + i]));
        float sx = amax * (1.0f / 127.0f), inv = 127.0f / amax;
        float sum = 0; // 整数和 <= 255*128, f32 精确
        for (int i = 0; i < 128; i++) {
            int q = (int)lrintf(x[g * 128 + i] * inv) + 128;
            xu[g * 128 + i] = (uint8_t)q;
            sum += q;
        }
        float tt = (sum - 16384.0f) * -8.0f * sx; // -8*Sx*sx (运算顺序与 kernel 一致)
        for (int j = 0; j < 8; j++) {
            aux[g * 8 + j] = sx;
            aux[KS * 8 + g * 8 + j] = tt;
            aux[2 * KS * 8 + g * 8 + j] = -128.0f * sx;
        }
    }
    for (int c = 0; c < 4; c++) // 旋转副本: 组 g 的 128B 放在副本 g%4 的 512*(g/4) 处
        for (int j = 0; j < KS / 4; j++) memcpy(rep + (size_t)c * k + j * 512, &xu[(4 * j + c) * 128], 128);
}

void embd_row_cpu(int tok, float* out) { // bf16 行 -> per-16 RTN int8 -> s*q, 数值与 get_embd_i8 + k_dq_i8 一致
    const uint16_t* p = (const uint16_t*)G_embd_t->data + (uint64_t)tok * D;
    float row[D];
    int8_t nat[D];
    float s[D / 16];
    for (int i = 0; i < D; i++) row[i] = Repack::bf2f(p[i]);
    Repack::quant_row_i8(row, D, nat, s);
    for (int i = 0; i < D; i++) out[i] = s[i / 16] * nat[i];
}

bool G_avx512 = __builtin_cpu_supports("avx512f");

// AVX512 版 host 计算 (标量 lrintf 链太慢): x += p0+p1 (p0 空则跳过) 并返回均方; 乘法逐元素与标量版逐位同, 平方和求和顺序不同 (~1ulp)
__attribute__((target("avx512f"))) float resid_norm_avx512(float* x, const float* p0, const float* p1, int n) {
    if (p0)
        for (int i = 0; i < n; i += 16) {
            __m512 v = _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_add_ps(_mm512_loadu_ps(p0 + i), _mm512_loadu_ps(p1 + i)));
            _mm512_storeu_ps(x + i, v);
        }
    __m512d acc = _mm512_setzero_pd();
    for (int i = 0; i < n; i += 8) { // 平方和保持 double 累加
        __m512d d = _mm512_cvtps_pd(_mm256_loadu_ps(x + i));
        acc = _mm512_fmadd_pd(d, d, acc);
    }
    return (float)(_mm512_reduce_add_pd(acc) / n);
}

__attribute__((target("avx512f"))) void scale_mul_avx512(const float* x, float rs, const float* w, float* xn, int n) {
    __m512 vr = _mm512_set1_ps(rs);
    for (int i = 0; i < n; i += 16) _mm512_storeu_ps(xn + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), vr), _mm512_loadu_ps(w + i)));
}

// 与 qx8_cpu 逐字节一致的 AVX512 版 (cvtps_epi32 的 RINT 就是 ties-even, 同 lrintf; 整数和精确)
__attribute__((target("avx512f"))) void qx8_avx512(const float* x, int k, uint8_t* rep, float* aux) {
    int KS = k / 128;
    uint8_t xu[8704];
    for (int g = 0; g < KS; g++) {
        const float* xg = x + g * 128;
        __m512 vm = _mm512_setzero_ps();
        for (int j = 0; j < 128; j += 16) vm = _mm512_max_ps(vm, _mm512_abs_ps(_mm512_loadu_ps(xg + j)));
        float amax = _mm512_reduce_max_ps(vm);
        if (amax < 1e-30f) amax = 1e-30f;
        float sx = amax * (1.0f / 127.0f), inv = 127.0f / amax;
        __m512 vinv = _mm512_set1_ps(inv);
        __m512i vsum = _mm512_setzero_si512(), v128 = _mm512_set1_epi32(128);
        for (int j = 0; j < 128; j += 16) {
            __m512i q = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xg + j), vinv)), v128);
            vsum = _mm512_add_epi32(vsum, q);
            _mm_storeu_si128((__m128i*)(xu + g * 128 + j), _mm512_cvtepi32_epi8(q));
        }
        float tt = ((float)_mm512_reduce_add_epi32(vsum) - 16384.0f) * -8.0f * sx;
        for (int j = 0; j < 8; j++) {
            aux[g * 8 + j] = sx;
            aux[KS * 8 + g * 8 + j] = tt;
            aux[2 * KS * 8 + g * 8 + j] = -128.0f * sx;
        }
    }
    for (int c = 0; c < 4; c++) // 旋转副本: 组 g 的 128B 放在副本 g%4 的 512*(g/4) 处
        for (int j = 0; j < KS / 4; j++) memcpy(rep + (size_t)c * k + j * 512, &xu[(4 * j + c) * 128], 128);
}

// decode 的汇合点: 取回两芯 partial -> CPU 残差累加 + RMSNorm (+qx8) -> 下发 (need_f32: ab gemv / lm_head 要 f32 xn)
// 两线程各自算一份相同的结果 (输入相同, 逐位一致), 免去"等对方线程算完"的第二个 barrier
void host_norm_push(const float* w, float* hidden_slot, bool with_partial, bool need_f32, bool need_quant, int32_t ring = 0, int32_t wait_seq = -1) {
    prof_mark(9);
    const float *p0 = nullptr, *p1 = nullptr;
    if (with_partial) {
        int par = ++DV->ar_seq & 1; // 奇偶双缓冲: 对方最多落后一个汇合点, 不会读到被覆盖的 partial
        int32_t ws = wait_seq < 0 ? DV->dn_seq : wait_seq; // gated 时 push 在下一段 launch 之后执行, 等待序号须由调用方在 launch 前捕获
        if (ws) wait_done(ws); // 本芯 partial 算完 (段尾 matmul 写的完成标志)
        else Acl::sync();
        Acl::d2h(G_ar_host[DV->idx][par], DV->d_out, (size_t)D * 4);
        dev_barrier(); // 唯一 barrier: 两边 partial 都已就位
        p0 = G_ar_host[0][par];
        p1 = G_ar_host[1][par];
    }
    float* x = DV->x_res.data();
    float ms;
    if (G_avx512) ms = resid_norm_avx512(x, p0, p1, D);
    else {
        if (p0)
            for (int i = 0; i < D; i++) x[i] += p0[i] + p1[i];
        double ss = 0;
        for (int i = 0; i < D; i++) ss += (double)x[i] * x[i];
        ms = (float)(ss / D);
    }
    if (hidden_slot && DV->idx == 0) memcpy(hidden_slot, x, (size_t)D * 4);
    float rs = 1.0f / sqrtf(ms + 1e-6f);
    if (G_avx512) scale_mul_avx512(x, rs, w, DV->xn_pin, D);
    else
        for (int i = 0; i < D; i++) DV->xn_pin[i] = x[i] * rs * w[i];
    if (need_quant) {
        if (G_avx512) qx8_avx512(DV->xn_pin, D, DV->qx_pin, (float*)(DV->qx_pin + XQ_AUX));
        else qx8_cpu(DV->xn_pin, D, DV->qx_pin, (float*)(DV->qx_pin + XQ_AUX));
    }
    pack_invalidate();
    if (need_quant) { // 一次上传 [xrep|xaux(|xn)], 把 qx8 缓存标成已就绪 (matmul 以 d_xn 指针为键, w8d 不读 f32 本体)
        Acl::h2d(DV->d_xq, DV->qx_pin, (size_t)(need_f32 ? XQ_TOTAL : XQ_XN));
        DV->qx_src = DV->d_xn;
        DV->qx_k = D;
        DV->qx_rep = DV->d_xq;
        DV->qx_aux = (uint8_t*)DV->d_xq + XQ_AUX;
    } else if (need_f32)
        Acl::h2d((uint8_t*)DV->d_xq + XQ_XN, DV->xn_pin, (size_t)D * 4);
    if (ring) Acl::h2d(DV->d_flag, &ring, 4); // doorbell: 数据就绪, 放行预下发的段首 kernel
    prof_mark(10);
}

void matmul(const Weight& w, void* x, int T, void* y, int xs = 0, int32_t done_seq = 0, int32_t gate_seq = 0) { // x: [T][k] f32 (行跨步 xs 元素, 0 = 紧凑), y: [T][n] f32; done_seq/gate_seq 非 0 时段尾写完成标志/段首自旋等 doorbell
    if (!xs) xs = w.k;
    if (w.fmt == Repack::FMT_F32) { // alpha/beta: 行常驻向量 GEMV, 每核 8 行
        prof_mark(9);
        Acl::ck(ACLRT_LAUNCH_KERNEL(k_gemv_f32t)((uint32_t)(w.n / 8), Acl::stream(), w.data, x, y, w.n, w.k, T), "launch gemv_f32t");
        prof_mark(2);
        return;
    }
    void* xp = nullptr;
    if (T > 1) {
        if (x != DV->pack_src || T != DV->pack_t || w.k != DV->pack_k || xs != DV->pack_xs) {
            prof_mark(9);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_pack_x)(8, Acl::stream(), x, DV->d_xp, T, w.k, xs), "launch pack_x");
            prof_mark(1);
            DV->pack_src = x;
            DV->pack_t = T;
            DV->pack_k = w.k;
            DV->pack_xs = xs;
        }
        xp = DV->d_xp;
    } else if (w.fmt == Repack::FMT_I4W8 && (x != DV->qx_src || w.k != DV->qx_k)) { // decode: x 量化 int8 (T=1 时只读行 0, 跨步无关), 同一 x 只做一次
        prof_mark(9);
        Acl::ck(ACLRT_LAUNCH_KERNEL(k_qx8)(8, Acl::stream(), x, DV->d_xrep, DV->d_xaux, w.k), "launch qx8");
        prof_mark(1);
        DV->qx_src = x;
        DV->qx_k = w.k;
        DV->qx_rep = DV->d_xrep;
        DV->qx_aux = DV->d_xaux;
    }
    prof_mark(9);
    aclError e;
    if (w.fmt == Repack::FMT_I8F) e = ACLRT_LAUNCH_KERNEL(k_gemm_i8)(8, Acl::stream(), w.data, x, xp, y, T, w.n, w.k, DV->d_flag, gate_seq, done_seq);
    else if (T == 1) e = ACLRT_LAUNCH_KERNEL(k_gemm_w8d)(8, Acl::stream(), w.data, DV->qx_rep, DV->qx_aux, y, w.n, w.k, DV->d_flag, gate_seq, done_seq);
    else e = ACLRT_LAUNCH_KERNEL(k_gemm_w8p)(8, Acl::stream(), w.data, xp, y, T, w.n, w.k);
    Acl::ck(e, "launch gemm");
    prof_mark(0);
}

void matmul_tok(const Weight& w, void* x, void* y, int32_t done_seq = 0, int32_t gate_seq = 0) { matmul(w, x, 1, y, 0, done_seq, gate_seq); }

void dq_rows(const Weight& w, void* d_rows, void* d_out, int nr) { // 反量化 nr 行到自然顺序 f32 (tok_embd, wave I8)
    Acl::ck(ACLRT_LAUNCH_KERNEL(k_dq_i8)(8, Acl::stream(), w.data, d_rows, d_out, nr, w.k, w.stride), "launch dq");
}

void add_norm(void* add, void* w, void* hidden, int T) {
    prof_mark(9);
    Acl::ck(ACLRT_LAUNCH_KERNEL(k_add_norm)(8, Acl::stream(), DV->d_x, add, w, DV->d_xn, hidden, T), "launch add_norm");
    prof_mark(7);
    pack_invalidate(); // xn 内容更新
}

void advance_pos(const int* pos, int T) { // 多芯时只由芯 0 更新共享位置, barrier 保证两边都用完旧值且看到新值
    dev_barrier();
    if (DV->idx == 0) {
        int mx = G_next_pos - 1;
        for (int i = 0; i < T * 3; i++) mx = std::max(mx, pos[i]);
        G_next_pos = mx + 1;
        G_n_past += T;
    }
    dev_barrier();
}

void eval_chunk(const int* tokens, int T, const float* embd, const uint8_t* mask, const int* pos3, bool need_logits) {
    int pos[CHUNK * 3];
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 3; j++) pos[t * 3 + j] = pos3 ? pos3[t * 3 + j] : G_next_pos + t;
    float cssn[CHUNK * 64]; // [cos T*32 | sin T*32] 连续, 一次上传
    for (int t = 0; t < T; t++) { // 与 CUDA 版 k_qk_rope 相同的 MRoPE 分段规则, cos/sin 在 host 算好
        for (int f = 0; f < 32; f++) {
            int sec = (f % 3 == 1 && f < 33) ? 1 : ((f % 3 == 2 && f < 30) ? 2 : 0);
            float ang = (float)pos[t * 3 + sec] * G_inv_freq[f];
            cssn[t * 32 + f] = cosf(ang);
            cssn[(T + t) * 32 + f] = sinf(ang);
        }
    }
    bool hostx = NDEV > 1 && T == 1; // decode: 公共部分 (残差/norm/qx8/embedding) 在 CPU 算
    if (DV->idx == 0) G_hid_host = hostx;
    Acl::sync(); // h2d 会绕开 stream 顺序, 必须等上一 chunk 的 kernel (qk_rope 读 cos/sin) 全部完成
    Acl::h2d(DV->d_cos, cssn, (size_t)T * 256);
    float* d_sin = (float*)DV->d_cos + (size_t)T * 32;
    if (hostx) { // 两线程各自算一份 x_res (结果逐位相同)
        if (mask && mask[0]) memcpy(DV->x_res.data(), embd, (size_t)D * 4);
        else embd_row_cpu(tokens[0], DV->x_res.data());
    } else {
        Acl::h2d(DV->d_tok, tokens, (size_t)T * 4);
        dq_rows(DV->tok_embd, DV->d_tok, DV->d_x, T); // token embedding
        if (mask) { // 图片 embedding 行直接覆盖 (embd 是紧凑的图片行, 只含 mask 为 1 的行, 与 CPU 版语义一致)
            Acl::sync();
            int j = 0;
            for (int t = 0; t < T; t++)
                if (mask[t]) Acl::h2d(fp(DV->d_x, (size_t)t * D), embd + (size_t)j++ * D, (size_t)D * 4);
        }
    }
    const char* stop_env = getenv("QWEN_NPU_STOP"); // 调试: 跑完第 N 层后停
    int stop_l = stop_env ? atoi(stop_env) : -1;
    // gated: 段 kernel 预下发 (段首在核上自旋等 doorbell), 汇合计算完一次 4B h2d 放行, 隐藏任务下发延迟; 与 sync 类调试钩子互斥
    bool gated = hostx && G_use_db && !G_prof_on && stop_l < 0 && !getenv("QWEN_NPU_PHASE");
    for (int l = 0; l < N_LAYER; l++) {
        Layer& L = DV->layers[l];
        int32_t g1 = gated ? ++DV->db_seq : 0, w1 = DV->dn_seq; // w1: 上一段的完成序号, 须在本段 launch 递增 dn_seq 前捕获
        if (gated) seed_qx(); // 汇合推迟到段 launch 之后, 先把 qx8 缓存指到 d_xq
        else if (hostx) host_norm_push(&G_hnorm[(size_t)l * 2 * D], G_hidden.data() + (size_t)l * D, l > 0, L.recr, true);
        else add_norm(l ? DV->d_out : nullptr, L.attn_norm, fp(DV->d_hidden, (size_t)l * D), T);
        if (L.recr) {
            matmul(L.qkv, DV->d_xn, T, DV->d_qkv, 0, 0, g1);
            matmul(L.ab, hostx ? (uint8_t*)DV->d_xq + XQ_XN : DV->d_xn, T, DV->d_ab); // gemv 要读 f32 xn 本体
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_conv)(8, Acl::stream(), DV->d_qkv, L.conv_state, L.conv_w, T), "launch conv");
            prof_mark(3);
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_dn_pre)(8, Acl::stream(), DV->d_qkv, DV->d_ab, L.ssm_a, L.dt_bias, DV->d_bd, T), "launch dn_pre");
            static int tokdn = getenv("QWEN_NPU_TOKDN") ? 1 : 0; // 调试: 强制 prefill 也走逐 token 版
            static int dnconly = getenv("QWEN_NPU_DNCONLY") ? atoi(getenv("QWEN_NPU_DNCONLY")) : -1; // 调试: 仅该层用 chunk 版
            bool use_tok = tokdn || (dnconly >= 0 && l != dnconly);
            if (T == 1 || use_tok) Acl::ck(ACLRT_LAUNCH_KERNEL(k_deltanet)(8, Acl::stream(), DV->d_qkv, DV->d_bd, L.ssm_norm, L.ssm_state, DV->d_o, T), "launch deltanet");
            else Acl::ck(ACLRT_LAUNCH_KERNEL(k_dnc)(8, Acl::stream(), DV->d_qkv, DV->d_bd, L.ssm_norm, L.ssm_state, DV->d_o, T), "launch dnc"); // chunk 化 cube 版
            prof_mark(4);
            pack_invalidate();
            matmul(L.ssm_out, DV->d_o, T, DV->d_out, 0, hostx && G_use_db ? ++DV->dn_seq : 0);
            if (!hostx) allreduce_out(T); // hostx 时 partial 由下一个 host_norm_push 取回累加
        } else {
            matmul(L.wqkv, DV->d_xn, T, DV->d_qg, 0, 0, g1);
            const char* ph2 = getenv("QWEN_NPU_PHASE");
            if (ph2 && l == stop_l && atoi(ph2) == 10) { Acl::sync(); return; }
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_qk_rope)(8, Acl::stream(), DV->d_qg, L.q_norm, L.k_norm, DV->d_cos, d_sin, L.kq, L.ksc, L.vq, L.vsc, G_n_past, T, G_max_ctx), "launch qk_rope");
            prof_mark(5);
            if (ph2 && l == stop_l && atoi(ph2) == 11) { Acl::sync(); return; }
            Acl::ck(ACLRT_LAUNCH_KERNEL(k_attn)(T == 1 ? 2 * N_KV_TP : 8, Acl::stream(), DV->d_qg, L.kq, L.ksc, L.vq, L.vsc, DV->d_att, DV->d_attp, G_n_past, T, G_max_ctx), "launch attn");
            if (T == 1) Acl::ck(ACLRT_LAUNCH_KERNEL(k_attn_merge)(N_KV_TP, Acl::stream(), DV->d_qg, DV->d_attp, DV->d_att), "launch attn_merge");
            prof_mark(6);
            if (ph2 && l == stop_l && atoi(ph2) == 12) { Acl::sync(); return; }
            pack_invalidate();
            matmul(L.wo, DV->d_att, T, DV->d_out, 0, hostx && G_use_db ? ++DV->dn_seq : 0);
            if (!hostx) allreduce_out(T);
        }
        if (gated) host_norm_push(&G_hnorm[(size_t)l * 2 * D], G_hidden.data() + (size_t)l * D, l > 0, L.recr, true, g1, w1); // 本段 (attn/recr) 的汇合, 敲铃 g1
        int32_t g2 = gated ? ++DV->db_seq : 0, w2 = DV->dn_seq;
        if (gated) seed_qx();
        else if (hostx) host_norm_push(&G_hnorm[(size_t)l * 2 * D + D], nullptr, true, false, true);
        else add_norm(DV->d_out, L.post_norm, nullptr, T);
        matmul(L.ffn_gu, DV->d_xn, T, DV->d_g, 0, 0, g2); // 一次 gemm 输出 [T][2*FF_N] = 行内 [g|u]
        const char* ph = getenv("QWEN_NPU_PHASE"); // 调试: 在 FFN 的某一步后直接停 (gate/up 已合并, 0/1 同点)
        if (ph && l == stop_l && (atoi(ph) == 0 || atoi(ph) == 1)) { Acl::sync(); return; }
        Acl::ck(ACLRT_LAUNCH_KERNEL(k_silu_mul)(8, Acl::stream(), DV->d_g, T, FF_N[DV->idx]), "launch silu_mul");
        prof_mark(8);
        if (ph && l == stop_l && atoi(ph) == 2) { Acl::sync(); return; }
        pack_invalidate();
        matmul(L.ffn_down, DV->d_g, T, DV->d_out, 2 * FF_N[DV->idx], hostx && G_use_db ? ++DV->dn_seq : 0); // 行跨步 2*FF_N, 只取 g 段
        if (!hostx) allreduce_out(T);
        if (gated) host_norm_push(&G_hnorm[(size_t)l * 2 * D + D], nullptr, true, false, true, g2, w2); // FFN 段的汇合, 敲铃 g2
        if (l == stop_l) {
            Acl::sync();
            advance_pos(pos, T); // 提前返回也推进位置, 保持多 chunk 的 n_past 语义 (hostx 时 x_res 不跨 chunk, 不必取回 partial)
            return;
        }
    }
    if (gated) { // lm_head 预下发 (门 g3), final norm 汇合后敲铃
        int32_t g3 = 0, w3 = DV->dn_seq;
        if (need_logits) {
            g3 = ++DV->db_seq;
            matmul_tok(DV->output, (uint8_t*)DV->d_xq + XQ_XN, DV->d_logits, ++DV->dn_seq, g3);
        }
        host_norm_push(&G_hnorm[(size_t)N_LAYER * 2 * D], G_hidden.data() + (size_t)N_LAYER * D, true, true, false, g3, w3);
        advance_pos(pos, T);
        if (!need_logits) return;
        wait_done(DV->dn_seq);
    } else {
        if (hostx) host_norm_push(&G_hnorm[(size_t)N_LAYER * 2 * D], G_hidden.data() + (size_t)N_LAYER * D, true, true, false);
        else add_norm(DV->d_out, DV->output_norm, fp(DV->d_hidden, (size_t)N_LAYER * D), T);
        advance_pos(pos, T);
        if (!need_logits) return;
        matmul_tok(DV->output, hostx ? (uint8_t*)DV->d_xq + XQ_XN : fp(DV->d_xn, (size_t)(T - 1) * D), DV->d_logits, hostx && G_use_db ? ++DV->dn_seq : 0);
        if (hostx && G_use_db) wait_done(DV->dn_seq);
        else Acl::sync();
    }
    Acl::d2h(G_logits + (size_t)DV->idx * VOCAB_TP, DV->d_logits, (size_t)VOCAB_TP * 4); // 各芯写 vocab 的各自一半
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
    Acl::init(device, NDEV);
    G_use_db = NDEV > 1 && !getenv("QWEN_NPU_NODB");
    if (!St::open(G_st, path)) return -1;
    G_max_ctx = max_ctx;
    for (int j = 0; j < LV; j++) G_perm48[j] = (j % LK) * 3 + j / LK; // tiled 头 j <- HF 头
    if (NDEV > 1) { // decode 公共计算在 host: norm 权重副本 + embedding 直接读 mmap
        G_hnorm.resize((size_t)(N_LAYER * 2 + 1) * D);
        G_embd_t = &find_t("model.language_model.embed_tokens.weight");
    }
    auto t1 = std::chrono::steady_clock::now();
    run_devs([&] { // 每芯一个线程: 构建本芯分片映射, 重打包并上传本芯的那一半权重
        int di = DV->idx;
        DV->vheads.resize(LV_TP); // 本芯 v 头 j -> 全局 tiled v 头 (qk 头取 [di*LK_TP, di*LK_TP+LK_TP), 保持局部 j%LK_TP 关系)
        for (int j = 0; j < LV_TP; j++) DV->vheads[j] = (j / LK_TP) * LK + di * LK_TP + j % LK_TP;
        DV->rows_qkv.resize(QKV_TP); // q/k 按 qk 头取段, v 段按头重排; conv1d 通道同此
        for (int r = 0; r < QKV_TP; r++) {
            if (r < KEY_TP) DV->rows_qkv[r] = di * KEY_TP + r;
            else if (r < 2 * KEY_TP) DV->rows_qkv[r] = KEY_DIM + di * KEY_TP + (r - KEY_TP);
            else DV->rows_qkv[r] = 2 * KEY_DIM + G_perm48[DV->vheads[(r - 2 * KEY_TP) / LHD]] * LHD + r % LHD;
        }
        DV->rows_z.resize(V_TP);
        for (int r = 0; r < V_TP; r++) DV->rows_z[r] = G_perm48[DV->vheads[r / LHD]] * LHD + r % LHD;
        DV->kblk_out.resize(LV_TP);
        for (int j = 0; j < LV_TP; j++) DV->kblk_out[j] = G_perm48[DV->vheads[j]];
        DV->rows_q.resize(Q_TP);
        for (int r = 0; r < Q_TP; r++) DV->rows_q[r] = di * Q_TP + r;
        DV->rows_kv.resize(KV_TP);
        for (int r = 0; r < KV_TP; r++) DV->rows_kv[r] = di * KV_TP + r;
        DV->kblk_wo.resize(ATT_TP / 128);
        for (int j = 0; j < ATT_TP / 128; j++) DV->kblk_wo[j] = di * (ATT_TP / 128) + j;
        DV->rows_ff.resize(FF_N[di]);
        for (int r = 0; r < FF_N[di]; r++) DV->rows_ff[r] = FF_OFF[di] + r;
        DV->kblk_ff.resize(FF_N[di] / 128);
        for (int j = 0; j < FF_N[di] / 128; j++) DV->kblk_ff[j] = FF_OFF[di] / 128 + j;
        DV->tok_embd = get_embd_i8("model.language_model.embed_tokens.weight", D, VOCAB); // embedding 每芯各一份 (x 复制)
        save_ref_i8("model.language_model.embed_tokens.weight", &DV->tok_embd);
        DV->output = get_w_i8_bf16("lm_head.weight", D, VOCAB_TP, VOCAB, di * VOCAB_TP); // lm_head 按 vocab 行切分
        save_ref_i8("lm_head.weight", &DV->output, di * VOCAB_TP);
        bool hk = di == 0 && NDEV > 1; // host 侧 norm 副本只需一份
        DV->output_norm = get_norm("model.language_model.norm.weight", D, true, hk ? &G_hnorm[(size_t)N_LAYER * 2 * D] : nullptr);
        void* conv_all = Acl::alloc((size_t)N_LAYER * 3 * QKV_TP * 4);
        void* ssm_all = Acl::alloc((size_t)N_LAYER * LV_TP * LHD * LHD * 4);
        vector<float> cwt(4 * QKV_TP);
        for (int l = 0; l < N_LAYER; l++) {
            Layer& L = DV->layers[l];
            string p = "model.language_model.layers." + std::to_string(l) + ".";
            L.recr = (l + 1) % 4 != 0;
            L.attn_norm = get_norm(p + "input_layernorm.weight", D, true, hk ? &G_hnorm[(size_t)l * 2 * D] : nullptr);
            L.post_norm = get_norm(p + "post_attention_layernorm.weight", D, true, hk ? &G_hnorm[(size_t)l * 2 * D + D] : nullptr);
            vector<QSeg> gu_segs = {{p + "mlp.gate_proj", FF_N[di], DV->rows_ff.data(), FF}, {p + "mlp.up_proj", FF_N[di], DV->rows_ff.data(), FF}};
            L.ffn_gu = get_w_q4_cat(gu_segs, D); // gate+up 行拼接 (按列切), down 按 k 切
            L.ffn_down = get_w_q4(p + "mlp.down_proj", FF_N[di], D, nullptr, DV->kblk_ff.data(), FF);
            if (l == 0) save_ref_q4(p + "mlp.gate+up", &L.ffn_gu, gu_segs, nullptr);
            if (l == 51) save_ref_q4(p + "mlp.down_proj", &L.ffn_down, {{p + "mlp.down_proj", D, nullptr}}, DV->kblk_ff.data(), FF);
            if (L.recr) {
                vector<QSeg> qkv_segs = {{p + "linear_attn.in_proj_qkv", QKV_TP, DV->rows_qkv.data(), QKV_DIM}, {p + "linear_attn.in_proj_z", V_TP, DV->rows_z.data(), V_DIM}};
                L.qkv = get_w_q4_cat(qkv_segs, D);
                L.ab = get_w_ab(p);
                L.ssm_out = get_w_q4(p + "linear_attn.out_proj", V_TP, D, nullptr, DV->kblk_out.data(), V_DIM);
                if (l == 0) {
                    save_ref_q4(p + "linear_attn.in_proj_qkv+z", &L.qkv, qkv_segs, nullptr);
                    save_ref_q4(p + "linear_attn.out_proj", &L.ssm_out, {{p + "linear_attn.out_proj", D, nullptr}}, DV->kblk_out.data(), V_DIM);
                    const St::Tensor& ta = find_t(p + "linear_attn.in_proj_a.weight");
                    const St::Tensor& tb = find_t(p + "linear_attn.in_proj_b.weight");
                    vector<float> aref((size_t)8 * D);
                    for (int i = 0; i < 4; i++) bf16_row_f32(ta, G_perm48[DV->vheads[i]], D, &aref[(size_t)i * D]);
                    for (int i = 4; i < 8; i++) bf16_row_f32(tb, G_perm48[DV->vheads[LV_TP - 8 + i]], D, &aref[(size_t)i * D]); // 合并矩阵尾 4 行 = β 的末 4 头
                    save_ref(p + "linear_attn.in_proj_a+b", &L.ab, aref);
                }
                vector<float> alog = load_f32(p + "linear_attn.A_log", LV);
                vector<float> a64(64, 0.0f), b64(64, 0.0f);
                for (int j = 0; j < LV_TP; j++) a64[j] = -expf(alog[G_perm48[DV->vheads[j]]]); // ssm_a = -exp(A_log), 按本芯 tiled 头序
                L.ssm_a = up_f32(a64.data(), 64);
                vector<float> dtb = load_f32(p + "linear_attn.dt_bias", LV);
                for (int j = 0; j < LV_TP; j++) b64[j] = dtb[G_perm48[DV->vheads[j]]];
                L.dt_bias = up_f32(b64.data(), 64);
                L.ssm_norm = get_norm(p + "linear_attn.norm.weight", LHD, false);
                vector<float> cw = load_f32(p + "linear_attn.conv1d.weight", (int64_t)QKV_DIM * 4); // [通道][1][4] -> [4][本芯通道], v 通道重排
                for (int c = 0; c < QKV_TP; c++)
                    for (int j = 0; j < 4; j++) cwt[j * QKV_TP + c] = cw[DV->rows_qkv[c] * 4 + j];
                L.conv_w = Acl::alloc((size_t)4 * QKV_TP * 4);
                Acl::h2d(L.conv_w, cwt.data(), (size_t)4 * QKV_TP * 4);
                L.conv_state = fp(conv_all, (size_t)l * 3 * QKV_TP);
                L.ssm_state = fp(ssm_all, (size_t)l * LV_TP * LHD * LHD);
                continue;
            }
            L.wqkv = get_w_q4_cat({{p + "self_attn.q_proj", Q_TP, DV->rows_q.data(), Q_DIM}, {p + "self_attn.k_proj", KV_TP, DV->rows_kv.data(), KV_DIM}, {p + "self_attn.v_proj", KV_TP, DV->rows_kv.data(), KV_DIM}}, D);
            L.wo = get_w_q4(p + "self_attn.o_proj", ATT_TP, D, nullptr, DV->kblk_wo.data(), ATT_DIM);
            L.q_norm = get_norm(p + "self_attn.q_norm.weight", HD, true);
            L.k_norm = get_norm(p + "self_attn.k_norm.weight", HD, true);
            L.kq = Acl::alloc((size_t)N_KV_TP * max_ctx * HD + 16384); // 末尾留 48 行 tile 读越量余量
            L.ksc = Acl::alloc((size_t)N_KV_TP * max_ctx * 64 + 16384);
            L.vq = Acl::alloc((size_t)N_KV_TP * max_ctx * HD + 16384);
            L.vsc = Acl::alloc((size_t)N_KV_TP * max_ctx * 64 + 16384);
        }
        DV->d_x = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
        DV->d_xn = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
        DV->d_out = Acl::alloc((size_t)CHUNK * D * 4 + PAD);
        DV->d_qkv = Acl::alloc((size_t)CHUNK * QKVZ_TP * 4 + PAD);
        DV->d_ab = Acl::alloc((size_t)CHUNK * 2 * LV_TP * 4 + PAD);
        DV->d_o = Acl::alloc((size_t)CHUNK * V_TP * 4 + PAD);
        DV->d_qg = Acl::alloc((size_t)CHUNK * QKVC_TP * 4 + PAD);
        DV->d_att = Acl::alloc((size_t)CHUNK * ATT_TP * 4 + PAD);
        DV->d_g = Acl::alloc((size_t)CHUNK * 2 * FF_N[di] * 4 + PAD); // [T][2*FF_N] 行内 [g|u]
        DV->d_u = Acl::alloc((size_t)CHUNK * FF_N[di] * 4 + PAD);
        DV->d_logits = Acl::alloc((size_t)VOCAB_TP * 4 + PAD);
        DV->d_hidden = Acl::alloc((size_t)(N_LAYER + 1) * D * 4);
        DV->d_tok = Acl::alloc((size_t)CHUNK * 4 + 64);
        DV->d_xp = Acl::alloc((size_t)(FF / 128) * (CHUNK / 16) * 2048 * 2); // 最大 k, m=CHUNK (沿用全尺寸, 留余量)
        DV->d_xrep = Acl::alloc((size_t)2 * 4 * 8704 + PAD);
        DV->d_xaux = Acl::alloc((size_t)(FF / 128) * 8 * 3 * 4 + PAD);
        DV->d_attp = Acl::alloc((size_t)(2 * N_KV_TP) * 16 * HD * 4 + 1024 + PAD);
        DV->d_cos = Acl::alloc((size_t)CHUNK * 256); // cos+sin 合并一块
        if (NDEV > 1) DV->d_xq = Acl::alloc((size_t)XQ_TOTAL + PAD);
        DV->d_bd = Acl::alloc((size_t)CHUNK * 2 * LV_TP * 4 + PAD); // k_dn_pre 的 beta/decay [T][2*LV_TP]
        DV->d_flag = Acl::alloc(4096); // doorbell 标志位
        if (NDEV > 1) {
            DV->d_peer = Acl::alloc((size_t)CHUNK * D * 4 + PAD); // all-reduce 对方 partial 落地
            for (int pa = 0; pa < 2; pa++) Acl::ck(aclrtMallocHost((void**)&G_ar_host[di][pa], (size_t)CHUNK * D * 4), "aclrtMallocHost ar");
            DV->x_res.resize(D);
            Acl::ck(aclrtMallocHost((void**)&DV->qx_pin, (size_t)XQ_TOTAL), "aclrtMallocHost qx");
            memset(DV->qx_pin, 0, (size_t)XQ_TOTAL); // 旋转副本的空隙字节保持 0
            DV->xn_pin = (float*)(DV->qx_pin + XQ_XN);
        }
    });
    double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "ℹ️ 权重重打包并上传完成, %.1fs\n", up);
    if (NDEV == 1) St::close(G_st); // 多芯 decode 的 embedding 查行直接读 mmap, 保持打开
    for (int i = 0; i < ROT / 2; i++) G_inv_freq[i] = 1.0f / powf(THETA, (float)(2 * i) / ROT);
    Acl::ck(aclrtMallocHost((void**)&G_logits, (size_t)VOCAB * 4), "aclrtMallocHost");
    memset(G_logits, 0, (size_t)VOCAB * 4);
    G_hidden.resize((size_t)(N_LAYER + 1) * D);
    G_work.resize(VOCAB);
    G_idx.resize(VOCAB);
    G_seen.resize(VOCAB);
    G_inited = true;
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (int di = 0; di < NDEV; di++) {
        dev_enter(di);
        size_t free_b = 0, total_b = 0;
        Acl::ck(aclrtGetMemInfo(ACL_HBM_MEM, &free_b, &total_b), "aclrtGetMemInfo");
        fprintf(stderr, "✅ 芯片 %d 初始化完成, 显存已用 %.2f / %.2f GB\n", device + di, (total_b - free_b) / 1073741824.0, total_b / 1073741824.0);
    }
    dev_enter(0);
    fprintf(stderr, "✅ 模型初始化完成 (%d 芯), 最大上下文 %d, 耗时 %.1fs\n", NDEV, max_ctx, sec);
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
    G_prof_on = getenv("QWEN_NPU_PROF") != nullptr;
    run_devs([&] { // 每芯一个线程跑完整 chunk 循环, all-reduce 处 barrier 汇合
        if (G_prof_on && DV->idx == 0) {
            Acl::sync();
            G_prof_t0 = std::chrono::steady_clock::now();
        }
        for (int i = 0, r = 0; i < n; i += CHUNK) { // r: 之前各 chunk 的图片行数 (embd 只含 mask 为 1 的行)
            int T = std::min(CHUNK, n - i);
            eval_chunk(tokens + i, T, embd ? embd + (size_t)r * D : nullptr, embd_mask ? embd_mask + i : nullptr, pos3 ? pos3 + i * 3 : nullptr, i + T == n);
            for (int t = i; embd_mask && t < i + T; t++) r += embd_mask[t];
        }
    });
    return 0;
}

void qwen_prof_dump() { // 打印并清零 QWEN_NPU_PROF 的逐 kernel 累计耗时
    for (int i = 0; i < 11; i++) {
        if (G_prof_ms[i] > 0) fprintf(stderr, "⏱️ %s: %.1f ms\n", G_prof_names[i], G_prof_ms[i]);
        G_prof_ms[i] = 0;
    }
}

int qwen_sample(float temperature, int top_k, float top_p, float min_p, float presence_penalty, const int* hist, int n_hist, int think_ok) {
    // 与 CPU 版一致: 只在可采样的 token 里采样: 普通文本 token [0, N_TEXT), think_ok (开了 thinking 且还在思考中) 时加上思考结束 token, 否则加上 im_end (思考中不能直接结束); 其余控制 token 和输出层的填充位不参与
    const int N_TEXT = 248044, THINK_END = 248069, IM_END = 248046;
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
    int* idx = G_idx.data();
    int m = 0; // 可采样的个数
    for (int i = 0; i < VOCAB; i++)
        if (i < N_TEXT || i == (think_ok ? THINK_END : IM_END)) idx[m++] = i;
    if (temperature <= 0) return *std::max_element(idx, idx + m, [&](int a, int b) { return l[a] < l[b]; }); // 相同时取 id 小的, 与原来一致
    int n = (top_k > 0 && top_k < m) ? top_k : m;
    auto cmp = [&](int a, int b) { return l[a] > l[b]; };
    if (n < m) std::nth_element(idx, idx + n, idx + m, cmp);
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
    for (int di = 0; di < NDEV; di++) {
        dev_enter(di);
        for (int l = 0; l < N_LAYER; l++) {
            Layer& L = DV->layers[l];
            if (!L.recr) continue;
            Acl::ck(aclrtMemset(L.conv_state, (size_t)3 * QKV_TP * 4, 0, (size_t)3 * QKV_TP * 4), "memset conv");
            Acl::ck(aclrtMemset(L.ssm_state, (size_t)LV_TP * LHD * LHD * 4, 0, (size_t)LV_TP * LHD * LHD * 4), "memset ssm");
        }
        Acl::ck(aclrtMemset(DV->d_flag, 4096, 0, 4096), "memset flag"); // doorbell 序号同步清零
        DV->dn_seq = 0;
        DV->db_seq = 0;
    }
    dev_enter(0);
    G_n_past = 0;
    G_next_pos = 0;
}

int qwen_n_past() { return G_n_past; }
float* qwen_logits() { return G_logits; }

float* qwen_hidden() { // hidden 是 all-reduce 之后的完整残差流, 各芯相同, 读芯 0; 多芯 decode 时直接在 host
    if (G_hid_host) return G_hidden.data();
    dev_enter(0);
    Acl::sync();
    Acl::d2h(G_hidden.data(), DV->d_hidden, G_hidden.size() * 4);
    return G_hidden.data();
}

int qwen_selftest() { // tok_embd: 反量化 vs host 参考逐位; matmul 权重: m=1 GEMM vs 参考行的 f32 点积 (fp16 计算, 阈值 5e-3); 逐芯依次测
    int fails = 0;
    for (int di = 0; di < NDEV; di++) {
    dev_enter(di);
    for (SelfRef& s : DV->selfrefs) {
        const Weight* w = s.w;
        int k = w->k;
        if (w->fmt == Repack::FMT_I8) { // tok_embd 反量化查行, 与 host 参考逐位一致
            Acl::h2d(DV->d_tok, s.rows, 8 * 4);
            dq_rows(*w, DV->d_tok, DV->d_out, 8);
            Acl::sync();
            vector<float> got((size_t)8 * k);
            Acl::d2h(got.data(), DV->d_out, got.size() * 4);
            double max_diff = 0, max_abs = 0;
            for (size_t i = 0; i < got.size(); i++) {
                max_diff = std::max(max_diff, (double)std::fabs(got[i] - s.ref[i]));
                max_abs = std::max(max_abs, (double)std::fabs(s.ref[i]));
            }
            bool ok = max_diff <= 1e-6 * max_abs;
            fails += !ok;
            fprintf(stderr, "%s 芯%d %s fmt=%d [%d, %d] 反量化最大差 %.3g (最大绝对值 %.3g)\n", ok ? "✅" : "❌", di, s.name.c_str(), w->fmt, k, w->n, max_diff, max_abs);
            continue;
        }
        vector<float> x(k), y((size_t)w->n);
        for (auto& v : x) v = (float)((rng_next() >> 11) * (1.0 / 9007199254740992.0) * 2 - 1);
        Acl::h2d(DV->d_g, x.data(), (size_t)k * 4);
        pack_invalidate(); // x 内容换了, 指针没换, 需失效 qx8 缓存
        matmul_tok(*w, DV->d_g, DV->d_u);
        Acl::sync();
        Acl::d2h(y.data(), DV->d_u, (size_t)w->n * 4);
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
        fprintf(stderr, "%s 芯%d %s fmt=%d [%d, %d] GEMM 最大相对误差 %.3g\n", ok ? "✅" : "❌", di, s.name.c_str(), w->fmt, k, w->n, max_rel);
    }
    }
    dev_enter(0);
    return fails;
}

int qwen_dbg(const char* name, float* out, long long n) { // 调试: 读取中间激活缓冲 (多芯时默认芯 0, QWEN_NPU_DBGDEV 可选)
    const char* ed = getenv("QWEN_NPU_DBGDEV");
    dev_enter(ed ? atoi(ed) : 0);
    void* p = nullptr;
    string s = name;
    if (s == "x") p = DV->d_x;
    else if (s == "xn") p = DV->d_xn;
    else if (s == "out") p = DV->d_out;
    else if (s == "qkv") p = DV->d_qkv; // 含 z 段 ([T][QKVZ_TP])
    else if (s == "ab") p = DV->d_ab;
    else if (s == "o") p = DV->d_o;
    else if (s == "qg") p = DV->d_qg; // 含 k/v 段 ([T][QKVC_TP])
    else if (s == "att") p = DV->d_att;
    else if (s == "g") p = DV->d_g;
    else if (s == "u") p = fp(DV->d_g, FF_N[DV->idx]); // u 在合并缓冲的行内后半 (多行时按 2*FF_N 跨步)
    else if (s == "bd") p = DV->d_bd;
    else if (s == "sst") { // 调试: 某层的 DeltaNet 状态, 层号用 QWEN_NPU_SSTL
        const char* e = getenv("QWEN_NPU_SSTL");
        p = DV->layers[e ? atoi(e) : 0].ssm_state;
    }
    else return -1;
    Acl::sync();
    Acl::d2h(out, p, (size_t)n * 4);
    return 0;
}

} // extern "C"
