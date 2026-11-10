// Qwen3.8-27B 的全部 CUDA kernel, 只编译 device 代码 (fatbin), host 端通过 driver API 按名字加载
#include <cstdint>
#include <cuda_fp16.h>

namespace {

constexpr int D = 5120, HD = 256, LHD = 128, LK = 16, LV = 48;
constexpr int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM;
constexpr int N_HEAD = 24, N_KV = 4, Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD, ATT_DIM = N_HEAD * HD;
constexpr float EPS = 1e-6f;
constexpr int NT = 8; // matmul 每次处理的 token 数 (寄存器累加器个数)
enum : int { T_F32 = 0, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14 };

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return *(const uint16_t*)p; }
__device__ __forceinline__ uint2 ld8_a2(const uint8_t* p) { // 只保证 2 字节对齐时读 8 字节
    const uint16_t* q = (const uint16_t*)p;
    return make_uint2((uint32_t)q[0] | ((uint32_t)q[1] << 16), (uint32_t)q[2] | ((uint32_t)q[3] << 16));
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int NTHR> __device__ __forceinline__ float block_sum(float v, float* red) { // red: 共享内存 [NTHR / 32], 返回值所有线程一致
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
#pragma unroll
    for (int i = 0; i < NTHR / 32; i++) s += red[i];
    __syncthreads();
    return s;
}

__device__ __forceinline__ void scale_min_k4(const uint8_t* sc, int j, float d, float dmin, float& d1, float& m1) { // 8 个子块的 6bit scale/min 打包在 12 字节里
    int s, m;
    if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    d1 = d * s;
    m1 = dmin * m;
}

// 一个 warp 解一个 256 元素的超块, lane 负责第 [8*lane, 8*lane+8) 个
template <int TYPE> __device__ __forceinline__ void dq8(const uint8_t* row, int sb, int lane, float* w) {
    if constexpr (TYPE == T_F32) {
        const float4* p = (const float4*)(row + (size_t)sb * 1024 + lane * 32);
        float4 a = p[0], b = p[1];
        w[0] = a.x, w[1] = a.y, w[2] = a.z, w[3] = a.w, w[4] = b.x, w[5] = b.y, w[6] = b.z, w[7] = b.w;
    } else if constexpr (TYPE == T_Q8_0) { // 34 字节一块, 32 个元素, 一个超块对应 8 块
        const uint8_t* b = row + (size_t)(sb * 8 + lane / 4) * 34;
        float d = h2f(ld16(b));
        uint2 q = ld8_a2(b + 2 + (lane % 4) * 8);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            w[e] = (float)(int8_t)(q.x >> (8 * e)) * d;
            w[e + 4] = (float)(int8_t)(q.y >> (8 * e)) * d;
        }
    } else if constexpr (TYPE == T_Q4_K) { // d, dmin, scales[12], qs[128]; 每 64 个元素共用 32 字节, 前 32 个取低 4 位, 后 32 个取高 4 位
        const uint8_t* b = row + (size_t)sb * 144;
        float d1, m1;
        scale_min_k4(b + 4, lane / 4, h2f(ld16(b)), h2f(ld16(b + 2)), d1, m1);
        int o = (lane % 8) * 8, sh = o >= 32 ? 4 : 0;
        uint2 q = *(const uint2*)(b + 16 + 32 * (lane / 8) + (o & 31));
#pragma unroll
        for (int e = 0; e < 4; e++) {
            w[e] = fmaf(d1, (float)((q.x >> (8 * e + sh)) & 0xF), -m1);
            w[e + 4] = fmaf(d1, (float)((q.y >> (8 * e + sh)) & 0xF), -m1);
        }
    } else if constexpr (TYPE == T_Q5_K) { // d, dmin, scales[12], qh[32], qs[128]; 第 g 组 (64 个) 的第 5 位在 qh 的 bit 2g (前 32 个) / 2g+1 (后 32 个)
        const uint8_t* b = row + (size_t)sb * 176;
        float d1, m1;
        scale_min_k4(b + 4, lane / 4, h2f(ld16(b)), h2f(ld16(b + 2)), d1, m1);
        int g = lane / 8, o = (lane % 8) * 8, sh = o >= 32 ? 4 : 0, hb = 2 * g + (o >= 32);
        uint2 q = *(const uint2*)(b + 48 + 32 * g + (o & 31));
        uint2 h = *(const uint2*)(b + 16 + (o & 31));
#pragma unroll
        for (int e = 0; e < 4; e++) {
            w[e] = fmaf(d1, (float)(((q.x >> (8 * e + sh)) & 0xF) | (((h.x >> (8 * e + hb)) & 1) << 4)), -m1);
            w[e + 4] = fmaf(d1, (float)(((q.y >> (8 * e + sh)) & 0xF) | (((h.y >> (8 * e + hb)) & 1) << 4)), -m1);
        }
    } else if constexpr (TYPE == T_Q6_K) { // ql[128], qh[64], scales[16] (int8), d; 每 128 个一半, 半内 4 个 32 元素段 s, ql 取 (s&1) 段的低/高 4 位, qh 取 bit 2s
        const uint8_t* b = row + (size_t)sb * 210;
        int n = lane / 16, s = (lane % 16) / 4, l0 = (lane % 4) * 8;
        uint2 ql = ld8_a2(b + 64 * n + l0 + 32 * (s & 1));
        uint2 qh = ld8_a2(b + 128 + 32 * n + l0);
        float ds = h2f(ld16(b + 208)) * (float)(int8_t)b[192 + 8 * n + l0 / 16 + 2 * s];
        int sh = s >= 2 ? 4 : 0;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            w[e] = ds * (float)((int)(((ql.x >> (8 * e + sh)) & 0xF) | (((qh.x >> (8 * e + 2 * s)) & 3) << 4)) - 32);
            w[e + 4] = ds * (float)((int)(((ql.y >> (8 * e + sh)) & 0xF) | (((qh.y >> (8 * e + 2 * s)) & 3) << 4)) - 32);
        }
    }
}

// y[t][n] = sum_k W[n][k] * x[t][k]; block 256 线程 = 8 个 warp, 每个 warp 一行; 权重每 NT 个 token 解一次
template <int TYPE> __device__ __forceinline__ void matmul_body(const uint8_t* W, uint64_t rb, int k, int n, const float* x, int T, float* y) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= n) return;
    const uint8_t* wr = W + (size_t)row * rb;
    int nsb = k / 256;
    for (int t0 = 0; t0 < T; t0 += NT) {
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] = 0;
#pragma unroll 2
        for (int sb = 0; sb < nsb; sb++) {
            float w[8];
            dq8<TYPE>(wr, sb, lane, w);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t0 + t >= T) break;
                const float4* xp = (const float4*)(x + (size_t)(t0 + t) * k + sb * 256 + lane * 8);
                float4 a = xp[0], b = xp[1];
                float s = acc[t];
                s = fmaf(w[0], a.x, s);
                s = fmaf(w[1], a.y, s);
                s = fmaf(w[2], a.z, s);
                s = fmaf(w[3], a.w, s);
                s = fmaf(w[4], b.x, s);
                s = fmaf(w[5], b.y, s);
                s = fmaf(w[6], b.z, s);
                acc[t] = fmaf(w[7], b.w, s);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t0 + t >= T) break;
            float s = warp_sum(acc[t]);
            if (lane == 0) y[(size_t)(t0 + t) * n + row] = s;
        }
    }
}

template <int TYPE> __device__ __forceinline__ void dequant_body(const uint8_t* W, uint64_t rb, int k, const int* rows, float* out) { // block 一行, 8 个 warp 轮流解超块
    int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const uint8_t* wr = W + (size_t)rows[blockIdx.x] * rb;
    float* dst = out + (size_t)blockIdx.x * k;
    for (int sb = wp; sb < k / 256; sb += 8) {
        float w[8];
        dq8<TYPE>(wr, sb, lane, w);
#pragma unroll
        for (int e = 0; e < 8; e++) dst[sb * 256 + lane * 8 + e] = w[e];
    }
}

template <int TYPE> __device__ __forceinline__ void embed_body(const uint8_t* W, uint64_t rb, const int* tokens, const uint8_t* mask, const float* embd, float* x) { // block 一个 token
    int t = blockIdx.x;
    if (mask && mask[t]) {
        for (int i = threadIdx.x; i < D; i += blockDim.x) x[(size_t)t * D + i] = embd[(size_t)t * D + i];
        return;
    }
    int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const uint8_t* wr = W + (size_t)tokens[t] * rb;
    for (int sb = wp; sb < D / 256; sb += 8) {
        float w[8];
        dq8<TYPE>(wr, sb, lane, w);
#pragma unroll
        for (int e = 0; e < 8; e++) x[(size_t)t * D + sb * 256 + lane * 8 + e] = w[e];
    }
}

} // namespace

#define INST(NAME, TYPE)                                                                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) k_mm_##NAME(const uint8_t* W, uint64_t rb, int k, int n, const float* x, int T, float* y) { matmul_body<TYPE>(W, rb, k, n, x, T, y); } \
    extern "C" __global__ void __launch_bounds__(256) k_dq_##NAME(const uint8_t* W, uint64_t rb, int k, const int* rows, float* out) { dequant_body<TYPE>(W, rb, k, rows, out); }          \
    extern "C" __global__ void __launch_bounds__(256) k_embed_##NAME(const uint8_t* W, uint64_t rb, const int* tokens, const uint8_t* mask, const float* embd, float* x) { embed_body<TYPE>(W, rb, tokens, mask, embd, x); }
INST(f32, T_F32)
INST(q8_0, T_Q8_0)
INST(q4_k, T_Q4_K)
INST(q5_k, T_Q5_K)
INST(q6_k, T_Q6_K)

// x += add (add 可为空); y = rmsnorm(x) * w; hidden 非空时保存最后一个 token 的 x. grid T, block 1024, 每线程 5 个元素
extern "C" __global__ void __launch_bounds__(1024) k_add_norm(float* x, const float* add, const float* w, float* y, float* hidden, int T) {
    __shared__ float red[32];
    int t = blockIdx.x;
    float* xr = x + (size_t)t * D;
    float v[D / 1024], ss = 0;
#pragma unroll
    for (int i = 0; i < D / 1024; i++) {
        int c = threadIdx.x + i * 1024;
        float a = xr[c];
        if (add) {
            a += add[(size_t)t * D + c];
            xr[c] = a;
        }
        v[i] = a;
        ss += a * a;
    }
    ss = block_sum<1024>(ss, red);
    float r = 1.0f / sqrtf(ss / D + EPS);
#pragma unroll
    for (int i = 0; i < D / 1024; i++) {
        int c = threadIdx.x + i * 1024;
        y[(size_t)t * D + c] = v[i] * r * w[c];
        if (hidden && t == T - 1) hidden[c] = v[i];
    }
}

extern "C" __global__ void k_silu_mul(float* g, const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = silu(g[i]) * u[i];
}

// 因果 depthwise conv1d (k=4) + SiLU, 一个线程一个通道, state: [QKV_DIM][3] 旧的在前
extern "C" __global__ void k_conv(float* qkv, float* state, const float* cw, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= QKV_DIM) return;
    float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    float w0 = cw[c * 4], w1 = cw[c * 4 + 1], w2 = cw[c * 4 + 2], w3 = cw[c * 4 + 3];
    for (int t = 0; t < T; t++) {
        float xv = qkv[(size_t)t * QKV_DIM + c];
        float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
        s0 = s1;
        s1 = s2;
        s2 = xv;
        qkv[(size_t)t * QKV_DIM + c] = silu(o);
    }
    state[c * 3] = s0;
    state[c * 3 + 1] = s1;
    state[c * 3 + 2] = s2;
}

// Gated DeltaNet 递推: grid 48 (v 头), block 128; 线程 j 在寄存器里持有状态的第 j 列 S[0..127][j], 对 T 个 token 顺序递推
// 同一 block 内完成 q/k 的 L2 归一化、beta/g 计算和输出的门控 RMSNorm
extern "C" __global__ void __launch_bounds__(128, 1) k_deltanet(const float* qkv, const float* z, const float* a, const float* b, const float* ssm_a, const float* dt_bias, const float* ssm_norm, float* state, float* out, int T) {
    __shared__ float sq[LHD], sk[LHD], red[4];
    int h = blockIdx.x, j = threadIdx.x, kh = h % LK; // GGUF 的 v 头是 tiled 顺序, v 头 h 对应 qk 头 h % 16
    float* S = state + (size_t)h * LHD * LHD;
    float s[LHD];
#pragma unroll
    for (int i = 0; i < LHD; i++) s[i] = S[i * LHD + j];
    const float qscale = 1.0f / sqrtf((float)LHD);
    for (int t = 0; t < T; t++) {
        const float* row = qkv + (size_t)t * QKV_DIM;
        float qv = row[kh * LHD + j], kv = row[KEY_DIM + kh * LHD + j], vv = row[2 * KEY_DIM + h * LHD + j];
        float qn = block_sum<128>(qv * qv, red);
        float kn = block_sum<128>(kv * kv, red);
        sq[j] = qv * (1.0f / sqrtf(qn + 1e-6f) * qscale);
        sk[j] = kv * (1.0f / sqrtf(kn + 1e-6f));
        float beta = sigmoid(b[t * LV + h]);
        float decay = expf(ssm_a[h] * softplus(a[t * LV + h] + dt_bias[h]));
        __syncthreads();
        float u0 = 0, u1 = 0, u2 = 0, u3 = 0; // S *= exp(g); u = S^T k
#pragma unroll
        for (int i = 0; i < LHD; i += 4) {
            s[i] *= decay;
            s[i + 1] *= decay;
            s[i + 2] *= decay;
            s[i + 3] *= decay;
            u0 = fmaf(sk[i], s[i], u0);
            u1 = fmaf(sk[i + 1], s[i + 1], u1);
            u2 = fmaf(sk[i + 2], s[i + 2], u2);
            u3 = fmaf(sk[i + 3], s[i + 3], u3);
        }
        float delta = (vv - ((u0 + u1) + (u2 + u3))) * beta;
        float o0 = 0, o1 = 0, o2 = 0, o3 = 0; // S += k delta^T; o = S^T q
#pragma unroll
        for (int i = 0; i < LHD; i += 4) {
            s[i] = fmaf(sk[i], delta, s[i]);
            s[i + 1] = fmaf(sk[i + 1], delta, s[i + 1]);
            s[i + 2] = fmaf(sk[i + 2], delta, s[i + 2]);
            s[i + 3] = fmaf(sk[i + 3], delta, s[i + 3]);
            o0 = fmaf(sq[i], s[i], o0);
            o1 = fmaf(sq[i + 1], s[i + 1], o1);
            o2 = fmaf(sq[i + 2], s[i + 2], o2);
            o3 = fmaf(sq[i + 3], s[i + 3], o3);
        }
        float o = (o0 + o1) + (o2 + o3);
        float on = block_sum<128>(o * o, red); // 内部的 __syncthreads 也保证下一轮写 sq/sk 前所有线程都读完了
        float r = 1.0f / sqrtf(on / LHD + EPS);
        out[(size_t)t * V_DIM + h * LHD + j] = o * r * ssm_norm[j] * silu(z[(size_t)t * V_DIM + h * LHD + j]);
    }
#pragma unroll
    for (int i = 0; i < LHD; i++) S[i * LHD + j] = s[i];
}

// q/k 的 RMSNorm + 交错 MRoPE, 并把 k、v 写入 KV cache. grid (T, 24 + 4), block 256; y < 24 是 q 头, 其余是 k 头
extern "C" __global__ void __launch_bounds__(256) k_qk_rope(float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, const int* pos3, const float* inv_freq, float* kc, float* vc, int n_past) {
    __shared__ float sh[HD], red[8];
    int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    bool is_q = hh < N_HEAD;
    const float* p = is_q ? qg + (size_t)t * Q_DIM + hh * HD * 2 : k + (size_t)t * KV_DIM + (hh - N_HEAD) * HD; // q_proj 每个头是 [q(256), gate(256)]
    float xv = p[i];
    float ss = block_sum<256>(xv * xv, red);
    float y = xv * (1.0f / sqrtf(ss / HD + EPS)) * (is_q ? q_norm : k_norm)[i];
    sh[i] = y;
    __syncthreads();
    if (i < 64) { // 前 64 维 rotate_half: 第 f 维与 f+32 维配对; 频率 f%3==1 且 f<33 用 h 位置, f%3==2 且 f<30 用 w 位置, 其余用 t
        int f = i % 32;
        int sec = (f % 3 == 1 && f < 33) ? 1 : ((f % 3 == 2 && f < 30) ? 2 : 0);
        float ang = (float)pos3[t * 3 + sec] * inv_freq[f];
        float c = cosf(ang), sn = sinf(ang);
        y = i < 32 ? sh[i] * c - sh[i + 32] * sn : sh[i] * c + sh[i - 32] * sn;
    }
    if (is_q) {
        qg[(size_t)t * Q_DIM + hh * HD * 2 + i] = y;
        return;
    }
    int kh = hh - N_HEAD;
    size_t dst = (size_t)(n_past + t) * KV_DIM + kh * HD + i;
    kc[dst] = y;
    vc[dst] = v[(size_t)t * KV_DIM + kh * HD + i];
}

// 因果 GQA 注意力 + 输出门控. grid (T, 24), block 256 = 8 个 warp; 每个 warp 用 online softmax 处理 j = warp, warp+8, ... 的 key, 最后合并
extern "C" __global__ void __launch_bounds__(256) k_attn(const float* qg, const float* kc, const float* vc, float* att, int n_past) {
    __shared__ float sm[8], sl[8], sacc[8][HD];
    int t = blockIdx.x, h = blockIdx.y, kvh = h / (N_HEAD / N_KV), lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    int n = n_past + t + 1;
    const float* q = qg + (size_t)t * Q_DIM + h * HD * 2;
    float4 qa = *(const float4*)(q + lane * 8), qb = *(const float4*)(q + lane * 8 + 4);
    const float scale = 1.0f / sqrtf((float)HD);
    float m = -INFINITY, l = 0, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = wp; j < n; j += 8) {
        const float* kr = kc + (size_t)j * KV_DIM + kvh * HD + lane * 8;
        float4 ka = *(const float4*)kr, kb = *(const float4*)(kr + 4);
        float d = qa.x * ka.x + qa.y * ka.y + qa.z * ka.z + qa.w * ka.w + qb.x * kb.x + qb.y * kb.y + qb.z * kb.z + qb.w * kb.w;
        float sc = warp_sum(d) * scale;
        float mn = fmaxf(m, sc), c = expf(m - mn), pr = expf(sc - mn);
        l = l * c + pr;
        const float* vr = vc + (size_t)j * KV_DIM + kvh * HD + lane * 8;
        float4 va = *(const float4*)vr, vb = *(const float4*)(vr + 4);
        acc[0] = acc[0] * c + pr * va.x;
        acc[1] = acc[1] * c + pr * va.y;
        acc[2] = acc[2] * c + pr * va.z;
        acc[3] = acc[3] * c + pr * va.w;
        acc[4] = acc[4] * c + pr * vb.x;
        acc[5] = acc[5] * c + pr * vb.y;
        acc[6] = acc[6] * c + pr * vb.z;
        acc[7] = acc[7] * c + pr * vb.w;
        m = mn;
    }
#pragma unroll
    for (int e = 0; e < 8; e++) sacc[wp][lane * 8 + e] = acc[e];
    if (lane == 0) {
        sm[wp] = m;
        sl[wp] = l;
    }
    __syncthreads();
    int i = threadIdx.x;
    float M = sm[0];
    for (int w = 1; w < 8; w++) M = fmaxf(M, sm[w]);
    float L = 0, o = 0;
    for (int w = 0; w < 8; w++) { // 没分到 key 的 warp: m = -inf, l = 0, 权重为 0
        float f = expf(sm[w] - M);
        L += sl[w] * f;
        o += sacc[w][i] * f;
    }
    att[(size_t)t * ATT_DIM + h * HD + i] = o / L * sigmoid(q[HD + i]);
}
