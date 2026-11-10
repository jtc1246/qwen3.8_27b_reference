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

__device__ __forceinline__ uint32_t h2u(half2 h) { return *(uint32_t*)&h; }

// tensor core: D = A(16x16, fp16, 行优先) * B(16x8, fp16, 列优先) + C (fp32). lane = 4g + t
// a0: (g, 2t..2t+1), a1: (g+8, 2t..), a2: (g, 8+2t..), a3: (g+8, 8+2t..); b0: (k = 2t.., n = g), b1: (k = 8+2t.., n = g); c0/c1: (g, 2t..), c2/c3: (g+8, 2t..)
__device__ __forceinline__ void mma16816(float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t movt(uint32_t x) { // warp 内 8x8 fp16 矩阵转置: lane (g, t) 持有 (行 g, 列 2t..2t+1), 返回转置后同位置的元素
    uint32_t y;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
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

__device__ __forceinline__ void b4h(uint32_t v, half2 off, half2* w) { // v 的 4 个字节 (无符号小整数) 拼成 half 1024+b, 再减 off, 得到 w[0] = (b0,b1), w[1] = (b2,b3), 结果是精确整数
    uint32_t lo = __byte_perm(v, 0x64646464u, 0x5140), hi = __byte_perm(v, 0x64646464u, 0x5342);
    w[0] = __hsub2(*(half2*)&lo, off);
    w[1] = __hsub2(*(half2*)&hi, off);
}

// 同 dq8, 但输出 fp16: w[i] = 第 (2i, 2i+1) 个元素; 量化值转 half 是精确的, 只有乘 scale 时舍入一次
template <int TYPE> __device__ __forceinline__ void dq8h(const uint8_t* row, int sb, int lane, half2* w) {
    if constexpr (TYPE == T_F32) {
        const float4* p = (const float4*)(row + (size_t)sb * 1024 + lane * 32);
        float4 a = p[0], b = p[1];
        w[0] = __floats2half2_rn(a.x, a.y), w[1] = __floats2half2_rn(a.z, a.w), w[2] = __floats2half2_rn(b.x, b.y), w[3] = __floats2half2_rn(b.z, b.w);
    } else if constexpr (TYPE == T_Q8_0) {
        const uint8_t* b = row + (size_t)(sb * 8 + lane / 4) * 34;
        half2 d = __half2half2(__ushort_as_half(ld16(b)));
        uint2 q = ld8_a2(b + 2 + (lane % 4) * 8);
        const half2 off = __float2half2_rn(1152.0f); // int8 异或 0x80 后是 q+128
        b4h(q.x ^ 0x80808080u, off, w);
        b4h(q.y ^ 0x80808080u, off, w + 2);
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = __hmul2(w[i], d);
    } else if constexpr (TYPE == T_Q4_K || TYPE == T_Q5_K) {
        const uint8_t* b = row + (size_t)sb * (TYPE == T_Q4_K ? 144 : 176);
        float d1, m1;
        scale_min_k4(b + 4, lane / 4, h2f(ld16(b)), h2f(ld16(b + 2)), d1, m1);
        int g = lane / 8, o = (lane % 8) * 8, sh = o >= 32 ? 4 : 0;
        uint2 q = *(const uint2*)(b + (TYPE == T_Q4_K ? 16 : 48) + 32 * g + (o & 31));
        uint32_t vx = (q.x >> sh) & 0x0F0F0F0Fu, vy = (q.y >> sh) & 0x0F0F0F0Fu;
        if constexpr (TYPE == T_Q5_K) { // 第 5 位: 每个字节取 qh 的 bit hb
            int hb = 2 * g + (o >= 32);
            uint2 h = *(const uint2*)(b + 16 + (o & 31));
            vx |= ((h.x >> hb) & 0x01010101u) << 4;
            vy |= ((h.y >> hb) & 0x01010101u) << 4;
        }
        const half2 off = __float2half2_rn(1024.0f), dh = __float2half2_rn(d1), mh = __float2half2_rn(-m1);
        b4h(vx, off, w);
        b4h(vy, off, w + 2);
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = __hfma2(w[i], dh, mh);
    } else if constexpr (TYPE == T_Q6_K) {
        const uint8_t* b = row + (size_t)sb * 210;
        int n = lane / 16, s = (lane % 16) / 4, l0 = (lane % 4) * 8;
        uint2 ql = ld8_a2(b + 64 * n + l0 + 32 * (s & 1));
        uint2 qh = ld8_a2(b + 128 + 32 * n + l0);
        float ds = h2f(ld16(b + 208)) * (float)(int8_t)b[192 + 8 * n + l0 / 16 + 2 * s];
        int sh = s >= 2 ? 4 : 0;
        uint32_t vx = ((ql.x >> sh) & 0x0F0F0F0Fu) | (((qh.x >> (2 * s)) & 0x03030303u) << 4), vy = ((ql.y >> sh) & 0x0F0F0F0Fu) | (((qh.y >> (2 * s)) & 0x03030303u) << 4);
        const half2 off = __float2half2_rn(1056.0f), dh = __float2half2_rn(ds); // 6bit 值减 32
        b4h(vx, off, w);
        b4h(vy, off, w + 2);
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = __hmul2(w[i], dh);
    }
}

// 一个 warp 算权重的一行 wr 与 token t0..t0+NTT-1 的点积 (x 是 fp16), 结果 warp 归约后每个 lane 都有; 权重每 NTT 个 token 解一次 (解成 half2)
// 每个 lane 每个超块的 8 个乘积及其部分和全用 half (half2 累加后两半相加), 再转 fp32 累加; decode 用 NTT = 1, 寄存器少
template <int TYPE, int NTT> __device__ __forceinline__ void gemv_row(const uint8_t* wr, int k, const __half* x, int t0, int T, float* acc) {
    int lane = threadIdx.x & 31, nsb = k / 256;
#pragma unroll
    for (int t = 0; t < NTT; t++) acc[t] = 0;
#pragma unroll 4
    for (int sb = 0; sb < nsb; sb++) {
        half2 w[4];
        dq8h<TYPE>(wr, sb, lane, w);
#pragma unroll
        for (int t = 0; t < NTT; t++) {
            if (t0 + t >= T) break;
            uint4 xv = *(const uint4*)(x + (size_t)(t0 + t) * k + sb * 256 + lane * 8);
            half2 s = __hmul2(w[0], *(half2*)&xv.x);
            s = __hfma2(w[1], *(half2*)&xv.y, s);
            s = __hfma2(w[2], *(half2*)&xv.z, s);
            s = __hfma2(w[3], *(half2*)&xv.w, s);
            acc[t] += __half2float(__hadd(__low2half(s), __high2half(s)));
        }
    }
#pragma unroll
    for (int t = 0; t < NTT; t++) acc[t] = warp_sum(acc[t]);
}

template <int NTT> __device__ __forceinline__ void gemv_row_rt(int type, const uint8_t* wr, int k, const __half* x, int t0, int T, float* acc) { // 运行时按量化类型分发 (同一 warp 内一致)
    switch (type) {
    case T_F32: gemv_row<T_F32, NTT>(wr, k, x, t0, T, acc); break;
    case T_Q8_0: gemv_row<T_Q8_0, NTT>(wr, k, x, t0, T, acc); break;
    case T_Q4_K: gemv_row<T_Q4_K, NTT>(wr, k, x, t0, T, acc); break;
    case T_Q5_K: gemv_row<T_Q5_K, NTT>(wr, k, x, t0, T, acc); break;
    default: gemv_row<T_Q6_K, NTT>(wr, k, x, t0, T, acc); break;
    }
}

constexpr int GB_M = 64, GB_N = 64, GB_K = 64, GS = GB_K + 8; // tensor core GEMM 的 block tile (行 x token x k) 与 smem 行跨度 (half, 补 8 个防 bank 冲突)

__device__ __forceinline__ void ldm4(uint32_t* r, const __half* p) { // ldmatrix x4: lane l 提供第 l/8 个 8x8 矩阵第 l%8 行的地址
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// prefill 的矩阵乘 (tensor core, fp16 输入, fp32 累加): y[t][n] = sum_k W[n][k] * x[t][k]
// grid (T/64, n/64 向上取整), block 128 = 2x2 个 warp, 每个 warp 算 32 行 x 32 个 token; 沿 k 每步 64, 权重用 dq8h 反量化成 fp16 写 smem
template <int TYPE> __device__ __forceinline__ void gemm_body(const uint8_t* W, uint64_t rb, int k, int n, const __half* x, int T, float* y) {
    __shared__ __align__(16) __half sw[GB_M * GS], sx[GB_N * GS];
    int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, wm = wp >> 1, wn = wp & 1, g = lane >> 2, t = lane & 3;
    int t0 = blockIdx.x * GB_N, n0 = blockIdx.y * GB_M;
    float acc[2][4][4] = {};
    for (int k0 = 0; k0 < k; k0 += GB_K) {
        __syncthreads(); // 上一步的 smem 都读完了
#pragma unroll
        for (int i = 0; i < 4; i++) { // 每个线程 4 段: 第 r 行 (token) 的第 e 个 8 元素
            int idx = tid + i * 128, r = idx >> 3, e = idx & 7;
            uint4 wv = make_uint4(0, 0, 0, 0), xv = make_uint4(0, 0, 0, 0);
            if (n0 + r < n) dq8h<TYPE>(W + (size_t)(n0 + r) * rb, k0 >> 8, ((k0 & 255) >> 3) + e, (half2*)&wv);
            if (t0 + r < T) xv = *(const uint4*)(x + (size_t)(t0 + r) * k + k0 + e * 8);
            *(uint4*)&sw[r * GS + e * 8] = wv;
            *(uint4*)&sx[r * GS + e * 8] = xv;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GB_K; kk += 16) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) ldm4(a[mi], &sw[(wm * 32 + mi * 16 + (lane & 15)) * GS + kk + (lane >> 4) * 8]);
#pragma unroll
            for (int nj = 0; nj < 2; nj++) { // x4 的 4 个矩阵: (token 0-7, k 0-7), (0-7, 8-15), (8-15, 0-7), (8-15, 8-15) = 两个 n8 块的 b0/b1
                uint32_t r[4];
                ldm4(r, &sx[(wn * 32 + nj * 16 + (lane & 7) + (lane >> 4) * 8) * GS + kk + ((lane >> 3) & 1) * 8]);
                b[nj * 2][0] = r[0], b[nj * 2][1] = r[1], b[nj * 2 + 1][0] = r[2], b[nj * 2 + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++) mma16816(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[ni][0], b[ni][1]);
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int row = n0 + wm * 32 + mi * 16 + g, tok = t0 + wn * 32 + ni * 8 + t * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) { // c0/c1: (row, tok/tok+1), c2/c3: (row+8, ...)
                int rr = row + (e >> 1) * 8, tt = tok + (e & 1);
                if (rr < n && tt < T) y[(size_t)tt * n + rr] = acc[mi][ni][e];
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
    extern "C" __global__ void __launch_bounds__(128) k_gemm_##NAME(const uint8_t* W, uint64_t rb, int k, int n, const __half* x, int T, float* y) { gemm_body<TYPE>(W, rb, k, n, x, T, y); } \
    extern "C" __global__ void __launch_bounds__(256) k_dq_##NAME(const uint8_t* W, uint64_t rb, int k, const int* rows, float* out) { dequant_body<TYPE>(W, rb, k, rows, out); }          \
    extern "C" __global__ void __launch_bounds__(256) k_embed_##NAME(const uint8_t* W, uint64_t rb, const int* tokens, const uint8_t* mask, const float* embd, float* x) { embed_body<TYPE>(W, rb, tokens, mask, embd, x); }
INST(f32, T_F32)
INST(q8_0, T_Q8_0)
INST(q4_k, T_Q4_K)
INST(q5_k, T_Q5_K)
INST(q6_k, T_Q6_K)

struct MmGroup { // decode GEMV 的一组矩阵 (共享输入 x 和 k), 与 host 端布局一致
    uint64_t w[4], rb[4], y[4];
    int n[4], type[4];
    int cnt, glu; // glu (k_gemv_glu): 矩阵 0/1 是 ffn gate/up, 行按矩阵 0 分配, y[0] = silu(gate)·up (fp16); 否则 (k_gemv) 行按 cnt 个矩阵依次拼接, 输出 fp32
};

// decode (T < 16) 的矩阵乘: grid 总行数/2, block 64 = 2 个 warp, 每个 warp 一行; 一组矩阵一次 launch, 块小使各 SM 负载均衡
// GLU 和普通版分开, T = 1 和 T > 1 分开 (NTT): 寄存器少, n = 5120 的 2560 个 block 能一次全部驻留 (每 SM 最多 24 个 block)
template <bool GLU, int NTT> __device__ __forceinline__ void gemv_body(const MmGroup& g, int k, const __half* x, int T) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 2 + (threadIdx.x >> 5), m = 0;
    while (m < g.cnt - 1 && row >= g.n[m]) row -= g.n[m++];
    if (row >= g.n[m]) return;
    for (int t0 = 0; t0 < T; t0 += NTT) {
        float acc[NTT];
        gemv_row_rt<NTT>(g.type[m], (const uint8_t*)g.w[m] + (size_t)row * g.rb[m], k, x, t0, T, acc);
        if constexpr (GLU) {
            float au[NTT];
            gemv_row_rt<NTT>(g.type[1], (const uint8_t*)g.w[1] + (size_t)row * g.rb[1], k, x, t0, T, au);
            if (lane == 0)
#pragma unroll
                for (int t = 0; t < NTT; t++)
                    if (t0 + t < T) ((__half*)g.y[0])[(size_t)(t0 + t) * g.n[0] + row] = __float2half_rn(silu(acc[t]) * au[t]);
        } else if (lane == 0) {
#pragma unroll
            for (int t = 0; t < NTT; t++)
                if (t0 + t < T) ((float*)g.y[m])[(size_t)(t0 + t) * g.n[m] + row] = acc[t];
        }
    }
}

extern "C" __global__ void __launch_bounds__(64) k_gemv(MmGroup g, int k, const __half* x, int T) { gemv_body<false, NT>(g, k, x, T); }
extern "C" __global__ void __launch_bounds__(64) k_gemv_glu(MmGroup g, int k, const __half* x, int T) { gemv_body<true, NT>(g, k, x, T); }
extern "C" __global__ void __launch_bounds__(64) k_gemv1(MmGroup g, int k, const __half* x, int T) { gemv_body<false, 1>(g, k, x, T); }
extern "C" __global__ void __launch_bounds__(64) k_gemv1_glu(MmGroup g, int k, const __half* x, int T) { gemv_body<true, 1>(g, k, x, T); }

// x += add (add 可为空); y = rmsnorm(x) * w (fp16, 作为 matmul 输入); hidden 非空时保存最后一个 token 的 x. grid T, block 1024, 每线程 5 个元素
extern "C" __global__ void __launch_bounds__(1024) k_add_norm(float* x, const float* add, const float* w, __half* y, float* hidden, int T) {
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
        y[(size_t)t * D + c] = __float2half_rn(v[i] * r * w[c]);
        if (hidden && t == T - 1) hidden[c] = v[i];
    }
}

extern "C" __global__ void k_silu_mul(const float* g, const float* u, __half* h, int n) { // h = silu(g) * u, fp16 给 down 用
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half_rn(silu(g[i]) * u[i]);
}

// DeltaNet 预处理 (各 token 独立, 并行): 因果 depthwise conv1d (k=4) + SiLU, q/k 的 L2 归一化 (q 再乘 1/sqrt(128)), q·k, beta = sigmoid(b), decay = exp(ssm_a * softplus(a + dt_bias))
// grid (T, 16 个 qk 头), block 128; qkv: matmul 原始输出 [T][10240], cstate: conv 状态 [10240][3] (旧的在前, 此处只读); 输出 cv [T][10240], qk [T][16]; a/b [T][48] 原地改成 decay/beta
extern "C" __global__ void __launch_bounds__(128) k_dn_prep(const float* qkv, const float* cstate, const float* cw, float* cv, float* a, float* b, const float* ssm_a, const float* dt_bias, float* qk, int T) {
    __shared__ float red[4];
    int t = blockIdx.x, kh = blockIdx.y, j = threadIdx.x;
    auto conv = [&](int c) { // 第 t 个 token 的输入序号 t-3..t, 负的取 conv 状态
        float o = 0;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            int tt = t - 3 + d;
            o = fmaf(cw[c * 4 + d], tt >= 0 ? qkv[(size_t)tt * QKV_DIM + c] : cstate[c * 3 + 3 + tt], o);
        }
        return silu(o);
    };
    int cq = kh * LHD + j, ck = KEY_DIM + kh * LHD + j;
    float qv = conv(cq), kv = conv(ck);
    float qn = block_sum<128>(qv * qv, red), kn = block_sum<128>(kv * kv, red);
    qv *= 1.0f / sqrtf(qn + 1e-6f) * (1.0f / sqrtf((float)LHD));
    kv *= 1.0f / sqrtf(kn + 1e-6f);
    cv[(size_t)t * QKV_DIM + cq] = qv;
    cv[(size_t)t * QKV_DIM + ck] = kv;
    float d = block_sum<128>(qv * kv, red);
    if (j == 0) qk[t * LK + kh] = d;
    for (int i = j; i < V_DIM / LK; i += 128) { // 本 block 顺便算 1/16 的 v 通道
        int c = 2 * KEY_DIM + kh * (V_DIM / LK) + i;
        cv[(size_t)t * QKV_DIM + c] = conv(c);
    }
    if (kh == 0 && j < LV) {
        int idx = t * LV + j;
        b[idx] = sigmoid(b[idx]);
        a[idx] = expf(ssm_a[j] * softplus(a[idx] + dt_bias[j]));
    }
}

constexpr int DN_TT = 16, DN_KS = 36; // k_dn_rec 每批预取的 token 数; k/q 每个 32 行的行段在 smem 中占 36 个 float (防 bank 冲突)

__device__ __forceinline__ void cp_async(void* dst, const void* src, int bytes) { // global -> smem 异步拷贝 (4 / 16 字节)
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    if (bytes == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d), "l"(src));
}

// Gated DeltaNet 递推 (输入已由 k_dn_prep 处理), 状态和计算都是 fp32. grid (48 个 v 头, 4), block 128, 每个 block 32 列
// lane = 4 * 列 + 行段 r, 线程在寄存器里持有 S[32r .. 32r+31][j]; 每个 token (S 为旧状态):
// a = kᵀS, b = qᵀS (4 个行段 shuffle 求和), delta = (v - decay·a)·beta, o = decay·b + (q·k)·delta, S = decay·S + k·delta; o (未归一化) 写 of [T][6144]
// 每批 16 个 token 的 k/q/v/标量用 cp.async 预取到 smem (双缓冲); 开头顺便更新 conv 状态 (k_dn_prep 已读完旧状态): 新状态 = 输入序号 T-3..T-1 的原始 qkv
extern "C" __global__ void __launch_bounds__(128) k_dn_rec(const float* qkv, float* cstate, const float* cv, const float* a, const float* b, const float* qk, float* state, float* of, int T) {
    __shared__ __align__(16) float sk[2][DN_TT][4 * DN_KS], sq[2][DN_TT][4 * DN_KS], sv[2][DN_TT][32], sc[2][DN_TT][4]; // sc: beta, decay, q·k
    int gt = (blockIdx.y * gridDim.x + blockIdx.x) * 128 + threadIdx.x;
    if (gt < QKV_DIM) {
        float ns[3];
#pragma unroll
        for (int i = 0; i < 3; i++) {
            int tt = T - 3 + i;
            ns[i] = tt >= 0 ? qkv[(size_t)tt * QKV_DIM + gt] : cstate[gt * 3 + T + i];
        }
#pragma unroll
        for (int i = 0; i < 3; i++) cstate[gt * 3 + i] = ns[i];
    }
    int tid = threadIdx.x, h = blockIdx.x, kh = h % LK, lane = tid & 31, jl = (tid >> 5) * 8 + (lane >> 2), j = blockIdx.y * 32 + jl, r = lane & 3; // GGUF 的 v 头是 tiled 顺序, v 头 h 对应 qk 头 h % 16
    auto load = [&](int bi, int tb) { // 预取 token [tb, tb+16) 到第 bi 个缓冲
        int n = min(DN_TT, T - tb);
        for (int i = tid; i < n * 64; i += 128) { // k/q: 每 token 各 32 个 16 字节
            int tt = i >> 6, c = i & 63, isq = c >> 5, seg = c & 31; // seg: 128 维中的第 seg 个 4 元素
            const float* src = cv + (size_t)(tb + tt) * QKV_DIM + (isq ? 0 : KEY_DIM) + kh * LHD + seg * 4;
            cp_async(&(isq ? sq : sk)[bi][tt][(seg >> 3) * DN_KS + (seg & 7) * 4], src, 16);
        }
        for (int i = tid; i < n * 8; i += 128) { // v: 每 token 本 block 的 32 列 = 8 个 16 字节
            int tt = i >> 3, c = i & 7;
            cp_async(&sv[bi][tt][c * 4], cv + (size_t)(tb + tt) * QKV_DIM + 2 * KEY_DIM + h * LHD + blockIdx.y * 32 + c * 4, 16);
        }
        if (tid < n * 3) {
            int tt = tid / 3, w = tid % 3;
            cp_async(&sc[bi][tt][w], w == 0 ? b + (tb + tt) * LV + h : (w == 1 ? a + (tb + tt) * LV + h : qk + (tb + tt) * LK + kh), 4);
        }
        asm volatile("cp.async.commit_group;");
    };
    float* S = state + (size_t)h * LHD * LHD;
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = S[(r * 32 + i) * LHD + j];
    load(0, 0);
    for (int tb = 0, bi = 0; tb < T; tb += DN_TT, bi ^= 1) {
        if (tb + DN_TT < T) {
            load(bi ^ 1, tb + DN_TT);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int n = min(DN_TT, T - tb);
        for (int tt = 0; tt < n; tt++) {
            const float4* kp = (const float4*)&sk[bi][tt][r * DN_KS];
            const float4* qp = (const float4*)&sq[bi][tt][r * DN_KS];
            float vj = sv[bi][tt][jl], bt = sc[bi][tt][0], dc = sc[bi][tt][1], qkt = sc[bi][tt][2];
            float kk[32], qq[32], av[4] = {0, 0, 0, 0}, bv[4] = {0, 0, 0, 0};
#pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 x = kp[i], y = qp[i];
                kk[4 * i] = x.x, kk[4 * i + 1] = x.y, kk[4 * i + 2] = x.z, kk[4 * i + 3] = x.w;
                qq[4 * i] = y.x, qq[4 * i + 1] = y.y, qq[4 * i + 2] = y.z, qq[4 * i + 3] = y.w;
            }
#pragma unroll
            for (int i = 0; i < 32; i++) {
                av[i & 3] = fmaf(kk[i], s[i], av[i & 3]);
                bv[i & 3] = fmaf(qq[i], s[i], bv[i & 3]);
            }
            float sa = (av[0] + av[1]) + (av[2] + av[3]), sb = (bv[0] + bv[1]) + (bv[2] + bv[3]);
            sa += __shfl_xor_sync(0xffffffffu, sa, 1);
            sb += __shfl_xor_sync(0xffffffffu, sb, 1);
            sa += __shfl_xor_sync(0xffffffffu, sa, 2);
            sb += __shfl_xor_sync(0xffffffffu, sb, 2);
            float delta = (vj - dc * sa) * bt, o = dc * sb + qkt * delta;
#pragma unroll
            for (int i = 0; i < 32; i++) s[i] = fmaf(kk[i], delta, s[i] * dc);
            if (r == 0) of[(size_t)(tb + tt) * V_DIM + h * LHD + j] = o;
        }
        __syncthreads(); // 本缓冲读完, 下一轮才能往里预取
    }
#pragma unroll
    for (int i = 0; i < 32; i++) S[(r * 32 + i) * LHD + j] = s[i];
}

// DeltaNet 输出的门控 RMSNorm: out = rmsnorm(o) * ssm_norm * silu(z), fp16 (ssm_out 的输入). grid (T, 48), block 128
extern "C" __global__ void __launch_bounds__(128) k_dn_post(const float* of, const float* z, const float* ssm_norm, __half* out) {
    __shared__ float red[4];
    int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    size_t idx = (size_t)t * V_DIM + h * LHD + j;
    float o = of[idx], on = block_sum<128>(o * o, red);
    out[idx] = __float2half_rn(o * (1.0f / sqrtf(on / LHD + EPS)) * ssm_norm[j] * silu(z[idx]));
}

// KV cache 是 q8_0 (同 llama.cpp): 每行 (token) 1024 个 int8 [ctx][1024], 每 32 维一个 fp16 scale [ctx][32] (4 个头 x 8 块), 分开存放保证 16 字节对齐
__device__ __forceinline__ void kv_q8_store(float x, int8_t* q, __half* d) { // 一个 warp 正好是一个块, 语义同 llama.cpp quantize_row_q8_0
    float am = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    float dd = am / 127.0f, id = dd != 0.0f ? 1.0f / dd : 0.0f;
    *q = (int8_t)roundf(x * id);
    if ((threadIdx.x & 31) == 0) *d = __float2half_rn(dd);
}

__device__ __forceinline__ uint4 kv_dq8(uint2 q, half2 d) { // 8 个 int8 乘 scale 得到 8 个 half: int8 转 half 是精确的, 只在乘 d 时舍入一次, 与 half(q * d) 相同
    const half2 off = __float2half2_rn(1152.0f); // 异或 0x80 后是 q+128
    half2 w[4];
    b4h(q.x ^ 0x80808080u, off, w);
    b4h(q.y ^ 0x80808080u, off, w + 2);
    return make_uint4(h2u(__hmul2(w[0], d)), h2u(__hmul2(w[1], d)), h2u(__hmul2(w[2], d)), h2u(__hmul2(w[3], d)));
}

// q/k 的 RMSNorm + 交错 MRoPE, 并把 k、v 写入 KV cache (q8_0). grid (T, 24 + 4), block 256; y < 24 是 q 头, 其余是 k 头
extern "C" __global__ void __launch_bounds__(256) k_qk_rope(float* qg, const float* k, const float* v, const float* q_norm, const float* k_norm, const int* pos3, const float* inv_freq, int8_t* kc, int8_t* vc, __half* ks, __half* vs, const int* np) { // np: 显存里的 n_past (CUDA Graph 重放时会变)
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
    size_t row = (size_t)(*np + t), dst = row * KV_DIM + kh * HD + i, ds = row * 32 + kh * 8 + i / 32;
    kv_q8_store(y, kc + dst, ks + ds);
    kv_q8_store(v[(size_t)t * KV_DIM + kh * HD + i], vc + dst, vs + ds);
}

constexpr int FA_Q = 64, FA_K = 32, FS = HD + 8; // prefill flash attention: 每 block 64 个 query, 每次 32 个 key, smem 行跨度 (half)

__device__ __forceinline__ void ldm4t(uint32_t* r, const __half* p) { // ldmatrix x4 + 转置
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// prefill (T > 1) 因果 GQA 注意力 + 输出门控, flash attention + tensor core (fp16 输入, fp32 累加)
// grid (T/64 向上取整, 24), block 128 = 4 个 warp, 每个 warp 16 个 query; 动态 smem: sq [64][FS], sk/sv [2][32][FS] (共 101376 字节, 正好是单 block 上限)
// K/V 每 32 个 key 一轮, smem 是反量化后的 fp16, 双缓冲: 算本轮 Q·Kᵀ 时把下一轮 K 的 int8 读进寄存器, 算完写入另一个缓冲, 再同样处理 V (K、V 分开以减少寄存器)
// S = Q·Kᵀ (16x32), online softmax (fp32), P 转 fp16 后 O += P·V (16x256), V 用 ldmatrix.trans 得到 B fragment
extern "C" __global__ void __launch_bounds__(128) k_attn_pf(const float* qg, const int8_t* kc, const int8_t* vc, const __half* ks, const __half* vs, __half* att, int n_past, int T) {
    extern __shared__ __align__(16) __half smem[];
    __half *sq = smem, *skb = sq + FA_Q * FS, *svb = skb + 2 * FA_K * FS;
    int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, g = lane >> 2, t = lane & 3;
    int t0 = blockIdx.x * FA_Q, h = blockIdx.y, kvh = h / (N_HEAD / N_KV);
    for (int i = tid; i < FA_Q * HD / 4; i += 128) { // q 转 fp16 写 smem, 超出 T 的行写 0
        int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        float4 v = t0 + r < T ? *(const float4*)(qg + (size_t)(t0 + r) * Q_DIM + h * HD * 2 + c) : make_float4(0, 0, 0, 0);
        *(half2*)&sq[r * FS + c] = __floats2half2_rn(v.x, v.y);
        *(half2*)&sq[r * FS + c + 2] = __floats2half2_rn(v.z, v.w);
    }
    const float scale = 1.0f / sqrtf((float)HD);
    int qp0 = n_past + t0 + wp * 16 + g, qp1 = qp0 + 8; // 本线程两行 query (c0/c1 行, c2/c3 行) 的绝对位置
    int nk = n_past + min(t0 + FA_Q, T); // 本 block 要看的 key 数
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0, l1 = 0, o[32][4] = {}; // o[d]: 维度 8d + 2t + {0,1}
    uint4 rq[4]; // 一轮 K 或 V 的 int8: 32 行 x 256 维, 每线程 4 个 16 字节 (第 i 个: 行 i/16, 维度 (i%16)*16 起)
    __half rd[4];
    auto fetch = [&](const int8_t* c, const __half* s, int kb) { // key [kb, kb+32) 的 K 或 V 读进寄存器; 超出 nk 的行夹到 nk-1 (分数被 mask, 权重为 0)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = tid + j * 128;
            size_t row = min(kb + (i >> 4), nk - 1);
            rq[j] = *(const uint4*)(c + row * KV_DIM + kvh * HD + (i & 15) * 16);
            rd[j] = s[row * 32 + kvh * 8 + (i & 15) / 2];
        }
    };
    auto store = [&](__half* dst) { // 寄存器里的一轮反量化成 fp16 写入 smem 缓冲 dst [32][FS]
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = tid + j * 128;
            half2 d = __half2half2(rd[j]);
            __half* p = dst + (i >> 4) * FS + (i & 15) * 16;
            *(uint4*)p = kv_dq8(make_uint2(rq[j].x, rq[j].y), d);
            *(uint4*)(p + 8) = kv_dq8(make_uint2(rq[j].z, rq[j].w), d);
        }
    };
    fetch(kc, ks, 0);
    store(skb);
    fetch(vc, vs, 0);
    store(svb);
    for (int kb = 0, bi = 0; kb < nk; kb += FA_K, bi ^= 1) {
        bool more = kb + FA_K < nk;
        if (more) fetch(kc, ks, kb + FA_K); // 下一轮的 K 在算本轮 Q·Kᵀ 时读取
        __syncthreads(); // 本轮 K/V 已写入 smem, 且所有线程都算完了上一轮 (另一个缓冲可以覆盖); 第一轮: sq 也写完了
        const __half *sk = skb + bi * FA_K * FS, *sv = svb + bi * FA_K * FS;
        float s[4][4] = {}; // s[ni]: key kb + 8ni + 2t + {0,1}; [0..1] 是 qp0 行, [2..3] 是 qp1 行
#pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t a[4];
            ldm4(a, &sq[(wp * 16 + (lane & 15)) * FS + kk + (lane >> 4) * 8]);
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                uint32_t b[4];
                ldm4(b, &sk[(nj * 16 + (lane & 7) + (lane >> 4) * 8) * FS + kk + ((lane >> 3) & 1) * 8]);
                mma16816(s[nj * 2][0], s[nj * 2][1], s[nj * 2][2], s[nj * 2][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma16816(s[nj * 2 + 1][0], s[nj * 2 + 1][1], s[nj * 2 + 1][2], s[nj * 2 + 1][3], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
        if (more) {
            store(skb + (bi ^ 1) * FA_K * FS);
            fetch(vc, vs, kb + FA_K); // 下一轮的 V 在算 softmax 和 P·V 时读取
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) { // 因果 mask; key 0 对所有行都有效, 所以第一轮 max 有限
                int key = kb + ni * 8 + t * 2 + (e & 1);
                s[ni][e] = key <= (e < 2 ? qp0 : qp1) ? s[ni][e] * scale : -INFINITY;
                if (e < 2) mx0 = fmaxf(mx0, s[ni][e]);
                else mx1 = fmaxf(mx1, s[ni][e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float c0 = expf(m0 - mx0), c1 = expf(m1 - mx1), ps0 = 0, ps1 = 0;
        m0 = mx0, m1 = mx1;
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            s[ni][0] = expf(s[ni][0] - mx0), s[ni][1] = expf(s[ni][1] - mx0), s[ni][2] = expf(s[ni][2] - mx1), s[ni][3] = expf(s[ni][3] - mx1);
            ps0 += s[ni][0] + s[ni][1];
            ps1 += s[ni][2] + s[ni][3];
        }
        l0 = l0 * c0 + ps0, l1 = l1 * c1 + ps1;
#pragma unroll
        for (int d = 0; d < 32; d++) o[d][0] *= c0, o[d][1] *= c0, o[d][2] *= c1, o[d][3] *= c1;
#pragma unroll
        for (int j = 0; j < 2; j++) { // P 的 A fragment 直接由 S 的累加器得到: key 16j..16j+15
            uint32_t pa0 = h2u(__floats2half2_rn(s[2 * j][0], s[2 * j][1])), pa1 = h2u(__floats2half2_rn(s[2 * j][2], s[2 * j][3]));
            uint32_t pa2 = h2u(__floats2half2_rn(s[2 * j + 1][0], s[2 * j + 1][1])), pa3 = h2u(__floats2half2_rn(s[2 * j + 1][2], s[2 * j + 1][3]));
#pragma unroll
            for (int dp = 0; dp < 16; dp++) { // x4.trans 的 4 个矩阵: (key 0-7, 维度 16dp..+7), (key 8-15, 同), (key 0-7, 16dp+8..), (key 8-15, 同)
                uint32_t b[4];
                ldm4t(b, &sv[(j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FS + dp * 16 + (lane >> 4) * 8]);
                mma16816(o[dp * 2][0], o[dp * 2][1], o[dp * 2][2], o[dp * 2][3], pa0, pa1, pa2, pa3, b[0], b[1]);
                mma16816(o[dp * 2 + 1][0], o[dp * 2 + 1][1], o[dp * 2 + 1][2], o[dp * 2 + 1][3], pa0, pa1, pa2, pa3, b[2], b[3]);
            }
        }
        if (more) store(svb + (bi ^ 1) * FA_K * FS);
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    int tok0 = t0 + wp * 16 + g, tok1 = tok0 + 8;
#pragma unroll
    for (int d = 0; d < 32; d++) {
        int dim = d * 8 + t * 2;
        if (tok0 < T) {
            const float* gp = qg + (size_t)tok0 * Q_DIM + h * HD * 2 + HD + dim;
            *(half2*)&att[(size_t)tok0 * ATT_DIM + h * HD + dim] = __floats2half2_rn(o[d][0] / l0 * sigmoid(gp[0]), o[d][1] / l0 * sigmoid(gp[1]));
        }
        if (tok1 < T) {
            const float* gp = qg + (size_t)tok1 * Q_DIM + h * HD * 2 + HD + dim;
            *(half2*)&att[(size_t)tok1 * ATT_DIM + h * HD + dim] = __floats2half2_rn(o[d][2] / l1 * sigmoid(gp[0]), o[d][3] / l1 * sigmoid(gp[1]));
        }
    }
}

constexpr int ATT_SPLIT = 78; // decode 注意力沿序列最多切的段数: 4 x 78 = 312 个 block = 156 SM x 2 (寄存器限制每 SM 2 个), 一波跑完

__device__ __forceinline__ void att_split(int n, int& kps, int& ns) { kps = (max(128, (n + ATT_SPLIT - 1) / ATT_SPLIT) + 63) / 64 * 64, ns = (n + kps - 1) / kps; } // 每段至少 128 个 key, 段数不超过 ATT_SPLIT

// decode (T = 1) 注意力, flash-decoding + tensor core (fp16 输入, fp32 累加). grid (4 个 kv 头, ATT_SPLIT), block 128 = 4 个 warp; n = *np + 1, 超出段数的 block 直接退出
// 一个 block 处理一个 kv 头下的 6 个 q 头 (K/V 只读一次) 和 [sp*kps, min(n, sp*kps+kps)) 段的 key, 段内每 16 个 key 一组轮流分给 4 个 warp
// S = Q·Kᵀ: A 的 16 行中前 6 行是 q 头, 其余为 0; head 维重排: 第 c 个 32 维块里 lane (g, t) 读 K/V 第 g 行的 [c*32 + 8t, +8), int8 8 字节连续, 乘该块 scale 反量化成 fp16
// O += P·V: P 由 S 的累加器直接转 fp16 作为 A; V 的 B fragment 由 movmatrix 转置得到, 结果中 lane (g, t) 持有 q 头 g 的维度 [c*32 + 8t, +8)
// 输出未归一化的 po [nsplit][24][256] 和 pml [nsplit][24] = (max, sum), 由 k_attn_merge 合并
extern "C" __global__ void __launch_bounds__(128) k_attn_dec(const float* qg, const int8_t* kc, const int8_t* vc, const __half* ks, const __half* vs, float* po, float2* pml, const int* np) {
    constexpr int G = N_HEAD / N_KV;
    __shared__ float sm[4][G], sl[4][G], so[4][G][HD];
    int kvh = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, wp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3, n = *np + 1, kps, ns;
    att_split(n, kps, ns);
    if (sp >= ns) return;
    int j0 = sp * kps, j1 = min(n, j0 + kps);
    uint32_t qa[8][4] = {}; // q 头 g 的维度 [c*32 + 8t, +8), g >= 6 为 0
    if (g < G) {
#pragma unroll
        for (int c = 0; c < 8; c++) {
            const float* qp = qg + (kvh * G + g) * HD * 2 + c * 32 + t * 8;
            float4 a = *(const float4*)qp, b = *(const float4*)(qp + 4);
            qa[c][0] = h2u(__floats2half2_rn(a.x, a.y)), qa[c][1] = h2u(__floats2half2_rn(a.z, a.w)), qa[c][2] = h2u(__floats2half2_rn(b.x, b.y)), qa[c][3] = h2u(__floats2half2_rn(b.z, b.w));
        }
    }
    const float scale = 1.0f / sqrtf((float)HD);
    float m = -INFINITY, l = 0, o[8][4][2] = {}, dz0 = 0, dz1 = 0; // o[c][r]: q 头 g 的维度 c*32 + 8t + 2r + {0,1}; dz: 补 0 行的输出, 丢弃
    for (int jb = j0 + wp * 16; jb < j1; jb += 64) {
        int ka = min(jb + g, n - 1), kb = min(jb + 8 + g, n - 1); // 本 lane 读的两行 key, 越界的夹到 n-1 (分数置 -inf)
        { // 本组的 V 和下一组的 K 预取到 L2 (每 lane 一个 128 字节行): V 要等 softmax 算完才读, 不预取时每组要串行等两次显存延迟
            size_t pv = (size_t)min(jb + (lane >> 1), n - 1) * KV_DIM, pk = (size_t)min(jb + 64 + (lane >> 1), n - 1) * KV_DIM;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(vc + pv + kvh * HD + (lane & 1) * 128));
            if (jb + 64 < j1) asm volatile("prefetch.global.L2 [%0];" ::"l"(kc + pk + kvh * HD + (lane & 1) * 128));
        }
        const int8_t* kra = kc + (size_t)ka * KV_DIM + kvh * HD + t * 8;
        const int8_t* krb = kc + (size_t)kb * KV_DIM + kvh * HD + t * 8;
        uint4 da = *(const uint4*)(ks + (size_t)ka * 32 + kvh * 8), db = *(const uint4*)(ks + (size_t)kb * 32 + kvh * 8); // 两行各自 8 个 32 维块的 scale
        float s[4] = {}; // key jb+2t, jb+2t+1, jb+8+2t, jb+9+2t 的分数
#pragma unroll
        for (int c = 0; c < 8; c++) { // 第 c 个 32 维块正好是一个 q8_0 块
            uint4 x = kv_dq8(*(const uint2*)(kra + c * 32), __half2half2(((const __half*)&da)[c])), y = kv_dq8(*(const uint2*)(krb + c * 32), __half2half2(((const __half*)&db)[c]));
            mma16816(s[0], s[1], dz0, dz1, qa[c][0], 0, qa[c][1], 0, x.x, x.y);
            mma16816(s[0], s[1], dz0, dz1, qa[c][2], 0, qa[c][3], 0, x.z, x.w);
            mma16816(s[2], s[3], dz0, dz1, qa[c][0], 0, qa[c][1], 0, y.x, y.y);
            mma16816(s[2], s[3], dz0, dz1, qa[c][2], 0, qa[c][3], 0, y.z, y.w);
        }
        float mx = m;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int key = jb + (i >> 1) * 8 + t * 2 + (i & 1);
            s[i] = key < j1 ? s[i] * scale : -INFINITY;
            mx = fmaxf(mx, s[i]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1)); // 同一行 (q 头) 的 4 个 lane 求 max; key jb 一定有效, mx 有限
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float cr = expf(m - mx), ps = 0;
        m = mx;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            s[i] = expf(s[i] - mx);
            ps += s[i];
        }
        l = l * cr + ps;
        uint32_t pa0 = h2u(__floats2half2_rn(s[0], s[1])), pa2 = h2u(__floats2half2_rn(s[2], s[3]));
        const int8_t* vra = vc + (size_t)ka * KV_DIM + kvh * HD + t * 8;
        const int8_t* vrb = vc + (size_t)kb * KV_DIM + kvh * HD + t * 8;
        da = *(const uint4*)(vs + (size_t)ka * 32 + kvh * 8), db = *(const uint4*)(vs + (size_t)kb * 32 + kvh * 8);
#pragma unroll
        for (int c = 0; c < 8; c++) {
            uint4 x = kv_dq8(*(const uint2*)(vra + c * 32), __half2half2(((const __half*)&da)[c])), y = kv_dq8(*(const uint2*)(vrb + c * 32), __half2half2(((const __half*)&db)[c]));
            uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
            for (int r = 0; r < 4; r++) {
                o[c][r][0] *= cr;
                o[c][r][1] *= cr;
                mma16816(o[c][r][0], o[c][r][1], dz0, dz1, pa0, 0, pa2, 0, movt(xs[r]), movt(ys[r]));
            }
        }
    }
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (g < G) {
#pragma unroll
        for (int c = 0; c < 8; c++)
#pragma unroll
            for (int r = 0; r < 4; r++) {
                so[wp][g][c * 32 + t * 8 + r * 2] = o[c][r][0];
                so[wp][g][c * 32 + t * 8 + r * 2 + 1] = o[c][r][1];
            }
        if (t == 0) {
            sm[wp][g] = m;
            sl[wp][g] = l;
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * HD; idx += 128) { // 合并 4 个 warp; 段内至少有一个 key, M 有限, 没分到 key 的 warp (m = -inf) 权重为 0
        int gg = idx / HD, i = idx % HD;
        float M = fmaxf(fmaxf(sm[0][gg], sm[1][gg]), fmaxf(sm[2][gg], sm[3][gg])), L = 0, oo = 0;
#pragma unroll
        for (int w = 0; w < 4; w++) {
            float f = expf(sm[w][gg] - M);
            L += sl[w][gg] * f;
            oo += so[w][gg][i] * f;
        }
        int h = kvh * G + gg;
        po[((size_t)sp * N_HEAD + h) * HD + i] = oo;
        if (i == 0) pml[sp * N_HEAD + h] = make_float2(M, L);
    }
}

// 合并 k_attn_dec 各段: grid 24 (q 头), block 256 (维度); att = o / L * sigmoid(gate), 写 fp16
extern "C" __global__ void __launch_bounds__(256) k_attn_merge(const float* qg, const float* po, const float2* pml, const int* np, __half* att) {
    int h = blockIdx.x, i = threadIdx.x, kps, nsplit;
    att_split(*np + 1, kps, nsplit);
    float M = -INFINITY, L = 0, o = 0;
    for (int s = 0; s < nsplit; s++) M = fmaxf(M, pml[s * N_HEAD + h].x);
    for (int s = 0; s < nsplit; s++) {
        float2 ml = pml[s * N_HEAD + h];
        float f = expf(ml.x - M);
        L += ml.y * f;
        o += po[((size_t)s * N_HEAD + h) * HD + i] * f;
    }
    att[h * HD + i] = __float2half_rn(o / L * sigmoid(qg[h * HD * 2 + HD + i]));
}
