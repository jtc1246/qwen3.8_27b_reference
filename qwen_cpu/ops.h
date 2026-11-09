#ifndef QWEN_CPU_OPS_H
#define QWEN_CPU_OPS_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include "pool.h"
#include "quants.h"

namespace {
namespace Ops {

const int MAX_THREADS = 256;
const int K_MAX = 17408; // 所有矩阵中最大的输入维度 (ffn_down)
const int ROWS = 4; // 每个并行任务处理的输出行数

struct Weight {
    const uint8_t* data = nullptr;
    uint32_t type = 0;
    int k = 0; // 输入维度, 每行长度
    int n = 0; // 输出维度, 行数
    size_t rb = 0; // 每行字节数
};

float* G_tbuf[MAX_THREADS]; // 每线程反量化缓冲, ROWS * K_MAX

void init_buffers(int n_threads) {
    for (int i = 0; i < n_threads; i++) G_tbuf[i] = (float*)aligned_alloc(64, sizeof(float) * ROWS * K_MAX);
}

namespace OpsInternal {

template <int NT> inline void micro_kernel(const float* const* wr, const float* x, int k, float* y, int ldy) { // NT 个 token × 4 行
    __m512 acc[NT][ROWS];
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < ROWS; r++) acc[t][r] = _mm512_setzero_ps();
    for (int i = 0; i < k; i += 16) {
        __m512 w0 = _mm512_loadu_ps(wr[0] + i), w1 = _mm512_loadu_ps(wr[1] + i), w2 = _mm512_loadu_ps(wr[2] + i), w3 = _mm512_loadu_ps(wr[3] + i);
        for (int t = 0; t < NT; t++) {
            __m512 xv = _mm512_loadu_ps(x + (size_t)t * k + i);
            acc[t][0] = _mm512_fmadd_ps(w0, xv, acc[t][0]);
            acc[t][1] = _mm512_fmadd_ps(w1, xv, acc[t][1]);
            acc[t][2] = _mm512_fmadd_ps(w2, xv, acc[t][2]);
            acc[t][3] = _mm512_fmadd_ps(w3, xv, acc[t][3]);
        }
    }
    for (int t = 0; t < NT; t++)
        for (int r = 0; r < ROWS; r++) y[(size_t)t * ldy + r] = _mm512_reduce_add_ps(acc[t][r]);
}

} // namespace OpsInternal

void matmul(const Weight& w, const float* x, int T, float* y) { // y[t][n] = sum_k W[n][k] * x[t][k], x 为 T×k, y 为 T×n
    Pool::run(w.n / ROWS, [&](int item, int tid) {
        int n0 = item * ROWS;
        const float* wr[ROWS];
        for (int r = 0; r < ROWS; r++) {
            const uint8_t* src = w.data + (size_t)(n0 + r) * w.rb;
            if (w.type == Gguf::T_F32) {
                wr[r] = (const float*)src;
                continue;
            }
            float* dst = G_tbuf[tid] + (size_t)r * K_MAX;
            Quants::dequant_row(w.type, src, dst, w.k);
            wr[r] = dst;
        }
        int t = 0;
        for (; t + 4 <= T; t += 4) OpsInternal::micro_kernel<4>(wr, x + (size_t)t * w.k, w.k, y + (size_t)t * w.n + n0, w.n);
        switch (T - t) {
            case 3: OpsInternal::micro_kernel<3>(wr, x + (size_t)t * w.k, w.k, y + (size_t)t * w.n + n0, w.n); break;
            case 2: OpsInternal::micro_kernel<2>(wr, x + (size_t)t * w.k, w.k, y + (size_t)t * w.n + n0, w.n); break;
            case 1: OpsInternal::micro_kernel<1>(wr, x + (size_t)t * w.k, w.k, y + (size_t)t * w.n + n0, w.n); break;
        }
    });
}

inline float dot(const float* a, const float* b, int n) { // n 为 16 的倍数
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    return _mm512_reduce_add_ps(acc);
}

inline void rmsnorm(const float* x, const float* w, float* y, int n, float eps) { // y = x / rms(x) * w, w 已经包含 +1
    float s = dot(x, x, n);
    float r = 1.0f / sqrtf(s / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}

inline float silu(float x) { return x / (1.0f + expf(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); } // 与 torch softplus 默认 threshold=20 一致

} // namespace Ops
} // namespace

#endif
