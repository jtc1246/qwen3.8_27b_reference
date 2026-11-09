#ifndef QWEN_CPU_QUANTS_H
#define QWEN_CPU_QUANTS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include "gguf.h"

namespace {
namespace Quants {

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
struct BlockQ4K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BlockQ5K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct BlockQ6K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(BlockQ8_0) == 34 && sizeof(BlockQ4K) == 144 && sizeof(BlockQ5K) == 176 && sizeof(BlockQ6K) == 210, "block 大小不对");

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

size_t row_bytes(uint32_t type, int64_t k) {
    switch (type) {
        case Gguf::T_F32: return k * 4;
        case Gguf::T_Q8_0: return k / 32 * sizeof(BlockQ8_0);
        case Gguf::T_Q4_K: return k / 256 * sizeof(BlockQ4K);
        case Gguf::T_Q5_K: return k / 256 * sizeof(BlockQ5K);
        case Gguf::T_Q6_K: return k / 256 * sizeof(BlockQ6K);
    }
    fprintf(stderr, "❌ 不支持的量化类型 %u\n", type);
    _Exit(1);
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) { // 8 个子块的 6bit scale/min 打包在 12 字节里
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
        return;
    }
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
}

namespace Ref { // 标量参考实现, 与 ggml 的 dequantize_row_* 逐行对应, 用于自测

void q8_0(const void* src, float* y, int64_t k) {
    const BlockQ8_0* x = (const BlockQ8_0*)src;
    for (int64_t i = 0; i < k / 32; i++) {
        float d = fp16(x[i].d);
        for (int j = 0; j < 32; j++) *y++ = x[i].qs[j] * d;
    }
}

void q4_k(const void* src, float* y, int64_t k) {
    const BlockQ4K* x = (const BlockQ4K*)src;
    for (int64_t i = 0; i < k / 256; i++) {
        const uint8_t* q = x[i].qs;
        float d = fp16(x[i].d), mn = fp16(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void q5_k(const void* src, float* y, int64_t k) {
    const BlockQ5K* x = (const BlockQ5K*)src;
    for (int64_t i = 0; i < k / 256; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        float d = fp16(x[i].d), mn = fp16(x[i].dmin);
        int is = 0;
        uint8_t sc, m, u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void q6_k(const void* src, float* y, int64_t k) {
    const BlockQ6K* x = (const BlockQ6K*)src;
    for (int64_t i = 0; i < k / 256; i++) {
        float d = fp16(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

} // namespace Ref

namespace QuantsInternal {

inline __m512 i8x16_to_ps(__m128i v) { return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v)); }
inline __m512 u8x16_to_ps(__m128i v) { return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v)); }

void q8_0(const void* src, float* y, int64_t k) {
    const BlockQ8_0* x = (const BlockQ8_0*)src;
    for (int64_t i = 0; i < k / 32; i++) {
        __m512 d = _mm512_set1_ps(fp16(x[i].d));
        __m256i q = _mm256_loadu_si256((const __m256i*)x[i].qs);
        _mm512_storeu_ps(y, _mm512_mul_ps(i8x16_to_ps(_mm256_castsi256_si128(q)), d));
        _mm512_storeu_ps(y + 16, _mm512_mul_ps(i8x16_to_ps(_mm256_extracti128_si256(q, 1)), d));
        y += 32;
    }
}

void q4_k(const void* src, float* y, int64_t k) {
    const BlockQ4K* x = (const BlockQ4K*)src;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    for (int64_t i = 0; i < k / 256; i++) {
        float d = fp16(x[i].d), mn = fp16(x[i].dmin);
        for (int j = 0; j < 4; j++) {
            uint8_t sc, m;
            get_scale_min_k4(2 * j, x[i].scales, &sc, &m);
            __m512 d1 = _mm512_set1_ps(d * sc), m1 = _mm512_set1_ps(mn * m);
            get_scale_min_k4(2 * j + 1, x[i].scales, &sc, &m);
            __m512 d2 = _mm512_set1_ps(d * sc), m2 = _mm512_set1_ps(mn * m);
            __m256i qb = _mm256_loadu_si256((const __m256i*)(x[i].qs + 32 * j));
            __m256i lo = _mm256_and_si256(qb, m4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m4);
            _mm512_storeu_ps(y + 0, _mm512_fmsub_ps(d1, u8x16_to_ps(_mm256_castsi256_si128(lo)), m1));
            _mm512_storeu_ps(y + 16, _mm512_fmsub_ps(d1, u8x16_to_ps(_mm256_extracti128_si256(lo, 1)), m1));
            _mm512_storeu_ps(y + 32, _mm512_fmsub_ps(d2, u8x16_to_ps(_mm256_castsi256_si128(hi)), m2));
            _mm512_storeu_ps(y + 48, _mm512_fmsub_ps(d2, u8x16_to_ps(_mm256_extracti128_si256(hi, 1)), m2));
            y += 64;
        }
    }
}

void q6_k(const void* src, float* y, int64_t k) {
    const BlockQ6K* x = (const BlockQ6K*)src;
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3), c32 = _mm256_set1_epi8(32);
    for (int64_t i = 0; i < k / 256; i++) {
        float d = fp16(x[i].d);
        for (int n = 0; n < 2; n++) { // 每次处理 128 个
            const uint8_t* ql = x[i].ql + 64 * n;
            const uint8_t* qh = x[i].qh + 32 * n;
            const int8_t* sc = x[i].scales + 8 * n;
            __m256i l0 = _mm256_loadu_si256((const __m256i*)ql);
            __m256i l1 = _mm256_loadu_si256((const __m256i*)(ql + 32));
            __m256i h = _mm256_loadu_si256((const __m256i*)qh);
            __m256i q[4]; // 先按字节 and 再移位, epi16 移位不会把相邻字节的位带进来
            q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(h, m2), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m2), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m2), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m2), 4));
            for (int s = 0; s < 4; s++) {
                __m256i qs = _mm256_sub_epi8(q[s], c32);
                _mm512_storeu_ps(y + 32 * s, _mm512_mul_ps(i8x16_to_ps(_mm256_castsi256_si128(qs)), _mm512_set1_ps(d * sc[2 * s])));
                _mm512_storeu_ps(y + 32 * s + 16, _mm512_mul_ps(i8x16_to_ps(_mm256_extracti128_si256(qs, 1)), _mm512_set1_ps(d * sc[2 * s + 1])));
            }
            y += 128;
        }
    }
}

} // namespace QuantsInternal

void dequant_row(uint32_t type, const void* src, float* y, int64_t k) { // k 必须是块大小的整数倍
    switch (type) {
        case Gguf::T_F32: memcpy(y, src, k * 4); return;
        case Gguf::T_Q8_0: QuantsInternal::q8_0(src, y, k); return;
        case Gguf::T_Q4_K: QuantsInternal::q4_k(src, y, k); return;
        case Gguf::T_Q5_K: Ref::q5_k(src, y, k); return; // 只有 2 个 tensor 用, 先用标量
        case Gguf::T_Q6_K: QuantsInternal::q6_k(src, y, k); return;
    }
    fprintf(stderr, "❌ dequant 不支持类型 %u\n", type);
    _Exit(1);
}

} // namespace Quants
} // namespace

#endif
