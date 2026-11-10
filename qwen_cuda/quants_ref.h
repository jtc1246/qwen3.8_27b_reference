#ifndef QWEN_CUDA_QUANTS_REF_H
#define QWEN_CUDA_QUANTS_REF_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "gguf.h"

namespace {
namespace Quants { // host 端: 量化块结构、每行字节数、标量参考反量化 (与 ggml dequantize_row_* 逐行对应, 自测用)

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
struct BlockQ4K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BlockQ5K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct BlockQ6K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(BlockQ8_0) == 34 && sizeof(BlockQ4K) == 144 && sizeof(BlockQ5K) == 176 && sizeof(BlockQ6K) == 210, "block 大小不对");

inline float fp16(uint16_t h) { // 软件实现 fp16 -> fp32, 含非规格化数
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, bits;
    if (e == 31)
        bits = s | 0x7f800000u | (m << 13);
    else if (e)
        bits = s | ((e + 112) << 23) | (m << 13);
    else if (m) {
        e = 113;
        while (!(m & 1024)) {
            m <<= 1;
            e--;
        }
        bits = s | (e << 23) | ((m & 1023) << 13);
    } else
        bits = s;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

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

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
        return;
    }
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
}

namespace Ref {

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

void dequant_row(uint32_t type, const uint8_t* src, float* y, int64_t k) {
    switch (type) {
        case Gguf::T_F32: memcpy(y, src, k * 4); return;
        case Gguf::T_Q8_0: q8_0(src, y, k); return;
        case Gguf::T_Q4_K: q4_k(src, y, k); return;
        case Gguf::T_Q5_K: q5_k(src, y, k); return;
        case Gguf::T_Q6_K: q6_k(src, y, k); return;
    }
    fprintf(stderr, "❌ dequant 不支持类型 %u\n", type);
    _Exit(1);
}

} // namespace Ref
} // namespace Quants
} // namespace

#endif
