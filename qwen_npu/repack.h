#ifndef QWEN_NPU_REPACK_H
#define QWEN_NPU_REPACK_H

// GGUF 权重重打包成 NPU 友好格式 (全部可逐位还原原始反量化值, scale/bias 用 f32 存, 与 CPU 参考的 fp32 中间值一致):
//   FMT_I8  (q8_0/q6_k): [wave0 int8 k/2][wave1 int8 k/2][scale f32 ngp16]
//   FMT_I8B (q5_k):      [wave0][wave1][scale f32 ngp16][bias f32 ngp16], w = s*q + b
//   FMT_I4  (q4_k):      [byteA k/4][byteB k/4][scale f32 ngp32][bias f32 ngp32], q 为 0..15 的 nibble
//   FMT_F32 (f32):       自然顺序 f32
// wave 布局: 自然 8 元素块 b -> I8 系: wave(b&1) 槽 b/2; I4: wave(b&3) 槽 b/4; byteA = wave0 | wave1<<4, byteB = wave2 | wave3<<4
// scale 分组: I8 系每 16 个元素一组, I4 每 32 个一组, 组数向上取整到 64 的倍数 (ngp), 填充组 scale/bias = 0
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "gguf.h"
#include "quants_ref.h"

namespace {
namespace Repack {

enum { FMT_F32 = 0, FMT_I8 = 1, FMT_I8B = 2, FMT_I4 = 3 };

int fmt_for(uint32_t t) {
    switch (t) {
        case Gguf::T_F32: return FMT_F32;
        case Gguf::T_Q8_0: return FMT_I8;
        case Gguf::T_Q6_K: return FMT_I8;
        case Gguf::T_Q5_K: return FMT_I8B;
        case Gguf::T_Q4_K: return FMT_I4;
    }
    fprintf(stderr, "❌ 不支持重打包的量化类型 %u\n", t);
    _Exit(1);
}

uint64_t pad64(uint64_t x) { return (x + 63) / 64 * 64; }

uint64_t row_stride(int fmt, int k) {
    switch (fmt) {
        case FMT_F32: return (uint64_t)k * 4;
        case FMT_I8: return pad64((uint64_t)k + pad64(k / 16) * 4);
        case FMT_I8B: return pad64((uint64_t)k + pad64(k / 16) * 8);
        case FMT_I4: return pad64((uint64_t)k / 2 + pad64(k / 32) * 8);
    }
    _Exit(1);
}

// 一行自然顺序的 int8 值 + 组 scale/bias -> 按格式写入 dst
inline void pack_row(int fmt, int k, const int8_t* nat, const float* s, const float* b, uint8_t* dst) {
    if (fmt == FMT_I8 || fmt == FMT_I8B) {
        int h = k / 2;
        int8_t* v = (int8_t*)dst;
        for (int blk = 0; blk < k / 8; blk++)
            for (int e = 0; e < 8; e++) v[(blk & 1) * h + (blk / 2) * 8 + e] = nat[blk * 8 + e];
        int ngp = (int)pad64(k / 16);
        float* sd = (float*)(dst + k);
        memset(sd, 0, (uint64_t)ngp * (fmt == FMT_I8B ? 8 : 4));
        memcpy(sd, s, (uint64_t)(k / 16) * 4);
        if (fmt == FMT_I8B) memcpy(sd + ngp, b, (uint64_t)(k / 16) * 4);
        return;
    }
    if (fmt == FMT_I4) {
        int q4 = k / 4;
        uint8_t* va = dst;
        uint8_t* vb = dst + q4;
        for (int i = 0; i < q4; i++) {
            int slot = i / 8, e = i % 8;
            va[i] = (uint8_t)(nat[(4 * slot + 0) * 8 + e] | (nat[(4 * slot + 1) * 8 + e] << 4));
            vb[i] = (uint8_t)(nat[(4 * slot + 2) * 8 + e] | (nat[(4 * slot + 3) * 8 + e] << 4));
        }
        int ngp = (int)pad64(k / 32);
        float* sd = (float*)(dst + k / 2);
        memset(sd, 0, (uint64_t)ngp * 8);
        memcpy(sd, s, (uint64_t)(k / 32) * 4);
        memcpy(sd + ngp, b, (uint64_t)(k / 32) * 4);
        return;
    }
    fprintf(stderr, "❌ pack_row 不支持格式 %d\n", fmt);
    _Exit(1);
}

// 把 src (GGUF 原始格式) 的第 row0 起 nrows 行重打包到 dst (连续 nrows 行)
inline void repack_rows(uint32_t t, const uint8_t* src, int k, int row0, int nrows, uint8_t* dst) {
    int fmt = fmt_for(t);
    uint64_t srb = Quants::row_bytes(t, k), drb = row_stride(fmt, k);
    std::vector<int8_t> nat(k);
    std::vector<float> s(k / 16 + 1), b(k / 16 + 1);
    for (int r = 0; r < nrows; r++) {
        const uint8_t* sp = src + (uint64_t)(row0 + r) * srb;
        uint8_t* dp = dst + (uint64_t)r * drb;
        if (t == Gguf::T_F32) {
            memcpy(dp, sp, (uint64_t)k * 4);
            continue;
        }
        if (t == Gguf::T_Q8_0) { // 32 一块: d; per-16 scale 重复两份
            for (int i = 0; i < k / 32; i++) {
                const Quants::BlockQ8_0* x = (const Quants::BlockQ8_0*)sp + i;
                float d = Quants::fp16(x->d);
                s[2 * i] = d;
                s[2 * i + 1] = d;
                memcpy(&nat[i * 32], x->qs, 32);
            }
        } else if (t == Gguf::T_Q6_K) { // 256 超块: 16 个 per-16 scale = d * sc[i]
            for (int i = 0; i < k / 256; i++) {
                const Quants::BlockQ6K* x = (const Quants::BlockQ6K*)sp + i;
                float d = Quants::fp16(x->d);
                for (int j = 0; j < 16; j++) s[i * 16 + j] = d * x->scales[j];
                int8_t* y = &nat[i * 256];
                const uint8_t* ql = x->ql;
                const uint8_t* qh = x->qh;
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; l++) {
                        y[n + l] = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        y[n + l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        y[n + l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        y[n + l + 96] = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    }
                    ql += 64;
                    qh += 32;
                }
            }
        } else if (t == Gguf::T_Q5_K) { // 256 超块: 8 个 per-32 (d1, m1), 复制成 per-16; q 0..31
            for (int i = 0; i < k / 256; i++) {
                const Quants::BlockQ5K* x = (const Quants::BlockQ5K*)sp + i;
                float d = Quants::fp16(x->d), mn = Quants::fp16(x->dmin);
                const uint8_t* ql = x->qs;
                const uint8_t* qh = x->qh;
                int8_t* y = &nat[i * 256];
                uint8_t u1 = 1, u2 = 2, sc, m;
                for (int j = 0; j < 8; j += 2) {
                    Quants::get_scale_min_k4(j, x->scales, &sc, &m);
                    float d1 = d * sc, m1 = mn * m;
                    Quants::get_scale_min_k4(j + 1, x->scales, &sc, &m);
                    float d2 = d * sc, m2 = mn * m;
                    int g = i * 16 + j * 2; // per-16 组号
                    s[g] = s[g + 1] = d1;
                    b[g] = b[g + 1] = -m1;
                    s[g + 2] = s[g + 3] = d2;
                    b[g + 2] = b[g + 3] = -m2;
                    for (int l = 0; l < 32; l++) y[j * 32 + l] = (int8_t)((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0));
                    for (int l = 0; l < 32; l++) y[j * 32 + 32 + l] = (int8_t)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0));
                    ql += 32;
                    u1 <<= 2;
                    u2 <<= 2;
                    y += 0; // y 基址不动, 用 j*32 偏移
                }
            }
        } else if (t == Gguf::T_Q4_K) { // 256 超块: 8 个 per-32 (d1, m1); q 0..15
            for (int i = 0; i < k / 256; i++) {
                const Quants::BlockQ4K* x = (const Quants::BlockQ4K*)sp + i;
                float d = Quants::fp16(x->d), mn = Quants::fp16(x->dmin);
                const uint8_t* q = x->qs;
                int8_t* y = &nat[i * 256];
                uint8_t sc, m;
                for (int j = 0; j < 8; j += 2) {
                    Quants::get_scale_min_k4(j, x->scales, &sc, &m);
                    s[i * 8 + j] = d * sc;
                    b[i * 8 + j] = -mn * m;
                    Quants::get_scale_min_k4(j + 1, x->scales, &sc, &m);
                    s[i * 8 + j + 1] = d * sc;
                    b[i * 8 + j + 1] = -mn * m;
                    for (int l = 0; l < 32; l++) y[j * 32 + l] = (int8_t)(q[l] & 0xF);
                    for (int l = 0; l < 32; l++) y[j * 32 + 32 + l] = (int8_t)(q[l] >> 4);
                    q += 32;
                }
            }
        } else {
            fprintf(stderr, "❌ repack_rows 不支持类型 %u\n", t);
            _Exit(1);
        }
        pack_row(fmt, k, nat.data(), s.data(), b.data(), dp);
    }
}

} // namespace Repack
} // namespace

#endif
