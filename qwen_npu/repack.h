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

enum { FMT_F32 = 0, FMT_I8 = 1, FMT_I8B = 2, FMT_I4 = 3, FMT_I8F = 4, FMT_I8BF = 5, FMT_I4F = 6 };

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

// cube GEMM 用的 fractal 布局格式 (matmul 权重)
int fmt2_for(uint32_t t) {
    switch (t) {
        case Gguf::T_F32: return FMT_F32;
        case Gguf::T_Q8_0: return FMT_I8F;
        case Gguf::T_Q6_K: return FMT_I8F;
        case Gguf::T_Q5_K: return FMT_I8BF;
        case Gguf::T_Q4_K: return FMT_I4F;
    }
    fprintf(stderr, "❌ 不支持重打包的量化类型 %u\n", t);
    _Exit(1);
}

int sb2_ni(int fmt) { return fmt == FMT_I8F ? 2304 : (fmt == FMT_I8BF ? 2560 : 1536); } // 每 16 行每 128k 字节
int qb2_ni(int fmt) { return fmt == FMT_I4F ? 1024 : 2048; }

inline uint16_t f2h(float f) { // f32 -> fp16 (RNE)
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sg = (u >> 16) & 0x8000, e = (u >> 23) & 0xFF, mnt = u & 0x7FFFFF;
    if (e == 0) return (uint16_t)sg;
    int ne = (int)e - 127 + 15;
    if (ne <= 0) { // subnormal / 0
        if (ne < -10) return (uint16_t)sg;
        uint32_t full = mnt | 0x800000;
        int shift = 14 - ne;
        uint32_t mant = full >> shift, rest = full & ((1u << shift) - 1), halfbit = 1u << (shift - 1);
        if (rest > halfbit || (rest == halfbit && (mant & 1))) mant++;
        return (uint16_t)(sg | mant);
    }
    if (ne >= 31) return (uint16_t)(sg | 0x7BFF); // 溢出按最大值截断 (量化 scale 不会出现)
    uint32_t mant = mnt >> 13, rest = mnt & 0x1FFF;
    if (rest > 0x1000 || (rest == 0x1000 && (mant & 1))) mant++;
    return (uint16_t)(sg | ((uint32_t)(ne << 10) + mant)); // mant 上溢到 1024 时正确进位
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

// 解码一行 GGUF 量化数据: nat 为整数值 (q4_k 为 0..15), s/b 为 per-16 的 f32 scale/bias
inline void decode_row(uint32_t t, const uint8_t* sp, int k, int8_t* nat_p, float* s_p, float* b_p) {
    int8_t* nat = nat_p;
    float* s = s_p;
    float* b = b_p;
    {
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
            fprintf(stderr, "❌ decode_row 不支持类型 %u\n", t);
            _Exit(1);
        }
    }
}

// 把 src (GGUF 原始格式) 的第 row0 起 nrows 行重打包到 dst (连续 nrows 行, 旧 wave 布局, tok_embd 用)
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
        decode_row(t, sp, k, nat.data(), s.data(), b.data());
        pack_row(fmt, k, nat.data(), s.data(), b.data(), dp);
    }
}

// 把第 row0 起 nb 行 (16 的倍数) 重打包为 k_gemm 的 fractal 布局块 (k/128 个切片) 写入 dst
// 切片 = [8 ki][NI ni] 个 512B fractal ([16k][16r] k 主序) + scale [ki][ni][16 half] (+bias 同样)
// I4F: 切片按半片组织 (半片 = 4 个 ki), 字节 i = lo(半片前 2ki 元素 i) | hi(后 2ki 元素 i)<<4
inline void repack2_block(uint32_t t, const uint8_t* src, int k, int row0, int nb, uint8_t* dst) {
    int fmt = fmt2_for(t);
    int KS = k / 128, NI = nb / 16, ng = k / 16;
    uint64_t srb = Quants::row_bytes(t, k);
    int SB = sb2_ni(fmt), QB = qb2_ni(fmt);
    std::vector<int8_t> nat((size_t)16 * k);
    std::vector<float> s((size_t)16 * ng), b((size_t)16 * ng, 0);
    for (int ni = 0; ni < NI; ni++) {
        for (int rr = 0; rr < 16; rr++)
            decode_row(t, src + (uint64_t)(row0 + ni * 16 + rr) * srb, k, &nat[(size_t)rr * k], &s[(size_t)rr * ng], &b[(size_t)rr * ng]);
        for (int kc = 0; kc < KS; kc++) {
            uint8_t* sl = dst + (uint64_t)kc * NI * SB;
            if (fmt != FMT_I4F) {
                for (int ki = 0; ki < 8; ki++) {
                    int8_t* fp2 = (int8_t*)sl + (ki * NI + ni) * 256;
                    for (int kk = 0; kk < 16; kk++)
                        for (int rr = 0; rr < 16; rr++) fp2[kk * 16 + rr] = nat[(size_t)rr * k + kc * 128 + ki * 16 + kk];
                }
            } else {
                for (int h = 0; h < 2; h++) {
                    uint8_t* hb = sl + (size_t)h * NI * 512;
                    for (int kil = 0; kil < 2; kil++)
                        for (int kk = 0; kk < 16; kk++)
                            for (int rr = 0; rr < 16; rr++) {
                                int lo = nat[(size_t)rr * k + kc * 128 + (h * 4 + kil) * 16 + kk];
                                int hi = nat[(size_t)rr * k + kc * 128 + (h * 4 + 2 + kil) * 16 + kk];
                                hb[((kil * NI + ni) * 16 + kk) * 16 + rr] = (uint8_t)(lo | (hi << 4));
                            }
                }
            }
            int gdiv = fmt == FMT_I4F ? 32 : 16; // q4_k 的 decode_row 产出 per-32 scale, 其余 per-16
            uint16_t* sp2 = (uint16_t*)(sl + (size_t)NI * QB);
            for (int ki = 0; ki < 8; ki++)
                for (int rr = 0; rr < 16; rr++) sp2[(ki * NI + ni) * 16 + rr] = f2h(s[(size_t)rr * ng + (kc * 128 + ki * 16) / gdiv]);
            if (fmt != FMT_I8F) {
                uint16_t* bp2 = sp2 + 8 * NI * 16;
                for (int ki = 0; ki < 8; ki++)
                    for (int rr = 0; rr < 16; rr++) bp2[(ki * NI + ni) * 16 + rr] = f2h(b[(size_t)rr * ng + (kc * 128 + ki * 16) / gdiv]);
            }
        }
    }
}

} // namespace Repack
} // namespace

#endif
