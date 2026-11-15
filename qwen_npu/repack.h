#ifndef QWEN_NPU_REPACK_H
#define QWEN_NPU_REPACK_H

// RedHat INT4 checkpoint (compressed-tensors, int4 对称 group128) -> NPU 格式重打包 (host 侧):
//   FMT_I8   (wave 布局, tok_embd): [wave0 int8 k/2][wave1 int8 k/2][scale f32 ngp16], 只给 k_dq_i8 查行, 反量化逐位精确
//   FMT_I8F  (fractal, lm_head): int8 + per-16 fp16 scale (host 从 bf16 RTN 量化)
//   FMT_I4SF (fractal, 主干 int4): nibble(q+8, 0..15) + fp16 scale (per-128 组复制成 per-16), w = s*(nib-8), 无 bias 区
// wave 布局: 自然 8 元素块 b -> wave(b&1) 槽 b/2; scale 每 16 元素一组, 组数向上取整到 64 (ngp), 填充组 scale = 0
// fractal 布局 (对应 kernels gemm_body): 行按 8 核均分, 核内 256 行分块, 块内按 128-k 切片;
// 每切片 = [8 ki][NI ni] 个 512B fractal ([16k][16r] k 主序) + scale 区 [ki][ni][16 half];
// I4SF 切片按半片组织 (半片 = 4 个 ki), 字节 i = lo(半片前 2 ki 的元素 i) | hi(后 2 ki 的元素 i) << 4
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
namespace Repack {

enum { FMT_F32 = 0, FMT_I8 = 1, FMT_I8B = 2, FMT_I4 = 3, FMT_I8F = 4, FMT_I8BF = 5, FMT_I4F = 6, FMT_I4SF = 7 };

int sb2_ni(int fmt) { // 每 16 行每 128k 字节
    if (fmt == FMT_I8F) return 2304;
    if (fmt == FMT_I4SF) return 1280;
    fprintf(stderr, "❌ sb2_ni 不支持格式 %d\n", fmt);
    _Exit(1);
}
int qb2_ni(int fmt) { return fmt == FMT_I4SF ? 1024 : 2048; }

inline float bf2f(uint16_t b) { // bf16 -> f32 (精确)
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

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
    if (fmt == FMT_F32) return (uint64_t)k * 4;
    if (fmt == FMT_I8) return pad64((uint64_t)k + pad64(k / 16) * 4);
    fprintf(stderr, "❌ row_stride 不支持格式 %d\n", fmt);
    _Exit(1);
}

// 一行自然顺序 int8 + per-16 f32 scale -> wave FMT_I8 写入 dst
inline void pack_row_i8(int k, const int8_t* nat, const float* s, uint8_t* dst) {
    int h = k / 2;
    int8_t* v = (int8_t*)dst;
    for (int blk = 0; blk < k / 8; blk++)
        for (int e = 0; e < 8; e++) v[(blk & 1) * h + (blk / 2) * 8 + e] = nat[blk * 8 + e];
    int ngp = (int)pad64(k / 16);
    float* sd = (float*)(dst + k);
    memset(sd, 0, (uint64_t)ngp * 4);
    memcpy(sd, s, (uint64_t)(k / 16) * 4);
}

// 一行 f32 按 per-16 对称 RTN 量化成 int8 (dequant = s * q, 与 k_dq_i8 逐位一致)
inline void quant_row_i8(const float* w, int k, int8_t* nat, float* s) {
    for (int g = 0; g < k / 16; g++) {
        float amax = 0;
        for (int j = 0; j < 16; j++) amax = std::fabs(w[g * 16 + j]) > amax ? std::fabs(w[g * 16 + j]) : amax;
        float d = amax / 127.0f;
        s[g] = d;
        float inv = d > 0 ? 1.0f / d : 0.0f;
        for (int j = 0; j < 16; j++) {
            long q = lrintf(w[g * 16 + j] * inv);
            nat[g * 16 + j] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
    }
}

// 16 行 (块内组号 ni) 写入 fractal 块 dst (覆盖全部 k/128 个切片)
// nat: [16][k] (I4SF 为 0..15 的 nibble 值, I8F 为 int8), s: [16][k/16] f32
inline void pack2_16rows(int fmt, int k, int NI, int ni, const int8_t* nat, const float* s, uint8_t* dst) {
    int KS = k / 128, ng = k / 16;
    int SB = sb2_ni(fmt), QB = qb2_ni(fmt);
    for (int kc = 0; kc < KS; kc++) {
        uint8_t* sl = dst + (uint64_t)kc * NI * SB;
        if (fmt == FMT_I8F) {
            for (int ki = 0; ki < 8; ki++) {
                int8_t* fp2 = (int8_t*)sl + (ki * NI + ni) * 256;
                for (int kk = 0; kk < 16; kk++)
                    for (int rr = 0; rr < 16; rr++) fp2[kk * 16 + rr] = nat[(size_t)rr * k + kc * 128 + ki * 16 + kk];
            }
        } else { // I4SF
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
        uint16_t* sp2 = (uint16_t*)(sl + (size_t)NI * QB);
        for (int ki = 0; ki < 8; ki++) {
            float div = (fmt == FMT_I4SF && (ki & 2)) ? 0.0625f : 1.0f; // I4SF hi 平面 (ki%4 为 2/3) And 解码后带 x16, scale 预除
            for (int rr = 0; rr < 16; rr++) sp2[(ki * NI + ni) * 16 + rr] = f2h(s[(size_t)rr * ng + kc * 8 + ki] * div);
        }
    }
}

} // namespace Repack
} // namespace

#endif
