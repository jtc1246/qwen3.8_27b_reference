// Qwen3.8-27B 的全部 Ascend C kernel (310P), 全 fp32 计算, 权重为 repack.h 定义的 I8/I8B/I4/F32 格式
// 激活/缓存均为 GM 上的 fp32; matmul 为 T=1 的 GEMV, prefill 由 host 逐 token 调用
#include "kernel_operator.h"

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::RoundMode;

namespace {

constexpr int D = 5120, FF = 17408;
constexpr int N_HEAD = 24, N_KV = 4, HD = 256;
constexpr int LK = 16, LV = 48, LHD = 128;
constexpr int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM;
constexpr int Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD;
constexpr float EPS = 1e-6f;
constexpr uint64_t FULL = 0xffffffffffffffffULL;
enum { FMT_F32 = 0, FMT_I8 = 1, FMT_I8B = 2, FMT_I4 = 3 };

__aicore__ inline int pad64i(int x) { return (x + 63) / 64 * 64; }

// 自然顺序 x[k] -> NW 路 wave 布局: dst[w*(k/NW) + s*8 + e] = src[(s*NW+w)*8 + e]
template <int NW> __aicore__ inline void perm_x(const LocalTensor<float>& dst, const LocalTensor<float>& src, int k)
{
    int part = k / NW;
    AscendC::UnaryRepeatParams up(1, (uint16_t)NW, 8, (uint8_t)(8 * NW));
    for (int w = 0; w < NW; w++)
        AscendC::Adds<float, false>(dst[w * part], src[w * 8], 0.0f, FULL, (uint8_t)(part / 64), up);
}

template <int FMT> __aicore__ inline void gemv_body(GM_ADDR w, GM_ADDR x, GM_ADDR y, int n, int k, int stride)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> rq;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> yq;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    int al = FMT == FMT_F32 ? k : (FMT == FMT_I4 ? k / 4 : k / 2); // 乘积累加向量长度
    int ng = FMT == FMT_F32 ? k / 8 : (FMT == FMT_I4 ? k / 32 : k / 16); // 组数
    int ngp = FMT == FMT_F32 ? ng : pad64i(ng);
    int nst = ngp / 64; // stage1 输出个数 (<=64)
    int cores = (int)AscendC::GetBlockNum();
    int rows = n / cores, r0 = rows * (int)AscendC::GetBlockIdx();
    int rb_n = stride <= 8192 ? 8 : (stride <= 16384 ? 2 : 1); // 每批行数
    while (rb_n > 1 && rows % rb_n) rb_n /= 2;
    int xtr = FMT == FMT_I4 ? al * 4 : 0; // I4 额外的 t1/ti 缓冲
    pipe.InitBuffer(rq, 2, (uint32_t)(rb_n * stride));
    pipe.InitBuffer(yq, 2, 64);
    pipe.InitBuffer(tb, (uint32_t)(k * 4 + al * 2 + xtr + al * 8 + ngp * 8 + 4096));
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> xp = ub.ReinterpretCast<float>(); // [k]
    LocalTensor<half> wh = ub[k * 4].ReinterpretCast<half>(); // [al] (I4 时为字节值/低 nibble)
    LocalTensor<half> t1 = ub[k * 4 + al * 2].ReinterpretCast<half>(); // I4: [al] 高 nibble
    LocalTensor<int16_t> ti = ub[k * 4 + al * 2 + xtr / 2].ReinterpretCast<int16_t>(); // I4: [al]
    LocalTensor<float> wf = ub[k * 4 + al * 2 + xtr].ReinterpretCast<float>(); // [al]
    LocalTensor<float> acc = ub[k * 4 + al * 2 + xtr + al * 4].ReinterpretCast<float>(); // [al]
    LocalTensor<float> gs = ub[k * 4 + al * 2 + xtr + al * 8].ReinterpretCast<float>(); // [ngp]
    LocalTensor<float> xs = ub[k * 4 + al * 2 + xtr + al * 8 + ngp * 4].ReinterpretCast<float>(); // [ngp]
    LocalTensor<float> bb = ub[k * 4 + al * 2 + xtr + al * 8 + ngp * 8].ReinterpretCast<float>(); // [16][64]
    GlobalTensor<uint8_t> gw;
    gw.SetGlobalBuffer((__gm__ uint8_t*)w, (uint64_t)n * stride);
    GlobalTensor<float> gx;
    gx.SetGlobalBuffer((__gm__ float*)x, k);
    GlobalTensor<float> gy;
    gy.SetGlobalBuffer((__gm__ float*)y, n + 64);
    AscendC::SetVectorMask<float>(FULL, FULL);
    AscendC::UnaryRepeatParams cp8(1, 1, 8, 4); // 窄->宽一倍的转换 (i8->f16, f16->f32)
    AscendC::BinaryRepeatParams bp(1, 1, 1, 8, 8, 8);
    if (ngp > ng) { // 组数补齐区清零, 否则未初始化 UB 可能是 inf/NaN, 0 乘出 NaN
        AscendC::Duplicate(gs[ng], 0.0f, ngp - ng);
        AscendC::Duplicate(xs[ng], 0.0f, ngp - ng);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (FMT == FMT_F32) {
        AscendC::DataCopy(xp, gx, k);
        AscendC::PipeBarrier<PIPE_ALL>();
    } else {
        LocalTensor<float> xt = wf; // wf+acc 区当临时
        if (FMT == FMT_I4) { // wf+acc 只有 k*2 字节, 分两块加载重排
            AscendC::UnaryRepeatParams up4(1, 4, 8, 32);
            for (int c = 0; c < 2; c++) {
                AscendC::DataCopy(xt, gx[c * (k / 2)], k / 2);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int wv_i = 0; wv_i < 4; wv_i++)
                    AscendC::Adds<float, false>(xp[wv_i * al + c * (al / 2)], xt[wv_i * 8], 0.0f, FULL, (uint8_t)(al / 2 / 64), up4);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        } else {
            AscendC::DataCopy(xt, gx, k);
            AscendC::PipeBarrier<PIPE_ALL>();
            perm_x<2>(xp, xt, k);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (FMT == FMT_I8B || FMT == FMT_I4) { // 组内 x 和 (bias 项用)
            AscendC::Add<float, false>(acc, xp, xp[al], FULL, (uint8_t)(al / 64), bp);
            if (FMT == FMT_I4) {
                AscendC::Add<float, false>(acc, acc, xp[2 * al], FULL, (uint8_t)(al / 64), bp);
                AscendC::Add<float, false>(acc, acc, xp[3 * al], FULL, (uint8_t)(al / 64), bp);
            }
            AscendC::BlockReduceSum<float, false>(xs, acc, al / 64, 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    int nb = rows / rb_n;
    LocalTensor<uint8_t> rt = rq.AllocTensor<uint8_t>();
    AscendC::DataCopy(rt, gw[(uint64_t)r0 * stride], rb_n * stride);
    rq.EnQue(rt);
    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) {
            LocalTensor<uint8_t> nt = rq.AllocTensor<uint8_t>();
            AscendC::DataCopy(nt, gw[(uint64_t)(r0 + (b + 1) * rb_n) * stride], rb_n * stride);
            rq.EnQue(nt);
        }
        LocalTensor<uint8_t> batch = rq.DeQue<uint8_t>();
        for (int i = 0; i < rb_n; i++) {
            int r = b * rb_n + i;
            LocalTensor<uint8_t> rbuf = batch[i * stride];
            if (FMT == FMT_F32) {
                AscendC::Mul<float, false>(acc, rbuf.ReinterpretCast<float>(), xp, FULL, (uint8_t)(k / 64), bp);
                AscendC::BlockReduceSum<float, false>(gs, acc, k / 64, 0, 1, 1, 8);
            } else if (FMT == FMT_I8 || FMT == FMT_I8B) {
                int h = al;
                LocalTensor<int8_t> q = rbuf.ReinterpretCast<int8_t>();
                AscendC::Cast<half, int8_t, false>(wh, q, RoundMode::CAST_NONE, FULL, (uint8_t)(h / 128), cp8);
                AscendC::Cast<float, half, false>(wf, wh, RoundMode::CAST_NONE, FULL, (uint8_t)(h / 64), cp8);
                AscendC::Mul<float, false>(acc, wf, xp, FULL, (uint8_t)(h / 64), bp);
                AscendC::Cast<half, int8_t, false>(wh, q[h], RoundMode::CAST_NONE, FULL, (uint8_t)(h / 128), cp8);
                AscendC::Cast<float, half, false>(wf, wh, RoundMode::CAST_NONE, FULL, (uint8_t)(h / 64), cp8);
                AscendC::MulAddDst<float, float, false>(acc, wf, xp[h], FULL, (uint8_t)(h / 64), bp);
                AscendC::BlockReduceSum<float, false>(gs, acc, h / 64, 0, 1, 1, 8);
                AscendC::Mul<float, false>(gs, gs, rbuf[k].ReinterpretCast<float>(), FULL, (uint8_t)(ngp / 64), bp);
                if (FMT == FMT_I8B)
                    AscendC::MulAddDst<float, float, false>(gs, rbuf[k + ngp * 4].ReinterpretCast<float>(), xs, FULL, (uint8_t)(ngp / 64), bp);
            } else { // FMT_I4: 平面 p 的低 nibble = wave 2p, 高 nibble = wave 2p+1
                int q4 = al;
                AscendC::UnaryRepeatParams cpa(1, 1, 8, 8);
                AscendC::BinaryRepeatParams bph(1, 1, 1, 8, 8, 8);
                for (int p = 0; p < 2; p++) {
                    AscendC::Cast<half, uint8_t, false>(wh, rbuf[p * q4], RoundMode::CAST_NONE, FULL, (uint8_t)(q4 / 128), cp8); // wh = 字节值 b
                    AscendC::Muls<half, false>(t1, wh, (half)0.0625f, FULL, (uint8_t)(q4 / 128), cpa);
                    AscendC::Adds<half, false>(t1, t1, (half)-0.46875f, FULL, (uint8_t)(q4 / 128), cpa);
                    AscendC::Cast<int16_t, half, false>(ti, t1, RoundMode::CAST_RINT, FULL, (uint8_t)(q4 / 128), cpa);
                    AscendC::Cast<half, int16_t, false>(t1, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(q4 / 128), cpa); // t1 = 高 nibble
                    AscendC::Cast<float, half, false>(wf, t1, RoundMode::CAST_NONE, FULL, (uint8_t)(q4 / 64), cp8);
                    if (p == 0) AscendC::Mul<float, false>(acc, wf, xp[q4], FULL, (uint8_t)(q4 / 64), bp);
                    else AscendC::MulAddDst<float, float, false>(acc, wf, xp[3 * q4], FULL, (uint8_t)(q4 / 64), bp);
                    AscendC::Muls<half, false>(t1, t1, (half)-16.0f, FULL, (uint8_t)(q4 / 128), cpa);
                    AscendC::Add<half, false>(t1, t1, wh, FULL, (uint8_t)(q4 / 128), bph); // t1 = 低 nibble = b - 16*高
                    AscendC::Cast<float, half, false>(wf, t1, RoundMode::CAST_NONE, FULL, (uint8_t)(q4 / 64), cp8);
                    AscendC::MulAddDst<float, float, false>(acc, wf, xp[p == 0 ? 0 : 2 * q4], FULL, (uint8_t)(q4 / 64), bp);
                }
                AscendC::BlockReduceSum<float, false>(gs, acc, q4 / 64, 0, 1, 1, 8);
                AscendC::Mul<float, false>(gs, gs, rbuf[k / 2].ReinterpretCast<float>(), FULL, (uint8_t)(ngp / 64), bp);
                AscendC::MulAddDst<float, float, false>(gs, rbuf[k / 2 + ngp * 4].ReinterpretCast<float>(), xs, FULL, (uint8_t)(ngp / 64), bp);
            }
            AscendC::WholeReduceSum<float, false>(bb[(r & 15) * 64], gs, 0, nst, 1, 1, 8);
            if ((r & 15) == 15 || r == rows - 1) {
                LocalTensor<float> yb = yq.AllocTensor<float>();
                AscendC::WholeReduceSum<float, true>(yb, bb, nst, (r & 15) + 1, 1, 1, 8);
                AscendC::SetVectorMask<float>(FULL, FULL);
                yq.EnQue(yb);
                yb = yq.DeQue<float>();
                AscendC::DataCopy(gy[r0 + (r & ~15)], yb, 16);
                yq.FreeTensor(yb);
            }
        }
        rq.FreeTensor(batch);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}



// 反量化 nr 行到自然顺序 f32 (embedding / 自测用, 性能不敏感)
// wave -> 自然顺序通过 DataCopy 的 32B 块跨步写 GM 完成, 省掉整行自然序缓冲
template <int FMT> __aicore__ inline void dq_body(GM_ADDR w, GM_ADDR rows, GM_ADDR out, int nr, int k, int stride)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    constexpr int NW = FMT == FMT_I4 ? 4 : 2;
    int part = FMT == FMT_F32 ? k : k / NW;
    int ng = FMT == FMT_I4 ? k / 32 : k / 16;
    int ngp = pad64i(ng);
    int rb = (stride + 63) / 64 * 64;
    // tb: rowbuf | wv[k] f32 | wh[part] h | t1[part] h | ti[part] i16 | se[part] f32 | be[part] f32 | ids
    pipe.InitBuffer(tb, (uint32_t)(rb + k * 4 + part * 6 + part * 8 + 1024 + 256));
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<uint8_t> rbuf = ub;
    LocalTensor<float> wv = ub[rb].ReinterpretCast<float>();
    LocalTensor<half> wh = ub[rb + k * 4].ReinterpretCast<half>();
    LocalTensor<half> t1 = ub[rb + k * 4 + part * 2].ReinterpretCast<half>();
    LocalTensor<int16_t> ti = ub[rb + k * 4 + part * 4].ReinterpretCast<int16_t>();
    LocalTensor<float> se = ub[rb + k * 4 + part * 6].ReinterpretCast<float>();
    LocalTensor<float> be = ub[rb + k * 4 + part * 6 + part * 4].ReinterpretCast<float>();
    LocalTensor<int32_t> ids = ub[rb + k * 4 + part * 6 + part * 8].ReinterpretCast<int32_t>();
    GlobalTensor<uint8_t> gw;
    gw.SetGlobalBuffer((__gm__ uint8_t*)w, 1); // 行号可能很大, 用 64 位偏移
    GlobalTensor<int32_t> gr;
    gr.SetGlobalBuffer((__gm__ int32_t*)rows, nr);
    GlobalTensor<float> go;
    go.SetGlobalBuffer((__gm__ float*)out, (uint64_t)nr * k);
    AscendC::DataCopy(ids, gr, (nr + 7) / 8 * 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    AscendC::UnaryRepeatParams cp8(1, 1, 8, 4);
    AscendC::UnaryRepeatParams cpa(1, 1, 8, 8);
    AscendC::BinaryRepeatParams bp(1, 1, 1, 8, 8, 8);
    AscendC::SetVectorMask<float>(FULL, FULL);
    AscendC::DataCopyParams wout((uint16_t)(part / 8), 1, 0, NW - 1); // UB 连续 -> GM 每 32B 隔 NW-1 块
    for (int r = core; r < nr; r += cores) {
        uint64_t off = (uint64_t)ids.GetValue(r) * stride;
        AscendC::DataCopy(rbuf, gw[off], stride);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (FMT == FMT_F32) {
            AscendC::DataCopy(go[(uint64_t)r * k], rbuf.ReinterpretCast<float>(), k);
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }
        if (FMT == FMT_I8 || FMT == FMT_I8B) {
            LocalTensor<int8_t> q = rbuf.ReinterpretCast<int8_t>();
            AscendC::Cast<half, int8_t, false>(wh, q, RoundMode::CAST_NONE, FULL, (uint8_t)(part / 128), cp8);
            AscendC::Cast<float, half, false>(wv, wh, RoundMode::CAST_NONE, FULL, (uint8_t)(part / 64), cp8);
            AscendC::Cast<half, int8_t, false>(wh, q[part], RoundMode::CAST_NONE, FULL, (uint8_t)(part / 128), cp8);
            AscendC::Cast<float, half, false>(wv[part], wh, RoundMode::CAST_NONE, FULL, (uint8_t)(part / 64), cp8);
            AscendC::Brcb(se, rbuf[k].ReinterpretCast<float>(), (uint8_t)(ng / 8), AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<float, false>(wv, wv, se, FULL, (uint8_t)(part / 64), bp);
            AscendC::Mul<float, false>(wv[part], wv[part], se, FULL, (uint8_t)(part / 64), bp);
            if (FMT == FMT_I8B) {
                AscendC::Brcb(be, rbuf[k + ngp * 4].ReinterpretCast<float>(), (uint8_t)(ng / 8), AscendC::BrcbRepeatParams(1, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add<float, false>(wv, wv, be, FULL, (uint8_t)(part / 64), bp);
                AscendC::Add<float, false>(wv[part], wv[part], be, FULL, (uint8_t)(part / 64), bp);
            }
        } else { // FMT_I4
            for (int p = 0; p < 2; p++) {
                AscendC::Cast<half, uint8_t, false>(wh, rbuf[p * part], RoundMode::CAST_NONE, FULL, (uint8_t)(part / 128), cp8);
                AscendC::Muls<half, false>(t1, wh, (half)0.0625f, FULL, (uint8_t)(part / 128), cpa);
                AscendC::Adds<half, false>(t1, t1, (half)-0.46875f, FULL, (uint8_t)(part / 128), cpa);
                AscendC::Cast<int16_t, half, false>(ti, t1, RoundMode::CAST_RINT, FULL, (uint8_t)(part / 128), cpa);
                AscendC::Cast<half, int16_t, false>(t1, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(part / 128), cpa);
                AscendC::Cast<float, half, false>(wv[(2 * p + 1) * part], t1, RoundMode::CAST_NONE, FULL, (uint8_t)(part / 64), cp8);
                AscendC::Muls<half, false>(t1, t1, (half)-16.0f, FULL, (uint8_t)(part / 128), cpa);
                AscendC::Add<half, false>(t1, t1, wh, FULL, (uint8_t)(part / 128), bp);
                AscendC::Cast<float, half, false>(wv[2 * p * part], t1, RoundMode::CAST_NONE, FULL, (uint8_t)(part / 64), cp8);
            }
            AscendC::Brcb(se, rbuf[k / 2].ReinterpretCast<float>(), (uint8_t)(ng / 8), AscendC::BrcbRepeatParams(1, 8));
            AscendC::Brcb(be, rbuf[k / 2 + ngp * 4].ReinterpretCast<float>(), (uint8_t)(ng / 8), AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            for (int wi = 0; wi < 4; wi++) {
                AscendC::Mul<float, false>(wv[wi * part], wv[wi * part], se, FULL, (uint8_t)(part / 64), bp);
                AscendC::Add<float, false>(wv[wi * part], wv[wi * part], be, FULL, (uint8_t)(part / 64), bp);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int wi = 0; wi < NW; wi++) { // wave wi 的 32B 块依次写到自然位置
            AscendC::DataCopy(go[(uint64_t)r * k + wi * 8], wv[wi * part], wout);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

} // namespace

extern "C" __global__ __aicore__ void k_gemv_f32(GM_ADDR w, GM_ADDR x, GM_ADDR y, int32_t n, int32_t k, int32_t stride) { gemv_body<FMT_F32>(w, x, y, n, k, stride); }
extern "C" __global__ __aicore__ void k_gemv_i8(GM_ADDR w, GM_ADDR x, GM_ADDR y, int32_t n, int32_t k, int32_t stride) { gemv_body<FMT_I8>(w, x, y, n, k, stride); }
extern "C" __global__ __aicore__ void k_gemv_i8b(GM_ADDR w, GM_ADDR x, GM_ADDR y, int32_t n, int32_t k, int32_t stride) { gemv_body<FMT_I8B>(w, x, y, n, k, stride); }
extern "C" __global__ __aicore__ void k_gemv_i4(GM_ADDR w, GM_ADDR x, GM_ADDR y, int32_t n, int32_t k, int32_t stride) { gemv_body<FMT_I4>(w, x, y, n, k, stride); }
extern "C" __global__ __aicore__ void k_dq_f32(GM_ADDR w, GM_ADDR rows, GM_ADDR out, int32_t nr, int32_t k, int32_t stride) { dq_body<FMT_F32>(w, rows, out, nr, k, stride); }
extern "C" __global__ __aicore__ void k_dq_i8(GM_ADDR w, GM_ADDR rows, GM_ADDR out, int32_t nr, int32_t k, int32_t stride) { dq_body<FMT_I8>(w, rows, out, nr, k, stride); }
extern "C" __global__ __aicore__ void k_dq_i8b(GM_ADDR w, GM_ADDR rows, GM_ADDR out, int32_t nr, int32_t k, int32_t stride) { dq_body<FMT_I8B>(w, rows, out, nr, k, stride); }
extern "C" __global__ __aicore__ void k_dq_i4(GM_ADDR w, GM_ADDR rows, GM_ADDR out, int32_t nr, int32_t k, int32_t stride) { dq_body<FMT_I4>(w, rows, out, nr, k, stride); }

// x += add (可空) + RMSNorm; T 个 token 分核轮转; hidden 非空时保存最后一个 token 的 x (归一化前)
extern "C" __global__ __aicore__ void k_add_norm(GM_ADDR x, GM_ADDR add, GM_ADDR w, GM_ADDR y, GM_ADDR hidden, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    pipe.InitBuffer(tb, D * 4 * 4 + 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> wb = ub.ReinterpretCast<float>(); // [D]
    LocalTensor<float> a = ub[D * 4].ReinterpretCast<float>();
    LocalTensor<float> p = ub[D * 8].ReinterpretCast<float>();
    LocalTensor<float> yb = ub[D * 12].ReinterpretCast<float>();
    LocalTensor<float> ss = ub[D * 16].ReinterpretCast<float>(); // [64]
    GlobalTensor<float> gx, ga, gw, gy, gh;
    gx.SetGlobalBuffer((__gm__ float*)x, (uint64_t)T * D);
    if (add) ga.SetGlobalBuffer((__gm__ float*)add, (uint64_t)T * D);
    gw.SetGlobalBuffer((__gm__ float*)w, D);
    gy.SetGlobalBuffer((__gm__ float*)y, (uint64_t)T * D);
    if (hidden) gh.SetGlobalBuffer((__gm__ float*)hidden, D);
    AscendC::DataCopy(wb, gw, D);
    AscendC::PipeBarrier<PIPE_ALL>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    for (int t = core; t < T; t += cores) {
        AscendC::DataCopy(a, gx[(uint64_t)t * D], D);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (add) {
            AscendC::DataCopy(p, ga[(uint64_t)t * D], D);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(a, a, p, D);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(gx[(uint64_t)t * D], a, D); // 残差写回
        }
        AscendC::Mul(p, a, a, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float>(yb, p, D / 64, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(ss, yb, 64, D / 512, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(ss[32], ss, D / 512, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ss[32], ss[32], 1.0f / D, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(ss[32], ss[32], EPS, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(ss[32], ss[32], 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float rs = 1.0f / ss.GetValue(32);
        AscendC::Muls(yb, a, rs, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yb, yb, wb, D);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gy[(uint64_t)t * D], yb, D);
        if (hidden && t == T - 1) AscendC::DataCopy(gh, a, D);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// g = silu(g) * u, 逐元素; total 必须是 512 的倍数
extern "C" __global__ __aicore__ void k_silu_mul(GM_ADDR g, GM_ADDR u, int32_t total)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    constexpr int CS = 8192;
    pipe.InitBuffer(tb, CS * 4 * 3 + 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> gb = ub.ReinterpretCast<float>();
    LocalTensor<float> ubuf = ub[CS * 4].ReinterpretCast<float>();
    LocalTensor<float> t = ub[CS * 8].ReinterpretCast<float>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int share = total / cores, base = share * core;
    GlobalTensor<float> gg, gu;
    gg.SetGlobalBuffer((__gm__ float*)g, total);
    gu.SetGlobalBuffer((__gm__ float*)u, total);
    for (int o = 0; o < share; o += CS) {
        int c = share - o < CS ? share - o : CS;
        AscendC::DataCopy(gb, gg[base + o], c);
        AscendC::DataCopy(ubuf, gu[base + o], c);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(t, gb, -1.0f, c);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(t, t, c);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(t, t, 1.0f, c);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(gb, gb, t, c);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gb, gb, ubuf, c);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gg[base + o], gb, c);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// 因果 depthwise conv1d (核宽 4) + SiLU; state GM 布局 [3][QKV_DIM], cw GM 布局 [4][QKV_DIM] (host 已重排)
extern "C" __global__ __aicore__ void k_conv(GM_ADDR qkv, GM_ADDR state, GM_ADDR cw, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int C = QKV_DIM / cores, c0 = C * core;
    pipe.InitBuffer(tb, (uint32_t)(C * 4 * 10 + 1024));
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> w4 = ub.ReinterpretCast<float>(); // [4][C]
    LocalTensor<float> bufs = ub[C * 16].ReinterpretCast<float>(); // 4 个轮转缓冲 [C]
    LocalTensor<float> o = ub[C * 32].ReinterpretCast<float>();
    LocalTensor<float> tt = ub[C * 36].ReinterpretCast<float>();
    GlobalTensor<float> gq, gs, gw;
    gq.SetGlobalBuffer((__gm__ float*)qkv, (uint64_t)T * QKV_DIM);
    gs.SetGlobalBuffer((__gm__ float*)state, 3 * QKV_DIM);
    gw.SetGlobalBuffer((__gm__ float*)cw, 4 * QKV_DIM);
    for (int j = 0; j < 4; j++) AscendC::DataCopy(w4[j * C], gw[j * QKV_DIM + c0], C);
    LocalTensor<float> s0 = bufs, s1 = bufs[C], s2 = bufs[2 * C], xn = bufs[3 * C];
    AscendC::DataCopy(s0, gs[c0], C);
    AscendC::DataCopy(s1, gs[QKV_DIM + c0], C);
    AscendC::DataCopy(s2, gs[2 * QKV_DIM + c0], C);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int t = 0; t < T; t++) {
        AscendC::DataCopy(xn, gq[(uint64_t)t * QKV_DIM + c0], C);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(o, w4, s0, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(o, w4[C], s1, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(o, w4[2 * C], s2, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(o, w4[3 * C], xn, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tt, o, -1.0f, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tt, tt, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tt, tt, 1.0f, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(o, o, tt, C);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gq[(uint64_t)t * QKV_DIM + c0], o, C);
        AscendC::PipeBarrier<PIPE_ALL>();
        LocalTensor<float> old = s0;
        s0 = s1;
        s1 = s2;
        s2 = xn;
        xn = old;
    }
    AscendC::DataCopy(gs[c0], s0, C);
    AscendC::DataCopy(gs[QKV_DIM + c0], s1, C);
    AscendC::DataCopy(gs[2 * QKV_DIM + c0], s2, C);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Gated DeltaNet 递推: 每核 6 个 v 头, 每头状态 S[128][128] 常驻 UB, 对 T 个 token 顺序递推
// state GM: [48][128][128]; qkv: conv 后的 [T][QKV_DIM]; z: [T][V_DIM]; a/b: [T][48]; out: [T][V_DIM]
extern "C" __global__ __aicore__ void k_deltanet(GM_ADDR qkv, GM_ADDR z, GM_ADDR a, GM_ADDR b, GM_ADDR ssm_a, GM_ADDR dt_bias, GM_ADDR ssm_norm, GM_ADDR state, GM_ADDR out, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    pipe.InitBuffer(tb, 64 * 1024 * 2 + 24 * 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> S = ub.ReinterpretCast<float>(); // [128][128]
    LocalTensor<float> P = ub[65536].ReinterpretCast<float>(); // [128][128]
    int o2 = 131072;
    LocalTensor<float> kb = ub[o2].ReinterpretCast<float>(); // [128*8] Brcb
    LocalTensor<float> qr = ub[o2 + 4096].ReinterpretCast<float>(); // 以下各 [128] (512B)
    LocalTensor<float> kr = ub[o2 + 4608].ReinterpretCast<float>();
    LocalTensor<float> vr = ub[o2 + 5120].ReinterpretCast<float>();
    LocalTensor<float> zr = ub[o2 + 5632].ReinterpretCast<float>();
    LocalTensor<float> sq = ub[o2 + 6144].ReinterpretCast<float>();
    LocalTensor<float> sk = ub[o2 + 6656].ReinterpretCast<float>();
    LocalTensor<float> dl = ub[o2 + 7168].ReinterpretCast<float>();
    LocalTensor<float> tm = ub[o2 + 7680].ReinterpretCast<float>();
    LocalTensor<float> red = ub[o2 + 8192].ReinterpretCast<float>(); // [64]
    LocalTensor<float> snorm = ub[o2 + 8448].ReinterpretCast<float>(); // [128]
    LocalTensor<float> ar = ub[o2 + 8960].ReinterpretCast<float>(); // [64] 其实 48
    LocalTensor<float> br = ub[o2 + 9216].ReinterpretCast<float>();
    LocalTensor<float> sa = ub[o2 + 9472].ReinterpretCast<float>(); // ssm_a [48]
    LocalTensor<float> db = ub[o2 + 9728].ReinterpretCast<float>(); // dt_bias [48]
    LocalTensor<float> sp = ub[o2 + 9984].ReinterpretCast<float>(); // softplus / decay 行 [48]
    LocalTensor<float> sp2 = ub[o2 + 10240].ReinterpretCast<float>();
    GlobalTensor<float> gq, gz, ga, gb, gsa, gdb, gsn, gst, go;
    gq.SetGlobalBuffer((__gm__ float*)qkv, (uint64_t)T * QKV_DIM);
    gz.SetGlobalBuffer((__gm__ float*)z, (uint64_t)T * V_DIM);
    ga.SetGlobalBuffer((__gm__ float*)a, (uint64_t)T * LV);
    gb.SetGlobalBuffer((__gm__ float*)b, (uint64_t)T * LV);
    gsa.SetGlobalBuffer((__gm__ float*)ssm_a, LV);
    gdb.SetGlobalBuffer((__gm__ float*)dt_bias, LV);
    gsn.SetGlobalBuffer((__gm__ float*)ssm_norm, LHD);
    gst.SetGlobalBuffer((__gm__ float*)state, (uint64_t)LV * LHD * LHD);
    go.SetGlobalBuffer((__gm__ float*)out, (uint64_t)T * V_DIM);
    AscendC::DataCopy(snorm, gsn, LHD);
    AscendC::DataCopy(sa, gsa, 64); // LV=48, 多拷到 64 无妨 (GM 有后续数据)
    AscendC::DataCopy(db, gdb, 64);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetVectorMask<float>(FULL, FULL);
    const float QSCALE = 0.08838834764831845f; // 1/sqrt(128)
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int hpc = LV / cores;
    AscendC::BinaryRepeatParams pmul(1, 1, 0, 16, 16, 1); // P = S ⊙ kb (行 × 行标量块)
    AscendC::BinaryRepeatParams pmad(1, 1, 0, 16, 0, 1); // S += delta(行向量) × kb
    for (int hh = 0; hh < hpc; hh++) {
        int h = core * hpc + hh, kh = h % LK;
        AscendC::DataCopy(S, gst[(uint64_t)h * LHD * LHD], LHD * LHD);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int t = 0; t < T; t++) {
            AscendC::DataCopy(qr, gq[(uint64_t)t * QKV_DIM + kh * LHD], LHD);
            AscendC::DataCopy(kr, gq[(uint64_t)t * QKV_DIM + KEY_DIM + kh * LHD], LHD);
            AscendC::DataCopy(vr, gq[(uint64_t)t * QKV_DIM + 2 * KEY_DIM + h * LHD], LHD);
            AscendC::DataCopy(zr, gz[(uint64_t)t * V_DIM + h * LHD], LHD);
            AscendC::DataCopy(ar, ga[(uint64_t)t * LV], 64);
            AscendC::DataCopy(br, gb[(uint64_t)t * LV], 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            // beta 行 = sigmoid(br); decay 行 = exp(ssm_a * softplus(ar + dt_bias))
            AscendC::Add(sp, ar, db, 48);
            AscendC::Muls(sp2, br, -1.0f, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(sp2, sp2, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sp2, sp2, 1.0f, 48); // 1+e^-b, beta = 1/它 (之后标量取倒数)
            AscendC::Mins(tm, sp, 20.0f, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tm, tm, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tm, tm, 1.0f, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Ln(tm, tm, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(tm, tm, sp, 48); // softplus (x>20 时取 x)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tm, tm, sa, 48);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tm, tm, 48); // decay 行
            // q/k 的 L2 范数
            AscendC::Mul(sq, qr, qr, LHD);
            AscendC::Mul(sk, kr, kr, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(red, sq, 64, 2, 1, 1, 8);
            AscendC::WholeReduceSum<float>(red[8], sk, 64, 2, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(red[16], red, 2, 1, 1, 1, 8);
            AscendC::WholeReduceSum<float>(red[24], red[8], 2, 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(red[16], red[16], 1e-6f, 1);
            AscendC::Adds(red[24], red[24], 1e-6f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(red[16], red[16], 1);
            AscendC::Sqrt(red[24], red[24], 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            float rq = 1.0f / red.GetValue(16) * QSCALE;
            float rk = 1.0f / red.GetValue(24);
            float beta = 1.0f / sp2.GetValue(h);
            float decay = tm.GetValue(h);
            AscendC::Muls(sq, qr, rq, LHD);
            AscendC::Muls(sk, kr, rk, LHD);
            AscendC::Muls(S, S, decay, 8192);
            AscendC::Muls(S[8192], S[8192], decay, 8192);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(kb, sk, 16, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<float, false>(P, S, kb, FULL, 128, pmul); // P = S ⊙ k 行标量
            AscendC::Mul<float, false>(P[64], S[64], kb, FULL, 128, pmul);
            AscendC::PipeBarrier<PIPE_V>();
            for (int w = 8192; w >= 128; w >>= 1) { // 列求和树 -> P[0..127] = u = S^T k
                AscendC::Add(P, P, P[w], w);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Sub(dl, vr, P, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(dl, dl, beta, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst<float, float, false>(S, dl, kb, FULL, 128, pmad); // S += k ⊗ delta
            AscendC::MulAddDst<float, float, false>(S[64], dl[64], kb, FULL, 128, pmad);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(kb, sq, 16, AscendC::BrcbRepeatParams(1, 8)); // 复用 kb 放 q 的广播
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<float, false>(P, S, kb, FULL, 128, pmul);
            AscendC::Mul<float, false>(P[64], S[64], kb, FULL, 128, pmul);
            AscendC::PipeBarrier<PIPE_V>();
            for (int w = 8192; w >= 128; w >>= 1) { // P[0..127] = o = S^T q
                AscendC::Add(P, P, P[w], w);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(tm, P, P, LHD); // 门控 RMSNorm
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(red, tm, 64, 2, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(red[16], red, 2, 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(red[16], red[16], 1.0f / LHD, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(red[16], red[16], EPS, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(red[16], red[16], 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            float ro = 1.0f / red.GetValue(16);
            AscendC::Muls(tm, zr, -1.0f, LHD); // silu(z)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tm, tm, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tm, tm, 1.0f, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(tm, zr, tm, LHD);
            AscendC::Muls(P, P, ro, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(P, P, snorm, LHD);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(P, P, tm, LHD);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(go[(uint64_t)t * V_DIM + h * LHD], P, LHD);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::DataCopy(gst[(uint64_t)h * LHD * LHD], S, LHD * LHD);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// q/k 头 RMSNorm + 交错 MRoPE + 写 KV cache; cos/sin 行 [T][32] 由 host 按 pos3 预先算好
// 单元 = (t, 头), 头 0..23 为 q, 24..27 为 k (k 同时把 v 写入 cache)
extern "C" __global__ __aicore__ void k_qk_rope(GM_ADDR qg, GM_ADDR kx, GM_ADDR vx, GM_ADDR q_norm, GM_ADDR k_norm, GM_ADDR cosr, GM_ADDR sinr, GM_ADDR kc, GM_ADDR vc, int32_t n_past, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    pipe.InitBuffer(tb, 16 * 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> qnw = ub.ReinterpretCast<float>(); // [256]
    LocalTensor<float> knw = ub[1024].ReinterpretCast<float>();
    LocalTensor<float> pr = ub[2048].ReinterpretCast<float>(); // 当前头 [256]
    LocalTensor<float> tp = ub[3072].ReinterpretCast<float>();
    LocalTensor<float> yr = ub[4096].ReinterpretCast<float>();
    LocalTensor<float> cs = ub[5120].ReinterpretCast<float>(); // [32]+[32]
    LocalTensor<float> red = ub[5504].ReinterpretCast<float>(); // 5504 = 5120+384, 64 个
    LocalTensor<float> vr = ub[5760].ReinterpretCast<float>();
    GlobalTensor<float> gqg, gkx, gvx, gqn, gkn, gcs, gsn, gkc, gvc;
    gqg.SetGlobalBuffer((__gm__ float*)qg, (uint64_t)T * Q_DIM);
    gkx.SetGlobalBuffer((__gm__ float*)kx, (uint64_t)T * KV_DIM);
    gvx.SetGlobalBuffer((__gm__ float*)vx, (uint64_t)T * KV_DIM);
    gqn.SetGlobalBuffer((__gm__ float*)q_norm, HD);
    gkn.SetGlobalBuffer((__gm__ float*)k_norm, HD);
    gcs.SetGlobalBuffer((__gm__ float*)cosr, (uint64_t)T * 32);
    gsn.SetGlobalBuffer((__gm__ float*)sinr, (uint64_t)T * 32);
    gkc.SetGlobalBuffer((__gm__ float*)kc, 1); // 行号大, 64 位偏移
    gvc.SetGlobalBuffer((__gm__ float*)vc, 1);
    AscendC::DataCopy(qnw, gqn, HD);
    AscendC::DataCopy(knw, gkn, HD);
    AscendC::PipeBarrier<PIPE_ALL>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int units = T * (N_HEAD + N_KV);
    for (int uidx = core; uidx < units; uidx += cores) {
        int t = uidx / (N_HEAD + N_KV), u = uidx % (N_HEAD + N_KV);
        bool is_q = u < N_HEAD;
        if (is_q) AscendC::DataCopy(pr, gqg[(uint64_t)t * Q_DIM + u * HD * 2], HD);
        else AscendC::DataCopy(pr, gkx[(uint64_t)t * KV_DIM + (u - N_HEAD) * HD], HD);
        AscendC::DataCopy(cs, gcs[(uint64_t)t * 32], 32);
        AscendC::DataCopy(cs[32], gsn[(uint64_t)t * 32], 32);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(tp, pr, pr, HD);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(red, tp, 64, HD / 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(red[32], red, HD / 64, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(red[32], red[32], 1.0f / HD, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(red[32], red[32], EPS, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(red[32], red[32], 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float rs = 1.0f / red.GetValue(32);
        AscendC::Muls(yr, pr, rs, HD);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yr, yr, is_q ? qnw : knw, HD);
        AscendC::PipeBarrier<PIPE_V>();
        // 前 64 维 rotate_half: y[i] = sh[i]c - sh[i+32]s, y[i+32] = sh[i+32]c + sh[i]s
        AscendC::Mul(tp, yr, cs, 32); // sh[0:32]*c
        AscendC::Mul(tp[32], yr[32], cs, 32); // sh[32:64]*c
        AscendC::Mul(tp[64], yr, cs[32], 32); // sh[0:32]*s
        AscendC::Mul(tp[96], yr[32], cs[32], 32); // sh[32:64]*s
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(yr, tp, tp[96], 32);
        AscendC::Add(yr[32], tp[32], tp[64], 32);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (is_q) {
            AscendC::DataCopy(gqg[(uint64_t)t * Q_DIM + u * HD * 2], yr, HD);
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }
        int kh = u - N_HEAD;
        uint64_t dst = (uint64_t)(n_past + t) * KV_DIM + kh * HD;
        AscendC::DataCopy(vr, gvx[(uint64_t)t * KV_DIM + kh * HD], HD);
        AscendC::DataCopy(gkc[dst], yr, HD);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gvc[dst], vr, HD);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// 因果 GQA 注意力 + 输出门控; 单元 = (t, q头), 每批 64 个 key 在线 softmax
extern "C" __global__ __aicore__ void k_attn(GM_ADDR qg, GM_ADDR kc, GM_ADDR vc, GM_ADDR att, int32_t n_past, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    constexpr int J = 64;
    pipe.InitBuffer(tb, J * HD * 4 * 2 + J * 64 * 4 + 16384);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> Kb = ub.ReinterpretCast<float>(); // [J][256]
    LocalTensor<float> Vb = ub[J * HD * 4].ReinterpretCast<float>(); // [J][256]
    LocalTensor<float> Mq = ub[J * HD * 8].ReinterpretCast<float>(); // [J][64]
    int o3 = J * HD * 8 + J * 64 * 4;
    LocalTensor<float> qr = ub[o3].ReinterpretCast<float>(); // [256]
    LocalTensor<float> gr = ub[o3 + 1024].ReinterpretCast<float>(); // gate [256]
    LocalTensor<float> ob = ub[o3 + 2048].ReinterpretCast<float>(); // o 累加 [256]
    LocalTensor<float> sc = ub[o3 + 3072].ReinterpretCast<float>(); // scores [64] + c 位
    LocalTensor<float> pb = ub[o3 + 3456].ReinterpretCast<float>(); // Brcb(p) [J*8]
    LocalTensor<float> red = ub[o3 + 5504].ReinterpretCast<float>(); // [256] (sigmoid 阶段用满)
    LocalTensor<float> wk = ub[o3 + 6528].ReinterpretCast<float>(); // ReduceMax 工作区 / 常量 1 [256]
    GlobalTensor<float> gqg, gkc, gvc, gat;
    gqg.SetGlobalBuffer((__gm__ float*)qg, (uint64_t)T * Q_DIM);
    gkc.SetGlobalBuffer((__gm__ float*)kc, 1);
    gvc.SetGlobalBuffer((__gm__ float*)vc, 1);
    gat.SetGlobalBuffer((__gm__ float*)att, (uint64_t)T * N_HEAD * HD);
    AscendC::SetVectorMask<float>(FULL, FULL);
    const float SCALE = 0.0625f; // 1/sqrt(256)
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    bool dbg = T < 0; // 调试: T=-1 当 T=1 跑, 且每批把 sc/树结果/ob 快照写到 att 行 1 之后
    if (dbg) T = 1;
    int units = T * N_HEAD;
    AscendC::BinaryRepeatParams smul(1, 1, 1, 8, 32, 0); // score 的四分之一乘: src1 (q 段) 每 repeat 重复
    AscendC::BinaryRepeatParams vmul(1, 1, 0, 8, 32, 1); // Mq = V 段 ⊙ p 行标量
    AscendC::DataCopyParams cpk(J, HD * 4 / 32, (KV_DIM - HD) * 4 / 32, 0); // [J][256] 从 cache 收集
    for (int uidx = core; uidx < units; uidx += cores) {
        int t = uidx / N_HEAD, h = uidx % N_HEAD, kvh = h / (N_HEAD / N_KV);
        int n = n_past + t + 1;
        AscendC::DataCopy(qr, gqg[(uint64_t)t * Q_DIM + h * HD * 2], HD);
        AscendC::DataCopy(gr, gqg[(uint64_t)t * Q_DIM + h * HD * 2 + HD], HD);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate(ob, 0.0f, HD);
        float m = -3e38f, l = 0;
        for (int j0 = 0; j0 < n; j0 += J) {
            int jn = n - j0 < J ? n - j0 : J;
            AscendC::DataCopyParams cpx(jn, HD * 4 / 32, (KV_DIM - HD) * 4 / 32, 0);
            AscendC::PipeBarrier<PIPE_ALL>(); // 必须等上一批的向量读取完成, 否则覆盖正在读的 Kb/Vb
            AscendC::DataCopy(Kb, gkc[(uint64_t)j0 * KV_DIM + kvh * HD], cpx);
            AscendC::DataCopy(Vb, gvc[(uint64_t)j0 * KV_DIM + kvh * HD], cpx);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (jn < J) { // 尾批: V 余下行清零, 否则后面固定 64 行的加权会乘到未初始化 UB
                AscendC::Duplicate(Vb[jn * HD], 0.0f, (J - jn) * HD);
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (int q4 = 0; q4 < 4; q4++) { // Mq[j][0:64] = K[j][q4*64:+64] ⊙ q 段
                if (q4 == 0) AscendC::Mul<float, false>(Mq, Kb, qr, FULL, (uint8_t)jn, smul);
                else AscendC::MulAddDst<float, float, false>(Mq, Kb[q4 * 64], qr[q4 * 64], FULL, (uint8_t)jn, smul);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::WholeReduceSum<float>(sc, Mq, 64, jn, 1, 1, 8); // 每 key 的点积
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(sc, sc, SCALE, jn);
            AscendC::PipeBarrier<PIPE_V>();
            if (jn < J) AscendC::Duplicate(sc[jn], -3e38f, J - jn);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(red, sc, wk, jn, false);
            AscendC::PipeBarrier<PIPE_ALL>();
            float mb = red.GetValue(0);
            float mn = m > mb ? m : mb;
            sc.SetValue(J, m); // 多算一个槽位, 下面统一减 mn 后 exp 得 c = exp(m-mn)
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Adds(sc, sc, -mn, J + 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(sc, sc, J + 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(red, sc, 64, 1, 1, 1, 8); // sum p (前 64 个)
            AscendC::PipeBarrier<PIPE_ALL>();
            float c = sc.GetValue(J), ps = red.GetValue(0);
            l = l * c + ps;
            m = mn;
            AscendC::Brcb(pb, sc, 8, AscendC::BrcbRepeatParams(1, 8));
            AscendC::Muls(ob, ob, c, HD);
            AscendC::PipeBarrier<PIPE_V>();
            if (dbg && h == 0 && j0 == 128) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + 3 * 1024 + 512], ob, HD); // 重缩放后
                AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + 3 * 1024], pb, 512); // pb
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (int q4 = 0; q4 < 4; q4++) { // o[q4] += sum_j p_j V[j][q4]
                AscendC::Mul<float, false>(Mq, Vb[q4 * 64], pb, FULL, (uint8_t)J, vmul);
                AscendC::PipeBarrier<PIPE_V>();
                if (dbg && h == 0 && j0 == 128 && q4 == 0) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + 4 * 1024], Vb, 64); // Vb 行0 前 64
                    AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + 4 * 1024 + 64], Mq, 128); // Mq 行0 行1
                    AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + 4 * 1024 + 192], Mq[2 * 64], 64); // Mq 行2
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                for (int w = J * 64 / 2; w >= 64; w >>= 1) {
                    AscendC::Add(Mq, Mq, Mq[w], w);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (dbg && h == 0 && q4 == 0) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + (j0 / J) * 1024 + 128], Mq, 64); // 树结果
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::Add(ob[q4 * 64], ob[q4 * 64], Mq, 64);
                AscendC::PipeBarrier<PIPE_V>();
                if (dbg && h == 0 && j0 == 128 && q4 == 0) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + 3 * 1024 + 768], ob, HD); // q0 加完后
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            if (dbg && h == 0) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + (j0 / J) * 1024], sc, 72); // exp 后的 p + c
                AscendC::DataCopy(gat[(uint64_t)(N_HEAD * HD) + (j0 / J) * 1024 + 256], ob, HD); // ob 快照
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        AscendC::Muls(red, gr, -1.0f, HD); // sigmoid(gate)
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(red, red, HD);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(red, red, 1.0f, HD);
        AscendC::Duplicate(wk, 1.0f, HD);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(red, wk, red, HD); // sigmoid = 1/(1+e^-g); red/wk 都借 256 个 float 的空间
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ob, ob, 1.0f / l, HD);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(ob, ob, red, HD);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gat[(uint64_t)t * N_HEAD * HD + h * HD], ob, HD);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
