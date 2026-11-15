// Qwen3.8-27B 的全部 Ascend C kernel (310P)
// matmul 走 cube: 权重为 repack.h 的 fractal 量化布局 (I8F/I8BF/I4F), 向量单元反量化到 fp16, mmad fp16 累加 fp32
// k_gemm 统一 decode (m=1, x 为 f32 向量) 和 prefill (m<=256, x 由 k_pack_x 转 fp16 zZ 布局)
// 其余 kernel (attention/DeltaNet/conv/norm 等) 全 fp32 向量计算; 激活/缓存均为 GM 上的 fp32
#include "kernel_operator.h"

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::RoundMode;
using AscendC::HardEvent;

namespace {

constexpr int D = 5120, FF = 17408;
constexpr int N_HEAD = 24, N_KV = 4, HD = 256;
constexpr int LK = 16, LV = 48, LHD = 128;
constexpr int KEY_DIM = LK * LHD, V_DIM = LV * LHD, QKV_DIM = 2 * KEY_DIM + V_DIM;
constexpr int Q_DIM = N_HEAD * HD * 2, KV_DIM = N_KV * HD;
constexpr float EPS = 1e-6f;
constexpr uint64_t FULL = 0xffffffffffffffffULL;
enum { FMT_F32 = 0, FMT_I8 = 1, FMT_I8B = 2, FMT_I4 = 3, FMT_I8F = 4, FMT_I8BF = 5, FMT_I4F = 6 };

__aicore__ inline int pad64i(int x) { return (x + 63) / 64 * 64; }

// 量化权重 fractal 布局 (见 repack.h) 的 cube GEMM: 向量反量化 fp16 -> L1 -> L0B -> mmad fp16 累加 fp32
// m=1 (decode): x 为 f32 [k], kernel 内转 fp16 后常驻 L0A (gemv 模式, 按 512B fractal 对齐取片)
// m>1 (prefill): xp 为 k_pack_x 产出的 fp16 zZ 切片布局 [kc][mi][ki][16m][16k], 按切片流进 L1/L0A
// 流水: MTE2(量化切片) -> V(反量化, scale/bias 用 src1 块复用逐 fractal 乘加) -> MTE3(L1) -> MTE1(L0B 转置加载) -> mmad 累加 L0C -> 输出
template <int FMT> __aicore__ inline void gemm_body(GM_ADDR w, GM_ADDR x, GM_ADDR xp, GM_ADDR y, int32_t m, int32_t n, int32_t k)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buf;
    AscendC::TBuf<AscendC::TPosition::A1> l1_buf;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buf;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buf;
    pipe.InitBuffer(ub_buf, 230 * 1024);
    pipe.InitBuffer(l1_buf, 256 * 1024);
    pipe.InitBuffer(l0a_buf, 64 * 1024);
    pipe.InitBuffer(l0b_buf, 64 * 1024);
    pipe.InitBuffer(l0c_buf, 256 * 1024);
    LocalTensor<uint8_t> ub = ub_buf.Get<uint8_t>();
    LocalTensor<uint8_t> l1 = l1_buf.Get<uint8_t>();
    LocalTensor<half> l0a = l0a_buf.Get<half>();
    LocalTensor<half> l0b = l0b_buf.Get<half>();
    LocalTensor<float> l0c = l0c_buf.Get<float>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int nr = n / cores, KS = k / 128, M1 = (m + 15) / 16;
    constexpr int QB = FMT == FMT_I4F ? 1024 : 2048; // 每 ni(16行) 每切片量化字节
    constexpr int SB = FMT == FMT_I8F ? 2304 : (FMT == FMT_I8BF ? 2560 : 1536); // 每 ni 每切片总字节
    uint64_t core_off = (uint64_t)core * (nr / 16) * KS * SB;
    GlobalTensor<uint8_t> gw;
    gw.SetGlobalBuffer((__gm__ uint8_t*)w + core_off, (uint64_t)(nr / 16) * KS * SB);
    GlobalTensor<float> gx;
    GlobalTensor<half> gxp;
    if (m == 1) gx.SetGlobalBuffer((__gm__ float*)x, k);
    else gxp.SetGlobalBuffer((__gm__ half*)xp, (uint64_t)KS * M1 * 2048);
    GlobalTensor<float> gy;
    gy.SetGlobalBuffer((__gm__ float*)y, (uint64_t)m * n);
    LocalTensor<uint8_t> qbuf0 = ub, qbuf1 = ub[40960]; // 切片双缓冲 (I8BF 最大 40960)
    LocalTensor<half> wh0 = ub[81920].ReinterpretCast<half>(), wh1 = ub[81920 + 32768].ReinterpretCast<half>();
    LocalTensor<float> co0 = ub[147456].ReinterpretCast<float>(), co1 = ub[147456 + 16384].ReinterpretCast<float>();
    LocalTensor<half> th = ub[180224].ReinterpretCast<half>(); // I4F 解码临时
    LocalTensor<half> tm = ub[196608].ReinterpretCast<half>();
    LocalTensor<int16_t> ti = ub[212992].ReinterpretCast<int16_t>();
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::UnaryRepeatParams cp8(1, 1, 8, 4);
    AscendC::UnaryRepeatParams cpa(1, 1, 8, 8);
    AscendC::BinaryRepeatParams bph(1, 1, 1, 8, 8, 8);
    AscendC::BinaryRepeatParams bsc(1, 1, 0, 16, 16, 1); // scale 乘: 每 fractal 两个 repeat 分两次调用, src1 块内复用
    if (m == 1) { // x: f32 -> fp16 -> L1 (每 128-k 切片 512B 对齐) -> L0A 常驻
        LocalTensor<float> xf = ub.ReinterpretCast<float>(); // 借 qbuf 区
        LocalTensor<half> xh = ub[81920].ReinterpretCast<half>(); // 借 wh 区
        AscendC::DataCopy(xf, gx, k);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int o = 0; o < k; o += 16320) { // 255 rep 上限分段
            int c = k - o < 16320 ? k - o : 16320;
            AscendC::Cast<half, float, false>(xh[o], xf[o], RoundMode::CAST_NONE, FULL, (uint8_t)(c / 64), AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(l1[131072].ReinterpretCast<half>(), xh, AscendC::DataCopyParams((uint16_t)KS, 8, 0, 8));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LoadData(l0a, l1[131072].ReinterpretCast<half>(), AscendC::LoadData2DParams(0, (uint8_t)(KS < 120 ? KS : 120), 1, 0, 0, false, 0));
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    // 事件 id 由 TPipe 分配 (避免与框架内部冲突), free 方向先置位
    AscendC::TEventID eq[2], eqf[2], ewh[2], ewf[2], eco[2], ecf[2], ebl[2], eblf[2], exa[2], exaf[2];
    for (int i = 0; i < 2; i++) {
        eq[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        eqf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
        ewh[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        ewf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
        eco[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        ecf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
        ebl[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE1>();
        eblf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE3>();
        exa[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
        exaf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eqf[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ewf[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ecf[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[i]);
    }
    AscendC::TEventID em = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
    AscendC::TEventID emf = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
    AscendC::TEventID emv = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_V>();
    AscendC::TEventID evm = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_M>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(emf);
    AscendC::SetFlag<AscendC::HardEvent::V_M>(evm);
    if (m > 1) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(exaf[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(exaf[1]);
    }
    int nblk = (nr + 255) / 256;
    uint64_t blk_base = 0;
    for (int b = 0; b < nblk; b++) {
        int nb = nr - b * 256 < 256 ? nr - b * 256 : 256;
        int NI = nb / 16;
        int snb = NI * SB, qnb = NI * QB;
        int nfrac = 8 * NI; // 每切片 fractal 数
        for (int kc = 0; kc < KS; kc++) {
            int p = kc & 1;
            LocalTensor<uint8_t> qbuf = p ? qbuf1 : qbuf0;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eqf[p]);
            AscendC::DataCopy(qbuf, gw[blk_base + (uint64_t)kc * snb], snb);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eq[p]);
            if (m > 1) { // x 切片 GM -> L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(exaf[p]);
                AscendC::DataCopy(l1[131072 + p * 65536].ReinterpretCast<half>(), gxp[(uint64_t)kc * M1 * 2048], M1 * 2048);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(exa[p]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eq[p]);
            for (int h = 0; h < 2; h++) { // 反量化, 每次半片
                LocalTensor<half> wh = h ? wh1 : wh0;
                int hb = h * qnb / 2, hf = h * nfrac / 2; // 半片字节偏移 / 起始 fractal
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ewf[h]);
                if (FMT != FMT_I4F) {
                    AscendC::Cast<half, int8_t, false>(wh, qbuf[hb].ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, FULL, (uint8_t)(NI * 8), cp8);
                } else { // nibble 半片: 字节 i = lo(半片前 2ki 的元素 i) | hi(后 2ki)<<4
                    int nby = NI * 512;
                    AscendC::Cast<half, uint8_t, false>(th, qbuf[hb], RoundMode::CAST_NONE, FULL, (uint8_t)(nby / 128), cp8);
                    AscendC::Muls<half, false>(tm, th, (half)0.0625f, FULL, (uint8_t)(nby / 128), cpa);
                    AscendC::Adds<half, false>(tm, tm, (half)-0.46875f, FULL, (uint8_t)(nby / 128), cpa);
                    AscendC::Cast<int16_t, half, false>(ti, tm, RoundMode::CAST_RINT, FULL, (uint8_t)(nby / 128), cpa);
                    AscendC::Cast<half, int16_t, false>(wh[nby], ti, RoundMode::CAST_NONE, FULL, (uint8_t)(nby / 128), cpa); // hi
                    AscendC::Muls<half, false>(tm, wh[nby], (half)-16.0f, FULL, (uint8_t)(nby / 128), cpa);
                    AscendC::Add<half, false>(wh, th, tm, FULL, (uint8_t)(nby / 128), bph); // lo = b - 16*hi
                }
                LocalTensor<half> sc = qbuf[qnb].ReinterpretCast<half>(); // scale 块区 (每 fractal 16 half)
                AscendC::Mul<half, false>(wh, wh, sc[hf * 16], FULL, (uint8_t)(nfrac / 2), bsc);
                AscendC::Mul<half, false>(wh[128], wh[128], sc[hf * 16], FULL, (uint8_t)(nfrac / 2), bsc);
                if (FMT != FMT_I8F) { // bias 块区
                    LocalTensor<half> bi = qbuf[qnb + nfrac * 32].ReinterpretCast<half>();
                    AscendC::Add<half, false>(wh, wh, bi[hf * 16], FULL, (uint8_t)(nfrac / 2), bsc);
                    AscendC::Add<half, false>(wh[128], wh[128], bi[hf * 16], FULL, (uint8_t)(nfrac / 2), bsc);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ewh[h]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eqf[p]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[p]);
            for (int h = 0; h < 2; h++) { // fp16 权重半片 -> L1
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ewh[h]);
                AscendC::DataCopy(l1[p * 65536 + h * (nfrac / 2) * 512].ReinterpretCast<half>(), h ? wh1 : wh0, (nfrac / 2) * 256);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ewf[h]);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(ebl[p]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(ebl[p]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(emf);
            AscendC::LoadData(l0b, l1[p * 65536].ReinterpretCast<half>(), AscendC::LoadData2DParams(0, (uint16_t)nfrac, 1, 0, 0, true, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[p]);
            if (m == 1 && KS > 120 && kc % 120 == 0 && !(b == 0 && kc == 0)) // 大 k: L0A 按 120 fractal 分段重载 (gemv mmad 读 A 为 512B 粒度, 留余量)
                AscendC::LoadData(l0a, l1[131072 + kc * 512].ReinterpretCast<half>(), AscendC::LoadData2DParams(0, (uint8_t)(KS - kc < 120 ? KS - kc : 120), 1, 0, 0, false, 0));
            if (m > 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(exa[p]);
                AscendC::LoadData(l0a, l1[131072 + p * 65536].ReinterpretCast<half>(), AscendC::LoadData2DParams(0, (uint16_t)(M1 * 8), 1, 0, 0, false, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(exaf[p]);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(em);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(em);
            if (kc == 0) AscendC::WaitFlag<AscendC::HardEvent::V_M>(evm); // 上一块的 L0C 输出已读完
            AscendC::MmadParams mm((uint16_t)m, (uint16_t)nb, 128, 0, false, kc == 0);
            if (m == 1) AscendC::Mmad(l0c, l0a[kc % 120 * 256], l0b, mm);
            else AscendC::Mmad(l0c, l0a, l0b, mm);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(emf);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_V>(emv);
        AscendC::WaitFlag<AscendC::HardEvent::M_V>(emv);
        AscendC::DataCopyEnhancedParams ep;
        ep.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        if (m == 1) { // y = 各 C fractal 的行 0, 跨步收集写出
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ecf[0]);
            AscendC::DataCopy(co0, l0c, AscendC::DataCopyParams(1, (uint16_t)NI, 0, 0), ep);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eco[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eco[0]);
            AscendC::DataCopy(gy[(uint64_t)core * nr + b * 256], co0, AscendC::DataCopyParams((uint16_t)NI, 2, 30, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ecf[0]);
        } else { // C [N1][M1][16m][16n] -> y [m][n] 自然布局
            for (int ni = 0; ni < NI; ni++) {
                int c = ni & 1;
                LocalTensor<float> co = c ? co1 : co0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ecf[c]);
                AscendC::DataCopy(co, l0c[(ni * M1) * 256], AscendC::DataCopyParams(1, (uint16_t)M1, 0, 0), ep);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eco[c]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eco[c]);
                for (int mi = 0; mi < M1; mi++) {
                    int mrows = m - mi * 16 < 16 ? m - mi * 16 : 16;
                    AscendC::DataCopy(gy[(uint64_t)(mi * 16) * n + core * nr + b * 256 + ni * 16], co[mi * 256], AscendC::DataCopyParams((uint16_t)mrows, 2, 0, (uint16_t)(n / 8 - 2)));
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ecf[c]);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_M>(evm);
        blk_base += (uint64_t)KS * snb;
    }
    for (int i = 0; i < 2; i++) { // 清理未消费事件
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eqf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ewf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ecf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[i]);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(emf);
    AscendC::WaitFlag<AscendC::HardEvent::V_M>(evm);
    if (m > 1) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(exaf[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(exaf[1]);
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

extern "C" __global__ __aicore__ void k_gemm_i8(GM_ADDR w, GM_ADDR x, GM_ADDR xp, GM_ADDR y, int32_t m, int32_t n, int32_t k) { gemm_body<FMT_I8F>(w, x, xp, y, m, n, k); }
extern "C" __global__ __aicore__ void k_gemm_i8b(GM_ADDR w, GM_ADDR x, GM_ADDR xp, GM_ADDR y, int32_t m, int32_t n, int32_t k) { gemm_body<FMT_I8BF>(w, x, xp, y, m, n, k); }
extern "C" __global__ __aicore__ void k_gemm_i4(GM_ADDR w, GM_ADDR x, GM_ADDR xp, GM_ADDR y, int32_t m, int32_t n, int32_t k) { gemm_body<FMT_I4F>(w, x, xp, y, m, n, k); }
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
        AscendC::BlockReduceSum<float>(yb, p, D / 64, 64, 1, 1, 8);
        AscendC::WholeReduceSum<float>(ss, yb, 64, D / 512, 1, 1, 8);
        AscendC::WholeReduceSum<float>(ss[32], ss, D / 512, 1, 1, 1, 8);
        AscendC::Muls(ss[32], ss[32], 1.0f / D, 1);
        AscendC::Adds(ss[32], ss[32], EPS, 1);
        AscendC::Sqrt(ss[32], ss[32], 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float rs = 1.0f / ss.GetValue(32);
        AscendC::Muls(yb, a, rs, D);
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
        AscendC::Exp(t, t, c);
        AscendC::Adds(t, t, 1.0f, c);
        AscendC::Div(gb, gb, t, c);
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
        AscendC::MulAddDst(o, w4[C], s1, C);
        AscendC::MulAddDst(o, w4[2 * C], s2, C);
        AscendC::MulAddDst(o, w4[3 * C], xn, C);
        AscendC::Muls(tt, o, -1.0f, C);
        AscendC::Exp(tt, tt, C);
        AscendC::Adds(tt, tt, 1.0f, C);
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
            AscendC::Exp(sp2, sp2, 48);
            AscendC::Adds(sp2, sp2, 1.0f, 48); // 1+e^-b, beta = 1/它 (之后标量取倒数)
            AscendC::Mins(tm, sp, 20.0f, 48);
            AscendC::Exp(tm, tm, 48);
            AscendC::Adds(tm, tm, 1.0f, 48);
            AscendC::Ln(tm, tm, 48);
            AscendC::Max(tm, tm, sp, 48); // softplus (x>20 时取 x)
            AscendC::Mul(tm, tm, sa, 48);
            AscendC::Exp(tm, tm, 48); // decay 行
            // q/k 的 L2 范数
            AscendC::Mul(sq, qr, qr, LHD);
            AscendC::Mul(sk, kr, kr, LHD);
            AscendC::WholeReduceSum<float>(red, sq, 64, 2, 1, 1, 8);
            AscendC::WholeReduceSum<float>(red[8], sk, 64, 2, 1, 1, 8);
            AscendC::WholeReduceSum<float>(red[16], red, 2, 1, 1, 1, 8);
            AscendC::WholeReduceSum<float>(red[24], red[8], 2, 1, 1, 1, 8);
            AscendC::Adds(red[16], red[16], 1e-6f, 1);
            AscendC::Adds(red[24], red[24], 1e-6f, 1);
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
            AscendC::Brcb(kb, sk, 16, AscendC::BrcbRepeatParams(1, 8));
            AscendC::Mul<float, false>(P, S, kb, FULL, 128, pmul); // P = S ⊙ k 行标量
            AscendC::Mul<float, false>(P[64], S[64], kb, FULL, 128, pmul);
            for (int w = 8192; w >= 128; w >>= 1) { // 列求和树 -> P[0..127] = u = S^T k
                AscendC::Add(P, P, P[w], w);
            }
            AscendC::Sub(dl, vr, P, LHD);
            AscendC::Muls(dl, dl, beta, LHD);
            AscendC::MulAddDst<float, float, false>(S, dl, kb, FULL, 128, pmad); // S += k ⊗ delta
            AscendC::MulAddDst<float, float, false>(S[64], dl[64], kb, FULL, 128, pmad);
            AscendC::Brcb(kb, sq, 16, AscendC::BrcbRepeatParams(1, 8)); // 复用 kb 放 q 的广播
            AscendC::Mul<float, false>(P, S, kb, FULL, 128, pmul);
            AscendC::Mul<float, false>(P[64], S[64], kb, FULL, 128, pmul);
            for (int w = 8192; w >= 128; w >>= 1) { // P[0..127] = o = S^T q
                AscendC::Add(P, P, P[w], w);
            }
            AscendC::Mul(tm, P, P, LHD); // 门控 RMSNorm
            AscendC::WholeReduceSum<float>(red, tm, 64, 2, 1, 1, 8);
            AscendC::WholeReduceSum<float>(red[16], red, 2, 1, 1, 1, 8);
            AscendC::Muls(red[16], red[16], 1.0f / LHD, 1);
            AscendC::Adds(red[16], red[16], EPS, 1);
            AscendC::Sqrt(red[16], red[16], 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            float ro = 1.0f / red.GetValue(16);
            AscendC::Muls(tm, zr, -1.0f, LHD); // silu(z)
            AscendC::Exp(tm, tm, LHD);
            AscendC::Adds(tm, tm, 1.0f, LHD);
            AscendC::Div(tm, zr, tm, LHD);
            AscendC::Muls(P, P, ro, LHD);
            AscendC::Mul(P, P, snorm, LHD);
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
        AscendC::WholeReduceSum<float>(red, tp, 64, HD / 64, 1, 1, 8);
        AscendC::WholeReduceSum<float>(red[32], red, HD / 64, 1, 1, 1, 8);
        AscendC::Muls(red[32], red[32], 1.0f / HD, 1);
        AscendC::Adds(red[32], red[32], EPS, 1);
        AscendC::Sqrt(red[32], red[32], 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float rs = 1.0f / red.GetValue(32);
        AscendC::Muls(yr, pr, rs, HD);
        AscendC::Mul(yr, yr, is_q ? qnw : knw, HD);
        // 前 64 维 rotate_half: y[i] = sh[i]c - sh[i+32]s, y[i+32] = sh[i+32]c + sh[i]s
        AscendC::Mul(tp, yr, cs, 32); // sh[0:32]*c
        AscendC::Mul(tp[32], yr[32], cs, 32); // sh[32:64]*c
        AscendC::Mul(tp[64], yr, cs[32], 32); // sh[0:32]*s
        AscendC::Mul(tp[96], yr[32], cs[32], 32); // sh[32:64]*s
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

// 因果 GQA 注意力 + 输出门控; 单元 = (16-query 块, q头), K/V 批 (J=48 key) 在块内 query 间复用
// 两遍扫描: 第一遍只读 K 求每 query 的全局 max, 第二遍直接 exp 累加 (无在线重缩放), 全程无标量数据读取
// 因果与尾批通过把 score 置 -3e38 屏蔽; max/sum 以 [16][8] 块跨步存放, 块广播用 src1 跨步 0 技巧
extern "C" __global__ __aicore__ void k_attn(GM_ADDR qg, GM_ADDR kc, GM_ADDR vc, GM_ADDR att, int32_t n_past, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    constexpr int J = 48, QT = 16;
    pipe.InitBuffer(tb, 190 * 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> Kb = ub.ReinterpretCast<float>(); // [J][256] (结尾借放 gate 块)
    LocalTensor<float> Vb = ub[J * HD * 4].ReinterpretCast<float>(); // [J][256]
    LocalTensor<float> Mq = ub[J * HD * 8].ReinterpretCast<float>(); // [J][256] 分数/加权和工作区
    int o3 = J * HD * 12;
    LocalTensor<float> Qb = ub[o3].ReinterpretCast<float>(); // [QT][256]
    LocalTensor<float> Ob = ub[o3 + QT * HD * 4].ReinterpretCast<float>(); // [QT][256]
    int o4 = o3 + QT * HD * 8;
    LocalTensor<float> sc = ub[o4].ReinterpretCast<float>(); // scores [64] (后 16 恒屏蔽)
    LocalTensor<float> pb = ub[o4 + 256].ReinterpretCast<float>(); // Brcb(p) [J*8]
    LocalTensor<float> mb = ub[o4 + 256 + 1536].ReinterpretCast<float>(); // max 累计 [16][8]
    LocalTensor<float> lb = ub[o4 + 2304].ReinterpretCast<float>(); // sum 累计 [16][8]
    LocalTensor<float> mc = ub[o4 + 2816].ReinterpretCast<float>(); // 紧凑 max [16]
    LocalTensor<float> mnbr = ub[o4 + 3072].ReinterpretCast<float>(); // 广播块 [16][8]
    LocalTensor<float> lnbr = ub[o4 + 3584].ReinterpretCast<float>(); // 1/l 广播块 [16][8]
    LocalTensor<float> red = ub[o4 + 4096].ReinterpretCast<float>(); // [256]
    LocalTensor<float> wk = ub[o4 + 5120].ReinterpretCast<float>(); // ReduceMax 工作区 / 常量 1 [256]
    GlobalTensor<float> gqg, gkc, gvc, gat;
    gqg.SetGlobalBuffer((__gm__ float*)qg, (uint64_t)T * Q_DIM);
    gkc.SetGlobalBuffer((__gm__ float*)kc, 1);
    gvc.SetGlobalBuffer((__gm__ float*)vc, 1);
    gat.SetGlobalBuffer((__gm__ float*)att, (uint64_t)T * N_HEAD * HD);
    AscendC::SetVectorMask<float>(FULL, FULL);
    const float SCALE = 0.0625f; // 1/sqrt(256)
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int ntile = (T + QT - 1) / QT;
    int units = ntile * N_HEAD;
    AscendC::BinaryRepeatParams smul(1, 1, 1, 8, 32, 0); // score 四分之一乘: src1 (q 段) 每 repeat 重复
    AscendC::BinaryRepeatParams vmul(1, 1, 0, 32, 32, 1); // Mq[j][q4 段] = V 段 ⊙ p 行标量 (dst/src0 跨 256 行进)
    AscendC::BinaryRepeatParams bbr(1, 1, 0, 8, 8, 0); // src1 单块广播
    for (int uidx = core; uidx < units; uidx += cores) {
        int ti = uidx / N_HEAD, h = uidx % N_HEAD, kvh = h / (N_HEAD / N_KV);
        int tq0 = ti * QT, tqn = T - tq0 < QT ? T - tq0 : QT;
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(Qb, gqg[(uint64_t)tq0 * Q_DIM + h * HD * 2], AscendC::DataCopyParams((uint16_t)tqn, HD * 4 / 32, (Q_DIM - HD) * 4 / 32, 0));
        AscendC::Duplicate(Ob, 0.0f, QT * HD);
        AscendC::Duplicate(mb, -3e38f, QT * 8);
        AscendC::Duplicate(lb, 0.0f, QT * 8);
        AscendC::Duplicate(sc, -3e38f, 64); // [48..64) 恒为屏蔽值
        int nmax = n_past + tq0 + tqn;
        for (int j0 = 0; j0 < nmax; j0 += J) { // 第一遍: 全局 max (只读 K)
            int jn = nmax - j0 < J ? nmax - j0 : J;
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(Kb, gkc[(uint64_t)j0 * KV_DIM + kvh * HD], AscendC::DataCopyParams((uint16_t)jn, HD * 4 / 32, (KV_DIM - HD) * 4 / 32, 0));
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int tq = 0; tq < tqn; tq++) {
                int nq = n_past + tq0 + tq + 1; // 该 query 可见 key 数
                if (j0 >= nq) continue;
                int jv = nq - j0 < jn ? nq - j0 : jn;
                LocalTensor<float> qr = Qb[tq * HD];
                for (int q4 = 0; q4 < 4; q4++) {
                    if (q4 == 0) AscendC::Mul<float, false>(Mq, Kb, qr, FULL, (uint8_t)jn, smul);
                    else AscendC::MulAddDst<float, float, false>(Mq, Kb[q4 * 64], qr[q4 * 64], FULL, (uint8_t)jn, smul);
                }
                AscendC::WholeReduceSum<float>(sc, Mq, 64, jn, 1, 1, 8);
                if (jv < jn) AscendC::Duplicate(sc[jv], -1e38f, jn - jv); // 因果屏蔽 (×SCALE 前)
                AscendC::ReduceMax<float>(red, sc, wk, jn, false);
                AscendC::Max(mb[tq * 8], mb[tq * 8], red, 1);
            }
        }
        AscendC::BlockReduceMax<float>(mc, mb, 2, 64, 1, 1, 8); // [16][8] -> 紧凑 [16] (其余槽为 -3e38)
        AscendC::Muls(mc, mc, SCALE, QT); // 全局 max (按 SCALE 缩放后)
        AscendC::Brcb(mnbr, mc, 2, AscendC::BrcbRepeatParams(1, 8));
        for (int j0 = 0; j0 < nmax; j0 += J) { // 第二遍: exp 累加 (无重缩放)
            int jn = nmax - j0 < J ? nmax - j0 : J;
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyParams cpx((uint16_t)jn, HD * 4 / 32, (KV_DIM - HD) * 4 / 32, 0);
            AscendC::DataCopy(Kb, gkc[(uint64_t)j0 * KV_DIM + kvh * HD], cpx);
            AscendC::DataCopy(Vb, gvc[(uint64_t)j0 * KV_DIM + kvh * HD], cpx);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (jn < J) { // 尾批: V 余下行清零, 否则 p=0 乘未初始化 UB 可能得 NaN
                AscendC::Duplicate(Vb[jn * HD], 0.0f, (J - jn) * HD);
            }
            for (int tq = 0; tq < tqn; tq++) {
                int nq = n_past + tq0 + tq + 1;
                if (j0 >= nq) continue;
                int jv = nq - j0 < jn ? nq - j0 : jn;
                LocalTensor<float> qr = Qb[tq * HD];
                for (int q4 = 0; q4 < 4; q4++) {
                    if (q4 == 0) AscendC::Mul<float, false>(Mq, Kb, qr, FULL, (uint8_t)jn, smul);
                    else AscendC::MulAddDst<float, float, false>(Mq, Kb[q4 * 64], qr[q4 * 64], FULL, (uint8_t)jn, smul);
                }
                AscendC::WholeReduceSum<float>(sc, Mq, 64, jn, 1, 1, 8);
                AscendC::Muls(sc, sc, SCALE, jn);
                if (jv < J) AscendC::Duplicate(sc[jv], -3e38f, J - jv); // 因果 + 尾批屏蔽
                AscendC::Sub<float, false>(sc, sc, mnbr[tq * 8], FULL, 1, bbr); // 减全局 max (块广播)
                AscendC::Exp(sc, sc, 64); // 屏蔽位 exp(-大) = 0 ([48..64) 为垃圾, 下面只用前 48)
                AscendC::WholeReduceSum<float>(red, sc, J, 1, 1, 1, 8);
                AscendC::Add(lb[tq * 8], lb[tq * 8], red, 1);
                AscendC::Brcb(pb, sc, 6, AscendC::BrcbRepeatParams(1, 8));
                for (int q4 = 0; q4 < 4; q4++) { // Mq[j][256] = V[j] ⊙ p_j (4 段)
                    AscendC::Mul<float, false>(Mq[q4 * 64], Vb[q4 * 64], pb, FULL, (uint8_t)J, vmul);
                }
                for (int w = J / 2; w >= 3; w >>= 1) { // 行二分树 48->24->12->6->3
                    AscendC::Add(Mq, Mq, Mq[w * HD], w * HD);
                }
                AscendC::Add(Mq, Mq, Mq[HD], HD);
                AscendC::Add(Mq, Mq, Mq[2 * HD], HD);
                AscendC::Add(Ob[tq * HD], Ob[tq * HD], Mq, HD);
            }
        }
        AscendC::BlockReduceSum<float>(mc, lb, 2, 64, 1, 1, 8); // 紧凑 sum [16]
        AscendC::Duplicate(wk, 1.0f, QT);
        AscendC::Div(mc, wk, mc, QT); // 1/l
        AscendC::Brcb(lnbr, mc, 2, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_ALL>(); // 批循环结束, Kb 借放 gate 块
        AscendC::DataCopy(Kb, gqg[(uint64_t)tq0 * Q_DIM + h * HD * 2 + HD], AscendC::DataCopyParams((uint16_t)tqn, HD * 4 / 32, (Q_DIM - HD) * 4 / 32, 0));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int tq = 0; tq < tqn; tq++) {
            LocalTensor<float> ob = Ob[tq * HD];
            AscendC::Muls(red, Kb[tq * HD], -1.0f, HD); // sigmoid(gate)
            AscendC::Exp(red, red, HD);
            AscendC::Adds(red, red, 1.0f, HD);
            AscendC::Duplicate(wk, 1.0f, HD);
            AscendC::Div(red, wk, red, HD);
            AscendC::Mul<float, false>(ob, ob, lnbr[tq * 8], FULL, 4, bbr); // ob /= l (块广播)
            AscendC::Mul(ob, ob, red, HD);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(gat[(uint64_t)(tq0 + tq) * N_HEAD * HD + h * HD], ob, HD);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// x 打包: f32 [T][k] -> fp16 zZ 切片布局 [kc][mi][ki 0..7][16m][16k], 供 k_gemm m>1 用; 核间按 kc 切分
// MTE2 / V / MTE3 三级双缓冲流水
extern "C" __global__ __aicore__ void k_pack_x(GM_ADDR x, GM_ADDR xp, int32_t m, int32_t k)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    pipe.InitBuffer(tb, 32 * 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> xf0 = ub.ReinterpretCast<float>(), xf1 = ub[8192].ReinterpretCast<float>(); // [16][128]
    LocalTensor<half> xh0 = ub[16384].ReinterpretCast<half>(), xh1 = ub[20480].ReinterpretCast<half>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int KS = k / 128, M1 = (m + 15) / 16;
    GlobalTensor<float> gx;
    gx.SetGlobalBuffer((__gm__ float*)x, (uint64_t)m * k);
    GlobalTensor<half> gp;
    gp.SetGlobalBuffer((__gm__ half*)xp, (uint64_t)KS * M1 * 2048);
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::TEventID el[2], elf[2], es[2], esf[2];
    for (int i = 0; i < 2; i++) {
        el[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        elf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
        es[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        esf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(elf[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(esf[i]);
    }
    int idx = 0;
    for (int kc = core; kc < KS; kc += cores) {
        for (int mi = 0; mi < M1; mi++) {
            int p = idx & 1;
            idx++;
            int mrows = m - mi * 16 < 16 ? m - mi * 16 : 16;
            LocalTensor<float> xf = p ? xf1 : xf0;
            LocalTensor<half> xh = p ? xh1 : xh0;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(elf[p]);
            AscendC::DataCopy(xf, gx[(uint64_t)(mi * 16) * k + kc * 128], AscendC::DataCopyParams((uint16_t)mrows, 16, (uint16_t)((k - 128) / 8), 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(el[p]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(el[p]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(esf[p]);
            if (mrows < 16) { // 尾块补零
                AscendC::Duplicate(xh, (half)0, 2048);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast<half, float, false>(xh, xf, RoundMode::CAST_NONE, FULL, (uint8_t)(mrows * 2), AscendC::UnaryRepeatParams(1, 1, 4, 8));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(elf[p]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(es[p]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(es[p]);
            for (int ki = 0; ki < 8; ki++) // [16m][128k] -> fractal [ki][16m][16k]
                AscendC::DataCopy(gp[(uint64_t)kc * M1 * 2048 + mi * 2048 + ki * 256], xh[ki * 16], AscendC::DataCopyParams(16, 1, 7, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(esf[p]);
        }
    }
    for (int i = 0; i < 2; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(elf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(esf[i]);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// f32 权重 (alpha/beta, 自然行布局) 的 GEMV, T 个 token 循环, 行常驻 UB; 每核 8 行, block_dim 需为 n/8
extern "C" __global__ __aicore__ void k_gemv_f32t(GM_ADDR w, GM_ADDR x, GM_ADDR y, int32_t n, int32_t k, int32_t T)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    pipe.InitBuffer(tb, (uint32_t)(8 * k * 4 + k * 8 + 8192));
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<float> wr = ub.ReinterpretCast<float>(); // [8][k]
    LocalTensor<float> xb = ub[8 * k * 4].ReinterpretCast<float>(); // [k]
    LocalTensor<float> acc = ub[8 * k * 4 + k * 4].ReinterpretCast<float>(); // [k]
    LocalTensor<float> bs = ub[8 * k * 4 + k * 8].ReinterpretCast<float>(); // [k/8 组和, <=2720]... 放尾部
    LocalTensor<float> bb = ub[8 * k * 4 + k * 8 + 2944].ReinterpretCast<float>(); // [8][64] 两级部分和
    LocalTensor<float> yb = ub[8 * k * 4 + k * 8 + 2944 + 2048].ReinterpretCast<float>(); // [8]
    int core = (int)AscendC::GetBlockIdx();
    int r0 = core * 8;
    GlobalTensor<float> gw, gx, gy;
    gw.SetGlobalBuffer((__gm__ float*)w, (uint64_t)n * k);
    gx.SetGlobalBuffer((__gm__ float*)x, (uint64_t)T * k);
    gy.SetGlobalBuffer((__gm__ float*)y, (uint64_t)T * n);
    AscendC::DataCopy(wr, gw[(uint64_t)r0 * k], 8 * k);
    AscendC::PipeBarrier<PIPE_ALL>();
    int nst = k / 512; // 每行两级归约后的部分和个数 (k 须为 512 倍数)
    for (int t = 0; t < T; t++) {
        AscendC::DataCopy(xb, gx[(uint64_t)t * k], k);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int r = 0; r < 8; r++) {
            AscendC::Mul(acc, wr[r * k], xb, k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float>(bs, acc, k / 64, 64, 1, 1, 8); // per-8 和 [k/8]
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(bb[r * 64], bs, 64, nst, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceSum<float>(yb, bb, nst, 8, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gy[(uint64_t)t * n + r0], yb, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
