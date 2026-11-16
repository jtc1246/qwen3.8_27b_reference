// Qwen3.8-27B 的全部 Ascend C kernel (310P)
// matmul 走 cube: 权重为 repack.h 的 fractal 量化布局 (I8F/I4SF 等), 向量单元反量化到 fp16, mmad fp16 累加 fp32
// k_gemm 统一 decode (m=1, x 为 f32 向量) 和 prefill (m<=256, x 由 k_pack_x 转 fp16 zZ 布局)
// attention: KV cache q8 (u8+per-128 fp16 scale), 反量化 fp16 后 Q·K^T / P·V 走 fp16 mmad (k_attn/k_attn_merge)
// 其余 kernel (DeltaNet/conv/norm 等) 全 fp32 向量计算; 激活均为 GM 上的 fp32
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
constexpr int QKVZ = QKV_DIM + V_DIM; // DeltaNet 的 qkv+z 合并矩阵每 token 输出宽度
constexpr int QKVC = Q_DIM + 2 * KV_DIM; // 全注意力 q+k+v 合并矩阵每 token 输出宽度
constexpr float EPS = 1e-6f;
constexpr uint64_t FULL = 0xffffffffffffffffULL;
enum { FMT_F32 = 0, FMT_I8 = 1, FMT_I8B = 2, FMT_I4 = 3, FMT_I8F = 4, FMT_I8BF = 5, FMT_I4F = 6, FMT_I4SF = 7 };

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
    constexpr int QB = (FMT == FMT_I4F || FMT == FMT_I4SF) ? 1024 : 2048; // 每 ni(16行) 每切片量化字节
    constexpr int SB = FMT == FMT_I8F ? 2304 : (FMT == FMT_I8BF ? 2560 : (FMT == FMT_I4SF ? 1280 : 1536)); // 每 ni 每切片总字节
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
    LocalTensor<half> th = ub[180224].ReinterpretCast<half>(); // nibble 解码临时
    LocalTensor<half> tm = ub[196608].ReinterpretCast<half>();
    LocalTensor<int16_t> ti = ub[212992].ReinterpretCast<int16_t>();
    LocalTensor<uint16_t> mk = ub[229376].ReinterpretCast<uint16_t>(); // I4SF And 掩码 [16]0x0F0F | [16]0xF0F0
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::UnaryRepeatParams cp8(1, 1, 8, 4);
    AscendC::UnaryRepeatParams cpa(1, 1, 8, 8);
    AscendC::BinaryRepeatParams bph(1, 1, 1, 8, 8, 8);
    AscendC::BinaryRepeatParams bsc(1, 1, 0, 16, 16, 1); // scale 乘: 每 fractal 两个 repeat 分两次调用, src1 块内复用
    AscendC::BinaryRepeatParams bmk(1, 1, 0, 8, 8, 0); // And: src1 单块复用且不前进
    if (FMT == FMT_I4SF) {
        AscendC::Duplicate(mk, (uint16_t)0x0F0F, 16);
        AscendC::Duplicate(mk[16], (uint16_t)0xF0F0, 16);
    }
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
                if (FMT != FMT_I4F && FMT != FMT_I4SF) {
                    AscendC::Cast<half, int8_t, false>(wh, qbuf[hb].ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, FULL, (uint8_t)(NI * 8), cp8);
                } else if (FMT == FMT_I4SF) { // And 平面解码: 字节 = lo | hi<<4 (nibble 为 q+8); hi 平面天然带 x16, scale 已在 repack 时预除
                    int nby = NI * 512;
                    AscendC::And<uint16_t, false>(th.ReinterpretCast<uint16_t>(), qbuf[hb].ReinterpretCast<uint16_t>(), mk, FULL, (uint8_t)(nby / 256), bmk);
                    AscendC::And<uint16_t, false>(tm.ReinterpretCast<uint16_t>(), qbuf[hb].ReinterpretCast<uint16_t>(), mk[16], FULL, (uint8_t)(nby / 256), bmk);
                    AscendC::Cast<half, uint8_t, false>(wh, th.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, FULL, (uint8_t)(nby / 128), cp8); // lo = q+8
                    AscendC::Cast<half, uint8_t, false>(wh[nby], tm.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, FULL, (uint8_t)(nby / 128), cp8); // 16*(q+8)
                    AscendC::Adds<half, false>(wh, wh, (half)-8.0f, FULL, (uint8_t)(nby / 128), cpa);
                    AscendC::Adds<half, false>(wh[nby], wh[nby], (half)-128.0f, FULL, (uint8_t)(nby / 128), cpa); // 16q
                } else { // I4F (GGUF 时代): nibble 乘法链
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
                if (FMT != FMT_I8F && FMT != FMT_I4SF) { // bias 块区
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
extern "C" __global__ __aicore__ void k_gemm_i4s(GM_ADDR w, GM_ADDR x, GM_ADDR xp, GM_ADDR y, int32_t m, int32_t n, int32_t k) { gemm_body<FMT_I4SF>(w, x, xp, y, m, n, k); }
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

// 因果 depthwise conv1d (核宽 4) + SiLU; qkv 为合并输出 [T][QKVZ], 只处理前 QKV_DIM 通道 (z 段不卷积)
// state GM 布局 [3][QKV_DIM], cw GM 布局 [4][QKV_DIM] (host 已重排)
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
    gq.SetGlobalBuffer((__gm__ float*)qkv, (uint64_t)T * QKVZ);
    gs.SetGlobalBuffer((__gm__ float*)state, 3 * QKV_DIM);
    gw.SetGlobalBuffer((__gm__ float*)cw, 4 * QKV_DIM);
    for (int j = 0; j < 4; j++) AscendC::DataCopy(w4[j * C], gw[j * QKV_DIM + c0], C);
    LocalTensor<float> s0 = bufs, s1 = bufs[C], s2 = bufs[2 * C], xn = bufs[3 * C];
    AscendC::DataCopy(s0, gs[c0], C);
    AscendC::DataCopy(s1, gs[QKV_DIM + c0], C);
    AscendC::DataCopy(s2, gs[2 * QKV_DIM + c0], C);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int t = 0; t < T; t++) {
        AscendC::DataCopy(xn, gq[(uint64_t)t * QKVZ + c0], C);
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
        AscendC::DataCopy(gq[(uint64_t)t * QKVZ + c0], o, C);
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
// state GM: [48][128][128]; qkvz: conv 后的合并输出 [T][QKVZ] (前 QKV_DIM 为 qkv, 后 V_DIM 为 z); ab: [T][96] (前 48 a, 后 48 b); out: [T][V_DIM]
extern "C" __global__ __aicore__ void k_deltanet(GM_ADDR qkvz, GM_ADDR ab, GM_ADDR ssm_a, GM_ADDR dt_bias, GM_ADDR ssm_norm, GM_ADDR state, GM_ADDR out, int32_t T)
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
    GlobalTensor<float> gq, gab, gsa, gdb, gsn, gst, go;
    gq.SetGlobalBuffer((__gm__ float*)qkvz, (uint64_t)T * QKVZ);
    gab.SetGlobalBuffer((__gm__ float*)ab, (uint64_t)T * 96 + 64);
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
            AscendC::DataCopy(qr, gq[(uint64_t)t * QKVZ + kh * LHD], LHD);
            AscendC::DataCopy(kr, gq[(uint64_t)t * QKVZ + KEY_DIM + kh * LHD], LHD);
            AscendC::DataCopy(vr, gq[(uint64_t)t * QKVZ + 2 * KEY_DIM + h * LHD], LHD);
            AscendC::DataCopy(zr, gq[(uint64_t)t * QKVZ + QKV_DIM + h * LHD], LHD);
            AscendC::DataCopy(ar, gab[(uint64_t)t * 96], 64);
            AscendC::DataCopy(br, gab[(uint64_t)t * 96 + 48], 64);
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

// ==================== 全注意力: KV cache q8 (i8 对称 + per-128 fp16 scale) ====================
// cache 布局 (每层): kq/vq [N_KV][mctx][HD] i8; ksc/vsc [N_KV][mctx][2][16] half (每组 scale 复制 16 份, 供 src1BlkStride=0 块乘)
// 反量化 = Cast i8->half + Mul(scale), 全 fp16 (与 CUDA q8_0 版 kv_dq8 同构), 之后喂 fp16 mmad; decode 的 V 连 Mul 都省掉 (scale 折进 P)

// row x [256] f32 原地量化 -> tu i8 (±127 对称) + sh [2][16] half (scale 复制 16 份); 量化用 fp16 舍入后的 scale, RINT 后钳 ±127
__aicore__ inline void quant_row_256(LocalTensor<float>& x, LocalTensor<float>& ab, LocalTensor<float>& red,
    LocalTensor<float>& sf, LocalTensor<float>& sf2, LocalTensor<half>& sh, LocalTensor<int32_t>& ti,
    LocalTensor<half>& th, LocalTensor<int8_t>& tu)
{
    AscendC::Abs(ab, x, 256);
    AscendC::BlockReduceMax<float>(red, ab, 4, 64, 1, 1, 8); // [32] per-8 max
    AscendC::PipeBarrier<PIPE_ALL>();
    float a0 = 1e-30f, a1 = 1e-30f;
    for (int i = 0; i < 16; i++) { float v = red.GetValue(i); a0 = v > a0 ? v : a0; }
    for (int i = 16; i < 32; i++) { float v = red.GetValue(i); a1 = v > a1 ? v : a1; }
    AscendC::Duplicate(sf, a0 / 127.0f, 16);
    AscendC::Duplicate(sf[16], a1 / 127.0f, 16);
    AscendC::Cast(sh, sf, RoundMode::CAST_NONE, 32); // 存储用 fp16 scale
    AscendC::Cast(sf2, sh, RoundMode::CAST_NONE, 32);
    AscendC::PipeBarrier<PIPE_ALL>();
    float s0 = sf2.GetValue(0), s1 = sf2.GetValue(16);
    AscendC::Muls(x, x, s0 > 0 ? 1.0f / s0 : 0.0f, 128); // fp16 scale 下溢成 0 时整组按 0 处理
    AscendC::Muls(x[128], x[128], s1 > 0 ? 1.0f / s1 : 0.0f, 128);
    AscendC::Cast(ti, x, RoundMode::CAST_RINT, 256);
    AscendC::Cast(x, ti, RoundMode::CAST_NONE, 256);
    AscendC::Maxs(x, x, -127.0f, 256);
    AscendC::Mins(x, x, 127.0f, 256);
    AscendC::Cast(th, x, RoundMode::CAST_NONE, 256); // 整数 ±127 精确
    AscendC::Cast(tu, th, RoundMode::CAST_NONE, 256); // half->int8 (±127 范围精确, 实测可用)
    AscendC::PipeBarrier<PIPE_ALL>();
}

// q/k 头 RMSNorm + 交错 MRoPE + K/V 量化写入 q8 cache; cos/sin 行 [T][32] 由 host 按 pos3 预先算好
// qkv 为合并输出 [T][QKVC] (q+gate | k | v); 单元 = (t, 头), 头 0..23 为 q, 24..27 为 k (k 同时量化并写 v)
extern "C" __global__ __aicore__ void k_qk_rope(GM_ADDR qkv, GM_ADDR q_norm, GM_ADDR k_norm, GM_ADDR cosr, GM_ADDR sinr, GM_ADDR kq, GM_ADDR ksc, GM_ADDR vq, GM_ADDR vsc, int32_t n_past, int32_t T, int32_t mctx)
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
    LocalTensor<int32_t> ti = ub[6784].ReinterpretCast<int32_t>(); // 量化 RINT 临时 [256]
    LocalTensor<float> sf = ub[7808].ReinterpretCast<float>(); // scale f32 [32]
    LocalTensor<float> sf2 = ub[7936].ReinterpretCast<float>(); // fp16 往返后 [32]
    LocalTensor<half> sh = ub[8064].ReinterpretCast<half>(); // scale half [2][16]
    LocalTensor<half> th = ub[8192].ReinterpretCast<half>(); // [256]
    LocalTensor<int8_t> tu = ub[8704].ReinterpretCast<int8_t>(); // i8 [256]
    GlobalTensor<float> gqg, gqn, gkn, gcs, gsn;
    GlobalTensor<int8_t> gkq, gvq;
    GlobalTensor<half> gks, gvs;
    gqg.SetGlobalBuffer((__gm__ float*)qkv, (uint64_t)T * QKVC);
    gqn.SetGlobalBuffer((__gm__ float*)q_norm, HD);
    gkn.SetGlobalBuffer((__gm__ float*)k_norm, HD);
    gcs.SetGlobalBuffer((__gm__ float*)cosr, (uint64_t)T * 32);
    gsn.SetGlobalBuffer((__gm__ float*)sinr, (uint64_t)T * 32);
    gkq.SetGlobalBuffer((__gm__ int8_t*)kq, 1); // 偏移大, 用 64 位
    gks.SetGlobalBuffer((__gm__ half*)ksc, 1);
    gvq.SetGlobalBuffer((__gm__ int8_t*)vq, 1);
    gvs.SetGlobalBuffer((__gm__ half*)vsc, 1);
    AscendC::DataCopy(qnw, gqn, HD);
    AscendC::DataCopy(knw, gkn, HD);
    AscendC::PipeBarrier<PIPE_ALL>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int units = T * (N_HEAD + N_KV);
    for (int uidx = core; uidx < units; uidx += cores) {
        int t = uidx / (N_HEAD + N_KV), u = uidx % (N_HEAD + N_KV);
        bool is_q = u < N_HEAD;
        if (is_q) AscendC::DataCopy(pr, gqg[(uint64_t)t * QKVC + u * HD * 2], HD);
        else AscendC::DataCopy(pr, gqg[(uint64_t)t * QKVC + Q_DIM + (u - N_HEAD) * HD], HD);
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
            AscendC::DataCopy(gqg[(uint64_t)t * QKVC + u * HD * 2], yr, HD);
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }
        int kh = u - N_HEAD;
        uint64_t pos = (uint64_t)kh * mctx + (n_past + t); // 该头 cache 平面内的行号
        AscendC::DataCopy(vr, gqg[(uint64_t)t * QKVC + Q_DIM + KV_DIM + kh * HD], HD);
        quant_row_256(yr, tp, red, sf, sf2, sh, ti, th, tu); // K 量化 (yr 原地)
        AscendC::DataCopy(gkq[pos * HD], tu, HD);
        AscendC::DataCopy(gks[pos * 32], sh, 32);
        AscendC::PipeBarrier<PIPE_ALL>();
        quant_row_256(vr, tp, red, sf, sf2, sh, ti, th, tu); // V 量化
        AscendC::DataCopy(gvq[pos * HD], tu, HD);
        AscendC::DataCopy(gvs[pos * 32], sh, 32);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// 因果 GQA 注意力, cube 版: KV cache q8 反量化 fp16 后, Q·K^T 与 P·V 都走 fp16 mmad (fp32 累加)
// scores 按转置方向算 (scoresT[j][m] = K·Q^T, A=K 流式, B=Q^T 常驻 L1), softmax 的行归约变成整列向量操作, 无需行广播
// decode (T=1): 单元 = (kv头, ctx 两段) 共 8 个, 单遍在线 softmax, MTE2 双缓冲流水; 部分结果 (O/m/l) 写 attp, 由 k_attn_merge 合并 + 门控
// prefill (T>1): 单元 = (kv头, 16-query 块), m = 6头x16token = 96 行; 两遍扫描 (K 二读走共享缓存), 输出在 L0C 跨 tile 累加, 门控在 kernel 内
// 尾批/因果屏蔽: scoresT 行置 -3e38 (因果边界行按 16-lane 周期掩码); V 尾行清零防 NaN
extern "C" __global__ __aicore__ void k_attn(GM_ADDR qg, GM_ADDR kq, GM_ADDR ksc, GM_ADDR vq, GM_ADDR vsc, GM_ADDR att, GM_ADDR attp, int32_t n_past, int32_t T, int32_t mctx)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buf;
    AscendC::TBuf<AscendC::TPosition::A1> l1_buf;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buf;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buf;
    constexpr int JT = 48, M1J = JT / 16; // 每 tile key 数
    pipe.InitBuffer(ub_buf, 236 * 1024);
    pipe.InitBuffer(l1_buf, 128 * 1024);
    pipe.InitBuffer(l0a_buf, 64 * 1024);
    pipe.InitBuffer(l0b_buf, 64 * 1024);
    pipe.InitBuffer(l0c_buf, 160 * 1024);
    LocalTensor<uint8_t> ub = ub_buf.Get<uint8_t>();
    LocalTensor<uint8_t> l1 = l1_buf.Get<uint8_t>();
    LocalTensor<half> l0a = l0a_buf.Get<half>();
    LocalTensor<half> l0b = l0b_buf.Get<half>();
    LocalTensor<float> l0c = l0c_buf.Get<float>();
    // L1: Q^T 常驻 [16ki][M1 或 N1] | l1_k | l1_v | l1_p
    LocalTensor<half> l1_q = l1.ReinterpretCast<half>();
    LocalTensor<half> l1_k = l1[49152].ReinterpretCast<half>();
    LocalTensor<half> l1_v = l1[73728].ReinterpretCast<half>();
    LocalTensor<half> l1_p = l1[98304].ReinterpretCast<half>();
    LocalTensor<half> l0a_k = l0a; // scores 的 A (K tile), 24KB
    LocalTensor<half> l0a_p = l0a[16384]; // out 的 A (P), 9KB
    LocalTensor<float> l0c_s = l0c; // scores C, 最大 18KB
    LocalTensor<float> l0c_o = l0c[8192]; // out C, 最大 96KB (f32 下标)
    GlobalTensor<float> gqg, gat, gop, gml;
    GlobalTensor<int8_t> gkq, gvq;
    GlobalTensor<half> gks, gvs;
    gqg.SetGlobalBuffer((__gm__ float*)qg, (uint64_t)T * QKVC);
    gat.SetGlobalBuffer((__gm__ float*)att, (uint64_t)T * N_HEAD * HD);
    gkq.SetGlobalBuffer((__gm__ int8_t*)kq, 1);
    gks.SetGlobalBuffer((__gm__ half*)ksc, 1);
    gvq.SetGlobalBuffer((__gm__ int8_t*)vq, 1);
    gvs.SetGlobalBuffer((__gm__ half*)vsc, 1);
    if (attp) {
        gop.SetGlobalBuffer((__gm__ float*)attp, 8 * 16 * HD);
        gml.SetGlobalBuffer((__gm__ float*)attp + 8 * 16 * HD, 256);
    }
    const float SCALE = 0.0625f; // 1/sqrt(256)
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::UnaryRepeatParams cp8(1, 1, 8, 4);
    AscendC::BinaryRepeatParams bdq(1, 1, 0, 8, 8, 1); // 反量化乘 scale: src1 块复用, 每 repeat (半行) 前进一块
    AscendC::DataCopyEnhancedParams epm;
    epm.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    if (T == 1) { // ==================== decode ====================
        constexpr int JTD = 96, M1D = JTD / 16; // decode tile 加大到 96, 摊薄每 tile 固定开销
        // UB 布局 (字节); qf 与 co 共用区 (qf 只在 tile 循环前用)
        LocalTensor<float> qf = ub.ReinterpretCast<float>(); // [16][256] f32 (Q 装载临时)
        LocalTensor<float> co = ub.ReinterpretCast<float>(); // [16][256] (scores/out 共用 drain 区)
        LocalTensor<half> qh = ub[16384].ReinterpretCast<half>(); // [16][256]
        LocalTensor<int8_t> kqs = ub[24576].ReinterpretCast<int8_t>(); // 单缓冲 staging (反量化后立即释放, DMA 与后续链重叠)
        LocalTensor<int8_t> vqs = ub[49152].ReinterpretCast<int8_t>();
        LocalTensor<half> kss = ub[73728].ReinterpretCast<half>();
        LocalTensor<half> vss = ub[79872].ReinterpretCast<half>();
        LocalTensor<half> Kh = ub[86016].ReinterpretCast<half>(); // [96][256]
        LocalTensor<half> Vh = ub[135168].ReinterpretCast<half>();
        LocalTensor<float> sT = ub[184320].ReinterpretCast<float>(); // [96][16] 紧凑布局
        LocalTensor<float> tmp = ub[190464].ReinterpretCast<float>(); // [48][16]
        LocalTensor<half> ph = ub[193536].ReinterpretCast<half>(); // [96][16]
        LocalTensor<float> Oacc = ub[196608].ReinterpretCast<float>(); // [16][256]
        LocalTensor<float> runm = ub[212992].ReinterpretCast<float>(); // 以下各 [16]
        LocalTensor<float> tmax = ub[213056].ReinterpretCast<float>();
        LocalTensor<float> corr = ub[213120].ReinterpretCast<float>();
        LocalTensor<float> lrun = ub[213184].ReinterpretCast<float>();
        LocalTensor<float> tmax4 = ub[213248].ReinterpretCast<float>(); // [64] = tmax 复制 4 份 (Sub 广播用)
        LocalTensor<float> corr4 = ub[213504].ReinterpretCast<float>(); // [64] = corr 复制 4 份 (OaccT 重缩放用)
        LocalTensor<half> ph0 = ub[214016].ReinterpretCast<half>(); // P' = P ⊙ sv_g, 两个 d-half 各一份 [96][16]
        LocalTensor<half> ph1 = ub[217088].ReinterpretCast<half>();
        LocalTensor<half> vsk = ub[220160].ReinterpretCast<half>(); // vss 常驻副本 [96][2][16] (staging 可早释放)
        // L1 (decode 专用偏移): Q^T 8KB | K 48KB | V 48KB | P'x2 各 3KB
        LocalTensor<half> l1_kd = l1[8192].ReinterpretCast<half>();
        LocalTensor<half> l1_vd = l1[57344].ReinterpretCast<half>();
        LocalTensor<half> l1_pd = l1[106496].ReinterpretCast<half>();
        LocalTensor<half> l1_pd2 = l1[109568].ReinterpretCast<half>();
        int nmax = n_past + 1;
        int s0 = (nmax / 2 + JTD - 1) / JTD * JTD;
        if (s0 > nmax) s0 = nmax;
        int kvh = core / 2, seg = core & 1;
        int j0s = seg ? s0 : 0, j1s = seg ? nmax : s0;
        // Q 准备: [6][256] -> *SCALE -> fp16 [16][256] (pad 行清零) -> L1 fractal
        AscendC::DataCopy(qf, gqg[(uint64_t)(kvh * 6) * HD * 2], AscendC::DataCopyParams(6, 32, 32, 0));
        AscendC::Duplicate(qh, (half)0, 4096);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(qf, qf, SCALE, 6 * HD);
        AscendC::Cast(qh, qf, RoundMode::CAST_NONE, 6 * HD);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int ki = 0; ki < 16; ki++)
            AscendC::DataCopy(l1_q[ki * 256], qh[ki * 16], AscendC::DataCopyParams(16, 1, 15, 0));
        AscendC::Duplicate(Oacc, 0.0f, 4096);
        AscendC::Duplicate(runm, -3e38f, 16);
        AscendC::Duplicate(lrun, 0.0f, 16);
        AscendC::PipeBarrier<PIPE_ALL>();
        // 事件
        AscendC::TEventID eq = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        AscendC::TEventID eqf = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        AscendC::SetFlag<HardEvent::V_MTE2>(eqf);
        AscendC::TEventID ekh = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        AscendC::TEventID ekhf = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        AscendC::TEventID evh = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        AscendC::TEventID evhf = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        AscendC::TEventID eph = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        AscendC::TEventID ephf = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        AscendC::TEventID ebk = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE1>();
        AscendC::TEventID ebkf = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE3>();
        AscendC::TEventID ebv = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE1>();
        AscendC::TEventID ebvf = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE3>();
        AscendC::TEventID ebp = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE1>();
        AscendC::TEventID ebpf = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE3>();
        AscendC::TEventID em = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
        AscendC::TEventID emf = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
        AscendC::TEventID ecs = GetTPipePtr()->AllocEventID<HardEvent::M_V>();
        AscendC::TEventID ecsf = GetTPipePtr()->AllocEventID<HardEvent::V_M>();
        AscendC::TEventID eco = GetTPipePtr()->AllocEventID<HardEvent::M_V>();
        AscendC::TEventID ecof = GetTPipePtr()->AllocEventID<HardEvent::V_M>();
        AscendC::SetFlag<HardEvent::MTE3_V>(ekhf);
        AscendC::SetFlag<HardEvent::MTE3_V>(evhf);
        AscendC::SetFlag<HardEvent::MTE3_V>(ephf);
        AscendC::SetFlag<HardEvent::MTE1_MTE3>(ebkf);
        AscendC::SetFlag<HardEvent::MTE1_MTE3>(ebvf);
        AscendC::SetFlag<HardEvent::MTE1_MTE3>(ebpf);
        AscendC::SetFlag<HardEvent::M_MTE1>(emf);
        AscendC::SetFlag<HardEvent::V_M>(ecsf);
        AscendC::SetFlag<HardEvent::V_M>(ecof);
        for (int j0 = j0s; j0 < j1s; j0 += JTD) {
            int jvl = j1s - j0 < JTD ? j1s - j0 : JTD;
            AscendC::WaitFlag<HardEvent::V_MTE2>(eqf);
            AscendC::DataCopy(kqs, gkq[((uint64_t)kvh * mctx + j0) * HD], JTD * HD);
            AscendC::DataCopy(kss, gks[((uint64_t)kvh * mctx + j0) * 32], JTD * 32);
            AscendC::DataCopy(vqs, gvq[((uint64_t)kvh * mctx + j0) * HD], JTD * HD);
            AscendC::DataCopy(vss, gvs[((uint64_t)kvh * mctx + j0) * 32], JTD * 32);
            AscendC::SetFlag<HardEvent::MTE2_V>(eq);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eq);
            // K 反量化 fp16 (i8 直接 Cast, 无零点)
            AscendC::WaitFlag<HardEvent::MTE3_V>(ekhf);
            AscendC::Cast<half, int8_t, false>(Kh, kqs, RoundMode::CAST_NONE, FULL, 192, cp8);
            AscendC::Mul<half, false>(Kh, Kh, kss, FULL, 192, bdq);
            // V 只 Cast 成整数 half (scale 折进 P'), 尾行清零
            AscendC::WaitFlag<HardEvent::MTE3_V>(evhf);
            AscendC::Cast<half, int8_t, false>(Vh, vqs, RoundMode::CAST_NONE, FULL, 192, cp8);
            if (jvl < JTD) AscendC::Duplicate(Vh[jvl * HD], (half)0, (JTD - jvl) * HD);
            AscendC::Adds(vsk, vss, (half)0, JTD * 32); // vss 拷到常驻副本 (P' 折叠在 softmax 后才用)
            AscendC::SetFlag<HardEvent::V_MTE2>(eqf); // staging 释放, 下一 tile 的 DMA 与余下链重叠
            // Kh/Vh -> L1 列片
            AscendC::SetFlag<HardEvent::V_MTE3>(ekh);
            AscendC::WaitFlag<HardEvent::V_MTE3>(ekh);
            AscendC::WaitFlag<HardEvent::MTE1_MTE3>(ebkf);
            for (int ki = 0; ki < 16; ki++)
                AscendC::DataCopy(l1_kd[ki * M1D * 256], Kh[ki * 16], AscendC::DataCopyParams(JTD, 1, 15, 0));
            AscendC::SetFlag<HardEvent::MTE3_V>(ekhf);
            AscendC::SetFlag<HardEvent::MTE3_MTE1>(ebk);
            AscendC::SetFlag<HardEvent::V_MTE3>(evh);
            AscendC::WaitFlag<HardEvent::V_MTE3>(evh);
            AscendC::WaitFlag<HardEvent::MTE1_MTE3>(ebvf);
            for (int db = 0; db < 16; db++)
                AscendC::DataCopy(l1_vd[db * M1D * 256], Vh[db * 16], AscendC::DataCopyParams(JTD, 1, 15, 0));
            AscendC::SetFlag<HardEvent::MTE3_V>(evhf);
            AscendC::SetFlag<HardEvent::MTE3_MTE1>(ebv);
            // L0 装载 + scores mmad: scoresT[j][m] = K·Q^T
            AscendC::WaitFlag<HardEvent::MTE3_MTE1>(ebk);
            AscendC::WaitFlag<HardEvent::M_MTE1>(emf);
            AscendC::LoadData(l0b, l1_q, AscendC::LoadData2DParams(0, 16, 1, 0, 0, false, 0));
            for (int jb = 0; jb < M1D; jb++)
                AscendC::LoadData(l0a_k[jb * 16 * 256], l1_kd[jb * 256], AscendC::LoadData2DParams(0, 16, M1D, 0, 0, false, 0));
            AscendC::SetFlag<HardEvent::MTE1_MTE3>(ebkf);
            AscendC::SetFlag<HardEvent::MTE1_M>(em);
            AscendC::WaitFlag<HardEvent::MTE1_M>(em);
            AscendC::WaitFlag<HardEvent::V_M>(ecsf);
            AscendC::Mmad(l0c_s, l0a_k, l0b, AscendC::MmadParams(JTD, 16, 256, 0, false, true));
            AscendC::SetFlag<HardEvent::M_MTE1>(emf);
            AscendC::SetFlag<HardEvent::M_V>(ecs);
            // drain scores + 在线 softmax (列向, sT [96][16] 紧凑)
            AscendC::WaitFlag<HardEvent::M_V>(ecs);
            AscendC::DataCopy(co, l0c_s, AscendC::DataCopyParams(1, M1D, 0, 0), epm);
            AscendC::SetFlag<HardEvent::V_M>(ecsf);
            AscendC::SetVectorMask<float>(0, 0xFFFFULL);
            for (int jb = 0; jb < M1D; jb++) // 紧凑化 [16j][16m] -> sT [96][16]
                AscendC::Adds<float, false>(sT[jb * 16 * 16], co[jb * 256], 0.0f, FULL, 16, AscendC::UnaryRepeatParams(1, 1, 2, 2));
            AscendC::SetVectorMask<half>(FULL, FULL);
            if (jvl < JTD) AscendC::Duplicate(sT[jvl * 16], -3e38f, (JTD - jvl) * 16); // 尾行屏蔽
            AscendC::Max(tmp, sT, sT[48 * 16], 48 * 16); // 行折叠树求列 max
            AscendC::Max(tmp, tmp, tmp[24 * 16], 24 * 16);
            AscendC::Max(tmp, tmp, tmp[12 * 16], 12 * 16);
            AscendC::Max(tmp, tmp, tmp[6 * 16], 6 * 16);
            AscendC::Max(tmp, tmp, tmp[3 * 16], 3 * 16);
            AscendC::Max(tmp, tmp, tmp[16], 16);
            AscendC::Max(tmax, tmp, tmp[32], 16);
            AscendC::Max(tmax, tmax, runm, 16); // newm
            AscendC::Sub(corr, runm, tmax, 16);
            AscendC::Exp(corr, corr, 16);
            AscendC::Adds(runm, tmax, 0.0f, 16);
            AscendC::Adds(tmax4, tmax, 0.0f, 16); // tmax 复制 4 份供整 repeat 广播
            AscendC::Adds(tmax4[16], tmax, 0.0f, 16);
            AscendC::Adds(tmax4[32], tmax, 0.0f, 16);
            AscendC::Adds(tmax4[48], tmax, 0.0f, 16);
            AscendC::Sub<float, false>(sT, sT, tmax4, FULL, JTD * 16 / 64, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
            AscendC::Exp(sT, sT, JTD * 16);
            AscendC::Add(tmp, sT, sT[48 * 16], 48 * 16); // 行折叠树求列 sum
            AscendC::Add(tmp, tmp, tmp[24 * 16], 24 * 16);
            AscendC::Add(tmp, tmp, tmp[12 * 16], 12 * 16);
            AscendC::Add(tmp, tmp, tmp[6 * 16], 6 * 16);
            AscendC::Add(tmp, tmp, tmp[3 * 16], 3 * 16);
            AscendC::Add(tmp, tmp, tmp[16], 16);
            AscendC::Add(tmp, tmp, tmp[32], 16); // tsum 在 tmp[0..15]
            AscendC::Mul(lrun, lrun, corr, 16);
            AscendC::Add(lrun, lrun, tmp, 16);
            AscendC::Adds(corr4, corr, 0.0f, 16); // corr 复制 4 份: OaccT 的 m 在 lane 维, 整 repeat 广播
            AscendC::Adds(corr4[16], corr, 0.0f, 16);
            AscendC::Adds(corr4[32], corr, 0.0f, 16);
            AscendC::Adds(corr4[48], corr, 0.0f, 16);
            AscendC::Mul<float, false>(Oacc, Oacc, corr4, FULL, 64, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0)); // OaccT *= corr
            // P^T -> fp16 -> P' = P ⊙ sv_g (V scale 折叠, 每 d-half 一份) -> L1
            AscendC::WaitFlag<HardEvent::MTE3_V>(ephf);
            AscendC::Cast(ph, sT, RoundMode::CAST_NONE, JTD * 16); // 连续, 一条 L2 Cast
            AscendC::Mul<half, false>(ph0, ph, vsk, FULL, 12, AscendC::BinaryRepeatParams(1, 1, 2, 8, 8, 16)); // src1 [j][2][16] 隔块取 g=0
            AscendC::Mul<half, false>(ph1, ph, vsk[16], FULL, 12, AscendC::BinaryRepeatParams(1, 1, 2, 8, 8, 16));
            AscendC::SetFlag<HardEvent::V_MTE3>(eph);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eph);
            AscendC::WaitFlag<HardEvent::MTE1_MTE3>(ebpf);
            AscendC::DataCopy(l1_pd, ph0, JTD * 16);
            AscendC::DataCopy(l1_pd2, ph1, JTD * 16);
            AscendC::SetFlag<HardEvent::MTE3_V>(ephf);
            AscendC::SetFlag<HardEvent::MTE3_MTE1>(ebp);
            AscendC::WaitFlag<HardEvent::MTE3_MTE1>(ebp);
            AscendC::WaitFlag<HardEvent::MTE3_MTE1>(ebv);
            // out 双 mmad (outT[d][m] = V^T·P'_g): A = V 整数片转置加载 (复用 scores A 区), B = P'_g
            AscendC::WaitFlag<HardEvent::M_MTE1>(emf);
            AscendC::LoadData(l0a_k, l1_vd, AscendC::LoadData2DParams(0, 8 * M1D, 1, 0, 0, true, 0));
            AscendC::LoadData(l0b, l1_pd, AscendC::LoadData2DParams(0, M1D, 1, 0, 0, true, 0));
            AscendC::SetFlag<HardEvent::MTE1_M>(em);
            AscendC::WaitFlag<HardEvent::MTE1_M>(em);
            AscendC::WaitFlag<HardEvent::V_M>(ecof);
            AscendC::Mmad(l0c_o, l0a_k, l0b, AscendC::MmadParams(128, 16, JTD, 0, false, true));
            AscendC::SetFlag<HardEvent::M_MTE1>(emf);
            AscendC::WaitFlag<HardEvent::M_MTE1>(emf);
            AscendC::LoadData(l0a_k, l1_vd[8 * M1D * 256], AscendC::LoadData2DParams(0, 8 * M1D, 1, 0, 0, true, 0));
            AscendC::LoadData(l0b, l1_pd2, AscendC::LoadData2DParams(0, M1D, 1, 0, 0, true, 0));
            AscendC::SetFlag<HardEvent::MTE1_MTE3>(ebpf);
            AscendC::SetFlag<HardEvent::MTE1_MTE3>(ebvf);
            AscendC::SetFlag<HardEvent::MTE1_M>(em);
            AscendC::WaitFlag<HardEvent::MTE1_M>(em);
            AscendC::Mmad(l0c_o[2048], l0a_k, l0b, AscendC::MmadParams(128, 16, JTD, 0, false, true));
            AscendC::SetFlag<HardEvent::M_MTE1>(emf);
            AscendC::SetFlag<HardEvent::M_V>(eco);
            // drain out (两半共 16 fractal 连续) + 一条 Add 累加
            AscendC::WaitFlag<HardEvent::M_V>(eco);
            AscendC::DataCopy(co, l0c_o, AscendC::DataCopyParams(1, 16, 0, 0), epm);
            AscendC::SetFlag<HardEvent::V_M>(ecof);
            AscendC::Add(Oacc, Oacc, co, 4096);
        }
        AscendC::WaitFlag<HardEvent::V_MTE2>(eqf);
        AscendC::WaitFlag<HardEvent::MTE3_V>(ekhf);
        AscendC::WaitFlag<HardEvent::MTE3_V>(evhf);
        AscendC::WaitFlag<HardEvent::MTE3_V>(ephf);
        AscendC::WaitFlag<HardEvent::MTE1_MTE3>(ebkf);
        AscendC::WaitFlag<HardEvent::MTE1_MTE3>(ebvf);
        AscendC::WaitFlag<HardEvent::MTE1_MTE3>(ebpf);
        AscendC::WaitFlag<HardEvent::M_MTE1>(emf);
        AscendC::WaitFlag<HardEvent::V_M>(ecsf);
        AscendC::WaitFlag<HardEvent::V_M>(ecof);
        AscendC::PipeBarrier<PIPE_ALL>();
        int slot = kvh * 2 + seg; // 部分结果: O [16][256] + m/l 各 [16]
        AscendC::DataCopy(gop[(uint64_t)slot * 4096], Oacc, 4096);
        AscendC::DataCopy(gml[slot * 16], runm, 16);
        AscendC::DataCopy(gml[128 + slot * 16], lrun, 16);
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }
    // ==================== prefill (T>1): 两遍扫描, 保守 barrier ====================
    LocalTensor<float> qf = ub.ReinterpretCast<float>(); // [16][256] (Q/gate 装载临时)
    LocalTensor<half> qh = ub[16384].ReinterpretCast<half>(); // [96][256]
    int so0 = 65536; // 单 slot staging: kq 12288 | vq 12288 | ks 3072 | vs 3072
    LocalTensor<int8_t> kqs = ub[so0].ReinterpretCast<int8_t>();
    LocalTensor<int8_t> vqs = ub[so0 + 12288].ReinterpretCast<int8_t>();
    LocalTensor<half> kss = ub[so0 + 24576].ReinterpretCast<half>();
    LocalTensor<half> vss = ub[so0 + 27648].ReinterpretCast<half>();
    LocalTensor<half> Kh = ub[96256].ReinterpretCast<half>(); // [48][256]
    LocalTensor<half> Vh = ub[120832].ReinterpretCast<half>();
    LocalTensor<float> sT = ub[145408].ReinterpretCast<float>(); // [48][128]
    LocalTensor<float> tmp = ub[169984].ReinterpretCast<float>(); // [24][128]
    LocalTensor<half> ph = ub[182272].ReinterpretCast<half>(); // [48][128]
    LocalTensor<float> co = ub[194560].ReinterpretCast<float>(); // scores drain 18KB / finalize co_h 16KB
    LocalTensor<float> ob = ub[212992].ReinterpretCast<float>(); // finalize [16][256]
    LocalTensor<float> runm = ub[229376].ReinterpretCast<float>(); // [128]
    LocalTensor<float> tmax = ub[229888].ReinterpretCast<float>();
    LocalTensor<float> lrun = ub[230400].ReinterpretCast<float>();
    LocalTensor<float> linvd = ub[230912].ReinterpretCast<float>(); // [96][8]
    int ntq = (T + 15) / 16;
    for (int uidx = core; uidx < N_KV * ntq; uidx += cores) {
        int kvh = uidx / ntq, ti = uidx % ntq, h0 = kvh * 6;
        int tq0 = ti * 16, tqn = T - tq0 < 16 ? T - tq0 : 16;
        int nmax = n_past + tq0 + tqn;
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate(qh, (half)0, 96 * HD);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int h = 0; h < 6; h++) { // Q [96][256] = 6头 x 16token, *SCALE, fp16
            AscendC::DataCopy(qf, gqg[(uint64_t)tq0 * QKVC + (h0 + h) * HD * 2], AscendC::DataCopyParams((uint16_t)tqn, 32, (QKVC - HD) * 4 / 32, 0));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(qf, qf, SCALE, tqn * HD);
            AscendC::Cast(qh[h * 16 * 256], qf, RoundMode::CAST_NONE, tqn * HD);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        for (int ki = 0; ki < 16; ki++)
            AscendC::DataCopy(l1_q[ki * 6 * 256], qh[ki * 16], AscendC::DataCopyParams(96, 1, 15, 0));
        AscendC::Duplicate(runm, -3e38f, 128);
        AscendC::Duplicate(lrun, 0.0f, 128);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int pass = 0; pass < 2; pass++) {
            for (int j0 = 0; j0 < nmax; j0 += JT) {
                int jvl = nmax - j0 < JT ? nmax - j0 : JT;
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(kqs, gkq[((uint64_t)kvh * mctx + j0) * HD], JT * HD);
                AscendC::DataCopy(kss, gks[((uint64_t)kvh * mctx + j0) * 32], JT * 32);
                if (pass) {
                    AscendC::DataCopy(vqs, gvq[((uint64_t)kvh * mctx + j0) * HD], JT * HD);
                    AscendC::DataCopy(vss, gvs[((uint64_t)kvh * mctx + j0) * 32], JT * 32);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast<half, int8_t, false>(Kh, kqs, RoundMode::CAST_NONE, FULL, 96, cp8);
                AscendC::Mul<half, false>(Kh, Kh, kss, FULL, 96, bdq);
                if (pass) {
                    AscendC::Cast<half, int8_t, false>(Vh, vqs, RoundMode::CAST_NONE, FULL, 96, cp8);
                    AscendC::Mul<half, false>(Vh, Vh, vss, FULL, 96, bdq);
                    if (jvl < JT) AscendC::Duplicate(Vh[jvl * HD], (half)0, (JT - jvl) * HD);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int ki = 0; ki < 16; ki++)
                    AscendC::DataCopy(l1_k[ki * M1J * 256], Kh[ki * 16], AscendC::DataCopyParams(JT, 1, 15, 0));
                if (pass)
                    for (int db = 0; db < 16; db++)
                        AscendC::DataCopy(l1_v[db * M1J * 256], Vh[db * 16], AscendC::DataCopyParams(JT, 1, 15, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::LoadData(l0b, l1_q, AscendC::LoadData2DParams(0, 96, 1, 0, 0, false, 0));
                for (int jb = 0; jb < M1J; jb++)
                    AscendC::LoadData(l0a_k[jb * 16 * 256], l1_k[jb * 256], AscendC::LoadData2DParams(0, 16, M1J, 0, 0, false, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Mmad(l0c_s, l0a_k, l0b, AscendC::MmadParams(JT, 96, 256, 0, false, true));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(co, l0c_s, AscendC::DataCopyParams(1, 6 * M1J, 0, 0), epm);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetVectorMask<float>(0, 0xFFFFULL);
                for (int nb = 0; nb < 6; nb++) // 紧凑化 -> sT [48][128]
                    for (int jb = 0; jb < M1J; jb++)
                        AscendC::Adds<float, false>(sT[jb * 16 * 128 + nb * 16], co[(nb * M1J + jb) * 256], 0.0f, FULL, 16, AscendC::UnaryRepeatParams(1, 1, 16, 2));
                AscendC::SetVectorMask<half>(FULL, FULL);
                if (jvl < JT) AscendC::Duplicate(sT[jvl * 128], -3e38f, (JT - jvl) * 128); // 尾行
                for (int r = 0; r < jvl; r++) { // 因果边界行: token 列 tq < thresh 屏蔽 (16-lane 周期掩码)
                    int thresh = j0 + r - n_past - tq0;
                    if (thresh <= 0) continue;
                    uint64_t m64 = ((1ULL << thresh) - 1) * 0x0001000100010001ULL;
                    AscendC::SetVectorMask<float>(0, m64);
                    AscendC::Duplicate<float, false>(sT[r * 128], -3e38f, FULL, 2, 1, 8);
                }
                AscendC::SetVectorMask<half>(FULL, FULL);
                AscendC::Max(tmp, sT, sT[24 * 128], 24 * 128); // 列 max
                AscendC::Max(tmp, tmp, tmp[12 * 128], 12 * 128);
                AscendC::Max(tmp, tmp, tmp[6 * 128], 6 * 128);
                AscendC::Max(tmp, tmp, tmp[3 * 128], 3 * 128);
                AscendC::Max(tmp, tmp, tmp[128], 128);
                AscendC::Max(tmax, tmp, tmp[2 * 128], 128);
                if (!pass) { // 第一遍: 只更新全局 max
                    AscendC::Max(runm, runm, tmax, 128);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    continue;
                }
                AscendC::Sub<float, false>(sT, sT, runm, FULL, JT, AscendC::BinaryRepeatParams(1, 1, 1, 16, 16, 0)); // 减全局 max (行 = 2 repeat)
                AscendC::Sub<float, false>(sT[64], sT[64], runm[64], FULL, JT, AscendC::BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                AscendC::Exp(sT, sT, JT * 128);
                AscendC::Add(tmp, sT, sT[24 * 128], 24 * 128); // 列 sum
                AscendC::Add(tmp, tmp, tmp[12 * 128], 12 * 128);
                AscendC::Add(tmp, tmp, tmp[6 * 128], 6 * 128);
                AscendC::Add(tmp, tmp, tmp[3 * 128], 3 * 128);
                AscendC::Add(tmp, tmp, tmp[128], 128);
                AscendC::Add(tmp, tmp, tmp[2 * 128], 128);
                AscendC::Add(lrun, lrun, tmp, 128);
                AscendC::Cast<half, float, false>(ph, sT, RoundMode::CAST_NONE, FULL, 96, AscendC::UnaryRepeatParams(1, 1, 4, 8));
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int mb = 0; mb < 6; mb++) // P^T -> L1 [M1=6][K1=3]
                    AscendC::DataCopy(l1_p[mb * M1J * 256], ph[mb * 16], AscendC::DataCopyParams(JT, 1, 7, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::LoadData(l0a_p, l1_p, AscendC::LoadData2DParams(0, 6 * M1J, 1, 0, 0, true, 0));
                for (int db = 0; db < 16; db++)
                    AscendC::LoadData(l0b[db * 256], l1_v[db * M1J * 256], AscendC::LoadData2DParams(0, M1J, 1, 0, 15, true, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Mmad(l0c_o, l0a_p, l0b, AscendC::MmadParams(96, HD, JT, 0, false, j0 == 0));
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        // finalize: 1/l 广播一次, 逐头 drain + /l + 门控 + 写出
        AscendC::Duplicate(tmax, 1.0f, 128);
        AscendC::Div(tmax, tmax, lrun, 128);
        AscendC::Brcb(linvd, tmax, 12, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int mb = 0; mb < 6; mb++) {
            for (int nb = 0; nb < 16; nb++)
                AscendC::DataCopy(co[nb * 256], l0c_o[(nb * 6 + mb) * 256], AscendC::DataCopyParams(1, 1, 0, 0), epm);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetVectorMask<float>(0, 0xFFFFULL);
            for (int nb = 0; nb < 16; nb++) // 紧凑化 -> ob [16tq][256]
                AscendC::Adds<float, false>(ob[nb * 16], co[nb * 256], 0.0f, FULL, 16, AscendC::UnaryRepeatParams(1, 1, 32, 2));
            AscendC::SetVectorMask<half>(FULL, FULL);
            for (int q4 = 0; q4 < 4; q4++) // ob /= l (行标量)
                AscendC::Mul<float, false>(ob[q4 * 64], ob[q4 * 64], linvd[mb * 16 * 8], FULL, 16, AscendC::BinaryRepeatParams(1, 1, 0, 32, 32, 1));
            AscendC::DataCopy(qf, gqg[(uint64_t)tq0 * QKVC + (h0 + mb) * HD * 2 + HD], AscendC::DataCopyParams((uint16_t)tqn, 32, (QKVC - HD) * 4 / 32, 0)); // gate
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(qf, qf, -1.0f, tqn * HD); // sigmoid = 1/(1+e^-x)
            AscendC::Exp(qf, qf, tqn * HD);
            AscendC::Adds(qf, qf, 1.0f, tqn * HD);
            AscendC::Div(ob, ob, qf, tqn * HD);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(gat[(uint64_t)tq0 * N_HEAD * HD + (h0 + mb) * HD], ob, AscendC::DataCopyParams((uint16_t)tqn, 32, 0, (N_HEAD * HD - HD) * 4 / 32));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// decode 两段部分结果合并 (OaccT [16d片][16d][16m] 布局, m 在 lane 维, 加权全用整 repeat 广播) + 输出门控
// 最终 [d][m] -> [m][d] 的转置用单位阵 mmad 在 cube 上做 (fp16 一次舍入); 单元 = kv 头, block_dim 4
extern "C" __global__ __aicore__ void k_attn_merge(GM_ADDR qg, GM_ADDR attp, GM_ADDR att)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    AscendC::TBuf<AscendC::TPosition::A1> l1_buf;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buf;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buf;
    pipe.InitBuffer(tb, 80 * 1024);
    pipe.InitBuffer(l1_buf, 16 * 1024);
    pipe.InitBuffer(l0a_buf, 1024);
    pipe.InitBuffer(l0b_buf, 8192);
    pipe.InitBuffer(l0c_buf, 16384);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    LocalTensor<uint8_t> l1 = l1_buf.Get<uint8_t>();
    LocalTensor<half> l0a = l0a_buf.Get<half>();
    LocalTensor<half> l0b = l0b_buf.Get<half>();
    LocalTensor<float> l0c = l0c_buf.Get<float>();
    LocalTensor<float> O0 = ub.ReinterpretCast<float>(); // OaccT [16][16d][16m]
    LocalTensor<float> O1 = ub[16384].ReinterpretCast<float>();
    LocalTensor<float> co = ub[32768].ReinterpretCast<float>(); // 转置后 drain [16][16m][16d]
    LocalTensor<float> On = ub[49152].ReinterpretCast<float>(); // 紧凑 [16m][256]
    LocalTensor<float> gb = ub[65536].ReinterpretCast<float>(); // gate [6][256]
    LocalTensor<half> Oh = ub[72320].ReinterpretCast<half>(); // fp16 [4096]
    LocalTensor<float> m0 = ub[80512].ReinterpretCast<float>(); // 以下各 [16]
    LocalTensor<float> m1 = ub[80576].ReinterpretCast<float>();
    LocalTensor<float> l0 = ub[80640].ReinterpretCast<float>();
    LocalTensor<float> l1s = ub[80704].ReinterpretCast<float>();
    LocalTensor<float> mm = ub[80768].ReinterpretCast<float>();
    LocalTensor<float> c0 = ub[80832].ReinterpretCast<float>();
    LocalTensor<float> c1 = ub[80896].ReinterpretCast<float>();
    LocalTensor<float> lt = ub[80960].ReinterpretCast<float>();
    LocalTensor<float> w4 = ub[81024].ReinterpretCast<float>(); // [64] 加权/1 除 l 的 4 份复制
    LocalTensor<half> ih = ub[81280].ReinterpretCast<half>(); // 单位阵 [16][16]
    int kvh = (int)AscendC::GetBlockIdx();
    if (kvh >= N_KV) return;
    GlobalTensor<float> gqg, gop, gml, gat;
    gqg.SetGlobalBuffer((__gm__ float*)qg, QKVC);
    gop.SetGlobalBuffer((__gm__ float*)attp, 8 * 16 * HD);
    gml.SetGlobalBuffer((__gm__ float*)attp + 8 * 16 * HD, 256);
    gat.SetGlobalBuffer((__gm__ float*)att, N_HEAD * HD);
    AscendC::SetVectorMask<half>(FULL, FULL);
    int s0 = kvh * 2, s1 = s0 + 1;
    AscendC::DataCopy(O0, gop[(uint64_t)s0 * 4096], 4096);
    AscendC::DataCopy(O1, gop[(uint64_t)s1 * 4096], 4096);
    AscendC::DataCopy(m0, gml[s0 * 16], 16);
    AscendC::DataCopy(m1, gml[s1 * 16], 16);
    AscendC::DataCopy(l0, gml[128 + s0 * 16], 16);
    AscendC::DataCopy(l1s, gml[128 + s1 * 16], 16);
    AscendC::DataCopy(gb, gqg[(uint64_t)(kvh * 6) * HD * 2 + HD], AscendC::DataCopyParams(6, 32, 32, 0));
    AscendC::Duplicate(ih, (half)0, 256);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int m = 0; m < 16; m++) ih.SetValue(m * 16 + m, (half)1.0f);
    AscendC::Max(mm, m0, m1, 16);
    AscendC::Sub(c0, m0, mm, 16);
    AscendC::Exp(c0, c0, 16);
    AscendC::Sub(c1, m1, mm, 16);
    AscendC::Exp(c1, c1, 16);
    AscendC::Mul(lt, c0, l0, 16);
    AscendC::MulAddDst(lt, c1, l1s, 16);
    AscendC::Duplicate(w4, 1.0f, 16);
    AscendC::Div(lt, w4, lt, 16); // 1/l
    AscendC::Mul(c0, c0, lt, 16); // 归一权重
    AscendC::Mul(c1, c1, lt, 16);
    for (int r = 0; r < 4; r++) AscendC::Adds(w4[r * 16], c0, 0.0f, 16); // m 在 lane 维, 复制 4 份整 repeat 广播
    AscendC::Mul<float, false>(O0, O0, w4, FULL, 64, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
    for (int r = 0; r < 4; r++) AscendC::Adds(w4[r * 16], c1, 0.0f, 16);
    AscendC::Mul<float, false>(O1, O1, w4, FULL, 64, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
    AscendC::Add(O0, O0, O1, 4096);
    AscendC::Cast(Oh, O0, RoundMode::CAST_NONE, 4096);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(l1.ReinterpretCast<half>(), ih, 256); // 单位阵 + OT -> cube 转置
    AscendC::DataCopy(l1[512].ReinterpretCast<half>(), Oh, 4096);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::LoadData(l0a, l1.ReinterpretCast<half>(), AscendC::LoadData2DParams(0, 1, 1, 0, 0, false, 0));
    AscendC::LoadData(l0b, l1[512].ReinterpretCast<half>(), AscendC::LoadData2DParams(0, 16, 1, 0, 0, false, 0));
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Mmad(l0c, l0a, l0b, AscendC::MmadParams(16, HD, 16, 0, false, true)); // C[m][d] = I·OT
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyEnhancedParams epm;
    epm.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    AscendC::DataCopy(co, l0c, AscendC::DataCopyParams(1, 16, 0, 0), epm);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetVectorMask<float>(0, 0xFFFFULL);
    for (int nb = 0; nb < 16; nb++) // 紧凑化 [16m][16d] -> On [16m][256]
        AscendC::Adds<float, false>(On[nb * 16], co[nb * 256], 0.0f, FULL, 16, AscendC::UnaryRepeatParams(1, 1, 32, 2));
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::Muls(gb, gb, -1.0f, 6 * HD); // sigmoid 门控
    AscendC::Exp(gb, gb, 6 * HD);
    AscendC::Adds(gb, gb, 1.0f, 6 * HD);
    AscendC::Div(On, On, gb, 6 * HD);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(gat[(uint64_t)kvh * 6 * HD], On, 6 * HD);
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

// ==================== W4A8 路径 (int4 权重 + u8 x, u8xu8 mmad; 布局与推导见 docs/npu_impl.md) ====================
// 权重布局: 每核连续, 每 256 行块: 每 sp (组对 2u,2u+1): [4t][NI][512B 包字节] + meta [s'_lo fp16 NI*32B][s'_hi=s/16][W2_lo f32 NI*64B][W2_hi]
// 包字节 (fractal 内 32r+2n+p) = lo:q(组2u, k=t*32+p*16+r)+8 | hi:q(组2u+1, 同k)+8 << 4; W2 = h2f(存储s')*Sum(存储nib)
// decode (k_gemm_w8d): And x2 提两组转置源 fractal -> b8 转置加载 -> u8xu8 mmad 组内 s32 精确累加 -> 组 scale/修正到 f32
// prefill (k_gemm_w8p): 五平面 And 拆解反量化 fp16 -> mmad (同老路径精度); x 用 k_pack_x 的 fp16 zZ
// x 量化 (k_qx8): per-token per-128 组对称 int8 (存 u8, 零点 128), f32 域 RINT 与 host lrintf 逐字节一致

extern "C" __global__ __aicore__ void k_qx8(GM_ADDR x, GM_ADDR xrep, GM_ADDR xaux, int32_t k)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tb;
    pipe.InitBuffer(tb, 230 * 1024);
    LocalTensor<uint8_t> ub = tb.Get<uint8_t>();
    int KS = k / 128, KH = k >= 16384 ? 2 : 1, khalf = k / KH, KSH = KS / KH;
    // UB: xf [k]f32 70KB | ab [k]f32 70KB (abs/v/求和复用) | xh [k]half 35KB | m64 [KS][64]f32 35KB | xu [k]u8 17.5KB | sc 小量
    LocalTensor<float> xf = ub.ReinterpretCast<float>();
    LocalTensor<float> ab = ub[71680].ReinterpretCast<float>();
    LocalTensor<half> xh = ub[143360].ReinterpretCast<half>();
    LocalTensor<float> m64 = ub[178688].ReinterpretCast<float>();
    LocalTensor<uint8_t> xu = ub[214016];
    LocalTensor<float> sc = ub[231936].ReinterpretCast<float>(); // 到 235520 只有 896 f32: amax/sx/... 分开放
    LocalTensor<float> amax = sc; // [KS<=136]
    LocalTensor<float> sx = sc[160];
    LocalTensor<float> c127 = sc[320];
    LocalTensor<float> tt = sc[480]; // sinv -> 组和 -> ncx1
    LocalTensor<float> nc2 = sc[640];
    LocalTensor<float> sinvd = m64; // [KS][8] 借 m64 区 (与 m64 生命期错开: m64 第二次使用前 sinvd 已消费...) 不行, 放 xh 后段? 直接借 xu 后空间也不够 -- 用 m64 区尾部
    GlobalTensor<float> gx, ga;
    GlobalTensor<uint8_t> gr;
    gx.SetGlobalBuffer((__gm__ float*)x, k);
    gr.SetGlobalBuffer((__gm__ uint8_t*)xrep, (uint64_t)KH * 4 * khalf);
    ga.SetGlobalBuffer((__gm__ float*)xaux, (uint64_t)KS * 8 * 3);
    AscendC::DataCopy(xf, gx, k);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetVectorMask<float>(FULL, FULL);
    AscendC::Abs(ab, xf, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max<float, false>(m64, ab, ab[64], FULL, (uint8_t)KS, AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16)); // 组内两半逐元素 max
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float>(ab, m64, KS, 64, 1, 1, 8); // [KS][8] (借 ab 区, abs 已折叠进 m64)
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float>(amax, ab, KS / 8, 64, 1, 1, 8); // [KS]
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(amax, amax, 1e-30f, KS);
    AscendC::Duplicate(c127, 127.0f, KS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(sx, amax, 1.0f / 127.0f, KS);
    AscendC::Div(tt, c127, amax, KS); // sinv
    AscendC::PipeBarrier<PIPE_V>();
    sinvd = m64; // [KS][8], m64 的 amax 内容已消费
    AscendC::Brcb(sinvd, tt, (uint8_t)(KS / 8), AscendC::BrcbRepeatParams(1, 8));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BinaryRepeatParams bgs(1, 1, 0, 16, 16, 1); // 每 repeat 半组 (64), src1 块内复用, 每 repeat 前进一块
    AscendC::Mul<float, false>(ab, xf, sinvd, FULL, (uint8_t)KS, bgs);
    AscendC::Mul<float, false>(ab[64], xf[64], sinvd, FULL, (uint8_t)KS, bgs);
    AscendC::PipeBarrier<PIPE_V>();
    // f32 域 RINT 取整, 整数链全程精确 (与 host lrintf 逐字节一致)
    AscendC::Cast(xf.ReinterpretCast<int32_t>(), ab, RoundMode::CAST_RINT, k); // x 已消费, 借 xf 放 i32
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(ab, xf.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(ab, ab, 128.0f, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(xh, ab, RoundMode::CAST_NONE, k); // 整数 <=255, 精确
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(xu, xh, RoundMode::CAST_NONE, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(xh, xu, RoundMode::CAST_NONE, k); // u8 -> half (精确), 求 Sx
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(ab, xh, RoundMode::CAST_NONE, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, false>(m64, ab, ab[64], FULL, (uint8_t)KS, AscendC::BinaryRepeatParams(1, 1, 1, 8, 16, 16));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float>(tt, m64, 64, KS, 1, 1, 8); // sum(xu) 每组
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tt, tt, -16384.0f, KS); // Sx = sum - 128*128
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tt, tt, -8.0f, KS);
    AscendC::Muls(nc2, sx, -128.0f, KS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(tt, tt, sx, KS); // -8*Sx*sx
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<float> aux = ab; // [KS*8*3] <= 3264 f32, ab 区已消费完
    AscendC::Brcb(aux, sx, (uint8_t)(KS / 8), AscendC::BrcbRepeatParams(1, 8));
    AscendC::Brcb(aux[(uint32_t)KS * 8], tt, (uint8_t)(KS / 8), AscendC::BrcbRepeatParams(1, 8));
    AscendC::Brcb(aux[(uint32_t)KS * 16], nc2, (uint8_t)(KS / 8), AscendC::BrcbRepeatParams(1, 8));
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(ga, aux, KS * 8 * 3);
    for (int h = 0; h < KH; h++)
        for (int c = 0; c < 4; c++) // 副本 c: 半内组 g'=4j+c 的 128B 放到 512*j
            AscendC::DataCopy(gr[(uint64_t)(h * 4 + c) * khalf], xu[(h * KSH + c) * 128], AscendC::DataCopyParams((uint16_t)(KSH / 4), 4, 12, 12));
    AscendC::PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void k_gemm_w8d(GM_ADDR w, GM_ADDR xrep, GM_ADDR xaux, GM_ADDR y, int32_t n, int32_t k)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buf;
    AscendC::TBuf<AscendC::TPosition::A1> l1_buf;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buf;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buf;
    pipe.InitBuffer(ub_buf, 230 * 1024);
    pipe.InitBuffer(l1_buf, 200 * 1024);
    pipe.InitBuffer(l0a_buf, 64 * 1024);
    pipe.InitBuffer(l0b_buf, 64 * 1024);
    pipe.InitBuffer(l0c_buf, 256 * 1024);
    LocalTensor<uint8_t> ub = ub_buf.Get<uint8_t>();
    LocalTensor<uint8_t> l1 = l1_buf.Get<uint8_t>();
    LocalTensor<uint8_t> l0a = l0a_buf.Get<uint8_t>();
    LocalTensor<uint8_t> l0b = l0b_buf.Get<uint8_t>();
    LocalTensor<int32_t> l0c = l0c_buf.Get<int32_t>();
    int cores = (int)AscendC::GetBlockNum(), core = (int)AscendC::GetBlockIdx();
    int nr = n / cores, KS = k / 128, SP = KS / 2;
    int KH = k >= 16384 ? 2 : 1, khalf = k / KH, KSH = KS / KH; // k 两半时 L0A 重载
    // UB 布局
    LocalTensor<uint8_t> qbuf0 = ub, qbuf1 = ub[36864]; // 每 sp 最大 16*2176=34816
    LocalTensor<uint8_t> wb0 = ub[73728], wb1 = ub[106496]; // And 输出, 每组 32KB
    LocalTensor<int32_t> co0 = ub[139264].ReinterpretCast<int32_t>(), co1 = ub[155648].ReinterpretCast<int32_t>(); // drain 每组 16KB
    LocalTensor<float> acc = ub[172032].ReinterpretCast<float>(); // y 累加 [256]
    LocalTensor<float> cp = ub[173056].ReinterpretCast<float>(); // 紧凑 C f32 [256]
    LocalTensor<float> sf = ub[174080].ReinterpretCast<float>(); // s' f32 [256]
    LocalTensor<float> wf = ub[175104].ReinterpretCast<float>(); // W2 f32 [256]
    LocalTensor<float> sr = ub[176128].ReinterpretCast<float>(); // s_raw f32 [256]
    LocalTensor<float> xa = ub[177152].ReinterpretCast<float>(); // xaux 常驻 [KS][8]*3 <= 13056B
    LocalTensor<uint16_t> mk = ub[190464].ReinterpretCast<uint16_t>(); // And 掩码 [16]+[16]
    GlobalTensor<uint8_t> gw, gx;
    GlobalTensor<float> ga, gy;
    uint64_t wtotal = 0;
    int nblk = (nr + 255) / 256;
    for (int b = 0; b < nblk; b++) {
        int nb = nr - b * 256 < 256 ? nr - b * 256 : 256;
        wtotal += (uint64_t)SP * (nb / 16) * 2240;
    }
    gw.SetGlobalBuffer((__gm__ uint8_t*)w + (uint64_t)core * wtotal, wtotal);
    gx.SetGlobalBuffer((__gm__ uint8_t*)xrep, (uint64_t)KH * 4 * khalf);
    ga.SetGlobalBuffer((__gm__ float*)xaux, (uint64_t)KS * 8 * 3);
    gy.SetGlobalBuffer((__gm__ float*)y, n);
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::Duplicate(mk, (uint16_t)0x0F0F, 16);
    AscendC::Duplicate(mk[16], (uint16_t)0xF0F0, 16);
    AscendC::DataCopy(xa, ga, KS * 8 * 3);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::BinaryRepeatParams bmk(1, 1, 0, 8, 8, 0); // src1 单块复用不前进 (And 掩码 / 组标量广播)
    AscendC::BinaryRepeatParams bpp(1, 1, 1, 8, 8, 8);
    AscendC::UnaryRepeatParams cp8(1, 1, 8, 4);
    AscendC::UnaryRepeatParams cpa(1, 1, 8, 8);
    AscendC::UnaryRepeatParams cps(1, 1, 2, 32); // C fractal 行0 紧凑化: src 跨 1KB, dst 跨 64B
    int cur_h = -1;
    AscendC::TEventID eq[2], eqf[2], ewb[2], ewbf[2], ebl[2], eblf[2], em[2], emf[2], ec[2], ecf[2];
    for (int i = 0; i < 2; i++) {
        eq[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        eqf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
        ewb[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
        ewbf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_V>();
        ebl[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE1>();
        eblf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE3>();
        em[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
        emf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
        ec[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_M>(); // drain 读完 -> L0C 槽可重用
        ecf[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_V>(); // mmad 完成 -> 可 drain
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eqf[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ewbf[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[i]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(emf[i]);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(ec[i]);
    }
    uint64_t woff = 0;
    for (int b = 0; b < nblk; b++) {
        int nb = nr - b * 256 < 256 ? nr - b * 256 : 256;
        int NI = nb / 16, nfr = 4 * NI; // 每组 fractal 数
        int spb = nfr * 512 + NI * 192;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(acc, 0.0f, 256); // Level-2, 结束会把 mask 恢复为满
        AscendC::PipeBarrier<PIPE_V>();
        int pg[2][3], npend = 0; // 待 drain 组队列 {g, gi, qbuf 奇偶}, 滞后 2 组发射, 避免 drain 的 WaitFlag 挡住后续 And
// drain + epilogue (V); 宏展开 (aicore 不支持 lambda)
#define DRAIN(g_, gi_, qp_) { \
            int p2_ = (g_) & 1; \
            LocalTensor<uint8_t> qb_ = (qp_) ? qbuf1 : qbuf0; \
            LocalTensor<int32_t> co_ = p2_ ? co1 : co0; \
            AscendC::WaitFlag<AscendC::HardEvent::M_V>(ecf[p2_]); \
            AscendC::DataCopyEnhancedParams ep_; \
            ep_.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX; \
            AscendC::DataCopy(co_, l0c[p2_ * 16 * 256], AscendC::DataCopyParams(1, (uint16_t)NI, 0, 0), ep_); \
            AscendC::SetFlag<AscendC::HardEvent::V_M>(ec[p2_]); \
            AscendC::SetVectorMask<float>(0, 0xFFFF); \
            AscendC::Cast<float, int32_t, false>(cp, co_, RoundMode::CAST_NONE, FULL, (uint8_t)NI, cps); \
            AscendC::SetVectorMask<float>(FULL, FULL); \
            LocalTensor<half> mh_ = qb_[nfr * 512].ReinterpretCast<half>(); \
            LocalTensor<float> mw_ = qb_[nfr * 512 + NI * 64 + (gi_) * NI * 64].ReinterpretCast<float>(); \
            AscendC::Cast<float, half, false>(sf, mh_[(gi_) * NI * 16], RoundMode::CAST_NONE, FULL, (uint8_t)(NI / 4), cp8); \
            AscendC::PipeBarrier<PIPE_V>(); \
            AscendC::Mul<float, false>(sr, sf, xa[KS * 8 + (g_) * 8], FULL, (uint8_t)(NI / 4), bmk); /* s_raw*(-cx1) */ \
            AscendC::Muls<float, false>(sf, sf, (gi_) ? 1.0f / 16.0f : 1.0f, FULL, (uint8_t)(NI / 4), cpa); /* hi 组的 /16 在 f32 域派生 (fp16 存 s/16 会掉次正规被 flush) */ \
            AscendC::Mul<float, false>(wf, mw_, xa[2 * KS * 8 + (g_) * 8], FULL, (uint8_t)(NI / 4), bmk); \
            AscendC::PipeBarrier<PIPE_V>(); \
            AscendC::Mul<float, false>(sf, sf, xa[(g_) * 8], FULL, (uint8_t)(NI / 4), bmk); \
            AscendC::MulAddDst<float, float, false>(acc, cp, sf, FULL, (uint8_t)(NI / 4), bpp); \
            AscendC::Add<float, false>(acc, acc, wf, FULL, (uint8_t)(NI / 4), bpp); \
            AscendC::PipeBarrier<PIPE_V>(); \
            AscendC::Add<float, false>(acc, acc, sr, FULL, (uint8_t)(NI / 4), bpp); \
            AscendC::SetVectorMask<half>(FULL, FULL); \
            if ((gi_) == 1) AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eqf[qp_]); \
        }
        for (int sp = 0; sp < SP; sp++) {
            int p = sp & 1;
            int g0 = 2 * sp, h = g0 / KSH;
            if (h != cur_h) { // 装入该 k 半的 xrep 到 L0A (整段驻留)
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(l1[131072], gx[(uint64_t)h * 4 * khalf], 4 * khalf);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::LoadData(l0a, l1[131072], AscendC::LoadData2DParams(0, (uint16_t)(4 * khalf / 512), 1, 0, 0, false, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
                cur_h = h;
            }
            LocalTensor<uint8_t> qb = p ? qbuf1 : qbuf0;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eqf[p]);
            AscendC::DataCopy(qb, gw[woff + (uint64_t)sp * spb], spb);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eq[p]);
            for (int gi = 0; gi < 2; gi++) { // 组 2sp+gi: And -> L1 -> L0B -> mmad
                int g = g0 + gi, gh = g - h * KSH, p2 = g & 1;
                if (npend == 2) { // 先发滞后的 drain (不依赖本 sp 数据)
                    DRAIN(pg[0][0], pg[0][1], pg[0][2]);
                    pg[0][0] = pg[1][0];
                    pg[0][1] = pg[1][1];
                    pg[0][2] = pg[1][2];
                    npend = 1;
                }
                if (gi == 0) AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eq[p]);
                LocalTensor<uint8_t> wbx = p2 ? wb1 : wb0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ewbf[p2]);
                AscendC::And<uint16_t, false>(wbx.ReinterpretCast<uint16_t>(), qb.ReinterpretCast<uint16_t>(), mk[gi * 16], FULL, (uint8_t)(nfr * 2), bmk);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ewb[p2]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ewb[p2]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[p2]);
                AscendC::DataCopy(l1[p2 * 32768], wbx, nfr * 512);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ewbf[p2]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(ebl[p2]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(ebl[p2]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(emf[p2]);
                AscendC::LoadData(l0b[p2 * 32768], l1[p2 * 32768], AscendC::LoadData2DParams(0, (uint16_t)nfr, 1, 0, 0, true, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[p2]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(em[p2]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(em[p2]);
                AscendC::WaitFlag<AscendC::HardEvent::V_M>(ec[p2]);
                AscendC::MmadParams mm(1, (uint16_t)nb, 128, 0, false, true);
                AscendC::Mmad(l0c[p2 * 16 * 256], l0a[(gh & 3) * khalf + 512 * (gh >> 2)], l0b[p2 * 32768], mm);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(emf[p2]);
                AscendC::SetFlag<AscendC::HardEvent::M_V>(ecf[p2]);
                pg[npend][0] = g;
                pg[npend][1] = gi;
                pg[npend][2] = p;
                npend++;
            }
        }
        for (int i = 0; i < npend; i++) DRAIN(pg[i][0], pg[i][1], pg[i][2]); // 清尾
#undef DRAIN
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gy[(uint64_t)core * nr + b * 256], acc, nb);
        AscendC::PipeBarrier<PIPE_ALL>();
        woff += (uint64_t)SP * spb;
    }
    for (int i = 0; i < 2; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eqf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ewbf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eblf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(emf[i]);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(ec[i]);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// m>1 prefill: W4A8 布局 -> 五平面拆解反量化 fp16 -> mmad (fp16, fp32 累加整个 k)
// xp: k_pack_x 的 fp16 zZ [kc][mi][ki][16m][16k]; y [m][n] f32
extern "C" __global__ __aicore__ void k_gemm_w8p(GM_ADDR w, GM_ADDR xp, GM_ADDR y, int32_t m, int32_t n, int32_t k)
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
    int nr = n / cores, KS = k / 128, SP = KS / 2, M1 = (m + 15) / 16;
    // UB: qbuf 单缓冲 35KB | tmp_i16 32KB | t2 32KB | S_even 32KB | S_odd 32KB | sdup 2KB | co 2x16KB
    LocalTensor<uint8_t> qb = ub;
    LocalTensor<int16_t> ti = ub[36864].ReinterpretCast<int16_t>();
    LocalTensor<half> t2 = ub[69632].ReinterpretCast<half>();
    LocalTensor<half> se = ub[102400].ReinterpretCast<half>();
    LocalTensor<half> so = ub[135168].ReinterpretCast<half>();
    LocalTensor<half> sd = ub[167936].ReinterpretCast<half>(); // sdup [4NI][16]
    LocalTensor<half> sr16 = ub[172032].ReinterpretCast<half>(); // s_raw 临时 [NI*16]
    LocalTensor<float> co0 = ub[176128].ReinterpretCast<float>(), co1 = ub[192512].ReinterpretCast<float>();
    LocalTensor<uint16_t> mk = ub[225280].ReinterpretCast<uint16_t>(); // 掩码 [16]x5: 0x000F 0x0F00 0x00F0 0x7000 0x8000
    uint64_t wtotal = 0;
    int nblk = (nr + 255) / 256;
    for (int b = 0; b < nblk; b++) {
        int nb = nr - b * 256 < 256 ? nr - b * 256 : 256;
        wtotal += (uint64_t)SP * (nb / 16) * 2240;
    }
    GlobalTensor<uint8_t> gw;
    GlobalTensor<half> gxp;
    GlobalTensor<float> gy;
    gw.SetGlobalBuffer((__gm__ uint8_t*)w + (uint64_t)core * wtotal, wtotal);
    gxp.SetGlobalBuffer((__gm__ half*)xp, (uint64_t)KS * M1 * 2048);
    gy.SetGlobalBuffer((__gm__ float*)y, (uint64_t)m * n);
    AscendC::SetVectorMask<half>(FULL, FULL);
    AscendC::Duplicate(mk, (uint16_t)0x000F, 16);
    AscendC::Duplicate(mk[16], (uint16_t)0x0F00, 16);
    AscendC::Duplicate(mk[32], (uint16_t)0x00F0, 16);
    AscendC::Duplicate(mk[48], (uint16_t)0x7000, 16);
    AscendC::Duplicate(mk[64], (uint16_t)0x8000, 16);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::BinaryRepeatParams bmk(1, 1, 0, 8, 8, 0); // And: src1 单块复用
    AscendC::BinaryRepeatParams bph(1, 1, 1, 8, 8, 8);
    AscendC::BinaryRepeatParams bsc(1, 1, 0, 16, 16, 1); // scale: 每 fractal 一块
    AscendC::UnaryRepeatParams cpa(1, 1, 8, 8);
    uint64_t woff = 0;
    for (int b = 0; b < nblk; b++) {
        int nb = nr - b * 256 < 256 ? nr - b * 256 : 256;
        int NI = nb / 16, nfr = 4 * NI; // 每 sp 的 chunk 数 (= 每组每平面 fractal 数)
        int spb = nfr * 512 + NI * 192;
        for (int g = 0; g < KS; g++) {
            int sp = g / 2, gi = g & 1;
            if (gi == 0) {
                AscendC::PipeBarrier<PIPE_ALL>(); // 单缓冲: 等上一 sp 的向量消费完
                AscendC::DataCopy(qb, gw[woff + (uint64_t)sp * spb], spb);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            int nby2 = nfr * 256; // u16 lane 数
            if (gi == 0) { // lo 组: And 0x000F -> se (x1), And 0x0F00 -> so (x256)
                AscendC::And<uint16_t, false>(ti.ReinterpretCast<uint16_t>(), qb.ReinterpretCast<uint16_t>(), mk, FULL, (uint8_t)(nby2 / 128), bmk);
                AscendC::Cast<half, int16_t, false>(se, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(nby2 / 128), cpa);
                AscendC::And<uint16_t, false>(ti.ReinterpretCast<uint16_t>(), qb.ReinterpretCast<uint16_t>(), mk[16], FULL, (uint8_t)(nby2 / 128), bmk);
                AscendC::Cast<half, int16_t, false>(so, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(nby2 / 128), cpa);
                AscendC::Muls<half, false>(so, so, (half)(1.0f / 256.0f), FULL, (uint8_t)(nby2 / 128), cpa);
            } else { // hi 组: And 0x00F0 -> se (x16); 0x7000/0x8000 相减 -> so (x4096)
                AscendC::And<uint16_t, false>(ti.ReinterpretCast<uint16_t>(), qb.ReinterpretCast<uint16_t>(), mk[32], FULL, (uint8_t)(nby2 / 128), bmk);
                AscendC::Cast<half, int16_t, false>(se, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(nby2 / 128), cpa);
                AscendC::Muls<half, false>(se, se, (half)(1.0f / 16.0f), FULL, (uint8_t)(nby2 / 128), cpa);
                AscendC::And<uint16_t, false>(ti.ReinterpretCast<uint16_t>(), qb.ReinterpretCast<uint16_t>(), mk[48], FULL, (uint8_t)(nby2 / 128), bmk);
                AscendC::Cast<half, int16_t, false>(t2, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(nby2 / 128), cpa);
                AscendC::And<uint16_t, false>(ti.ReinterpretCast<uint16_t>(), qb.ReinterpretCast<uint16_t>(), mk[64], FULL, (uint8_t)(nby2 / 128), bmk);
                AscendC::Cast<half, int16_t, false>(so, ti, RoundMode::CAST_NONE, FULL, (uint8_t)(nby2 / 128), cpa);
                AscendC::Sub<half, false>(so, t2, so, FULL, (uint8_t)(nby2 / 128), bph); // hi(p1)*4096
                AscendC::Muls<half, false>(so, so, (half)(1.0f / 4096.0f), FULL, (uint8_t)(nby2 / 128), cpa);
            }
            AscendC::Adds<half, false>(se, se, (half)-8.0f, FULL, (uint8_t)(nby2 / 128), cpa);
            AscendC::Adds<half, false>(so, so, (half)-8.0f, FULL, (uint8_t)(nby2 / 128), cpa);
            // s_raw -> sdup [4t][NI][16]
            LocalTensor<half> ms = qb[nfr * 512 + gi * NI * 32].ReinterpretCast<half>(); // meta 存的就是原始 s (平面值已被 Muls 2^-j 还原为 q+8)
            for (int t = 0; t < 4; t++) AscendC::Adds(sd[t * NI * 16], ms, (half)0.0f, NI * 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<half, false>(se, se, sd, FULL, (uint8_t)nfr, bsc);
            AscendC::Mul<half, false>(se[128], se[128], sd, FULL, (uint8_t)nfr, bsc);
            AscendC::Mul<half, false>(so, so, sd, FULL, (uint8_t)nfr, bsc);
            AscendC::Mul<half, false>(so[128], so[128], sd, FULL, (uint8_t)nfr, bsc);
            AscendC::PipeBarrier<PIPE_ALL>();
            // MTE3: se -> L1 偶 ki 行, so -> 奇 ki 行 (fractal 顺序 [8ki][NI])
            AscendC::DataCopy(l1.ReinterpretCast<half>(), se, AscendC::DataCopyParams(4, (uint16_t)(NI * 512 / 32), 0, (uint16_t)(NI * 512 / 32)));
            AscendC::DataCopy(l1[(uint32_t)NI * 512].ReinterpretCast<half>(), so, AscendC::DataCopyParams(4, (uint16_t)(NI * 512 / 32), 0, (uint16_t)(NI * 512 / 32)));
            // x 切片 -> L1 -> L0A
            AscendC::DataCopy(l1[131072].ReinterpretCast<half>(), gxp[(uint64_t)g * M1 * 2048], M1 * 2048);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::LoadData(l0a, l1[131072].ReinterpretCast<half>(), AscendC::LoadData2DParams(0, (uint16_t)(M1 * 8), 1, 0, 0, false, 0));
            AscendC::LoadData(l0b, l1.ReinterpretCast<half>(), AscendC::LoadData2DParams(0, (uint16_t)(8 * NI), 1, 0, 0, true, 0));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::MmadParams mm((uint16_t)m, (uint16_t)nb, 128, 0, false, g == 0);
            AscendC::Mmad(l0c, l0a, l0b, mm);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        // 输出: C [N1][M1][16m][16n] -> y [m][n] (与 fp16 kernel 相同)
        AscendC::DataCopyEnhancedParams ep;
        ep.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        for (int ni = 0; ni < nb / 16; ni++) {
            int c = ni & 1;
            LocalTensor<float> co = c ? co1 : co0;
            AscendC::DataCopy(co, l0c[(ni * M1) * 256], AscendC::DataCopyParams(1, (uint16_t)M1, 0, 0), ep);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int mi = 0; mi < M1; mi++) {
                int mrows = m - mi * 16 < 16 ? m - mi * 16 : 16;
                AscendC::DataCopy(gy[(uint64_t)(mi * 16) * n + core * nr + b * 256 + ni * 16], co[mi * 256], AscendC::DataCopyParams((uint16_t)mrows, 2, 0, (uint16_t)(n / 8 - 2)));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        woff += (uint64_t)SP * spb;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
