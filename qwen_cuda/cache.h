#ifndef QWEN_CUDA_CACHE_H
#define QWEN_CUDA_CACHE_H

// 前缀缓存 (磁盘): 缓存点在每条消息的 <|im_end|> 之后, key 由 python 按 token 序列 (图片用其 sha256) 算出
// 每个缓存点一个目录 <dir>/ab/cd/<key>: metadata.txt (文本, 最后写, 末行 END 才有效) + kvcache.bin (该段 KV) + 可选 state.bin (fp32 槽状态) / state.bin.compressed (后台量化)
// 线程: 写盘线程 (计算线程只把数据放进无上限队列) + 量化线程 (定期扫目录, 与计算线程无通信); 两者都是本 so 创建的 musl 线程, 不调 libcuda
// 计算线程 (prefill / decode) 是 python 的 glibc 线程: 文件操作全部直接 syscall, 不碰 musl 的 errno / 锁 / FILE
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <immintrin.h>
#include <thread>
#include <vector>

namespace {

namespace Sys { // x86_64 raw syscall, 返回值 < 0 是 -errno

inline long sc(long n, long a1 = 0, long a2 = 0, long a3 = 0, long a4 = 0, long a5 = 0, long a6 = 0) {
    long r;
    register long r10 __asm__("r10") = a4;
    register long r8 __asm__("r8") = a5;
    register long r9 __asm__("r9") = a6;
    __asm__ volatile("syscall" : "=a"(r) : "0"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9) : "rcx", "r11", "memory");
    return r;
}

const int O_RD = 0, O_WR = 1, O_CREAT_ = 0100, O_TRUNC_ = 01000, O_DIRECT_ = 040000, O_CLOEXEC_ = 02000000;
inline int open(const char* p, int flags, int mode = 0644) { return (int)sc(2, (long)p, flags | O_CLOEXEC_, mode); }
inline void close(int fd) { sc(3, fd); }
inline long pread(int fd, void* b, size_t n, uint64_t off) { return sc(17, fd, (long)b, (long)n, (long)off); }
inline long pwrite(int fd, const void* b, size_t n, uint64_t off) { return sc(18, fd, (long)b, (long)n, (long)off); }
inline long write(int fd, const void* b, size_t n) { return sc(1, fd, (long)b, (long)n); }
inline int mkdir(const char* p) { return (int)sc(83, (long)p, 0755); }
inline int rename(const char* a, const char* b) { return (int)sc(82, (long)a, (long)b); }
inline int unlink(const char* p) { return (int)sc(87, (long)p); }
inline int rmdir(const char* p) { return (int)sc(84, (long)p); }
inline int access(const char* p) { return (int)sc(21, (long)p, 0); } // F_OK
inline int touch(const char* p) { return (int)sc(280, -100, (long)p, 0, 0); } // utimensat(AT_FDCWD, p, NULL): atime/mtime 改为现在
inline int64_t fsize(int fd) { // fstat 的 st_size (x86_64 struct stat 偏移 48)
    alignas(8) uint8_t st[144];
    if (sc(5, fd, (long)st) < 0) return -1;
    int64_t s;
    memcpy(&s, st + 48, 8);
    return s;
}
inline void wake(void* addr) { sc(202, (long)addr, 129, 0x7fffffff); } // FUTEX_WAKE_PRIVATE
inline void wait(void* addr, uint32_t old) { sc(202, (long)addr, 128, old, 0); } // FUTEX_WAIT_PRIVATE
inline void sleep_ms(long ms) {
    long ts[2] = {ms / 1000, ms % 1000 * 1000000};
    sc(35, (long)ts, 0);
}

void* amap(size_t n) { // 匿名 mmap (页对齐, 已清零), 可在任意线程分配 / 释放; 失败直接退出
    long r = sc(9, 0, (long)n, 3, 0x22, -1, 0); // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS
    if (r < 0 && r > -4096) {
        fprintf(stderr, "❌ mmap %zu 字节失败: %ld\n", n, -r);
        _Exit(1);
    }
    return (void*)r;
}
inline void unmap(void* p, size_t n) {
    if (p) sc(11, (long)p, (long)n);
}

bool read_all(int fd, void* b, size_t n, uint64_t off) {
    for (size_t d = 0; d < n;) {
        long r = pread(fd, (uint8_t*)b + d, n - d, off + d);
        if (r <= 0) return false;
        d += r;
    }
    return true;
}

bool write_all(int fd, const void* b, size_t n, uint64_t off) {
    for (size_t d = 0; d < n;) {
        long r = pwrite(fd, (const uint8_t*)b + d, n - d, off + d);
        if (r <= 0) return false;
        d += r;
    }
    return true;
}

} // namespace Sys

namespace Sha { // SHA-256

const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

struct Ctx {
    uint32_t h[8];
    uint8_t buf[64];
    uint64_t len; // 已输入字节数
};

inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void block(Ctx& c, const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[i * 4] << 24 | (uint32_t)p[i * 4 + 1] << 16 | (uint32_t)p[i * 4 + 2] << 8 | p[i * 4 + 3];
    for (int i = 16; i < 64; i++) w[i] = w[i - 16] + (ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] + (ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3], e = c.h[4], f = c.h[5], g = c.h[6], h = c.h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
        uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
        h = g, g = f, f = e, e = d + t1, d = cc, cc = b, b = a, a = t1 + t2;
    }
    c.h[0] += a, c.h[1] += b, c.h[2] += cc, c.h[3] += d, c.h[4] += e, c.h[5] += f, c.h[6] += g, c.h[7] += h;
}

void init(Ctx& c) {
    const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(c.h, h0, sizeof(h0));
    c.len = 0;
}

void update(Ctx& c, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    while (n) {
        size_t r = c.len % 64, m = std::min(n, 64 - r);
        memcpy(c.buf + r, p, m);
        c.len += m, p += m, n -= m;
        if (r + m == 64) block(c, c.buf);
    }
}

void final(Ctx& c, uint8_t* out) {
    uint64_t bits = c.len * 8;
    uint8_t pad[72] = {0x80};
    size_t r = c.len % 64, np = (r < 56 ? 56 : 120) - r;
    for (int i = 0; i < 8; i++) pad[np + i] = (uint8_t)(bits >> (56 - 8 * i));
    update(c, pad, np + 8);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) out[i * 4 + j] = (uint8_t)(c.h[i] >> (24 - 8 * j));
}

void hash(const void* data, size_t n, uint8_t* out) {
    Ctx c;
    init(c);
    update(c, data, n);
    final(c, out);
}

void hex(const uint8_t* k, char* out) { // 32 字节 -> 64 个小写 hex + '\0'
    const char* d = "0123456789abcdef";
    for (int i = 0; i < 32; i++) out[i * 2] = d[k[i] >> 4], out[i * 2 + 1] = d[k[i] & 15];
    out[64] = 0;
}

} // namespace Sha

// 槽状态的 8 bit 编码 (notes/state_quant_experiments.md 第 5 节的最终方案): 每 32 个一块, 存块内最大值 amax (fp32) + 倍频程档位下标 (uint8) + 每个值 1 字节 (符号位 | 级 q)
// 第 q 级 = amax·2^((q-127)·o/127), o 取自 256 档 [2, 12]; 精确为 0 的值另记下标. 关闭 fp-contract: 与 GPU 解码 (kernels.cu k_state_dec) 逐位相同
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#pragma GCC optimize("fp-contract=off")
namespace Codec {

const int QK = 32, N_OCT = 256;

inline float oct(int k) { return (float)(k == 255 ? 12.0 : 2.0 + (double)k * (10.0 / 255.0)); } // 与 np.linspace(2, 12, 256).astype(np.float32) 逐位相同

inline __m256 exp2v(__m256 x) { // 2^x: x = n + f, f ∈ [-0.5, 0.5], 2^f 用 6 阶泰勒, 2^n 拼指数位
    __m256 n = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(x, n);
    __m256 p = _mm256_set1_ps(1.5403530e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

inline __m256 lev_v(__m256 amax, __m256 st, __m256 q) { return _mm256_mul_ps(amax, exp2v(_mm256_mul_ps(_mm256_sub_ps(q, _mm256_set1_ps(127)), st))); } // q 是 float 形式的整数

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

void encode_block(const float* x, const float* st_t, const float* inv_t, const float* mid_t, float* amax_out, uint8_t* idx_out, int8_t* code) { // st = o/127, inv = 1/st, mid = log2((1 + 2^st)/2)
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), zero = _mm256_setzero_ps();
    __m256 u[4], lr[4], nzm[4], bq[4];
    float amax = 0;
    for (int k = 0; k < 4; k++) {
        u[k] = _mm256_and_ps(_mm256_loadu_ps(x + k * 8), absm);
        float t[8];
        _mm256_storeu_ps(t, u[k]);
        for (int j = 0; j < 8; j++) amax = std::max(amax, t[j]);
    }
    int best_o = 0;
    for (int k = 0; k < 4; k++) bq[k] = _mm256_set1_ps(127);
    if (amax > 0) {
        for (int k = 0; k < 4; k++) { // log2(|x|/amax), 各档共用; 0 处填 0, 之后用 nzm 屏蔽
            float t[8];
            _mm256_storeu_ps(t, u[k]);
            for (int j = 0; j < 8; j++) t[j] = t[j] > 0 ? log2f(t[j] / amax) : 0;
            lr[k] = _mm256_loadu_ps(t);
            nzm[k] = _mm256_cmp_ps(u[k], zero, _CMP_GT_OQ);
        }
        const __m256 va = _mm256_set1_ps(amax), c127 = _mm256_set1_ps(127), c126 = _mm256_set1_ps(126), one = _mm256_set1_ps(1);
        float best_e = INFINITY;
        for (int c = 0; c < N_OCT; c++) {
            __m256 st = _mm256_set1_ps(st_t[c]), inv = _mm256_set1_ps(inv_t[c]), mid = _mm256_set1_ps(mid_t[c]), acc = zero, q[4];
            for (int k = 0; k < 4; k++) {
                __m256 q0 = _mm256_min_ps(_mm256_max_ps(_mm256_floor_ps(_mm256_fmadd_ps(lr[k], inv, c127)), zero), c126);
                __m256 r = _mm256_fnmadd_ps(_mm256_sub_ps(q0, c127), st, lr[k]); // 相对 q0 级的 log2 偏移
                q[k] = _mm256_add_ps(q0, _mm256_and_ps(_mm256_cmp_ps(r, mid, _CMP_GT_OQ), one)); // 超过线性中点取 q0 + 1
                __m256 d = _mm256_and_ps(_mm256_sub_ps(u[k], lev_v(va, st, q[k])), nzm[k]);
                acc = _mm256_fmadd_ps(d, d, acc);
            }
            float e = hsum(acc);
            if (e < best_e) {
                best_e = e, best_o = c;
                for (int k = 0; k < 4; k++) bq[k] = _mm256_and_ps(q[k], nzm[k]);
            }
        }
    }
    *amax_out = amax;
    *idx_out = (uint8_t)best_o;
    for (int k = 0; k < 4; k++) {
        int qi[8];
        _mm256_storeu_si256((__m256i*)qi, _mm256_cvtps_epi32(bq[k]));
        for (int j = 0; j < 8; j++) code[k * 8 + j] = (int8_t)((x[k * 8 + j] < 0 ? 0x80 : 0) | qi[j]);
    }
}

void decode_block(float amax, float st, const int8_t* code, float* out) { // 只用于自测 (正式解码在 GPU 上)
    const __m256 va = _mm256_set1_ps(amax), vst = _mm256_set1_ps(st);
    for (int k = 0; k < 4; k++) {
        __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(code + k * 8)));
        __m256 v = lev_v(va, vst, _mm256_cvtepi32_ps(_mm256_and_si256(c, _mm256_set1_epi32(127))));
        __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(c, _mm256_set1_epi32(128)), 24); // 第 7 位移到 float 符号位
        _mm256_storeu_ps(out + k * 8, _mm256_xor_ps(v, _mm256_castsi256_ps(sgn)));
    }
}

template <class F> void parallel(int64_t nb, int n_threads, F f) { // 块 [0, nb): n_threads 个线程用原子计数器每次领 256 块; 只在 musl 线程里调用
    const int64_t STEP = 256;
    std::atomic<int64_t> next{0};
    std::vector<std::thread> ths;
    for (int t = 0; t < n_threads; t++)
        ths.emplace_back([&] {
            for (int64_t b0; (b0 = next.fetch_add(STEP)) < nb;)
                for (int64_t i = b0; i < std::min(nb, b0 + STEP); i++) f(i);
        });
    for (auto& th : ths) th.join();
}

// 压缩文件布局 (各段 4096 对齐, 段内按顺序): [头 4096: "QSC1", u64 n, u64 nz][amax f32 × nb][idx u8 × nb][code i8 × n][zero_idx u32 × nz]
inline size_t al4k(size_t x) { return (x + 4095) & ~(size_t)4095; }
inline size_t off_amax() { return 4096; }
inline size_t off_idx(size_t n) { return off_amax() + al4k(n / QK * 4); }
inline size_t off_code(size_t n) { return off_idx(n) + al4k(n / QK); }
inline size_t off_zero(size_t n) { return off_code(n) + al4k(n); }
inline size_t comp_bytes(size_t n, size_t nz) { return off_zero(n) + al4k(nz * 4); }

size_t encode(const float* x, size_t n, uint8_t* out, size_t cap, int n_threads) { // n 是 32 的倍数, out 已清零且 4096 对齐; 返回写入字节数 (对齐后), 放不下返回 0
    float st[N_OCT], inv[N_OCT], mid[N_OCT];
    for (int c = 0; c < N_OCT; c++) {
        st[c] = oct(c) / 127.0f;
        inv[c] = 1.0f / st[c];
        mid[c] = log2f((1 + exp2f(st[c])) / 2);
    }
    float* amax = (float*)(out + off_amax());
    uint8_t* idx = out + off_idx(n);
    int8_t* code = (int8_t*)(out + off_code(n));
    parallel((int64_t)(n / QK), n_threads, [&](int64_t i) { encode_block(x + i * QK, st, inv, mid, amax + i, idx + i, code + i * QK); });
    uint32_t* zi = (uint32_t*)(out + off_zero(n));
    size_t nz = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] != 0) continue;
        if (off_zero(n) + (nz + 1) * 4 > cap) return 0;
        zi[nz++] = (uint32_t)i;
    }
    uint64_t hdr[2] = {n, nz};
    memcpy(out, "QSC1", 4);
    memcpy(out + 8, hdr, 16);
    return comp_bytes(n, nz);
}

void decode_cpu(const uint8_t* comp, float* out) { // 只用于自测, 与 GPU 解码对比
    uint64_t hdr[2];
    memcpy(hdr, comp + 8, 16);
    size_t n = hdr[0], nz = hdr[1];
    const float* amax = (const float*)(comp + off_amax());
    const uint8_t* idx = comp + off_idx(n);
    const int8_t* code = (const int8_t*)(comp + off_code(n));
    for (size_t i = 0; i < n / QK; i++) decode_block(amax[i], oct(idx[i]) / 127.0f, code + i * QK, out + i * QK);
    const uint32_t* zi = (const uint32_t*)(comp + off_zero(n));
    for (size_t i = 0; i < nz; i++) out[zi[i]] = 0;
}

} // namespace Codec
#pragma GCC pop_options

namespace Cache {

const int N_ATT = 16, KV_ROW = 1024, SC_ROW = 64; // 全注意力层数; 每 token 每层 K (或 V) 的 int8 字节数 / fp16 scale 字节数 (与 qwen.cpp 一致)
const size_t LAYER_TOK = 2 * (KV_ROW + SC_ROW), TOK_BYTES = N_ATT * LAYER_TOK; // 每 token 的 KV 字节数 (34816)
const int MAXP = 65536; // 一个请求最多的缓存点数, 超过则该请求不缓存
const int FLUSH = 256; // 输出阶段每生成这么多 token 写一段 KV
const int STATE_MIN = 512, BLK = 2048; // state 写入规则的两个长度阈值, 见 qwen.cpp plan_states
const size_t AL = 4096; // direct IO 的对齐
const size_t MBUF = 8 << 20; // metadata 最大字节数
const int ROLE_IDS[4] = {8678, 846, 74455, 13766};
const char* ROLE_NAMES[4] = {"system", "user", "assistant", "tool"};

inline size_t al(size_t x) { return (x + AL - 1) & ~(AL - 1); }

char G_dir[2048] = {}; // 缓存根目录 (已按后端区分), 空表示不启用
char G_model[512] = {}; // 模型标识 (文件名 + 字节数), 写入 metadata, 读取时不一致则视为无效
size_t G_state_bytes = 0;
int G_qthreads = 8;
int G_max_segs = 0; // 一个缓存点最多的 KV 段数
std::atomic<uint32_t> G_out_seq{0}; // 输出临时目录编号

// kvcache.bin: 若干段依次存放, 每段 n 个 token, 段内 [16 层][K n×1024 | V n×1024 | K scale n×64 | V scale n×64], 每段补齐到 4096; 段长列在 metadata 的 kv_segments
// 计算线程拷出的 KV 缓冲也是这个布局 (覆盖 token [a, b), 可能包含多个缓存点), 写盘时按点切出
inline size_t seg_bytes(int n) { return al((size_t)n * TOK_BYTES); }

void gather_seg(const uint8_t* src, int n, int s, int e, uint8_t* dst) { // src: 长 n 的缓冲, 取其中相对行 [s, e) 组成一段写到 dst
    const size_t ro[4] = {KV_ROW, KV_ROW, SC_ROW, SC_ROW};
    size_t m = e - s;
    for (int l = 0; l < N_ATT; l++) {
        const uint8_t* p = src + (size_t)l * n * LAYER_TOK;
        uint8_t* q = dst + l * m * LAYER_TOK;
        for (int k = 0; k < 4; k++) {
            memcpy(q, p + s * ro[k], m * ro[k]);
            p += n * ro[k], q += m * ro[k];
        }
    }
}

struct Plan { // 一个槽当前请求的缓存信息; 请求结束 (等所有写入完成) 之前不会被下一个请求覆盖, 写盘线程可以直接读
    int n_pts = -1; // -1: 本请求不缓存
    int* pos; // 第 i 个缓存点 = 前 pos[i] 个 token (第 i 段是 [pos[i-1], pos[i]))
    uint8_t (*hash)[32]; // 第 i 段的 sha256 (python 算)
    uint8_t (*key)[32]; // 第 i 个缓存点的 key = sha256(hash[0] || ... || hash[i]) (python 算)
    int* role; // 第 i 段的角色 token id (<|im_start|> 后一个), 只用于 metadata
    int* npos; // 第 i 个缓存点之后的 RoPE 位置 (next_pos)
    uint8_t* exist; // 查找时已有效存在, 不再写
    uint8_t* st; // 要写 state
    int hit = -1; // 命中的缓存点 (-1 未命中)
    int pf_next = 0; // prefill 中下一个还没交给写盘线程的缓存点
    int last_state = 0; // 最后一个有 state 的位置 (512 规则)
    int out_on = 0; // 输出结束时 (<|im_end|>) 生成缓存点
    int out_start = 0, out_flushed = 0, n_segs = 0, out_role = -1;
    uint32_t out_id = 0;
    uint64_t out_off = 0; // 临时 kvcache.bin 已写的字节数
    int* segs; // 输出缓存点各段 token 数
    Sha::Ctx out_sha; // 已输入 prompt 末尾 (最后一个缓存点之后) 的序列化, 结束时再输入输出 token
    std::atomic<int> w_kv_in{0}, w_st_in{0}, w_st_out{0}; // 写盘线程实际写成功的: 输入部分的 KV token 数 / state 数, 输出部分的 state 数 (0 或 1)
};

enum : int { K_PF = 0, K_SEG = 1, K_FIN = 2, K_ABORT = 3 };

struct Item { // 写盘队列的一项, 由计算线程用 amap 分配, 写盘线程处理后释放
    Item* next;
    int kind, slot;
    Plan* P;
    int p0, p1; // K_PF: 缓存点下标 [p0, p1), state (若有) 属于 p1 - 1
    int a, b; // kv 覆盖的 token 范围
    uint8_t* kv; // 布局见 gather_seg, 大小 kv_cap
    size_t kv_cap;
    uint8_t* state; // G_state_bytes, 可为空
    uint64_t off; // K_SEG / K_FIN: 本段在临时 kvcache.bin 中的偏移
    int npos; // K_FIN: 输出缓存点的 next_pos
    uint8_t key[32]; // K_FIN: 输出缓存点的 key
    std::atomic<int>* pend; // 该槽未完成的写入数, 完成后 -1
    std::atomic<int>* pend_pf; // 只有 prompt 的项: 未完成的 prompt 写入数
    std::atomic<uint32_t>* ver; // 完成后 +1 并唤醒 (qwen_next 在上面等)
};

std::atomic<Item*> G_head{nullptr}; // 无锁栈, 写盘线程每次全部取走再反转成先进先出
std::atomic<uint32_t> G_qseq{0};

void push(Item* it) { // 任意线程
    Item* h = G_head.load();
    do it->next = h;
    while (!G_head.compare_exchange_weak(h, it));
    G_qseq.fetch_add(1);
    Sys::wake(&G_qseq);
}

void key_dir(const uint8_t* key, char* out) { // <dir>/ab/cd/<64 hex>
    char h[65];
    Sha::hex(key, h);
    snprintf(out, 2200, "%s/%.2s/%.2s/%s", G_dir, h, h + 2, h);
}

struct Meta {
    int start = 0, end = 0, next_pos = 0, has_state = 0, n_segs = 0;
};

const char* parse_int(const char* s, int& v) { // 跳过空格读一个非负整数, 返回之后的位置; 不用 strtol (会写 musl errno, 这里可能在 glibc 线程)
    while (*s == ' ') s++;
    v = -1;
    if (*s < '0' || *s > '9') return s;
    long x = 0;
    for (; *s >= '0' && *s <= '9' && x < (1L << 40); s++) x = x * 10 + (*s - '0');
    v = x > 0x7fffffff ? -1 : (int)x;
    return s;
}

bool meta_read(const char* dir, Meta& m, int* segs, char* buf) { // 读 dir/metadata.txt; 必须完整 (末行 END) 且模型一致才有效; segs 容量 G_max_segs, buf 容量 MBUF
    char p[2300];
    snprintf(p, sizeof(p), "%s/metadata.txt", dir);
    int fd = Sys::open(p, Sys::O_RD);
    if (fd < 0) return false;
    int64_t sz = Sys::fsize(fd);
    bool ok = sz > 4 && sz < (int64_t)MBUF && Sys::read_all(fd, buf, sz, 0);
    Sys::close(fd);
    if (!ok) return false;
    buf[sz] = 0;
    if (strncmp(buf, "QWEN_PREFIX_CACHE 1\n", 20) || strcmp(buf + sz - 4, "END\n")) return false;
    const char* mp = strstr(buf, "\nmodel=");
    size_t ml = strlen(G_model);
    if (!mp || strncmp(mp + 7, G_model, ml) || mp[7 + ml] != '\n') return false;
    auto num = [&](const char* k, int& v) {
        const char* q = strstr(buf, k);
        if (!q) return false;
        parse_int(q + strlen(k), v);
        return v >= 0;
    };
    if (!num("\nstart=", m.start) || !num("\nend=", m.end) || !num("\nnext_pos=", m.next_pos) || !num("\nhas_state=", m.has_state) || !num("\nkv_segments=", m.n_segs)) return false;
    if (m.n_segs > G_max_segs) return false;
    const char* q = parse_int(strstr(buf, "\nkv_segments=") + 13, m.n_segs);
    int tot = 0;
    for (int i = 0; i < m.n_segs; i++) {
        q = parse_int(q, segs[i]);
        if (segs[i] <= 0) return false;
        tot += segs[i];
    }
    return tot == m.end - m.start;
}

// ---------------- 写盘线程 ----------------

namespace WriterInternal {

uint8_t* G_wbuf = nullptr; // 拼段用的缓冲 (按需扩大)
size_t G_wcap = 0;
char* G_mbuf = nullptr; // metadata 文本
char* G_rbuf = nullptr; // 读 metadata
int* G_segtmp = nullptr;

uint8_t* wbuf(size_t n) {
    if (n > G_wcap) {
        Sys::unmap(G_wbuf, G_wcap);
        G_wcap = std::max(n, (size_t)64 << 20);
        G_wbuf = (uint8_t*)Sys::amap(G_wcap);
    }
    return G_wbuf;
}

void mkdirs(const char* dir) { // dir = <root>/ab/cd/<key>, 依次创建 ab, ab/cd, key (已存在不报错)
    char p[2300];
    size_t r = strlen(G_dir);
    snprintf(p, sizeof(p), "%s", dir);
    for (size_t i = r + 1; p[i]; i++) {
        if (p[i] != '/') continue;
        p[i] = 0;
        Sys::mkdir(p);
        p[i] = '/';
    }
    Sys::mkdir(p);
}

bool write_direct(const char* path, const void* buf, size_t n, uint64_t off, bool trunc) { // n、off、buf 都 4096 对齐
    int fd = Sys::open(path, Sys::O_WR | Sys::O_CREAT_ | Sys::O_DIRECT_ | (trunc ? Sys::O_TRUNC_ : 0));
    if (fd < 0) {
        fprintf(stderr, "⚠️ 缓存写入失败, 打开 %s: %d\n", path, -fd);
        return false;
    }
    bool ok = Sys::write_all(fd, buf, n, off);
    Sys::close(fd);
    if (!ok) fprintf(stderr, "⚠️ 缓存写入失败 %s (%zu 字节)\n", path, n);
    return ok;
}

void rm_point(const char* dir) { // 删掉缓存点目录 (只有已知的文件)
    const char* fs[] = {"metadata.txt", "kvcache.bin", "state.bin", "state.bin.compressed", "state.bin.compressed.tmp"};
    char p[2300];
    for (const char* f : fs) {
        snprintf(p, sizeof(p), "%s/%s", dir, f);
        Sys::unlink(p);
    }
    Sys::rmdir(dir);
}

const char* role_name(int id) {
    for (int i = 0; i < 4; i++)
        if (ROLE_IDS[i] == id) return ROLE_NAMES[i];
    return "unknown";
}

bool write_meta(const char* dir, const uint8_t* key, const char* type, int start, int end, int next_pos, int has_state, const int* segs, int n_segs, const Plan& P, int n_chain) { // 普通 IO; 前面所有缓存点 (P 的前 n_chain 个) 列在 chain
    char h[65];
    Sha::hex(key, h);
    size_t n = snprintf(G_mbuf, MBUF, "QWEN_PREFIX_CACHE 1\nkey=%s\nmodel=%s\ntype=%s\nstart=%d\nend=%d\nlength=%d\nnext_pos=%d\nhas_state=%d\ncreated=%ld\nkv_segments=%d", h, G_model, type, start, end, end - start, next_pos, has_state, (long)time(nullptr), n_segs);
    for (int i = 0; i < n_segs && n < MBUF - 200; i++) n += snprintf(G_mbuf + n, MBUF - n, " %d", segs[i]);
    n += snprintf(G_mbuf + n, MBUF - n, "\nchain=%d\n", n_chain);
    for (int i = 0; i < n_chain && n < MBUF - 200; i++) {
        Sha::hex(P.key[i], h);
        n += snprintf(G_mbuf + n, MBUF - n, "%s %d\n", h, P.pos[i]);
    }
    if (n >= MBUF - 200) {
        fprintf(stderr, "⚠️ metadata 太大, 不写缓存点 %s\n", dir);
        return false;
    }
    n += snprintf(G_mbuf + n, MBUF - n, "END\n");
    char p[2300];
    snprintf(p, sizeof(p), "%s/metadata.txt", dir);
    int fd = Sys::open(p, Sys::O_WR | Sys::O_CREAT_ | Sys::O_TRUNC_);
    bool ok = fd >= 0 && Sys::write_all(fd, G_mbuf, n, 0);
    if (fd >= 0) Sys::close(fd);
    if (!ok) fprintf(stderr, "⚠️ 写 %s 失败\n", p);
    return ok;
}

bool valid(const char* dir) {
    Meta m;
    return meta_read(dir, m, G_segtmp, G_rbuf);
}

void do_pf(Item* it) { // prompt 的若干缓存点: 每点 kvcache.bin (+ state.bin), 最后 metadata.txt
    Plan& P = *it->P;
    char dir[2300], p[2400], type[64];
    for (int j = it->p0; j < it->p1; j++) {
        if (P.exist[j]) continue;
        key_dir(P.key[j], dir);
        if (valid(dir)) continue; // 其他请求已经写过
        int s = j ? P.pos[j - 1] : 0, e = P.pos[j], m = e - s;
        mkdirs(dir);
        uint8_t* b = wbuf(seg_bytes(m));
        memset(b + (size_t)m * TOK_BYTES, 0, seg_bytes(m) - (size_t)m * TOK_BYTES);
        gather_seg(it->kv, it->b - it->a, s - it->a, e - it->a, b);
        snprintf(p, sizeof(p), "%s/kvcache.bin", dir);
        if (!write_direct(p, b, seg_bytes(m), 0, true)) continue;
        int hs = it->state && j == it->p1 - 1 && P.st[j];
        snprintf(p, sizeof(p), "%s/state.bin", dir);
        if (hs && !write_direct(p, it->state, G_state_bytes, 0, true)) continue;
        snprintf(type, sizeof(type), "%s prompt", role_name(P.role[j]));
        if (!write_meta(dir, P.key[j], type, s, e, P.npos[j], hs, &m, 1, P, j)) continue;
        P.w_kv_in += m;
        P.w_st_in += hs;
    }
}

void out_tmp(const Item* it, char* out) { snprintf(out, 2300, "%s/tmp/s%d_%u", G_dir, it->slot, it->P->out_id); }

void do_seg(Item* it) { // 输出的一段 KV 追加到临时目录的 kvcache.bin
    char dir[2300], p[2400];
    out_tmp(it, dir);
    if (it->off == 0) Sys::mkdir(dir);
    int m = it->b - it->a;
    if (m <= 0) return;
    uint8_t* b = wbuf(seg_bytes(m));
    memset(b + (size_t)m * TOK_BYTES, 0, seg_bytes(m) - (size_t)m * TOK_BYTES);
    gather_seg(it->kv, m, 0, m, b);
    snprintf(p, sizeof(p), "%s/kvcache.bin", dir);
    write_direct(p, b, seg_bytes(m), it->off, it->off == 0);
}

void do_fin(Item* it) { // 输出结束: 最后一段 + state + metadata 写到临时目录, 再整个目录改名到 key 的位置
    Plan& P = *it->P;
    char tmp[2300], dir[2300], p[2400], type[64];
    do_seg(it);
    out_tmp(it, tmp);
    snprintf(p, sizeof(p), "%s/state.bin", tmp);
    if (it->state && !write_direct(p, it->state, G_state_bytes, 0, true)) {
        rm_point(tmp);
        return;
    }
    snprintf(type, sizeof(type), "%s output", role_name(P.out_role));
    int end = P.out_start;
    for (int i = 0; i < P.n_segs; i++) end += P.segs[i];
    if (!write_meta(tmp, it->key, type, P.out_start, end, it->npos, it->state != nullptr, P.segs, P.n_segs, P, P.n_pts)) {
        rm_point(tmp);
        return;
    }
    key_dir(it->key, dir);
    if (valid(dir)) { // 同样的输出之前已经缓存过
        rm_point(tmp);
        return;
    }
    rm_point(dir); // 可能有不完整的残留
    mkdirs(dir);
    Sys::rmdir(dir);
    int r = Sys::rename(tmp, dir);
    if (r < 0) {
        fprintf(stderr, "⚠️ 缓存目录改名失败 %s -> %s: %d\n", tmp, dir, -r);
        rm_point(tmp);
        return;
    }
    P.w_st_out = it->state != nullptr;
}

void done(Item* it) {
    if (it->pend_pf) it->pend_pf->fetch_sub(1);
    it->pend->fetch_sub(1);
    it->ver->fetch_add(1);
    Sys::wake(it->ver);
    Sys::unmap(it->kv, it->kv_cap);
    Sys::unmap(it->state, G_state_bytes);
    Sys::unmap(it, sizeof(Item));
}

void writer_main() {
    G_mbuf = (char*)Sys::amap(MBUF);
    G_rbuf = (char*)Sys::amap(MBUF + 1);
    G_segtmp = (int*)Sys::amap((size_t)G_max_segs * 4);
    while (true) {
        uint32_t seq = G_qseq.load();
        Item* h = G_head.exchange(nullptr);
        if (!h) {
            Sys::wait(&G_qseq, seq);
            continue;
        }
        Item* r = nullptr; // 反转成先进先出
        while (h) {
            Item* nx = h->next;
            h->next = r;
            r = h;
            h = nx;
        }
        while (r) {
            Item* nx = r->next;
            auto t0 = std::chrono::steady_clock::now();
            if (r->kind == K_PF) do_pf(r);
            else if (r->kind == K_SEG) do_seg(r);
            else if (r->kind == K_FIN) do_fin(r);
            else {
                char tmp[2300];
                out_tmp(r, tmp);
                rm_point(tmp);
            }
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (r->kind == K_PF || r->kind == K_FIN) fprintf(stderr, "ℹ️ 缓存写入 槽 %d %s: token [%d, %d)%s, %.3fs\n", r->slot, r->kind == K_PF ? "输入" : "输出", r->kind == K_PF ? (r->p0 ? r->P->pos[r->p0 - 1] : 0) : r->P->out_start, r->b, r->state ? " + state" : "", sec);
            done(r);
            r = nx;
        }
    }
}

// ---------------- 量化线程 ----------------

void quant_point(const char* dir, uint8_t* sbuf, uint8_t* cbuf, size_t ccap) { // state.bin -> state.bin.compressed (先写 .tmp 再改名, 然后删除 state.bin)
    char p[2400], q[2400];
    snprintf(p, sizeof(p), "%s/state.bin", dir);
    int fd = Sys::open(p, Sys::O_RD | Sys::O_DIRECT_);
    if (fd < 0) return;
    bool ok = Sys::fsize(fd) == (int64_t)G_state_bytes && Sys::read_all(fd, sbuf, G_state_bytes, 0);
    Sys::close(fd);
    if (!ok) {
        fprintf(stderr, "⚠️ 读取 %s 失败, 跳过量化\n", p);
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    memset(cbuf, 0, ccap);
    size_t n = Codec::encode((const float*)sbuf, G_state_bytes / 4, cbuf, ccap, G_qthreads);
    if (!n) {
        fprintf(stderr, "⚠️ %s 的 0 太多, 不量化\n", p);
        return;
    }
    snprintf(q, sizeof(q), "%s/state.bin.compressed.tmp", dir);
    if (!write_direct(q, cbuf, n, 0, true)) return;
    snprintf(p, sizeof(p), "%s/state.bin.compressed", dir);
    if (Sys::rename(q, p) < 0) return;
    snprintf(p, sizeof(p), "%s/state.bin", dir);
    Sys::unlink(p);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "ℹ️ 缓存 state 已量化 %s: %.1fMB -> %.1fMB, %.2fs\n", strrchr(dir, '/') + 1, G_state_bytes / 1048576.0, n / 1048576.0, sec);
}

bool is_hex(const char* s, size_t n) {
    if (strlen(s) != n) return false;
    for (size_t i = 0; i < n; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false;
    return true;
}

void quant_main() { // 常驻: 每 5 秒扫一遍 <dir>/ab/cd/<key>, 有效且只有 state.bin 的量化
    char* rbuf = (char*)Sys::amap(MBUF + 1);
    int* segs = (int*)Sys::amap((size_t)G_max_segs * 4);
    uint8_t* sbuf = (uint8_t*)Sys::amap(G_state_bytes);
    size_t ccap = Codec::comp_bytes(G_state_bytes / 4, G_state_bytes / 4 / 64); // 0 超过 1/64 时不量化
    uint8_t* cbuf = (uint8_t*)Sys::amap(ccap);
    char p1[2300], p2[2300], p3[2300], p[2400];
    while (true) {
        Sys::sleep_ms(5000);
        DIR* d1 = opendir(G_dir);
        if (!d1) continue;
        for (dirent* e1; (e1 = readdir(d1));) {
            if (!is_hex(e1->d_name, 2)) continue;
            snprintf(p1, sizeof(p1), "%s/%s", G_dir, e1->d_name);
            DIR* d2 = opendir(p1);
            if (!d2) continue;
            for (dirent* e2; (e2 = readdir(d2));) {
                if (!is_hex(e2->d_name, 2)) continue;
                snprintf(p2, sizeof(p2), "%s/%s", p1, e2->d_name);
                DIR* d3 = opendir(p2);
                if (!d3) continue;
                for (dirent* e3; (e3 = readdir(d3));) {
                    if (!is_hex(e3->d_name, 64)) continue;
                    snprintf(p3, sizeof(p3), "%s/%s", p2, e3->d_name);
                    snprintf(p, sizeof(p), "%s/state.bin", p3);
                    if (Sys::access(p)) continue;
                    snprintf(p, sizeof(p), "%s/state.bin.compressed", p3);
                    if (!Sys::access(p)) continue;
                    Meta m;
                    if (!meta_read(p3, m, segs, rbuf)) continue; // 还没写完 (或无效)
                    quant_point(p3, sbuf, cbuf, ccap);
                }
                closedir(d3);
            }
            closedir(d2);
        }
        closedir(d1);
    }
}

} // namespace WriterInternal

bool init(const char* dir, const char* model, size_t state_bytes, int max_ctx, int qthreads) { // 在 python 线程调用一次: 检查 direct IO 可用, 启动写盘线程和量化线程
    snprintf(G_dir, sizeof(G_dir), "%s", dir);
    snprintf(G_model, sizeof(G_model), "%s", model);
    G_state_bytes = state_bytes;
    G_qthreads = qthreads;
    G_max_segs = max_ctx / FLUSH + 2;
    char p[2300];
    snprintf(p, sizeof(p), "%s/tmp/direct_io_probe", G_dir);
    uint8_t* b = (uint8_t*)Sys::amap(AL);
    memset(b, 0x5a, AL);
    int fd = Sys::open(p, Sys::O_WR | Sys::O_CREAT_ | Sys::O_TRUNC_ | Sys::O_DIRECT_);
    bool ok = fd >= 0 && Sys::write_all(fd, b, AL, 0);
    if (fd >= 0) Sys::close(fd);
    memset(b, 0, AL);
    fd = Sys::open(p, Sys::O_RD | Sys::O_DIRECT_);
    ok = ok && fd >= 0 && Sys::read_all(fd, b, AL, 0) && b[0] == 0x5a && b[AL - 1] == 0x5a;
    if (fd >= 0) Sys::close(fd);
    Sys::unlink(p);
    Sys::unmap(b, AL);
    if (!ok) {
        fprintf(stderr, "❌ 缓存目录 %s 不支持 direct IO (O_DIRECT)\n", G_dir);
        G_dir[0] = 0;
        return false;
    }
    std::thread(WriterInternal::writer_main).detach();
    std::thread(WriterInternal::quant_main).detach();
    return true;
}

} // namespace Cache
} // namespace

#endif
