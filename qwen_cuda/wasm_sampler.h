#ifndef QWEN_CUDA_WASM_SAMPLER_H
#define QWEN_CUDA_WASM_SAMPLER_H

// 自定义采样: 请求带来的 wasm (用户 C++ 代码由 server 编译, 接口见 wasm_sampler/prelude.h) 在 WAMR 解释器里执行, 每个请求一个新实例
// 实例的生命周期: 加载 -> _initialize (全局构造) -> sampler_init(is_thinking, custom_data, len) 一次 -> 每步 sample(fp16 logits) 一次; 下一个请求前卸载
// 每个槽一个 worker 线程 (本 so 创建的 musl 线程), 该槽的 WAMR 调用都在它里面执行: 调用方 (prefill / decode) 是 python 的 glibc 线程, 不能用 musl 的 malloc / errno / 锁
// 调用方把任务写进槽的 Job, futex 唤醒 worker; 采样可以先提交再等 (decode 时各槽的 wasm 并行执行); 不同槽的实例互不相关, WAMR 内存池分配有锁
// 第一次加载模块时才初始化 WAMR 和启动线程, 不用自定义采样就不占任何资源; 编译时没有 WAMR (QWEN_WAMR 未定义) 时接口照常存在, 加载直接报错
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef QWEN_WAMR
#include <immintrin.h>
#include <pthread.h>
#include "wasm_export.h"
#endif

namespace {

namespace WasmSampler {

const int MAXS = 8; // 槽数上限
const int VOCAB = 248320;

#ifdef QWEN_WAMR

const uint32_t WASM_STACK = 1 << 20; // 每个实例的 wasm 栈 (WAMR exec env)
const int INSN_LIMIT = 2000000000; // 每次调用 (全局构造 / sampler_init / sample) 最多执行的 wasm 指令数, 超过即报错, 防止死循环卡住 decode
const uint32_t POOL_BYTES = 1u << 30; // WAMR 内部分配的内存池, mmap 后按需占用物理内存

struct Job {
    std::atomic<uint32_t> seq{0}, done{0}; // 调用方 seq + 1 提交 (worker 在 seq 上等), worker 做完后 done = seq (调用方在 done 上等)
    int op = 0; // 1: 卸载旧模块并加载新模块 (含 sampler_init), 2: 采样, 3: 卸载
    const uint8_t* wasm = nullptr; // op 1: 只在调用期间有效, worker 会复制
    uint32_t n_wasm = 0;
    const uint8_t* cdata = nullptr; // op 1: custom_data, 复制进线性内存
    uint64_t n_cdata = 0;
    int thinking = 0; // op 1
    const float* logits = nullptr; // op 2: [VOCAB] fp32, 转成 fp16 交给 sample
    int ret = 0; // op 2: sample 的返回值
    bool ok = false; // 是否成功, 失败时 err 是原因
    char err[512] = {};
    // 以下只有 worker 访问
    uint8_t* buf = nullptr;
    size_t buf_cap = 0;
    wasm_module_t mod = nullptr;
    wasm_module_inst_t inst = nullptr;
    wasm_exec_env_t env = nullptr;
    wasm_function_inst_t f_sample = nullptr;
    uint64_t a_logits = 0; // logits 缓冲在线性内存里的偏移 (memory.grow 后 native 地址可能变, 每次重新换算)
};
Job G_jobs[MAXS];
std::atomic<uint32_t> G_ready{0}; // 0 未启动, 1 可用, 2 初始化失败; 只在 prefill 里 (按 ticket 串行) 启动, 不会并发
char G_init_err[256] = {};

namespace WasmSamplerInternal {

inline long sc(long n, long a1, long a2, long a3, long a4 = 0, long a5 = 0, long a6 = 0) { // raw syscall, 调用方线程里不碰 musl 的 errno
    long r;
    register long r10 __asm__("r10") = a4;
    register long r8 __asm__("r8") = a5;
    register long r9 __asm__("r9") = a6;
    __asm__ volatile("syscall" : "=a"(r) : "0"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9) : "rcx", "r11", "memory");
    return r;
}
inline void fwait(std::atomic<uint32_t>& a, uint32_t old) { sc(202, (long)&a, 128, old, 0); } // FUTEX_WAIT_PRIVATE
inline void fwake(std::atomic<uint32_t>& a) { sc(202, (long)&a, 129, 0x7fffffff); } // FUTEX_WAKE_PRIVATE
inline void* mmap_anon(size_t n) {
    long r = sc(9, 0, (long)n, 3, 0x22, -1, 0); // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS
    return r < 0 && r > -4096 ? nullptr : (void*)r;
}

__attribute__((target("avx,f16c"))) void to_fp16(const float* src, uint16_t* dst, int n) { // n 是 8 的倍数; 就近舍入 (RNE)
    for (int i = 0; i < n; i += 8) _mm_storeu_si128((__m128i*)(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
}

void unload(Job& J) {
    if (J.env) wasm_runtime_destroy_exec_env(J.env);
    if (J.inst) wasm_runtime_deinstantiate(J.inst);
    if (J.mod) wasm_runtime_unload(J.mod);
    J.env = nullptr, J.inst = nullptr, J.mod = nullptr, J.f_sample = nullptr;
}

bool call(Job& J, wasm_function_inst_t f, int n_args, wasm_val_t* args, wasm_val_t* res, const char* what) { // 结果最多 1 个 (res 为空表示没有返回值)
    wasm_runtime_set_instruction_count_limit(J.env, INSN_LIMIT);
    if (wasm_runtime_call_wasm_a(J.env, f, res ? 1 : 0, res, n_args, args)) return true;
    const char* e = wasm_runtime_get_exception(J.inst);
    snprintf(J.err, sizeof(J.err), "%s 执行出错: %s", what, e ? e : "未知错误");
    wasm_runtime_clear_exception(J.inst);
    return false;
}

wasm_function_inst_t func(Job& J, const char* name) {
    wasm_function_inst_t f = wasm_runtime_lookup_function(J.inst, name);
    if (!f) snprintf(J.err, sizeof(J.err), "模块缺少导出函数 %s", name);
    return f;
}

bool check_addr(Job& J, uint64_t a, uint64_t bytes, const char* what) { // 整个缓冲区都要在线性内存内
    if (wasm_runtime_validate_app_addr(J.inst, a, bytes)) return true;
    wasm_runtime_clear_exception(J.inst);
    snprintf(J.err, sizeof(J.err), "%s 的地址 %llu (%llu 字节) 不在线性内存内", what, (unsigned long long)a, (unsigned long long)bytes);
    return false;
}

bool load(Job& J) {
    unload(J);
    if (J.buf_cap < J.n_wasm) {
        if (J.buf) sc(11, (long)J.buf, (long)J.buf_cap, 0);
        J.buf_cap = (J.n_wasm + 4095) & ~(size_t)4095;
        J.buf = (uint8_t*)mmap_anon(J.buf_cap);
        if (!J.buf) {
            J.buf_cap = 0;
            snprintf(J.err, sizeof(J.err), "分配 %u 字节失败", J.n_wasm);
            return false;
        }
    }
    memcpy(J.buf, J.wasm, J.n_wasm); // WAMR 要求模块存续期间 buffer 一直有效
    J.mod = wasm_runtime_load(J.buf, J.n_wasm, J.err, sizeof(J.err));
    if (!J.mod) return false;
    int n_imp = wasm_runtime_get_import_count(J.mod);
    if (n_imp) {
        wasm_import_t imp;
        wasm_runtime_get_import_type(J.mod, 0, &imp);
        snprintf(J.err, sizeof(J.err), "模块有 %d 个导入 (第一个 %s.%s), 不允许调用任何外部函数", n_imp, imp.module_name, imp.name);
        return false;
    }
    J.inst = wasm_runtime_instantiate(J.mod, WASM_STACK, 0, J.err, sizeof(J.err));
    if (!J.inst) return false;
    J.env = wasm_runtime_create_exec_env(J.inst, WASM_STACK);
    if (!J.env) {
        snprintf(J.err, sizeof(J.err), "创建 exec env 失败");
        return false;
    }
    wasm_function_inst_t f_start = wasm_runtime_lookup_function(J.inst, "_initialize"); // reactor 模块: 全局构造函数
    if (f_start && !call(J, f_start, 0, nullptr, nullptr, "全局构造 (_initialize)")) return false;
    wasm_function_inst_t f_ptr = func(J, "qs_logits_ptr"), f_alloc = func(J, "qs_alloc"), f_init = func(J, "qs_init");
    J.f_sample = func(J, "qs_sample");
    if (!f_ptr || !f_alloc || !f_init || !J.f_sample) return false;
    wasm_val_t r[1], a[3];
    if (!call(J, f_ptr, 0, nullptr, r, "qs_logits_ptr")) return false;
    J.a_logits = (uint32_t)r[0].of.i32;
    if (!check_addr(J, J.a_logits, (uint64_t)VOCAB * 2, "logits 缓冲")) return false;
    uint32_t a_data = 0;
    if (J.n_cdata) {
        if (J.n_cdata > 0x7fffffff) {
            snprintf(J.err, sizeof(J.err), "custom_data %llu 字节, 超过 wasm32 能放下的大小", (unsigned long long)J.n_cdata);
            return false;
        }
        a[0].kind = WASM_I32, a[0].of.i32 = (int32_t)J.n_cdata;
        if (!call(J, f_alloc, 1, a, r, "qs_alloc")) return false;
        a_data = (uint32_t)r[0].of.i32;
        if (!a_data) {
            snprintf(J.err, sizeof(J.err), "线性内存放不下 custom_data (%llu 字节)", (unsigned long long)J.n_cdata);
            return false;
        }
        if (!check_addr(J, a_data, J.n_cdata, "custom_data")) return false;
        memcpy(wasm_runtime_addr_app_to_native(J.inst, a_data), J.cdata, J.n_cdata);
    }
    a[0].kind = WASM_I32, a[0].of.i32 = J.thinking;
    a[1].kind = WASM_I32, a[1].of.i32 = (int32_t)a_data; // 没有 custom_data 时为 0 (nullptr)
    a[2].kind = WASM_I64, a[2].of.i64 = (int64_t)J.n_cdata;
    return call(J, f_init, 3, a, nullptr, "sampler_init");
}

bool sample(Job& J) {
    if (!J.f_sample) {
        snprintf(J.err, sizeof(J.err), "没有加载采样模块");
        return false;
    }
    if (!check_addr(J, J.a_logits, (uint64_t)VOCAB * 2, "logits 缓冲")) return false;
    to_fp16(J.logits, (uint16_t*)wasm_runtime_addr_app_to_native(J.inst, J.a_logits), VOCAB); // 用户代码可能 memory.grow, 每次重新换算地址
    wasm_val_t r[1];
    if (!call(J, J.f_sample, 0, nullptr, r, "sample")) return false;
    J.ret = r[0].of.i32;
    return true;
}

void* worker_main(void* arg) { // 槽 (intptr_t)arg 的 worker; 槽 0 的 worker 负责初始化 WAMR, 其余等它
    Job& J = G_jobs[(intptr_t)arg];
    if (!arg) {
        RuntimeInitArgs args;
        memset(&args, 0, sizeof(args));
        void* pool = mmap_anon(POOL_BYTES);
        args.mem_alloc_type = Alloc_With_Pool;
        args.mem_alloc_option.pool.heap_buf = pool;
        args.mem_alloc_option.pool.heap_size = POOL_BYTES;
        args.running_mode = Mode_Interp;
        bool ok = pool && wasm_runtime_full_init(&args);
        if (ok) wasm_runtime_set_log_level(WASM_LOG_LEVEL_ERROR);
        else snprintf(G_init_err, sizeof(G_init_err), "WAMR 初始化失败");
        G_ready.store(ok ? 1 : 2);
        fwake(G_ready);
    }
    for (uint32_t r; (r = G_ready.load()) == 0;) fwait(G_ready, r);
    if (G_ready.load() != 1) return nullptr;
    while (true) {
        uint32_t s = J.seq.load();
        if (s == J.done.load()) {
            fwait(J.seq, s);
            continue;
        }
        J.err[0] = 0;
        if (J.op == 1) J.ok = load(J);
        else if (J.op == 2) J.ok = sample(J);
        else unload(J), J.ok = true;
        if (J.op == 1 && !J.ok) unload(J);
        J.done.store(s);
        fwake(J.done);
    }
}

bool start() { // 第一次加载时调用: 启动每个槽的 worker 并等 WAMR 初始化完成
    if (G_ready.load()) return G_ready.load() == 1;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 16 << 20); // WAMR 解释器按 native 栈边界检查栈溢出, musl 默认线程栈只有 128KB
    for (intptr_t i = 0; i < MAXS; i++) {
        pthread_t th;
        if (pthread_create(&th, &at, worker_main, (void*)i)) {
            snprintf(G_init_err, sizeof(G_init_err), "wasm 采样线程创建失败");
            G_ready.store(2);
            fwake(G_ready);
            break;
        }
        pthread_detach(th);
    }
    for (uint32_t r; !(r = G_ready.load());) fwait(G_ready, r);
    if (G_ready.load() == 1) fprintf(stderr, "✅ wasm 自定义采样运行时 (WAMR) 已启动, %d 个线程\n", MAXS);
    else fprintf(stderr, "❌ %s, 自定义采样不可用\n", G_init_err);
    return G_ready.load() == 1;
}

void post(int sq, int op) { // 在调用方线程里提交任务, 不等
    Job& J = G_jobs[sq];
    J.op = op;
    J.seq.store(J.seq.load() + 1);
    fwake(J.seq);
}

bool wait(int sq) { // 等槽 sq 最近一次提交的任务做完
    Job& J = G_jobs[sq];
    for (uint32_t d; (d = J.done.load()) != J.seq.load();) fwait(J.done, d);
    return J.ok;
}

} // namespace WasmSamplerInternal

bool load(int sq, const uint8_t* wasm, uint32_t n, const uint8_t* cdata, uint64_t n_cdata, bool thinking) { // 新实例 + sampler_init; 失败时 error(sq) 是原因
    Job& J = G_jobs[sq];
    if (!WasmSamplerInternal::start()) {
        snprintf(J.err, sizeof(J.err), "wasm 运行时不可用: %s", G_init_err);
        return false;
    }
    J.wasm = wasm, J.n_wasm = n, J.cdata = cdata, J.n_cdata = n_cdata, J.thinking = thinking;
    WasmSamplerInternal::post(sq, 1);
    return WasmSamplerInternal::wait(sq);
}

void sample_post(int sq, const float* logits) { // 提交采样, 不等 (logits 在 sample_wait 返回前保持不变)
    G_jobs[sq].logits = logits;
    WasmSamplerInternal::post(sq, 2);
}

bool sample_wait(int sq, int& tok) { // tok: sample 的返回值 (未检查是否合法); 失败时 error(sq) 是原因
    bool ok = WasmSamplerInternal::wait(sq);
    tok = G_jobs[sq].ret;
    return ok;
}

void unload(int sq) {
    if (G_ready.load() != 1) return;
    WasmSamplerInternal::post(sq, 3);
    WasmSamplerInternal::wait(sq);
}

const char* error(int sq) { return G_jobs[sq].err; }

#else // 编译时没有 WAMR: 接口照常存在, 只有真正加载时报错

char G_err[MAXS][128];

bool load(int sq, const uint8_t*, uint32_t, const uint8_t*, uint64_t, bool) {
    snprintf(G_err[sq], sizeof(G_err[sq]), "本 so 编译时没有 WAMR (未设置 WAMR_DIR), 不支持自定义采样");
    return false;
}
void sample_post(int, const float*) {}
bool sample_wait(int sq, int& tok) {
    tok = -1;
    snprintf(G_err[sq], sizeof(G_err[sq]), "本 so 编译时没有 WAMR, 不支持自定义采样");
    return false;
}
void unload(int) {}
const char* error(int sq) { return G_err[sq]; }

#endif

} // namespace WasmSampler

} // namespace

#endif
