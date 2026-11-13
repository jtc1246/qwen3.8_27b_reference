// 自定义采样 (wasm) 的预置头文件: 服务端编译用户代码时用 -include 强制包含在最前面, 用户只需实现 sampler_init 和 sample
// 用户代码由 wasi-sdk 编译成 wasm32, 不允许任何导入 (不能调用 printf 等需要 WASI 的函数), 在服务端 WAMR 解释器里运行
// 每个请求一个新实例: 先调用一次 sampler_init, 之后每生成一个 token 调用一次 sample; 全局变量在一个请求内保持, 不跨请求
#ifndef QS_PRELUDE_H
#define QS_PRELUDE_H
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <new>

// libc++ 的断言失败 / operator new 默认实现会经 fprintf(stderr) 引入 WASI 导入, 这里替换成直接 trap (运行时报 unreachable)
namespace std { inline namespace _LIBCPP_ABI_NAMESPACE { // 不用 _LIBCPP_BEGIN_NAMESPACE_STD: 它会给函数加 abi_tag / hidden, 与 <__verbose_abort> 里的声明不一致
[[noreturn]] _LIBCPP_OVERRIDABLE_FUNC_VIS void __libcpp_verbose_abort(const char*, ...) noexcept { __builtin_trap(); }
} }
void* operator new(size_t n) {
    void* p = malloc(n ? n : 1);
    if (!p) __builtin_trap();
    return p;
}
void* operator new[](size_t n) { return operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept { return malloc(n ? n : 1); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return malloc(n ? n : 1); }
void* operator new(size_t n, std::align_val_t a) {
    void* p = aligned_alloc((size_t)a, (n + (size_t)a - 1) / (size_t)a * (size_t)a + (size_t)a * !n); // aligned_alloc 要求大小是对齐的整数倍
    if (!p) __builtin_trap();
    return p;
}
void* operator new[](size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }
void operator delete(void* p, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { free(p); }

const int QS_VOCAB = 248320; // logits 个数, 含控制 token 和输出层填充位
const int QS_N_TEXT = 248044; // 普通文本 token 是 [0, QS_N_TEXT)
const int QS_IM_END = 248046, QS_THINK_END = 248069;

// 返回值的合法性 (服务端检查, 不合法则请求报错): 普通文本; 思考中 (thinking 模式且还没返回过 QS_THINK_END) 是 QS_THINK_END (结束思考), 否则是 QS_IM_END (结束本次生成); 思考中不能直接返回 QS_IM_END, 其余控制 token (包括 endoftext 248044) 都不允许
// 何时结束思考、何时结束生成完全由用户决定, 没有额外的结束信号; 不返回结束 token 就一直生成到上下文用完
inline bool qs_allowed(int id, bool in_thinking) { return (id >= 0 && id < QS_N_TEXT) || id == (in_thinking ? QS_THINK_END : QS_IM_END); }
inline float qs_half_to_float(uint16_t h) { // IEEE fp16 -> float (含非规格化数、inf、nan)
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 31) f = s | 0x7f800000 | (m << 13);
    else if (e) f = s | ((e + 112) << 23) | (m << 13);
    else if (!m) f = s;
    else {
        for (e = 113; !(m & 0x400); e--) m <<= 1; // 非规格化数: 规格化后指数相应减小
        f = s | (e << 23) | ((m & 0x3ff) << 13);
    }
    return __builtin_bit_cast(float, f);
}

// 用户实现:
// sampler_init: 每个请求开始时调用一次; is_thinking_mode: 本次是否开了 thinking (生成从思考中开始); custom_data: 请求的 sampler_data (base64 解码后), 没有时为 nullptr / 0, 整个请求内有效
// sample: 每步调用一次, data 是本步的 QS_VOCAB 个 logits (fp16 二进制, 小端), 返回本步的 token
void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len);
int sample(void* data);

uint16_t qs_logits[QS_VOCAB];

extern "C" {
__attribute__((export_name("qs_logits_ptr"))) uint16_t* qs_logits_ptr() { return qs_logits; }
__attribute__((export_name("qs_alloc"))) void* qs_alloc(uint32_t n) { return malloc(n); } // 服务端用它在线性内存里放 custom_data
__attribute__((export_name("qs_init"))) void qs_init(int thinking, void* p, uint64_t n) { sampler_init(thinking != 0, p, n); }
__attribute__((export_name("qs_sample"))) int qs_sample() { return sample(qs_logits); }
}

#endif
