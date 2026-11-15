#ifndef QWEN_NPU_ACL_H
#define QWEN_NPU_ACL_H

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdio>
#include <cstdlib>

namespace {
namespace Acl { // AscendCL 运行时薄封装, 出错直接退出进程

aclrtStream G_stream = nullptr;
aclrtContext G_ctx = nullptr;

void ck(aclError r, const char* what) {
    if (r == ACL_SUCCESS) return;
    fprintf(stderr, "❌ ACL %s 失败: %d\n", what, (int)r);
    _Exit(1);
}

void init(int device) {
    ck(aclInit(nullptr), "aclInit");
    ck(aclrtSetDevice(device), "aclrtSetDevice");
    ck(aclrtCreateContext(&G_ctx, device), "aclrtCreateContext");
    ck(aclrtCreateStream(&G_stream), "aclrtCreateStream");
    int32_t major = 0, minor = 0, patch = 0;
    aclrtGetVersion(&major, &minor, &patch);
    const char* soc = aclrtGetSocName();
    fprintf(stderr, "ℹ️ NPU %d: %s, ACL %d.%d.%d\n", device, soc ? soc : "?", major, minor, patch);
}

void bind() { ck(aclrtSetCurrentContext(G_ctx), "aclrtSetCurrentContext"); } // python 可能从不同线程调用

void* alloc(size_t bytes) { // 分配并清零
    void* p = nullptr;
    ck(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    ck(aclrtMemset(p, bytes, 0, bytes), "aclrtMemset");
    return p;
}

void h2d(void* dst, const void* src, size_t bytes) { ck(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D"); }
void d2h(void* dst, const void* src, size_t bytes) { ck(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H"); }
void sync() { ck(aclrtSynchronizeStream(G_stream), "aclrtSynchronizeStream"); }

} // namespace Acl
} // namespace

#endif
