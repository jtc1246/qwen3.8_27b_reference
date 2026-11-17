#ifndef QWEN_NPU_ACL_H
#define QWEN_NPU_ACL_H

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdio>
#include <cstdlib>

namespace {
namespace Acl { // AscendCL 运行时薄封装 (支持多设备, 每线程绑定一个), 出错直接退出进程

aclrtStream G_streams[2] = {nullptr, nullptr};
aclrtContext G_ctxs[2] = {nullptr, nullptr};
thread_local int G_di = 0; // 当前线程绑定的设备序号 (0..ndev-1)

void ck(aclError r, const char* what) {
    if (r == ACL_SUCCESS) return;
    fprintf(stderr, "❌ ACL %s 失败: %d\n", what, (int)r);
    _Exit(1);
}

void init(int device, int ndev) { // 使用设备 device..device+ndev-1
    ck(aclInit(nullptr), "aclInit");
    for (int d = 0; d < ndev; d++) {
        ck(aclrtSetDevice(device + d), "aclrtSetDevice");
        ck(aclrtCreateContext(&G_ctxs[d], device + d), "aclrtCreateContext");
        // FAST_SYNC: 同步走忙等而非中断唤醒 (阻塞 sync 实测 ~95µs 唤醒延迟); FAST_LAUNCH: 快速下发
        ck(aclrtCreateStreamWithConfig(&G_streams[d], 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC), "aclrtCreateStreamWithConfig");
        int32_t major = 0, minor = 0, patch = 0;
        aclrtGetVersion(&major, &minor, &patch);
        const char* soc = aclrtGetSocName();
        fprintf(stderr, "ℹ️ NPU %d: %s, ACL %d.%d.%d\n", device + d, soc ? soc : "?", major, minor, patch);
    }
}

void bind(int di) { // python 可能从不同线程调用, 每次入口重新绑定
    G_di = di;
    ck(aclrtSetCurrentContext(G_ctxs[di]), "aclrtSetCurrentContext");
}

aclrtStream stream() { return G_streams[G_di]; }

void* alloc(size_t bytes) { // 分配并清零 (当前线程绑定的设备)
    void* p = nullptr;
    ck(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    ck(aclrtMemset(p, bytes, 0, bytes), "aclrtMemset");
    return p;
}

void h2d(void* dst, const void* src, size_t bytes) { ck(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D"); }
void d2h(void* dst, const void* src, size_t bytes) { ck(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H"); }
void sync() { ck(aclrtSynchronizeStream(stream()), "aclrtSynchronizeStream"); }

} // namespace Acl
} // namespace

#endif
