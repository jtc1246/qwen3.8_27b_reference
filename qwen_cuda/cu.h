#ifndef QWEN_CUDA_CU_H
#define QWEN_CUDA_CU_H

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

namespace {
namespace Cu { // CUDA driver API 的薄封装, 出错直接退出进程

CUcontext G_ctx = nullptr;
CUmodule G_mod = nullptr;

void ck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    fprintf(stderr, "❌ CUDA %s 失败: %d %s\n", what, (int)r, s ? s : "?");
    _Exit(1);
}

void init(int device, const void* image) {
    CUdevice dev;
    char name[256] = {};
    int ver = 0, cc_major = 0, cc_minor = 0;
    ck(cuInit(0), "cuInit");
    ck(cuDriverGetVersion(&ver), "cuDriverGetVersion");
    ck(cuDeviceGet(&dev, device), "cuDeviceGet");
    ck(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    ck(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cuDeviceGetAttribute");
    ck(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cuDeviceGetAttribute");
    ck(cuDevicePrimaryCtxRetain(&G_ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(cuCtxSetCurrent(G_ctx), "cuCtxSetCurrent");
    ck(cuModuleLoadData(&G_mod, image), "cuModuleLoadData");
    fprintf(stderr, "ℹ️ GPU %d: %s, sm_%d%d, 驱动 CUDA %d.%d\n", device, name, cc_major, cc_minor, ver / 1000, ver % 1000 / 10);
}

void bind() { ck(cuCtxSetCurrent(G_ctx), "cuCtxSetCurrent"); } // context 是每线程的, python 可能从不同线程调用

CUfunction func(const char* name) {
    CUfunction f;
    ck(cuModuleGetFunction(&f, G_mod, name), name);
    return f;
}

CUdeviceptr alloc(size_t bytes) { // 分配并清零
    CUdeviceptr p;
    ck(cuMemAlloc(&p, bytes), "cuMemAlloc");
    ck(cuMemsetD8(p, 0, bytes), "cuMemsetD8");
    return p;
}

void h2d(CUdeviceptr dst, const void* src, size_t bytes) { ck(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD"); }
void d2h(void* dst, CUdeviceptr src, size_t bytes) { ck(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH"); }

template <class... A> void launch(CUfunction f, unsigned gx, unsigned gy, unsigned block, A... args) { // 参数类型必须与 kernel 声明的大小一致 (指针用 CUdeviceptr)
    void* params[] = {(void*)&args..., nullptr};
    ck(cuLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, nullptr, params, nullptr), "cuLaunchKernel");
}

} // namespace Cu
} // namespace

#endif
