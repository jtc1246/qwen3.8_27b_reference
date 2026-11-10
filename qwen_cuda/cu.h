#ifndef QWEN_CUDA_CU_H
#define QWEN_CUDA_CU_H

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

namespace {
namespace Cu { // CUDA driver API 的薄封装, 出错直接退出进程

CUcontext G_ctx = nullptr;
CUmodule G_mod = nullptr;
CUstream G_stream = nullptr; // 所有 kernel 和拷贝都走这个非阻塞 stream (可被 CUDA Graph 捕获, 不与同进程 torch 的 legacy stream 隐式同步)

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
    ck(cuStreamCreate(&G_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
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
    ck(cuMemsetD8Async(p, 0, bytes, G_stream), "cuMemsetD8Async");
    return p;
}

void h2d(CUdeviceptr dst, const void* src, size_t bytes) { ck(cuMemcpyHtoDAsync(dst, src, bytes, G_stream), "cuMemcpyHtoDAsync"); } // 源是可分页内存时返回前已拷到暂存区, src 可立即复用
void d2h(void* dst, CUdeviceptr src, size_t bytes) { // 同步: 返回时数据已到 dst
    ck(cuMemcpyDtoHAsync(dst, src, bytes, G_stream), "cuMemcpyDtoHAsync");
    ck(cuStreamSynchronize(G_stream), "cuStreamSynchronize");
}

template <class... A> void launch(CUfunction f, unsigned gx, unsigned gy, unsigned block, A... args) { // 参数类型必须与 kernel 声明的大小一致 (指针用 CUdeviceptr)
    void* params[] = {(void*)&args..., nullptr};
    ck(cuLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, G_stream, params, nullptr), "cuLaunchKernel");
}

template <class... A> void launch_s(CUfunction f, unsigned gx, unsigned gy, unsigned block, unsigned smem, A... args) { // 同 launch, 带动态 smem 字节数
    void* params[] = {(void*)&args..., nullptr};
    ck(cuLaunchKernel(f, gx, gy, 1, block, 1, 1, smem, G_stream, params, nullptr), "cuLaunchKernel");
}

} // namespace Cu
} // namespace

#endif
