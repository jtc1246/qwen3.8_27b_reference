#ifndef QWEN_CUDA_CU_H
#define QWEN_CUDA_CU_H

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

namespace {
namespace Cu { // CUDA driver API 的薄封装, 出错直接退出进程

// stream 0/1: prefill 用整卡 / 分区, 2/3: decode 用整卡 / 分区; 分区是两个 green context (SM 不相交), 整卡是 primary context
// 都是非阻塞 stream (可被 CUDA Graph 捕获, 不与同进程 torch 的 legacy stream 隐式同步)
const int ST_PF = 0, ST_PF_PART = 1, ST_DEC = 2, ST_DEC_PART = 3;
CUdevice G_dev = 0;
CUcontext G_ctx = nullptr; // primary context
CUcontext G_ctxs[4] = {}; // 各 stream 所属的 context
CUstream G_sts[4] = {};
CUlibrary G_lib = nullptr; // kernel 用 cuLibrary 加载, 与 context 无关, launch 时按 stream 所属的 context 执行
CUstream G_stream = nullptr; // 当前 stream: launch / 默认拷贝都走它; 只由 prefill 线程 (和初始化) 使用, decode 线程只重放 graph, 显式指定 stream

void ck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    fprintf(stderr, "❌ CUDA %s 失败: %d %s\n", what, (int)r, s ? s : "?");
    _Exit(1);
}

void init(int device, const void* image, int dec_sm) { // dec_sm: decode 分区的 SM 数, 其余给 prefill
    CUdevice dev;
    char name[256] = {};
    int ver = 0, cc_major = 0, cc_minor = 0;
    ck(cuInit(0), "cuInit");
    ck(cuDriverGetVersion(&ver), "cuDriverGetVersion");
    ck(cuDeviceGet(&dev, device), "cuDeviceGet");
    G_dev = dev;
    ck(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    ck(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cuDeviceGetAttribute");
    ck(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cuDeviceGetAttribute");
    ck(cuDevicePrimaryCtxRetain(&G_ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(cuCtxSetCurrent(G_ctx), "cuCtxSetCurrent");
    ck(cuLibraryLoadData(&G_lib, image, nullptr, nullptr, 0, nullptr, nullptr, 0), "cuLibraryLoadData");
    ck(cuStreamCreate(&G_sts[ST_PF], CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    ck(cuStreamCreate(&G_sts[ST_DEC], CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    G_ctxs[ST_PF] = G_ctxs[ST_DEC] = G_ctx;
    CUdevResource all_res, res[2];
    CU_DEV_SM_RESOURCE_GROUP_PARAMS gp[2] = {};
    gp[0].smCount = dec_sm; // group 0: decode, group 1: 其余全部 (prefill)
    gp[0].coscheduledSmCount = 2; // 本卡默认粒度 8, 指定 2 后可以按 2 个 SM 分
    gp[1].coscheduledSmCount = 2;
    gp[1].flags = CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
    ck(cuDeviceGetDevResource(dev, &all_res, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
    ck(cuDevSmResourceSplit(res, 2, &all_res, nullptr, 0, gp), "cuDevSmResourceSplit");
    for (int i = 0; i < 2; i++) {
        int sid = i ? ST_PF_PART : ST_DEC_PART;
        CUdevResourceDesc desc;
        CUgreenCtx gctx;
        ck(cuDevResourceGenerateDesc(&desc, &res[i], 1), "cuDevResourceGenerateDesc");
        ck(cuGreenCtxCreate(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
        ck(cuCtxFromGreenCtx(&G_ctxs[sid], gctx), "cuCtxFromGreenCtx");
        ck(cuGreenCtxStreamCreate(&G_sts[sid], gctx, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
    }
    G_stream = G_sts[ST_PF];
    fprintf(stderr, "ℹ️ GPU %d: %s, sm_%d%d, 驱动 CUDA %d.%d, 共 %u 个 SM, prefill 和 decode 同时运行时分别用 %u / %u 个\n", device, name, cc_major, cc_minor, ver / 1000, ver % 1000 / 10, all_res.sm.smCount, res[1].sm.smCount, res[0].sm.smCount);
}

void use(int sid) { // 当前线程切到 stream sid (及其 context, context 是每线程的, python 可能从不同线程调用), 之后的 launch / 默认拷贝都走它
    ck(cuCtxSetCurrent(G_ctxs[sid]), "cuCtxSetCurrent");
    G_stream = G_sts[sid];
}

CUfunction func(const char* name) {
    CUkernel f;
    ck(cuLibraryGetKernel(&f, G_lib, name), name);
    return (CUfunction)f; // cuLaunchKernel 可以直接接受 CUkernel
}

CUdeviceptr alloc(size_t bytes) { // 分配并清零
    CUdeviceptr p;
    ck(cuMemAlloc(&p, bytes), "cuMemAlloc");
    ck(cuMemsetD8Async(p, 0, bytes, G_stream), "cuMemsetD8Async");
    return p;
}

void h2d(CUdeviceptr dst, const void* src, size_t bytes, CUstream st = G_stream) { ck(cuMemcpyHtoDAsync(dst, src, bytes, st), "cuMemcpyHtoDAsync"); } // 源是可分页内存时返回前已拷到暂存区, src 可立即复用
void d2h(void* dst, CUdeviceptr src, size_t bytes, CUstream st = G_stream) { // 同步: 返回时数据已到 dst
    ck(cuMemcpyDtoHAsync(dst, src, bytes, st), "cuMemcpyDtoHAsync");
    ck(cuStreamSynchronize(st), "cuStreamSynchronize");
}

template <class... A> void launch(CUfunction f, unsigned gx, unsigned gy, unsigned block, A... args) { // 参数类型必须与 kernel 声明的大小一致 (指针用 CUdeviceptr)
    void* params[] = {(void*)&args..., nullptr};
    ck(cuLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, G_stream, params, nullptr), "cuLaunchKernel");
}

template <class... A> void launch3(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned block, A... args) { // 三维 grid
    void* params[] = {(void*)&args..., nullptr};
    ck(cuLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, G_stream, params, nullptr), "cuLaunchKernel");
}

template <class... A> void launch_s(CUfunction f, unsigned gx, unsigned gy, unsigned block, unsigned smem, A... args) { // 同 launch, 带动态 smem 字节数
    void* params[] = {(void*)&args..., nullptr};
    ck(cuLaunchKernel(f, gx, gy, 1, block, 1, 1, smem, G_stream, params, nullptr), "cuLaunchKernel");
}

} // namespace Cu
} // namespace

#endif
