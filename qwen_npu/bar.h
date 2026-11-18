// doorbell 标志页的 BAR MMIO 直访: 310P 在 PCIe 上暴露一个 8GB 可预取 BAR (resource4), 映射两颗芯 GM 的
// 一部分 (高物理地址段)。标志页若落在可见区, host 对它的读写就是一条阻塞的 MMIO load/store (实测读 0.44us
// 写 0.14us), 替代 4B 同步 memcpy (~6us), doorbell 往返 12.2us -> 1.2us (实验 tests/npu_bar_db)。
// 需要宿主机对 resource4 放开权限 (chmod 666, 重启后会复位); 打不开/探测不到可见页时自动退回 memcpy 路径。
#ifndef QWEN_NPU_BAR_H
#define QWEN_NPU_BAR_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "acl.h"

namespace {
namespace Bar {

const size_t BAR_SZ = 8ull << 30;
uint8_t* G_base = nullptr; // 窗口映射基址 (null = 不可用)

bool open_map() { // 找华为 NPU (vendor 0x19e5) 的 8GB resource4 并 mmap (多卡时只取第一个, 探测不中会自然回退)
    DIR* d = opendir("/sys/bus/pci/devices");
    if (!d) return false;
    char path[320];
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", e->d_name);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        unsigned ven = 0;
        int ok = fscanf(f, "%x", &ven);
        fclose(f);
        if (ok != 1 || ven != 0x19e5) continue;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource4", e->d_name);
        struct stat st;
        if (stat(path, &st) != 0 || (size_t)st.st_size != BAR_SZ) continue;
        int fd = open(path, O_RDWR);
        if (fd < 0) { fprintf(stderr, "ℹ️ BAR 窗口 %s 无权限 (宿主机未放开), doorbell 走 memcpy\n", e->d_name); continue; }
        void* p = mmap(nullptr, BAR_SZ, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd); // mmap 自持引用
        if (p == MAP_FAILED) continue;
        G_base = (uint8_t*)p;
        fprintf(stderr, "ℹ️ BAR 窗口已映射 (%s)\n", e->d_name);
        closedir(d);
        return true;
    }
    closedir(d);
    return false;
}

uint64_t mix64(uint64_t z) { // splitmix64 终混
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// 探测一个 BAR 可见的 4KB GM 标志页 (当前设备, 需在模型全部分配之后调用, 可见区在高物理地址段):
// 按 32MB 块临时吃掉空闲显存 (默认全部, 留 2GB 余量; QWEN_NPU_BARGB 可设上限), 每 4KB 页填
// 64 位随机标签 (页首带页号), 扫窗口找完整可见页, 再写 nonce 经自己句柄回读确认所有权
// (防多进程同时探测误认他人页面); 命中的 32MB 块保留 (进程生命周期不释放), 其余全部释放。
// 返回标志页的 GM 地址并写 *host_ptr (BAR 侧地址), 失败返回 null。
const size_t PROBE_CHUNK = 32ull << 20;
void* probe_flag_page(volatile int32_t** host_ptr, int dev_idx) {
    *host_ptr = nullptr;
    if (!G_base) return nullptr;
    auto t0 = std::chrono::steady_clock::now();
    size_t free_b = 0, total_b = 0;
    Acl::ck(aclrtGetMemInfo(ACL_HBM_MEM, &free_b, &total_b), "aclrtGetMemInfo bar");
    size_t reserve = 2ull << 30;
    size_t budget = free_b > reserve ? free_b - reserve : 0;
    if (const char* e = getenv("QWEN_NPU_BARGB")) { // 上限 (GB), <=0 禁用探测
        long gb = atol(e);
        size_t cap = gb <= 0 ? 0 : (size_t)gb << 30;
        if (budget > cap) budget = cap;
    }
    size_t n_chunk = budget / PROBE_CHUNK;
    if (!n_chunk) {
        fprintf(stderr, "ℹ️ 芯 %d BAR 标志页探测跳过 (预算/空闲显存不足), doorbell 走 memcpy\n", dev_idx);
        return nullptr;
    }
    uint64_t tag = mix64((uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() ^ ((uint64_t)getpid() << 24) ^ ((uint64_t)dev_idx << 62) ^ (uint64_t)&free_b);
    size_t pgc = PROBE_CHUNK / 4096; // 每块页数
    std::vector<void*> chunks;
    std::vector<uint64_t> stage(PROBE_CHUNK / 8, tag); // 填充值 = tag, 页首 = tag + 全局页号 + 1
    chunks.reserve(n_chunk);
    for (size_t c = 0; c < n_chunk; c++) {
        void* p = nullptr;
        if (aclrtMalloc(&p, PROBE_CHUNK, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) break; // 吃不下就到此为止
        for (size_t pg = 0; pg < pgc; pg++) stage[pg * 512] = tag + c * pgc + pg + 1;
        Acl::h2d(p, stage.data(), PROBE_CHUNK);
        chunks.push_back(p);
    }
    size_t n_pages = chunks.size() * pgc;
    void* gm_page = nullptr;
    volatile int32_t* bar_page = nullptr;
    size_t keep = (size_t)-1, n_cand = 0;
    for (size_t off = 0; off < BAR_SZ && !gm_page; off += 4096) {
        uint64_t delta = *(volatile uint64_t*)(G_base + off) - tag;
        if (delta == 0 || delta > n_pages) continue;
        n_cand++;
        bool full = true;
        for (int j = 1; j < 512 && full; j++) full = *(volatile uint64_t*)(G_base + off + (size_t)j * 8) == tag;
        if (!full) continue;
        size_t idx = delta - 1;
        uint8_t* cand = (uint8_t*)chunks[idx / pgc] + (idx % pgc) * 4096;
        uint64_t nonce = mix64(tag ^ off);
        Acl::h2d(cand, &nonce, 8);
        for (int r = 0; r < 100000; r++) { // 所有权确认: 经自己句柄写入的 nonce 必须从 BAR 读到 (约 us 级可见)
            if (*(volatile uint64_t*)(G_base + off) == nonce) {
                gm_page = cand;
                bar_page = (volatile int32_t*)(G_base + off);
                keep = idx / pgc;
                break;
            }
        }
    }
    for (size_t c = 0; c < chunks.size(); c++)
        if (c != keep) Acl::ck(aclrtFree(chunks[c]), "free bar cand");
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!gm_page) {
        fprintf(stderr, "ℹ️ 芯 %d BAR 标志页探测未命中 (%zu 块 / 候选 %zu, %.1fs), doorbell 走 memcpy\n", dev_idx, chunks.size(), n_cand, sec);
        return nullptr;
    }
    Acl::ck(aclrtMemset(gm_page, 4096, 0, 4096), "memset bar flag");
    *host_ptr = bar_page;
    fprintf(stderr, "✅ 芯 %d doorbell 标志页走 BAR 直访 (窗口偏移 0x%llx, 探测 %zu 块 %.1fs)\n", dev_idx, (unsigned long long)((uint8_t*)bar_page - G_base), chunks.size(), sec);
    return gm_page;
}

} // namespace Bar
} // namespace

#endif
