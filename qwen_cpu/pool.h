#ifndef QWEN_CPU_POOL_H
#define QWEN_CPU_POOL_H

#include <atomic>
#include <thread>
#include <type_traits>
#include <immintrin.h>

namespace {
namespace Pool {

using Fn = void (*)(void* ctx, int item, int tid);

int G_n_threads = 1;
std::atomic<uint32_t> G_gen{0};
std::atomic<int> G_next{0};
std::atomic<int> G_pending{0};
Fn G_fn = nullptr;
void* G_ctx = nullptr;
int G_items = 0;

namespace PoolInternal {

void work(int tid) {
    while (true) {
        int i = G_next.fetch_add(1, std::memory_order_relaxed);
        if (i >= G_items) return;
        G_fn(G_ctx, i, tid);
    }
}

void worker_main(int tid) {
    uint32_t seen = 0;
    while (true) {
        uint32_t g;
        for (int s = 0;; s++) { // 先自旋一段时间, 之后 futex 休眠, 避免空闲时占满 CPU
            g = G_gen.load(std::memory_order_acquire);
            if (g != seen) break;
            if (s < 20000)
                _mm_pause();
            else
                G_gen.wait(seen, std::memory_order_acquire);
        }
        seen = g;
        work(tid);
        G_pending.fetch_sub(1, std::memory_order_release);
    }
}

} // namespace PoolInternal

void init(int n) {
    G_n_threads = n < 1 ? 1 : n;
    for (int i = 1; i < G_n_threads; i++) std::thread(PoolInternal::worker_main, i).detach();
}

template <class F> void run(int items, F&& f) { // f(item, tid), tid 范围 [0, G_n_threads)
    if (items <= 0) return;
    if (G_n_threads == 1 || items == 1) {
        for (int i = 0; i < items; i++) f(i, 0);
        return;
    }
    using FT = std::remove_reference_t<F>;
    G_fn = [](void* c, int i, int tid) { (*(FT*)c)(i, tid); };
    G_ctx = (void*)&f;
    G_items = items;
    G_next.store(0, std::memory_order_relaxed);
    G_pending.store(G_n_threads - 1, std::memory_order_relaxed);
    G_gen.fetch_add(1, std::memory_order_release);
    G_gen.notify_all();
    PoolInternal::work(0);
    while (G_pending.load(std::memory_order_acquire) != 0) _mm_pause();
}

} // namespace Pool
} // namespace

#endif
