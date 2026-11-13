// 示例: top-k + 温度采样, 外加对已生成 token 的 presence penalty; custom_data 的前 8 字节 (小端) 是随机种子, 没有时用 0
#include <algorithm>
#include <cmath>
#include <cstring>

const int K = 20;
const float TEMP = 0.7f, PP = 1.5f;
float G_l[QS_VOCAB];
int G_idx[QS_VOCAB];
bool G_seen[QS_VOCAB]; // 本次已生成过的 token
bool G_think = false; // 还在思考中
uint64_t G_rng = 0;

uint64_t next_rand() { // splitmix64
    uint64_t z = (G_rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) {
    G_think = is_thinking_mode;
    if (custom_data_len >= 8) memcpy(&G_rng, custom_data, 8);
}

int sample(void* data) {
    const uint16_t* h = (const uint16_t*)data;
    int m = 0;
    for (int i = 0; i < QS_VOCAB; i++) {
        if (!qs_allowed(i, G_think)) continue;
        G_l[i] = qs_half_to_float(h[i]) - (G_seen[i] ? PP : 0.0f);
        G_idx[m++] = i;
    }
    std::partial_sort(G_idx, G_idx + K, G_idx + m, [](int a, int b) { return G_l[a] > G_l[b]; });
    double p[K], sum = 0;
    for (int i = 0; i < K; i++) sum += p[i] = std::exp((double)(G_l[G_idx[i]] - G_l[G_idx[0]]) / TEMP);
    double r = (next_rand() >> 11) * (1.0 / 9007199254740992.0) * sum;
    int t = G_idx[K - 1];
    for (int i = 0; i < K; i++)
        if ((r -= p[i]) <= 0) {
            t = G_idx[i];
            break;
        }
    G_seen[t] = true;
    if (t == QS_THINK_END) G_think = false;
    return t;
}
