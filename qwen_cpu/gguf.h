#ifndef QWEN_CPU_GGUF_H
#define QWEN_CPU_GGUF_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
namespace Gguf {

using std::string;
using std::vector;
using std::unordered_map;

enum : uint32_t { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

struct Tensor {
    vector<int64_t> dims; // dims[0] 是最内层维度 (ggml ne0)
    uint32_t type;
    const uint8_t* data;
};

struct File {
    const uint8_t* base = nullptr;
    size_t size = 0;
    unordered_map<string, Tensor> tensors;
    unordered_map<string, double> kv_num; // 数值型标量
    unordered_map<string, string> kv_str; // 字符串标量, 数组不保存
};

namespace GgufInternal {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    void need(size_t n) {
        if ((size_t)(end - p) < n) {
            fprintf(stderr, "❌ GGUF 文件截断或格式错误\n");
            _Exit(1);
        }
    }
    template <class T> T get() {
        need(sizeof(T));
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    string get_str() {
        uint64_t n = get<uint64_t>();
        need(n);
        string s((const char*)p, n);
        p += n;
        return s;
    }
};

const size_t SCALAR_SIZE[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8}; // 按 gguf 值类型编号, 8=string 9=array 单独处理

void skip_value(Reader& r, uint32_t t) {
    if (t == 8) {
        r.get_str();
        return;
    }
    if (t == 9) {
        uint32_t it = r.get<uint32_t>();
        uint64_t n = r.get<uint64_t>();
        if (it == 8 || it == 9) {
            for (uint64_t i = 0; i < n; i++) skip_value(r, it);
            return;
        }
        r.need(n * SCALAR_SIZE[it]);
        r.p += n * SCALAR_SIZE[it];
        return;
    }
    r.need(SCALAR_SIZE[t]);
    r.p += SCALAR_SIZE[t];
}

double read_num(Reader& r, uint32_t t) {
    switch (t) {
        case 0: return r.get<uint8_t>();
        case 1: return r.get<int8_t>();
        case 2: return r.get<uint16_t>();
        case 3: return r.get<int16_t>();
        case 4: return r.get<uint32_t>();
        case 5: return r.get<int32_t>();
        case 6: return r.get<float>();
        case 7: return r.get<uint8_t>();
        case 10: return (double)r.get<uint64_t>();
        case 11: return (double)r.get<int64_t>();
        case 12: return r.get<double>();
    }
    return 0;
}

} // namespace GgufInternal

bool open(File& f, const char* path) {
    using namespace GgufInternal;
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "❌ 无法打开 %s\n", path);
        return false;
    }
    struct stat st;
    fstat(fd, &st);
    f.size = st.st_size;
    uint8_t* buf = (uint8_t*)malloc(f.size); // 整个文件读进自己分配的内存, 进程生命周期内不释放
    if (!buf) {
        fprintf(stderr, "❌ 内存分配失败 %zu 字节\n", f.size);
        return false;
    }
    for (size_t off = 0; off < f.size;) {
        size_t want = f.size - off < ((size_t)1 << 30) ? f.size - off : ((size_t)1 << 30);
        ssize_t n = pread(fd, buf + off, want, off);
        if (n <= 0) {
            fprintf(stderr, "❌ 读取失败 %s (偏移 %zu)\n", path, off);
            return false;
        }
        off += n;
    }
    close(fd);
    f.base = buf;
    Reader r{f.base, f.base + f.size};
    if (r.get<uint32_t>() != 0x46554747) { // "GGUF"
        fprintf(stderr, "❌ 不是 GGUF 文件: %s\n", path);
        return false;
    }
    uint32_t ver = r.get<uint32_t>();
    if (ver != 3) {
        fprintf(stderr, "❌ 只支持 GGUF v3, 当前 v%u\n", ver);
        return false;
    }
    uint64_t n_tensors = r.get<uint64_t>();
    uint64_t n_kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv; i++) {
        string key = r.get_str();
        uint32_t t = r.get<uint32_t>();
        if (t == 8)
            f.kv_str[key] = r.get_str();
        else if (t == 9)
            skip_value(r, t);
        else
            f.kv_num[key] = read_num(r, t);
    }
    vector<std::pair<string, uint64_t>> offsets;
    for (uint64_t i = 0; i < n_tensors; i++) {
        string name = r.get_str();
        Tensor t;
        uint32_t nd = r.get<uint32_t>();
        for (uint32_t d = 0; d < nd; d++) t.dims.push_back((int64_t)r.get<uint64_t>());
        t.type = r.get<uint32_t>();
        offsets.emplace_back(name, r.get<uint64_t>());
        f.tensors[name] = t;
    }
    uint64_t align = f.kv_num.count("general.alignment") ? (uint64_t)f.kv_num["general.alignment"] : 32;
    uint64_t data_start = ((uint64_t)(r.p - f.base) + align - 1) / align * align;
    for (auto& [name, off] : offsets) {
        if (data_start + off > f.size) {
            fprintf(stderr, "❌ tensor %s 偏移越界\n", name.c_str());
            return false;
        }
        f.tensors[name].data = f.base + data_start + off;
    }
    fprintf(stderr, "ℹ️ GGUF 加载完成: %zu 个 tensor, %zu 个数值 kv, 文件 %.2f GB\n", f.tensors.size(), f.kv_num.size(), f.size / 1073741824.0);
    return true;
}

} // namespace Gguf
} // namespace

#endif
