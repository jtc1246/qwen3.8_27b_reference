#ifndef QWEN_NPU_ST_H
#define QWEN_NPU_ST_H

// safetensors 加载 (mmap 只读): 8 字节头长度 + JSON 头 + 数据区, 建立 name -> {dtype, shape, data} 索引
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
namespace St {

using std::string;
using std::unordered_map;
using std::vector;

enum { DT_BF16 = 0, DT_F32 = 1, DT_I32 = 2, DT_I64 = 3 };

struct Tensor {
    int dtype = 0;
    vector<int64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t nbytes = 0;
};

struct File {
    const uint8_t* base = nullptr;
    size_t size = 0;
    unordered_map<string, Tensor> tensors;
};

namespace StInternal {

struct P { // JSON 头解析器, 格式错误直接退出
    const char* p;
    const char* end;
    void fail(const char* why) {
        fprintf(stderr, "❌ safetensors 头解析失败: %s\n", why);
        _Exit(1);
    }
    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    char peek() {
        if (p >= end) fail("意外结束");
        return *p;
    }
    void expect(char c) {
        ws();
        if (peek() != c) fail("意外字符");
        p++;
    }
    string str() {
        expect('"');
        string s;
        while (true) {
            if (p >= end) fail("字符串未闭合");
            char c = *p++;
            if (c == '"') return s;
            if (c == '\\') { // 名字里不会有转义, metadata 可能有, 简单处理
                if (p >= end) fail("转义截断");
                char e = *p++;
                if (e == 'n') s += '\n';
                else if (e == 't') s += '\t';
                else if (e == 'u') { p += 4; s += '?'; } // 不还原 unicode, 只跳过
                else s += e;
                continue;
            }
            s += c;
        }
    }
    int64_t num() {
        ws();
        bool neg = peek() == '-';
        if (neg) p++;
        int64_t v = 0;
        if (p >= end || *p < '0' || *p > '9') fail("不是数字");
        while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) fail("不支持浮点数");
        return neg ? -v : v;
    }
    void skip_value() { // metadata 用: 跳过任意值
        ws();
        char c = peek();
        if (c == '"') { str(); return; }
        if (c == '{') {
            p++;
            ws();
            if (peek() == '}') { p++; return; }
            while (true) {
                str();
                expect(':');
                skip_value();
                ws();
                if (peek() == ',') { p++; continue; }
                expect('}');
                return;
            }
        }
        if (c == '[') {
            p++;
            ws();
            if (peek() == ']') { p++; return; }
            while (true) {
                skip_value();
                ws();
                if (peek() == ',') { p++; continue; }
                expect(']');
                return;
            }
        }
        if (c == 't' || c == 'f' || c == 'n') {
            while (p < end && *p >= 'a' && *p <= 'z') p++;
            return;
        }
        num();
    }
};

} // namespace StInternal

inline int64_t numel(const Tensor& t) {
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    return n;
}

inline bool open(File& f, const char* path) {
    using namespace StInternal;
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "❌ 无法打开 %s\n", path);
        return false;
    }
    struct stat st;
    fstat(fd, &st);
    f.size = st.st_size;
    void* m = mmap(nullptr, f.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        fprintf(stderr, "❌ mmap 失败 %s\n", path);
        return false;
    }
    f.base = (const uint8_t*)m;
    if (f.size < 16) {
        fprintf(stderr, "❌ 文件太小, 不是 safetensors: %s\n", path);
        return false;
    }
    uint64_t hlen;
    memcpy(&hlen, f.base, 8);
    if (hlen > f.size - 8) {
        fprintf(stderr, "❌ 头长度越界, 不是 safetensors: %s\n", path);
        return false;
    }
    const uint8_t* data0 = f.base + 8 + hlen;
    uint64_t dsize = f.size - 8 - hlen;
    P ps{(const char*)f.base + 8, (const char*)f.base + 8 + hlen};
    ps.expect('{');
    while (true) {
        string name = ps.str();
        ps.expect(':');
        if (name == "__metadata__") {
            ps.skip_value();
        } else {
            Tensor t;
            int64_t off0 = -1, off1 = -1;
            ps.expect('{');
            while (true) {
                string key = ps.str();
                ps.expect(':');
                if (key == "dtype") {
                    string d = ps.str();
                    if (d == "BF16") t.dtype = DT_BF16;
                    else if (d == "F32") t.dtype = DT_F32;
                    else if (d == "I32") t.dtype = DT_I32;
                    else if (d == "I64") t.dtype = DT_I64;
                    else {
                        fprintf(stderr, "❌ 不支持的 dtype %s (%s)\n", d.c_str(), name.c_str());
                        _Exit(1);
                    }
                } else if (key == "shape") {
                    ps.expect('[');
                    ps.ws();
                    if (ps.peek() == ']') ps.p++;
                    else
                        while (true) {
                            t.shape.push_back(ps.num());
                            ps.ws();
                            if (ps.peek() == ',') { ps.p++; continue; }
                            ps.expect(']');
                            break;
                        }
                } else if (key == "data_offsets") {
                    ps.expect('[');
                    off0 = ps.num();
                    ps.expect(',');
                    off1 = ps.num();
                    ps.expect(']');
                } else {
                    ps.skip_value();
                }
                ps.ws();
                if (ps.peek() == ',') { ps.p++; continue; }
                ps.expect('}');
                break;
            }
            if (off0 < 0 || off1 < off0 || (uint64_t)off1 > dsize) {
                fprintf(stderr, "❌ tensor %s 偏移非法\n", name.c_str());
                _Exit(1);
            }
            t.data = data0 + off0;
            t.nbytes = off1 - off0;
            f.tensors[name] = std::move(t);
        }
        ps.ws();
        if (ps.peek() == ',') { ps.p++; continue; }
        ps.expect('}');
        break;
    }
    fprintf(stderr, "ℹ️ safetensors 加载完成: %zu 个 tensor, 文件 %.2f GB\n", f.tensors.size(), f.size / 1073741824.0);
    return true;
}

inline void close(File& f) {
    if (f.base) munmap((void*)f.base, f.size);
    f.base = nullptr;
}

} // namespace St
} // namespace

#endif
