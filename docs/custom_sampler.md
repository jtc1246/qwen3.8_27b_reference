# 自定义采样 (CUDA)

请求里带一段 C++ 代码, 服务端编译成 wasm 后代替内置采样: 每生成一个 token 调用一次, 拿到本步的全部 logits, 自己决定输出哪个 token、何时结束思考、何时结束生成。只有 cuda 后端支持。

第一部分是使用方式, 第二部分是内部实现。

## 一、使用

### 1. 启用

需要两样资源 (见 README 环境准备第 4 步): 编译 CUDA 版时的 WAMR 源码 (`WAMR_DIR`), 运行服务时的编译镜像 `sampler_rootfs.tar`。启动参数:

| 参数 | 说明 |
| --- | --- |
| `--sampler-rootfs <tar>` | 编译镜像, 不传则自定义采样不可用 |
| `--cache-dir <目录>` | 必须同时有; 编译结果缓存在 `<cache-dir>/wasm/` (经 router 启动时是 `<根目录>/cuda/wasm/`) |
| `--wasm-tmp-dir <目录>` | 编译用的临时目录, 默认 `/dev/shm`, 每次编译约占 360MB, 用完删除 |

```bash
python -u server.py --type cuda --gguf ... --hf-dir ... --cache-dir /path/to/cache --sampler-rootfs /path/to/sampler_rootfs.tar
python -u router.py --gguf ... --hf-dir ... --cache-dir /path/to/cache_root --sampler-rootfs /path/to/sampler_rootfs.tar # router 原样转发给后端
```

没有配置、镜像不存在、或 so 编译时没有 WAMR 时, 服务照常启动, 只有带 `sampler_cpp` 的请求报错。运行服务的机器需要 `tar`、`chroot`、`unshare` 命令和使用它们的权限。

### 2. API

在普通的 `POST /v1/chat/completions` 请求里加两个字段, 其余用法不变 (messages、图片、`enable_thinking` / `reasoning_effort`、`stream` 等):

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sampler_cpp` | string | C++ 源码, 最多 1MB (utf-8) |
| `sampler_data` | string, 可选 | base64, 解码后最多 1MB, 原样交给 `sampler_init` (`custom_data`); 只能和 `sampler_cpp` 一起用 |

- 有 `sampler_cpp` 时内置采样不起作用: `temperature`、`seed` 等都被忽略, 随机数需要自己生成 (种子可以放在 `sampler_data` 里)
- 经 router 请求时 `model` 要选 cuda 后端 (如 `qwen3.8-27b-cuda`); 发给 cpu 后端返回 400
- 返回格式与普通请求相同: 开 thinking 时思考结束 token 之前的内容在 `reasoning_content`, 之后在 `content`; 以 im_end 结束时 `finish_reason` 是 `stop`, 上下文用完是 `length`; `usage` 照常统计
- 出错时 (编译失败、函数缺失或签名不对、运行时越界 / 死循环 / trap、返回值不合法): 非流式返回 400 `{"error": {"message": "..."}}`; 流式在已经发出的 token 之后发一个 `data: {"error": {...}}` 事件。错误信息里有编译器输出或出错原因, 以及出错前已生成的 token 数
- 相同源码 (字符串完全相同) 只编译一次, 之后直接用缓存; 第一次编译约 0.3~2s

### 3. 代码怎么写

实现两个函数:

```cpp
void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len);
int sample(void* data);
```

- `sampler_init`: 每个请求开始时调用一次。`is_thinking_mode` 是本次是否开了 thinking (开了的话生成从思考中开始); `custom_data` / `custom_data_len` 是 `sampler_data` 解码后的二进制, 整个请求内有效, 不传时是 nullptr / 0
- `sample`: 每生成一个 token 调用一次 (包括第一个), `data` 是本步的 `QS_VOCAB` (248320) 个 logits, 每个是 fp16 (IEEE half, 小端), 用 `qs_half_to_float` 转成 float; 返回本步的 token id
- 每个请求一个全新的实例: 全局变量从初值开始, 在请求内一直保持, 不跨请求。已经输出了哪些 token、是否还在思考、随机数状态、计数器等都由代码自己用全局变量维护 (服务端不再传入历史)
- 何时结束完全由代码决定: 返回 im_end (`QS_IM_END`) 结束生成; thinking 模式下返回思考结束 token (`QS_THINK_END`) 结束思考、开始正文, 思考中不能直接返回 im_end (要先结束思考)。没有其他结束信号, 也没有长度限制, 一直不返回 im_end 就生成到上下文用完
- 以下内容已自动包含 ([wasm_sampler/prelude.h](../wasm_sampler/prelude.h)), 不需要也不应该自己声明:
  - 常量 `QS_VOCAB` (248320)、`QS_N_TEXT` (248044)、`QS_IM_END` (248046)、`QS_THINK_END` (248069)
  - `bool qs_allowed(int id, bool in_thinking)`: 返回值是否合法 (见下面的 token 范围)
  - `float qs_half_to_float(uint16_t h)`: fp16 转 float
- 编译: wasm32 (wasi-sdk, clang, C++23, `-O2`, 无异常、无 RTTI)。可以 include 标准 C / C++ 头文件, 用 STL 容器和算法、`new` / `malloc`; 不能调用需要系统接口的函数 (printf、文件、time、clock 等), 用了会编译失败 ("编译结果有导入")
- `sample` 和 `sampler_init` 必须是普通的 C++ 函数, 签名完全一致; 不能写成 `extern "C"`, 参数或返回值类型不对会编译失败并提示要求的签名
- 运行限制: 内存最多 1GB, 每次 `sample` / `sampler_init` (以及全局构造函数) 最多约 20 亿条 wasm 指令 (约 2s), 超出、越界访问、`__builtin_trap` 都会让本请求出错

### 4. token 范围

| id | 内容 |
| --- | --- |
| `[0, 248044)` | 普通文本 token (BPE 词表), 可以返回 |
| 248044 | endoftext, 不能返回 |
| 248045 | im_start, 不能返回 |
| 248046 | im_end, 不在思考中时可以返回: 结束本次生成; 思考中不能返回 |
| 248069 | 思考结束, 只有 thinking 模式下、思考还没结束时可以返回 |
| 248044 ~ 248076 中的其余 | 控制 token (图片、工具调用、思考开始等), 不能返回 |
| `[248077, 248320)` | 输出层的填充位, 没有对应的 token, 不能返回 |

`data` 里 248320 个位置都有值 (填充位的 logits 没有意义)。返回其他值, 在思考中返回 im_end, 或在非 thinking 模式 / 思考已经结束后返回思考结束 token, 本请求报错 (与内置采样的可采样集合相同, 开 thinking 的回复总是 思考 → 思考结束 → 正文 → im_end)。`qs_allowed(id, in_thinking)` 就是这条规则, `in_thinking` 需要自己维护 (初值是 `is_thinking_mode`, 返回思考结束 token 后变成 false)。

普通文本 token 是 byte-level BPE: 一个 token 是若干字节, 一个汉字 (3 字节) 可能被拆在两个 token 里。要按内容挑选 token (比如禁用某种文字) 时, 在客户端用仓库的 `qwen_tok` 查每个 id 的字节: `qwen_tok.load("tokenizer.json")` 之后 `qwen_tok.token_bytes(id)`; 文本转 id 用 `qwen_tok.encode(text)` (只产生普通文本 token)。

### 5. 样例

下面的 python 例子都用这个函数发请求 (只用标准库; 直连 server 时 `model` 会被忽略):

```python
import json, base64, struct, urllib.request, urllib.error

def chat(question, sampler_cpp, sampler_data=b"", thinking=False, base="http://127.0.0.1:8000/v1"):
    body = {"model": "qwen3.8-27b-cuda", "messages": [{"role": "user", "content": question}], "enable_thinking": thinking, "sampler_cpp": sampler_cpp}
    if sampler_data:
        body["sampler_data"] = base64.b64encode(sampler_data).decode()
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return json.load(e) # 400: {"error": {"message": ...}}
```

#### 5.1 贪心

每步取 logit 最大的合法 token:

```cpp
// 样例: 贪心
bool G_think = false; // 还在思考中

void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) { G_think = is_thinking_mode; }

int sample(void* data) {
    const uint16_t* h = (const uint16_t*)data;
    int best = -1;
    float bv = 0;
    for (int i = 0; i < QS_VOCAB; i++) {
        if (!qs_allowed(i, G_think)) continue;
        float v = qs_half_to_float(h[i]);
        if (best < 0 || v > bv) best = i, bv = v;
    }
    if (best == QS_THINK_END) G_think = false;
    return best;
}
```

```python
r = chat("用一句话介绍一下北京。", GREEDY)
print(r["choices"][0]["message"]["content"])
```

#### 5.2 top-k + 温度 + 随机种子

完整代码见 [wasm_sampler/examples/top_k.cpp](../wasm_sampler/examples/top_k.cpp): top-k 20、温度 0.7、对已输出 token 的 presence penalty 1.5, 自己实现 splitmix64 随机数, 种子是 `sampler_data` 的前 8 字节。同一个种子输出相同:

```python
r = chat("用一句话介绍一下北京。", open("wasm_sampler/examples/top_k.cpp").read(), struct.pack("<Q", 12345))
```

#### 5.3 限制思考长度

思考超过给定的 token 数就强制结束思考 (`sampler_data` 是 int32 小端的上限), 其余同贪心:

```cpp
// 样例: 限制思考长度
#include <cstring>

bool G_think = false;
int G_budget = 0, G_n = 0; // 思考 token 上限, 已生成的思考 token 数

void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) {
    G_think = is_thinking_mode;
    if (custom_data_len >= 4) memcpy(&G_budget, custom_data, 4);
}

int sample(void* data) {
    if (G_think && G_n >= G_budget) {
        G_think = false;
        return QS_THINK_END;
    }
    const uint16_t* h = (const uint16_t*)data;
    int best = -1;
    float bv = 0;
    for (int i = 0; i < QS_VOCAB; i++) {
        if (!qs_allowed(i, G_think)) continue;
        float v = qs_half_to_float(h[i]);
        if (best < 0 || v > bv) best = i, bv = v;
    }
    if (G_think) G_n++;
    if (best == QS_THINK_END) G_think = false;
    return best;
}
```

```python
r = chat("一个三位数, 各位数字之和是 16, 百位比个位大 3, 十位是 5, 求这个数。", BUDGET, struct.pack("<i", 64), thinking=True)
print(r["usage"]["completion_tokens"], r["choices"][0]["message"]["content"]) # 思考在第 64 个 token 处截断, 正文照常给出 754
```

#### 5.4 禁用一批 token

`sampler_data` 是 `QS_VOCAB / 8` 字节的位图, 第 i 位为 1 表示禁用 token i。这里在客户端把所有可能组成汉字、中文标点的 token 都禁掉, 中文问题通常会改用英文回答; 但问题很短时, 贪心下第一步最大的合法 token 可能就是 im_end (例如只问 "你叫什么名字？" 且不开 thinking 时直接返回空回答), 开 thinking 时更稳定 (思考会先变成英文):

```cpp
// 样例: 禁用一批 token
bool G_think = false;
const uint8_t* G_ban = nullptr; // 位图, 整个请求内有效

void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) {
    G_think = is_thinking_mode;
    if (custom_data_len >= QS_VOCAB / 8) G_ban = (const uint8_t*)custom_data;
}

int sample(void* data) {
    const uint16_t* h = (const uint16_t*)data;
    int best = -1;
    float bv = 0;
    for (int i = 0; i < QS_VOCAB; i++) {
        if (!qs_allowed(i, G_think) || (G_ban && (G_ban[i >> 3] >> (i & 7) & 1))) continue;
        float v = qs_half_to_float(h[i]);
        if (best < 0 || v > bv) best = i, bv = v;
    }
    if (best == QS_THINK_END) G_think = false;
    return best;
}
```

```python
import qwen_tok
qwen_tok.load("/path/to/Qwen3.8-27B/tokenizer.json")
ban = bytearray(248320 // 8)
for i in range(248044):
    if any(0xE3 <= b <= 0xE9 or b == 0xEF for b in qwen_tok.token_bytes(i)): # 汉字、中文标点、假名的 utf-8 首字节, 全角符号; 碎片 token 也禁, 拼不出中文
        ban[i >> 3] |= 1 << (i & 7)
r = chat("你叫什么名字？你能做什么？", BAN, bytes(ban))
print(r["choices"][0]["message"]["content"]) # Hello! I am Qwen (Tongyi Qianwen), ...
```

#### 5.5 输出指定内容

按 `sampler_data` 里的 token 序列 (int32 小端) 逐个返回, 用完返回 im_end (开 thinking 时序列里要有思考结束 token, 否则用完时还在思考中, 返回 im_end 会报错)。输出的文本与给定文本逐字节相同 (可以用来构造固定回复、测试客户端):

```cpp
// 样例: 输出指定内容
const int* G_tok = nullptr;
uint64_t G_n = 0, G_i = 0;

void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) {
    G_tok = (const int*)custom_data;
    G_n = custom_data_len / 4;
}

int sample(void* data) { return G_i < G_n ? G_tok[G_i++] : QS_IM_END; }
```

```python
ids = qwen_tok.encode("你好，世界！")
r = chat("随便说点什么", REPLAY, b"".join(struct.pack("<i", t) for t in ids))
print(r["choices"][0]["message"]["content"]) # 你好，世界！
```

开 thinking 时在序列中间放一个思考结束 token (248069), 之前的进 `reasoning_content`, 之后的进 `content`。

#### 5.6 常见错误

| 写法 | 结果 |
| --- | --- |
| 没写 `sample`, 或写成 `int sample(float*)` | 编译失败: "缺少 int sample(void* data) 的定义" |
| `float sample(void*)` | 编译失败: "返回值类型不对" |
| 调用 printf / time | 编译失败: "编译结果有导入" |
| 死循环 | "instruction limit exceeded" (约 2s 后) |
| 访问越界 / `__builtin_trap()` | "out of bounds memory access" / "unreachable" |
| 返回 endoftext (248044) 等控制 token | "sample 返回了不允许的 token 248044" |
| 非 thinking 模式返回思考结束 token | "sample 返回了思考结束 token 248069, 但本次不是 thinking 模式" |
| 思考中直接返回 im_end | "sample 在思考还没结束时返回了 im_end 248046 (要先返回思考结束 token 248069)" |

## 二、内部实现

涉及的文件: `server.py` (请求字段)、[wasm_sampler.py](../wasm_sampler.py) (编译和缓存)、[wasm_sampler/prelude.h](../wasm_sampler/prelude.h) (强制包含的头文件)、[qwen_cuda/wasm_sampler.h](../qwen_cuda/wasm_sampler.h) (运行时)、`qwen_cuda/qwen.cpp` (接入采样和 decode 循环)、[tests/test_wasm_sampler.py](../tests/test_wasm_sampler.py)。

### 1. 请求处理 (server.py)

`build_prompt` 里: `sampler_data` 严格 base64 解码 (有非法字符直接 400), 检查不超过 1MB; `wasm_sampler.compile(sampler_cpp)` 得到 wasm 字节 (编译错误转成 400)。这些都在拿到槽位之前完成, 之后 wasm 和 `sampler_data` 作为 `qwen_prefill` 的最后两组参数传给 C++ (`qwen.engine_args(..., wasm=, wasm_data=)`)。请求结束 (`engine_next` 读到空) 后用 `qwen_slot_error(槽)` 查是否出错, 出错时抛 `BadRequest`: 非流式返回 400, 流式在已发出的 token 后发 error 事件。

### 2. 编译 (wasm_sampler.py)

- 不检查源码内容, 只限制大小。编译期的隔离靠 chroot: 编译器能读到的只有镜像里的文件, `#include` / `#embed` 任意路径都只能看到镜像
- 缓存: `<cache-dir>/wasm/<源码字符串 utf-8 的 sha256>.wasm`, 命中直接返回, 永不删除; 先写临时名再改名, 并发编译同一份源码不会读到半个文件。key 只有源码, 改了 prelude、镜像或编译参数要手动清空这个目录。前缀缓存的后台线程只扫描两位十六进制的子目录, 不受 `wasm/` 影响
- 每次编译: 在 `--wasm-tmp-dir` 下建 `qwen_wasm_<pid>_<uuid>`, 把镜像 tar 整个解压进去, 写入 `work/user.cpp` 和 `work/prelude.h`, 然后

  ```bash
  ulimit -v 8388608 -f 262144 -t 60; exec unshare --net chroot <目录> /usr/bin/env -i LD_LIBRARY_PATH=/opt/wasi-sdk/lib /bin/sh -c 'cd /work && clang++ <CFLAGS> -include prelude.h -c user.cpp -o user.o && exec clang++ <LDFLAGS> user.o -o out.wasm'
  ```

  以 root 编译, 没有网络 (`unshare --net`), 不带服务端的环境变量 (`env -i`); 内存 8GB、输出文件 256MB、CPU 60s, 另有 70s 墙钟超时 (杀掉整个进程组)。chroot 里没有 /proc, ld.so 解析不了 RPATH 的 `$ORIGIN`, 所以用 `LD_LIBRARY_PATH`。编译和链接分开, 中间文件留在 `/work`, 不进系统临时目录; 全部相对路径, 编译输出里不出现服务端目录
- 编译参数: `--target=wasm32-wasip1 -std=c++23 -O2 -fno-exceptions -fno-rtti`; 链接 `-mexec-model=reactor` (导出 `_initialize` 做全局构造)、`-fno-exceptions` (用 noeh 版 libc++)、栈 1MB、线性内存最大 1GB
- 编译失败时把编译器输出 (最多 8000 字符) 返回给客户端; 输出里有 "undefined symbol: sample(void*)" 等特征时在前面加上中文提示和要求的签名
- 编译成功后解析 wasm 的 import 段, 有任何导入就拒绝 (运行时也会再检查一次)
- 用完 `shutil.rmtree` 删除整个目录, 编译之间不共享任何文件。启动时清理 `--wasm-tmp-dir` 下所属进程已不存在的 `qwen_wasm_*` (临时目录可能被多个服务共用, 不删其他进程的)
- 镜像 (`sampler_rootfs.tar`, 约 356MB): Rocky 9 官方容器镜像 (Container-Minimal) 的 rootfs, 加上精简的 wasi-sdk 34 (放在 `/opt/wasi-sdk`: clang / lld、libLLVM / libclang-cpp / libedit、clang 资源目录、wasm32-wasip1 的头文件和 noeh 库) 和空的 `/work`。由 `python wasm_sampler.py build-rootfs <Rocky 镜像 tar.xz> <wasi-sdk 目录> <输出 tar>` 生成: 从 OCI 包里流式读出唯一的 rootfs 层写进新 tar, 不解压到磁盘

### 3. prelude.h

用 `-include` 强制包含在用户代码之前, 提供常量、`qs_allowed`、`qs_half_to_float`、两个函数的声明, 以及服务端调用的导出函数:

| 导出 | 作用 |
| --- | --- |
| `_initialize` | reactor 模块的全局构造 (链接器生成) |
| `qs_logits_ptr()` | logits 缓冲 `uint16_t qs_logits[QS_VOCAB]` 的地址 |
| `qs_alloc(n)` | `malloc(n)`, 服务端用它在线性内存里放 custom_data |
| `qs_init(thinking, p, n)` | 调用 `sampler_init` |
| `qs_sample()` | 调用 `sample(qs_logits)` |

另外替换了 libc++ 的 `__libcpp_verbose_abort` 和全部 `operator new` / `delete`: libc++.a 里预编译的 `std::sort` 等带断言, 默认的断言失败处理和 `operator new` 的 new_handler 路径都会经 `fprintf(stderr)` 引入 WASI 导入; 替换成直接 `__builtin_trap()` (运行时报 unreachable) 和 `malloc` / `free` 后, 用 STL 不会产生任何导入。

### 4. 运行时 (qwen_cuda/wasm_sampler.h)

- WAMR 2.4.5 (Bytecode Alliance 的 wasm 运行时) 用 musl 静态编译进 `libqwen_cuda.so`, 只开 fast interpreter 和指令计数, 关掉 WASI / libc-builtin / 线程 / SIMD / AOT / JIT, 关掉硬件边界检查 (不安装信号处理函数, 不影响 python 进程)。不注册任何 native 函数, 有导入的模块直接拒绝。编译时没有 `WAMR_DIR` 则不定义 `QWEN_WAMR`, 接口照常存在, 加载时报错
- 线程: python 的 prefill / decode 线程是 glibc 创建的, 里面不能用 musl 的 malloc / errno / 锁, 所以 WAMR 的所有调用都放在本 so 自己创建的 musl 线程里: 每个槽一个 worker (共 8 个, 16MB 栈)。第一次加载模块时才创建线程并初始化 WAMR (在按 ticket 串行的 prefill 里, 不会并发), 不用自定义采样就不占任何资源
- 内存: WAMR 内部分配全部来自初始化时 mmap 的 1GB 内存池 (`Alloc_With_Pool`, 带锁, 多个 worker 同时用没问题), 线性内存由 WAMR 自己 mmap
- 调用方式: 每个槽一个 `Job` (`seq` / `done` 两个计数), 调用方写好参数后 `seq + 1` 并 futex 唤醒该槽的 worker, worker 做完 `done = seq` 并唤醒; 等待都是直接 futex syscall。采样可以先提交 (`sample_post`) 再等 (`sample_wait`)
- 加载 (每个请求一次, 在 `qwen_prefill` 开头): 卸载该槽上一个实例 → 复制 wasm 字节 (WAMR 要求模块存续期间 buffer 有效) → `wasm_runtime_load` → 检查没有导入 → 实例化 (wasm 栈 1MB) → 调 `_initialize` → 取 `qs_logits_ptr` 并检查整个缓冲区在线性内存内 → 有 custom_data 时 `qs_alloc` 后复制进去 → 调 `qs_init`。任何一步失败, 这个请求不做 prefill, 直接以 done 4 结束 (没有 token)。下一个请求不用自定义采样时卸载旧实例
- 采样: fp32 logits 用 F16C (`_mm256_cvtps_ph`, 就近舍入) 转成 fp16 直接写进线性内存 (每次重新换算地址, 用户代码可能 memory.grow), 调 `qs_sample`。每次调用前设置指令上限 20 亿条, 超过时 WAMR 报 "instruction limit exceeded"; trap、越界等异常的信息原样写进错误
- 返回值检查 (`qwen.cpp` 的 `wasm_check`): 只接受 `[0, 248044)`、思考中的思考结束 token、不在思考中时的 im_end (与内置采样相同); 思考状态由服务端按实际输出维护 (本请求开了 thinking 且还没输出思考结束 token)。不合法、或 wasm 出错时该槽以 done 4 结束, 原因写进槽的 `err`, `qwen_slot_error` 返回给 python
- 结束: im_end 在停止 token 列表里, 遇到即结束 (与内置采样相同, 会写输出缓存点); 生成上限是上下文 (`max_new` = ctx - prompt 长度)

### 5. 与 decode 循环的配合 (qwen.cpp)

- 第一个 token 在 prefill 线程上同步采样 (`sample_slot`)
- decode 每步所有槽一起算完之后采样: 这一批里没有自定义采样的槽时, 走原来的逐槽内置采样, 代码路径与没有这个功能时相同; 有至少一个时, 先给所有自定义采样的槽提交 `sample_post` (各自的 worker 线程并行执行), 内置采样的槽在 decode 线程上算, 最后逐个 `sample_wait` 收齐, 再按原来的逻辑处理停止、输出缓存等
- 慢的自定义采样会拖慢整批 decode (各槽进度对齐), 但多个自定义采样之间互不串行: 实测两个每步都做大量空循环的采样器同时跑, 总耗时 1.8s, 单个是 1.6s

### 6. 测试

`tests/test_wasm_sampler.py --base ... --hf-dir <tokenizer.json 所在目录>` (经 router 时加 `--model qwen3.8-27b-cuda`), 覆盖: 编译错误和各种签名错误的提示、include 宿主机文件 (chroot 里不存在)、调用 WASI 函数、源码 / sampler_data 超过 1MB、`sample` / `sampler_init` / 全局构造的死循环、越界、各种不合法返回值、第 6 次 sample 时 trap (非流式和流式)、按给定 token 序列输出 (开 / 关 thinking, 逐字节相同)、fp16 贪心与内置 temperature=0 输出相同、top_k 示例同一种子输出相同、与内置采样混合并发、两个慢采样器并行; `--long` 另测一直不返回 im_end 时生成到上下文用完。
