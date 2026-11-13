#!/usr/bin/env python3
# 自定义采样: 把请求带来的 C++ 代码 (不可信) 编译成 wasm, 缓存在磁盘 (<源码字符串 utf-8 的 sha256>.wasm, 永不删除; 改了 prelude / 镜像 / 编译参数需要手动清空缓存目录); 接口见 wasm_sampler/prelude.h
# 编译隔离: 每次编译把 chroot 镜像 (Rocky 9 rootfs + 精简的 wasi-sdk, 一个 tar) 解压到临时目录 (--wasm-tmp-dir 下, 默认 /dev/shm) 里独立的子目录, unshare --net + chroot 里以 root 编译, 用完删除, 编译之间不共享任何文件
# 不对源码做任何文本检查 (只限制大小), 编译期能读到的只有镜像里的文件; 编译后检查没有导入, 运行时限制在 C++ 侧 (WAMR)
# 只有真正收到自定义采样请求时才用到镜像和缓存目录, 没配置或不存在时只让这个请求报错
# 用法: python wasm_sampler.py build-rootfs <Rocky-9-Container-Minimal.*.tar.xz> <wasi-sdk 目录> <输出 tar>
import os, sys, json, uuid, shlex, signal, shutil, tarfile, hashlib, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PRELUDE = os.path.join(ROOT, "wasm_sampler", "prelude.h")
SDK = "/opt/wasi-sdk" # 镜像里 wasi-sdk 的位置
SDK_FILES = ["bin/clang-23", "bin/clang", "bin/clang++", "bin/lld", "bin/wasm-ld", "lib/libLLVM.so.23.1-wasi-sdk", "lib/libclang-cpp.so.23.1-wasi-sdk", "lib/libedit.so.0", "lib/libedit.so.0.0.76",
             "lib/clang/23/include", "lib/clang/23/lib/wasm32-unknown-wasip1", "share/wasi-sysroot/include/c++", "share/wasi-sysroot/include/wasm32-wasip1"] # 只要 wasm32-wasip1 + 无异常版本需要的部分
SDK_LIB_SKIP = {"eh", "llvm-lto"} # share/wasi-sysroot/lib/wasm32-wasip1 下不要的目录
CFLAGS = f"--target=wasm32-wasip1 --sysroot={SDK}/share/wasi-sysroot -std=c++23 -O2 -fno-exceptions -fno-rtti"
LDFLAGS = f"--target=wasm32-wasip1 --sysroot={SDK}/share/wasi-sysroot -fno-exceptions -mexec-model=reactor -Wl,-z,stack-size=1048576 -Wl,--max-memory=1073741824" # -fno-exceptions: 链接 noeh 版 libc++; 线性内存最大 1GB
TIMEOUT = 60 # 编译超时 (秒), 同时是 CPU 时间上限
MAX_SRC = 1 << 20 # 源码最多 1MB (utf-8 字节)
MAX_DATA = 1 << 20 # sampler_data 最多 1MB (解码后的二进制)
MAX_ERR = 8000 # 返回给客户端的编译输出最多字符数

G_image = None # chroot 镜像 tar, None 表示未配置
G_dir = None # 缓存目录: <sha256>.wasm
G_tmp = None # 编译用的临时目录放在这里: qwen_wasm_<pid>_<uuid>, 用完删除
SIGS = [("undefined symbol: sample(void*)", "缺少 int sample(void* data) 的定义 (函数名或参数类型不对)"),
        ("undefined symbol: sampler_init(bool, void*, unsigned long long)", "缺少 void sampler_init(bool is_thinking_mode, void* custom_data, uint64_t custom_data_len) 的定义 (函数名或参数类型不对)"),
        ("differ only in their return type", "返回值类型不对: 要求 int sample(void* data) 和 void sampler_init(bool, void*, uint64_t)"),
        ("different language linkage", "sample / sampler_init 不能声明成 extern \"C\"")] # 链接 / 编译报错里的特征 -> 提示


class CompileError(Exception):
    pass


def build_rootfs(rocky, sdk, out): # 从 Rocky 官方容器镜像 (OCI 格式, 单层) 取出 rootfs, 加上精简的 wasi-sdk 和空的 /work, 写成一个 tar; 全程流式, 不解压到磁盘
    with tarfile.open(rocky, "r:xz") as oci:
        idx = json.load(oci.extractfile("index.json"))
        man = json.load(oci.extractfile("blobs/sha256/" + idx["manifests"][0]["digest"].split(":")[1]))
        if len(man["layers"]) != 1:
            raise RuntimeError(f"❌ 镜像有 {len(man['layers'])} 层, 只支持单层")
        layer = oci.extractfile("blobs/sha256/" + man["layers"][0]["digest"].split(":")[1])
        with tarfile.open(out + ".tmp", "w", format=tarfile.PAX_FORMAT) as dst, tarfile.open(fileobj=layer, mode="r|gz") as src:
            n = 0
            for m in src:
                dst.addfile(m, src.extractfile(m) if m.isreg() else None)
                n += 1
            print(f"ℹ️ rootfs {n} 项", file=sys.stderr)
            def add(rel): # 目录递归加入, 软链接保持为软链接
                dst.add(os.path.join(sdk, rel), arcname=SDK[1:] + "/" + rel)
            for d in ("", "bin", "lib", "lib/clang", "lib/clang/23", "lib/clang/23/lib", "share", "share/wasi-sysroot", "share/wasi-sysroot/include", "share/wasi-sysroot/lib"):
                dst.add(os.path.join(sdk, d) if d else sdk, arcname=(SDK[1:] + "/" + d).rstrip("/"), recursive=False)
            for rel in SDK_FILES:
                add(rel)
            lib = "share/wasi-sysroot/lib/wasm32-wasip1"
            dst.add(os.path.join(sdk, lib), arcname=SDK[1:] + "/" + lib, recursive=False)
            for f in sorted(os.listdir(os.path.join(sdk, lib))):
                if f not in SDK_LIB_SKIP:
                    add(lib + "/" + f)
            w = tarfile.TarInfo("work")
            w.type, w.mode = tarfile.DIRTYPE, 0o755
            dst.addfile(w)
    os.replace(out + ".tmp", out)
    print(f"✅ 镜像已生成: {out}, {os.path.getsize(out) / 1048576:.0f}MB", file=sys.stderr)


def init(image, d, tmp): # 只记录路径, 不检查 (用到时再检查, 不影响不用自定义采样的请求)
    global G_image, G_dir, G_tmp
    G_image = os.path.abspath(image) if image else None
    G_dir = os.path.abspath(d) if d else None
    G_tmp = os.path.abspath(tmp)
    if G_image and os.path.isdir(G_tmp): # 清理已退出进程没删完的编译目录 (临时目录可能被多个服务共用, 只删进程已不存在的)
        for f in os.listdir(G_tmp):
            parts = f.split("_")
            if f.startswith("qwen_wasm_") and len(parts) == 4 and parts[2].isdigit() and not os.path.exists(f"/proc/{parts[2]}"):
                shutil.rmtree(os.path.join(G_tmp, f), ignore_errors=True)


def wasm_imports(data): # 返回 [(模块名, 字段名)], 只解析 import 段的名字
    def leb(i):
        r, s = 0, 0
        while True:
            x = data[i]
            i += 1
            r |= (x & 0x7f) << s
            s += 7
            if x < 0x80:
                return r, i
    def name(i):
        n, i = leb(i)
        return data[i:i + n].decode("utf-8", "replace"), i + n
    i = 8
    while i < len(data):
        sid = data[i]
        size, i = leb(i + 1)
        if sid != 2:
            i += size
            continue
        cnt, j = leb(i)
        mod, j = name(j)
        field, j = name(j)
        return [(mod, field)] + [("?", "?")] * (cnt - 1) # 只要知道有没有导入和第一个是什么
    return []


def compile(src): # 返回 wasm bytes
    if not isinstance(src, str):
        raise CompileError("sampler_cpp 必须是字符串")
    code = src.encode("utf-8")
    if len(code) > MAX_SRC:
        raise CompileError(f"源码 {len(code)} 字节, 超过上限 {MAX_SRC}")
    if G_dir is None:
        raise CompileError("服务端没有配置 --cache-dir, 自定义采样的编译缓存放在 <cache-dir>/wasm")
    path = os.path.join(G_dir, hashlib.sha256(code).hexdigest() + ".wasm")
    if os.path.isfile(path):
        return open(path, "rb").read()
    if G_image is None or not os.path.isfile(G_image):
        raise CompileError(f"服务端没有自定义采样的编译镜像 (--sampler-rootfs {G_image})")
    prelude = open(PRELUDE, "rb").read()
    os.makedirs(G_dir, exist_ok=True)
    tmp = os.path.join(G_tmp, f"qwen_wasm_{os.getpid()}_{uuid.uuid4().hex}")
    os.makedirs(tmp)
    try:
        r = subprocess.run(f"tar -xf {shlex.quote(G_image)} -C {shlex.quote(tmp)}", shell=True, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"❌ 解压 chroot 镜像失败: {r.stderr.decode(errors='replace')}")
        with open(os.path.join(tmp, "work", "user.cpp"), "wb") as f:
            f.write(code)
        with open(os.path.join(tmp, "work", "prelude.h"), "wb") as f:
            f.write(prelude)
        inner = f"cd /work && {SDK}/bin/clang++ {CFLAGS} -include prelude.h -c user.cpp -o user.o && exec {SDK}/bin/clang++ {LDFLAGS} user.o -o out.wasm" # 编译和链接分开, 中间文件留在 /work
        cmd = f"ulimit -v 8388608 -f 262144 -t {TIMEOUT}; exec unshare --net chroot {shlex.quote(tmp)} /usr/bin/env -i LD_LIBRARY_PATH={SDK}/lib /bin/sh -c {shlex.quote(inner)}"
        # 内存 8GB, 输出文件 256MB; chroot 里没有 /proc, ld.so 解析不了 RPATH 的 $ORIGIN, 用 LD_LIBRARY_PATH; env -i 不带入服务端的环境变量
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=TIMEOUT + 10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL) # 整个进程组 (shell / unshare / clang)
            p.communicate()
            raise CompileError(f"编译超时 ({TIMEOUT}s)")
        out = out.decode("utf-8", "replace")
        if p.returncode != 0:
            hints = "".join(h + "\n" for k, h in SIGS if k in out)
            raise CompileError(f"编译失败 (返回 {p.returncode}):\n{hints}{out[:MAX_ERR]}")
        data = open(os.path.join(tmp, "work", "out.wasm"), "rb").read()
        imp = wasm_imports(data)
        if imp:
            raise CompileError(f"编译结果有 {len(imp)} 个导入 (第一个 {imp[0][0]}.{imp[0][1]}): 不能调用需要 WASI 的函数 (printf、time 等)")
        with open(path + "." + uuid.uuid4().hex, "wb") as f: # 先写临时名再改名, 并发编译同一份源码也不会读到半个文件
            f.write(data)
        os.replace(f.name, path)
        return data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "build-rootfs":
        print("用法: python wasm_sampler.py build-rootfs <Rocky-9-Container-Minimal.*.tar.xz> <wasi-sdk 目录> <输出 tar>", file=sys.stderr)
        sys.exit(1)
    build_rootfs(sys.argv[2], sys.argv[3], sys.argv[4])
