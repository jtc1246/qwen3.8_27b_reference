#!/bin/bash
set -e
set -o pipefail
cd "$(readlink -f "$(dirname "$0")")"
source ~/.bashrc
if [ -z "$MUSL_BIN" ] && [ -f /share/develop/softwares/rc ]; then
    source /share/develop/softwares/rc # 本机 ~/.bashrc 没有加载 softwares rc
fi
if [ -z "$MUSL_BIN" ]; then
    echo "❌ 未设置 MUSL_BIN, 请参考 README 的环境准备" >&2
    exit 1
fi
if [ -z "$WAMR_DIR" ]; then
    echo "⚠️ 未设置 WAMR_DIR, 编译出的 so 不支持自定义采样 (其余功能不受影响), 请参考 README" >&2
elif [ ! -f "$WAMR_DIR/build-scripts/runtime_lib.cmake" ]; then
    echo "❌ WAMR_DIR=$WAMR_DIR 不是 wasm-micro-runtime 源码目录" >&2
    exit 1
fi
conda activate local_llm
rm -rf cmake_build
cmake -S . -B cmake_build
cmake --build cmake_build -j
echo "✅ 编译完成: $(readlink -f cmake_build/libqwen_cuda.so)"
