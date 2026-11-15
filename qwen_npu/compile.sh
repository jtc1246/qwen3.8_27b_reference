#!/bin/bash
set -e
set -o pipefail
cd "$(readlink -f "$(dirname "$0")")"
export CANN_HOME="${CANN_HOME:-/root/ascend-npu-test/cann/ascend-toolkit/latest}"
CMAKE_BIN="${CMAKE_BIN:-/root/ascend-npu-test/cmake-venv/bin/cmake}"
set +u
source "$CANN_HOME/set_env.sh"
set -u
rm -rf cmake_build
"$CMAKE_BIN" -S . -B cmake_build -DCMAKE_BUILD_TYPE=Release
"$CMAKE_BIN" --build cmake_build --parallel 8
