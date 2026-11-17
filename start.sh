#!/bin/bash

source ~/.bashrc
export CANN_HOME="${CANN_HOME:-/root/ascend-npu-test/cann/ascend-toolkit/latest}"
source "$CANN_HOME/set_env.sh" # npu 后端需要 CANN 环境

set -e
set -o pipefail

cd "$(readlink -f "$(dirname "$0")")"
conda activate local_llm

log_file="logs/$(date +%Y%m%d_%H%M%S_%3N).txt"

python -u router.py --gguf ../models/qwen38_27b_q4km/Qwen3.8-27B-Q4_K_M.gguf --st ../models/qwen38_27b_redhat_int4/model.safetensors --hf-dir ../models/qwen38_27b_hf --cache-dir ../kv_cache/ > "$log_file" 2>&1