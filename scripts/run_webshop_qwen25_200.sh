#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=/data2/zhangwenjian/model/Qwen2.5-1.5B-Instruct
MYVLLM=/data2/zhangwenjian/miniconda3/envs/myvllm
RECODE_ROOT=${RECODE_ROOT:-/data2/zhangwenjian/agent/ReCode}
JAVA_HOME=/data2/zhangwenjian/miniconda3/envs/recode/lib/jvm
PORT=${PORT:-56780}
GPU=${GPU:-2}
CONCURRENCY=${CONCURRENCY:-16}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/webshop_200_qwen25_1_5b_prompt_print}

CUDA_HOME="$MYVLLM/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME
export PATH="$JAVA_HOME/bin:$CUDA_HOME/bin:$MYVLLM/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CONDA_PREFIX="$CUDA_HOME"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib"
unset NVCC_CCBIN CUDAHOSTCXX GCC_EXEC_PREFIX COMPILER_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

"$MYVLLM/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  >"$OUTPUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 2
done

curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null

"/data2/zhangwenjian/miniconda3/envs/recode/bin/python" \
  examples/run_webshop_batch.py \
  --config configs/model_api_qwen25_vllm.yaml \
  --recode-root "$RECODE_ROOT" \
  --split test \
  --start-index 0 \
  --count 200 \
  --concurrency "$CONCURRENCY" \
  --env-max-steps 30 \
  --agent-max-steps 35 \
  --max-depth 2 \
  --max-concurrent-subagents 4 \
  --max-run-seconds 900 \
  --trace-jsonl "$OUTPUT_DIR/traces.jsonl" \
  --summary-json "$OUTPUT_DIR/summary.json"
