#!/usr/bin/env bash
set -euo pipefail

MODEL=/data2/zhangwenjian/model/Qwen3-1.7B
MYVLLM=/data2/zhangwenjian/miniconda3/envs/myvllm
CUDA_HOME="$MYVLLM/lib/python3.12/site-packages/nvidia/cu13"
PORT=${PORT:-56782}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$MYVLLM/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
unset NVCC_CCBIN CUDAHOSTCXX GCC_EXEC_PREFIX COMPILER_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CONDA_PREFIX="$CUDA_HOME"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
hash -r

exec "$MYVLLM/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS"
