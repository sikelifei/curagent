# Simplified Recursive Agent Harness

The current target, where every recursive node shares the same optional task
environment and children receive explicit task/context, is specified in
[`docs/simple_recursive_agent_target.md`](docs/simple_recursive_agent_target.md).

## Implementation

- Root and child nodes run the same `AgentNode` loop.
- The task tree has one shared `max_total_steps` limit and one `max_depth` limit.
- Every model output consumes one shared step before parsing or validation.
- A model-service failure produces no output, releases its reserved step, and is not retried.
- Exactly one tool call may execute per output; multiple calls execute nothing.
- Parse errors, schema errors, tool rejection strings, child failures, and execution exceptions are ordinary trajectory results. The model decides what to do next.
- The harness never repairs, rewrites, replays, substitutes, or automatically retries a call.
- `spawn_agent`, `spawn_agents`, and `finish` are the common recursive controls.
- A child receives explicit `task`, `context`, and optional `expected_output`; its prompt still
  uses the same six-field payload, and its result contains only `result` and `error`.
- `spawn_agents` waits for children sequentially in input order. There is no child-count or child-concurrency limit.
- Prompt feedback for long execution results and malformed output is truncated to about 1000 tokens; the external trace keeps the full value.
- Environments are optional. When present, every node receives the same reference and the same action tools. Reward and benchmark reporting remain in the WebShop evaluator.

The minimal decision prompt contains `task`, `context`, node-local `trajectory`, current
`observation` (or `null`), available `tools`, and shared `remaining_steps`.

## Layout

```text
curagent/core/          Generic agent loop, shared steps, scheduler, prompt, trace
curagent/environments/  Minimal environment interface and WebShop adapters
curagent/models/        OpenAI-compatible native tool-call adapter
curagent/harness/       Small or batch ReCode WebShop evaluation
curagent/tests/         Runtime invariant and integration tests
configs/                DeepSeek and local-vLLM model configurations
docs/                   Design and evaluation evidence
outputs/                Raw reports and traces (git-ignored)
```

## Tests

```bash
python -m unittest discover curagent/tests -v
python -m compileall -q curagent
git diff --check
```

## DeepSeek-V4-Flash small evaluation

The example config is complete except for the secret. Set the key through the environment:

```bash
export DASHSCOPE_API_KEY='your-key'

python -m curagent.harness.webshop_eval \
  --config configs/planner_api.example.yaml \
  --split test --start-id 0 --num-instances 3 \
  --episode-concurrency 2 \
  --max-total-steps 24 --max-depth 3 \
  --output outputs/curagent_webshop_deepseek_small.json \
  --trace-dir outputs/curagent_webshop_deepseek_small_traces
```

`configs/planner_api.local.yaml` is ignored by Git and may hold a local credential for direct
execution. The report is updated atomically after each completed episode.

## Local Qwen small evaluation

Start vLLM with native tool parsing:

```bash
source /data2/zhangwenjian/miniconda3/etc/profile.d/conda.sh
conda activate myvllm

export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export VLLM_USE_FLASHINFER_SAMPLER=0
unset NVCC_CCBIN CUDAHOSTCXX GCC_EXEC_PREFIX COMPILER_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

CUDA_VISIBLE_DEVICES=1 vllm serve /data2/zhangwenjian/model/Qwen3-4B \
  --host 127.0.0.1 --port 56789 \
  --served-model-name qwen3-4b \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.5 --max-model-len 32768 \
  --disable-uvicorn-access-log
```

Choose a currently free GPU. The tested `myvllm` environment uses CUDA 13; disabling the
optional FlashInfer sampler avoids falling back to an incompatible system `nvcc` during JIT.

Then run the same harness with the local model configuration:

```bash
python -m curagent.harness.webshop_eval \
  --config configs/curagent_vllm_qwen3.yaml \
  --split test --start-id 0 --num-instances 3 \
  --episode-concurrency 1 \
  --max-total-steps 24 --max-depth 3 \
  --output outputs/curagent_webshop_qwen3_4b_small.json \
  --trace-dir outputs/curagent_webshop_qwen3_4b_small_traces
```

Each episode records `total_steps`, observed tool calls, and `children_spawned`. These are
telemetry only; only total steps and depth constrain the task tree.

The previous strict-harness run is retained as historical evidence in
[`docs/webshop_qwen3_4b_eval_2026-07-14.md`](docs/webshop_qwen3_4b_eval_2026-07-14.md).
