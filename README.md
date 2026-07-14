# Strict Recursive Agent Harness

This repository implements the strict recursive tool-call harness specified in
[`designed.md`](designed.md).

## Runtime guarantees

- Root and child nodes run the same `AgentNode` loop.
- A model response must contain exactly one native tool call. Strict whole-document JSON is an explicit fallback protocol.
- Tool names and arguments are validated against JSON schema without coercion, extraction, completion, or action substitution.
- `spawn_agent`, `spawn_agents`, and `finish` are the common recursive controls.
- A child receives only its explicit JSON `task`, `context`, and optional `expected_output`.
- The parent receives only the final `SubagentResult` in its own tool receipt.
- Model, tool, child, depth, step, and concurrency budgets cover the complete task tree.
- `spawn_agents` validates every spec and reserves the full child budget before starting any child.
- Environment writes use observation versions and structured `ExecutionReceipt` values.
- Unknown effects are reconciled once and are never replayed automatically.
- Python is absent by default and can be enabled only as an isolated pure-computation tool.

## Layout

```text
.
├── designed.md        Original design specification
├── curagent/          Python package
│   ├── core/          Agent loop, strict calls, scheduler, budgets, prompts, traces
│   ├── environments/  Common interface, mock WebShop, ReCode WebShop
│   ├── executors/     Optional Python capability
│   ├── models/        OpenAI-compatible native tool-call client
│   ├── tasks/         Environment-only task modules
│   ├── harness/       ReCode WebShop evaluation
│   └── tests/         Runtime invariant tests
├── configs/           Local model configuration
├── docs/              Evaluation report
└── outputs/           Raw 200-instance report and traces (git-ignored)
```

## Tests

From this repository root:

```bash
python -m unittest discover curagent/tests -v
python -m compileall -q curagent
```

## Local WebShop evaluation

Start a local model server with native tool parsing:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve /data2/zhangwenjian/model/Qwen3-4B \
  --host 127.0.0.1 --port 56789 \
  --served-model-name qwen3-4b \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.7 --max-model-len 32768 \
  --disable-uvicorn-access-log --disable-log-requests
```

Run the requested 200-instance ReCode WebShop test evaluation:

```bash
python -m curagent.harness.webshop_eval \
  --config configs/curagent_vllm_qwen3.yaml \
  --split test --start-id 0 --num-instances 200 \
  --episode-concurrency 8 \
  --max-steps-per-agent 12 \
  --max-model-calls-total 24 \
  --max-tool-calls-total 24 \
  --max-depth 3 --max-children-total 6 --max-concurrency 4 \
  --output outputs/curagent_webshop_qwen3_4b_200_20260714.json \
  --trace-dir outputs/curagent_webshop_qwen3_4b_200_20260714_traces
```

The output is updated atomically after every completed episode. One episode
failure does not cancel the remaining batch. Completed metrics and failure
analysis are in [`docs/webshop_eval_2026-07-14.md`](docs/webshop_eval_2026-07-14.md).
