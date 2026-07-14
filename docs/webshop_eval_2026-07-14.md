# Strict Recursive Harness: ReCode WebShop Evaluation

## Scope

This is the requested 200-instance `test` evaluation for the strict recursive
tool-call harness.

- Benchmark: ReCode WebShop
- Instances: test ids `0..199`
- Protocol: `strict_recursive_tool_call_v1`
- Model: local `Qwen3-4B`
- Model API: vLLM 0.8.5, native tool calls, Hermes parser
- Model server context: 32,768 tokens
- External model API: not used

Raw artifacts are available locally in the ignored output directory:

- `outputs/curagent_webshop_qwen3_4b_200_20260714.json`
- `outputs/curagent_webshop_qwen3_4b_200_20260714_traces/`

The raw report contains 200 episodes with 200 unique task ids, and the trace
directory contains one JSON trace per episode.

## Command

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

Model request settings were temperature `0`, maximum output `512` tokens,
native tools enabled, parallel tool calls disabled at the client, and Qwen
thinking disabled. vLLM notes that its parser ignores the client parallel-call
flag, so the strict harness still validates the actual returned call count.

## Metrics

| Metric | Value |
| --- | ---: |
| Completed | 200 / 200 |
| Average reward | 0.5562083333 |
| Median reward | 0.6 |
| Full-reward count | 37 |
| Full-reward rate | 18.5% |
| Reward >= 0.5 | 137 (68.5%) |
| Zero reward | 20 (10.0%) |
| Status `ok` | 189 |
| Status `error` | 10 |
| Status `budget_exhausted` | 1 |
| Model calls | 753 total, 3.765 average |
| Tool calls | 732 total, 3.66 average |
| Children launched | 4 |
| Wall time | 277.878565 seconds |

`Full-reward count` uses WebShop reward `>= 1.0`. Status `ok` means the
environment reached a terminal purchase; it does not imply a full reward.

## Non-OK Episodes

- 9 episodes returned two to four native calls in one model response. The
  harness rejected each response exactly as required; it did not select one
  call, combine calls, or rewrite them.
- 1 episode exceeded the local model server's 32,768-token context while
  retaining the node's complete trajectory.
- 1 episode consumed the shared per-tree limit of 24 model calls and returned
  `budget_exhausted` rather than continuing recursion.

## Validation

```bash
python -m unittest discover curagent/tests -v
# 25 tests passed

python -m compileall -q curagent
git diff --check
```

The tests cover strict native/JSON parsing, duplicate and non-finite JSON
rejection, schema validation, one repair decision, infrastructure retry
classification, shared atomic budgets, recursive concurrency handoff, batch
preflight/barrier behavior, context and trace isolation, access modes,
versioned receipts, committed and unknown effects, and the optional Python
capability.
