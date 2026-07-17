# Oolong-Synth 1M Recursion Experiments

Date: 2026-07-18

## Scope

These are single-example engineering checks against the unfiltered 1M-token
bucket. They test whether prompt-selected decomposition can process the complete
semantic workload. They are not benchmark estimates.

The model API is `deepseek-v4-flash`. All runs use temperature 0, one episode
worker, 4,096 maximum output tokens per model call, 16 maximum concurrent
subagents, and source index 650 from the validation split.

## Sample

- Source index: 650
- Dataset: `spam`
- Declared `context_len`: 1,048,576 (2^20, the 1M bucket)
- Actual context characters: 2,689,330
- Complete data rows: 17,469
- Question: least common label among `ham` and `spam`
- Gold answer: `ham`
- Filtering: none; all 17,469 rows require semantic classification

`context_len` is the dataset's target token-length bucket, not 131K. The 131K
bucket is 131,072 (2^17).

## Prior Control

The same sample with a 12,000-character observation limit, depth 1, and a
900-second run limit failed with `TimeoutExceededError` after 982.12 seconds.
Output directory:
`outputs/oolong_synth/deepseek_v4_flash_adaptive_1m_full_semantic_20260717`.

## Sequential Runs

| Run | Observation limit | Max depth | Prompt flow | Result | Duration |
| --- | ---: | ---: | --- | --- | ---: |
| A | 200,000 | 1 | adaptive | running | - |
| B | 50,000 | 1 | adaptive | pending | - |
| C | 200,000 | 2 | recursive | pending | - |

Run A command:

```bash
python -u -m examples.run_oolong_synth \
  --config configs/model_api.local.yaml \
  --sample-count 100 \
  --min-context-len 1048576 --max-context-len 1048576 \
  --start-index 0 --count 1 \
  --episode-workers 1 \
  --agent-max-steps 16 \
  --max-depth 1 \
  --max-concurrent-subagents 16 \
  --max-run-seconds 3600 \
  --max-observation-chars 200000 \
  --request-timeout 600 \
  --temperature 0 --max-tokens 4096 \
  --bootstrap-samples 1000 \
  --output-dir outputs/oolong_synth/deepseek_v4_flash_1m_full_semantic_large_obs_20260718
```

Run B uses the same command and sample, with
`--max-observation-chars 50000` and its own output directory.

Run C uses the same command and sample, with `--max-depth 2`, the 200,000
observation limit, the recursive prompt flow, and its own output directory.

## Evidence To Capture

For every run record: submitted answer, score, duration, status/error, total
model calls and tokens, root steps, child counts by depth, chunk coverage, and
any truncated observations. Keep generated manifests, exact prompt snapshots,
JSONL traces, and summaries in the named output directory.

## Code Boundary

Prompt flows may decide filtering, chunk sizes, fan-out, recursion, and merging.
The generic `RecursiveAgent`, REPL observation truncation, subagent scheduler,
and depth implementation remain unchanged. A runtime prompt-flow selector may
choose between preserved prompt variants; it must not implement decomposition.
