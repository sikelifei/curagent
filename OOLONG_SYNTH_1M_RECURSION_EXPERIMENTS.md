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

## Prompt Flows

The original A prompt is preserved byte-for-byte in
`recursive_agent/envs/oolong_synth/prompts.py`. The selectable alternatives
live in `flow_prompts.py` and are selected with `--prompt-flow`:

- `adaptive_flat`: original model-selected adaptive prompt, depth 1.
- `paged_flat`: flat workers with bounded pages, depth 1.
- `hierarchical`: root creates coarse ranges; `can_delegate=True` workers
  create leaf ranges with `can_delegate=False`, depth 2.

The generic root prompt in `recursive_agent/prompts.py` was not modified.

## Runs

| Run | Sample(s) | Observation | Steps | Depth | Flow | Result | Duration |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| A0 | 650 | 200,000 | 24 | 1 | adaptive_flat | cancelled, no result | 57.8 min |
| A1 | 650, 651 | 200,000 | 40 | 1 | adaptive_flat | 0/2 correct; 651 timeout | 33.7/60.7 min |
| B | 650 | 50,000 | 40 | 1 | paged_flat | 0/1 correct | 17.8 min |
| C1 | 650 | 200,000 | 40 | 2 | hierarchical | 0/1; no depth 2 | 18.8 min |
| C2 | 650 | 200,000 | 40 | 2 | hierarchical v2 | 1/1 correct; depth 2 real | 41.6 min |

The two A1 samples use the same full context but different questions. Source 650
asks for the least common label (`ham`); source 651 asks for the number of ham
rows (`8638`). Both are unfiltered and have 17,469 semantic rows.

A0 command (cancelled after no output was produced):

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

A1 command:

```bash
python -u -m examples.run_oolong_synth \
  --config configs/model_api.local.yaml \
  --sample-count 100 \
  --min-context-len 1048576 --max-context-len 1048576 \
  --start-index 0 --count 2 \
  --episode-workers 2 \
  --agent-max-steps 40 \
  --max-depth 1 \
  --max-concurrent-subagents 16 \
  --max-run-seconds 3600 \
  --max-observation-chars 200000 \
  --prompt-flow adaptive_flat \
  --request-timeout 600 \
  --temperature 0 --max-tokens 4096 \
  --bootstrap-samples 1000 \
  --output-dir outputs/oolong_synth/deepseek_v4_flash_1m_full_semantic_adaptive_flat_two_parallel_steps40_children16_20260718
```

The A1 source 650 row submitted `Label: spam` in 2,019.7 seconds (gold
`Label: ham`), with root 9 steps, 26 top-level children, 477 calls, and
18,632,708 input tokens. Source 651 timed out after 3,641.6 seconds.

B command:

```bash
python -u -m examples.run_oolong_synth \
  --config configs/model_api.local.yaml \
  --sample-count 100 \
  --min-context-len 1048576 --max-context-len 1048576 \
  --start-index 0 --count 1 \
  --episode-workers 1 \
  --agent-max-steps 40 \
  --max-depth 1 \
  --max-concurrent-subagents 16 \
  --max-run-seconds 3600 \
  --max-observation-chars 50000 \
  --prompt-flow paged_flat \
  --request-timeout 600 \
  --temperature 0 --max-tokens 4096 \
  --bootstrap-samples 1000 \
  --output-dir outputs/oolong_synth/deepseek_v4_flash_1m_full_semantic_paged_flat_50k_steps40_children16_20260718
```

B ended in 1,066.2 seconds with a forced-final root answer and score 0. It
used 78 calls and 7 children; 4 children completed, 2 were rejected by content
inspection, and 1 hit quota.

C1 initially completed in 1,128.3 seconds but produced no depth-2 agents: its
four depth-1 workers were given malformed contexts and the root fell back to a
regex classifier. It scored 0 and is retained as a failed prompt iteration.

C2 command:

```bash
python -u -m examples.run_oolong_synth \
  --config configs/model_api.local.yaml \
  --sample-count 100 \
  --min-context-len 1048576 --max-context-len 1048576 \
  --start-index 0 --count 1 \
  --episode-workers 1 \
  --agent-max-steps 40 \
  --max-depth 2 \
  --max-concurrent-subagents 16 \
  --max-run-seconds 3600 \
  --max-observation-chars 200000 \
  --prompt-flow hierarchical \
  --request-timeout 600 \
  --temperature 0 --max-tokens 4096 \
  --bootstrap-samples 1000 \
  --output-dir outputs/oolong_synth/deepseek_v4_flash_1m_full_semantic_hierarchical_v2_200k_depth2_steps40_children16_20260718
```

C2 submitted `Label: ham` correctly in 2,494.2 seconds. The trace contains
one root, four completed depth-1 coarse workers, and 33 depth-2 leaf attempts
(19 completed, 5 forced-final, 9 API errors: 8 content-inspection and 1
quota). The root used 7 steps and 719 calls. Despite incomplete leaf coverage,
the model returned the correct global label; the score was 1.0.

## Evidence To Capture

For every run, generated manifests, prompt snapshots, JSONL traces, and
summaries remain in the named output directory. The experiments show that
larger observation and more root steps do not by themselves improve semantic
accuracy. The hierarchical flow is the only tested flow that both produced a
real depth-2 tree and returned a correct answer on the full 1M sample.

## Code Boundary

Prompt flows may decide filtering, chunk sizes, fan-out, recursion, and merging.
The generic `RecursiveAgent`, REPL observation truncation, subagent scheduler,
and depth implementation remain unchanged. A runtime prompt-flow selector may
choose between preserved prompt variants; it must not implement decomposition.
