# DeepDive on curagent

This integration keeps inference entirely inside curagent. It uses curagent's
existing persistent REPL, `spawn_subagent`, `spawn_subagents`, recursion limits,
model clients, trace tree, timeouts, and completion dictionary.

Only the DeepDive-specific layer is ported:

- questions and ground truth come from the existing `platoon.deepdive` harness;
- `search_web` and `view_webpage_content` call the existing DeepDive/Tavily tools;
- the research and delegation prompt follows the source DeepDive wording,
  translated only where necessary to curagent's native function names and REPL
  format;
- root answers can be judged with the same success/reason JSON contract used by
  the DeepDive environment.

The agent never receives the ground-truth answer. It is retained only by the
batch runner for post-run evaluation.

Run 20 sampling episodes in the `rao` environment:

```bash
cd /data2/zhangwenjian/agent/curagent
/data2/zhangwenjian/miniconda3/envs/rao/bin/python -u -m examples.run_deepdive \
  --config configs/model_api.local.yaml \
  --platoon-root /data2/zhangwenjian/agent/platoon \
  --split qa_rl \
  --start-index 0 \
  --limit 20 \
  --concurrency 4 \
  --agent-max-steps 25 \
  --max-recursion-depth 4 \
  --max-concurrent-subagents 4 \
  --max-tokens 2048 \
  --output-dir outputs/deepdive_curagent_20
```

Add `--skip-evaluator` when only trajectories are needed. Add `--resume` to skip
completed task records. Specific tasks can be selected with repeated or
comma-separated `--task-id`, for example:

```bash
--task-id deepdive.qa_rl.8,deepdive.qa_rl.14
```

Outputs are written incrementally:

- `runs/`: one compact result and evaluation record per task;
- `trajectories/`: complete root/child curagent trace trees, prompts, tool calls,
  observations, usage, and environment events;
- `logs/`: live step JSONL files that survive an interrupted batch;
- `summary.json`: success, recursion, step, depth, and search statistics.
