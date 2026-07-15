# Oolong Full Evaluation Records

This directory is reserved for full Oolong-real evaluation records.

The full runner writes:

- `manifest.json`: dataset source, model configuration path, concurrency, and run parameters.
- `summary.json`: resumable aggregate progress and final score.
- `call_issues.json`: agent status, tool-call counts, and REPL errors.
- `traces/episodes.jsonl`: one credential-free execution trace per evaluated sample.

Run with:

```bash
python -m examples.run_oolong_full \
  --config configs/model_api.local.yaml \
  --output-dir outputs/oolong/full_test \
  --episode-workers 16 \
  --max-concurrent-subagents 8 \
  --resume
```

The raw 9GB dataset is kept outside Git. Results and traces are also kept out of
Git because they contain large transcript-derived records; this README and the
runner implementation are versioned.
