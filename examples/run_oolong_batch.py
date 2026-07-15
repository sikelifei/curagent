"""Run and score a batch of Oolong-real examples with JSONL traces."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from recursive_agent.envs import run_registered_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--oolong-root", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="oolongbench/oolong-real")
    parser.add_argument("--config-name", default="dnd")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--agent-max-steps", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-concurrent-subagents", type=int, default=2)
    parser.add_argument("--max-run-seconds", type=float, default=900)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--trace-jsonl", default="outputs/oolong_real_traces.jsonl")
    parser.add_argument("--summary-json", default="outputs/oolong_real_summary.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("count must be positive")
    indices = list(range(args.start_index, args.start_index + args.count))
    trace_path = Path(args.trace_jsonl)
    summary_path = Path(args.summary_json)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_rows(trace_path) if args.resume else []
    rows_by_index = {int(row["instance_id"]): row for row in existing}
    pending = [index for index in indices if index not in rows_by_index]
    started = time.time()
    with trace_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index in pending:
            row = _run_one(args, index)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            rows_by_index[index] = row
            _write_json(
                summary_path,
                _build_summary(rows_by_index, indices, started, args.split),
            )
            print(
                f"[{len(rows_by_index)}/{len(indices)}] index={index} "
                f"ok={row['ok']} duration={row['duration_seconds']:.1f}s",
                flush=True,
            )

    summary = _build_summary(rows_by_index, indices, started, args.split)
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def _run_one(args: argparse.Namespace, index: int) -> dict[str, Any]:
    episode_started = time.time()
    try:
        run = run_registered_environment(
            "oolong",
            model_config=args.config,
            environment_kwargs={
                "oolong_root": args.oolong_root,
                "data_path": args.data_path,
                "dataset_name": args.dataset_name,
                "config_name": args.config_name,
                "split": args.split,
                "instance_id": index,
            },
            agent_kwargs={
                "max_steps": args.agent_max_steps,
                "max_depth": args.max_depth,
                "max_concurrent_subagents": args.max_concurrent_subagents,
                "max_run_seconds": args.max_run_seconds,
                "max_observation_chars": args.max_observation_chars,
            },
        )
        trace = run.to_trace_dict()
        return {
            "instance_id": index,
            "ok": True,
            "duration_seconds": time.time() - episode_started,
            "score": run.environment_report.get("score", 0.0),
            "run": trace,
        }
    except Exception as exc:
        return {
            "instance_id": index,
            "ok": False,
            "duration_seconds": time.time() - episode_started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _build_summary(
    rows_by_index: dict[int, dict[str, Any]],
    indices: list[int],
    started: float,
    split: str,
) -> dict[str, Any]:
    rows = [rows_by_index[index] for index in indices if index in rows_by_index]
    completed = [row for row in rows if row.get("ok")]
    scores = [float(row.get("score", 0.0) or 0.0) for row in completed]
    return {
        "environment": "oolong",
        "split": split,
        "requested_indices": indices,
        "recorded_rows": len(rows),
        "episodes_requested": len(indices),
        "episodes_completed": len(completed),
        "episodes_failed": len(rows) - len(completed),
        "submitted": sum(
            bool((row.get("run") or {}).get("environment_report", {}).get("submitted"))
            for row in completed
        ),
        "total_score": sum(scores),
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "elapsed_seconds": time.time() - started,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            rows.append(row)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
