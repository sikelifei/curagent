"""Run and audit a batch of ReCode WebShop episodes with full JSONL traces."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from recursive_agent.envs import (
    aggregate_trace_metrics,
    analyze_environment_trace,
    create_environment,
    run_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--recode-root", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--env-max-steps", type=int, default=30)
    parser.add_argument("--agent-max-steps", type=int, default=35)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-concurrent-subagents", type=int, default=4)
    parser.add_argument("--max-run-seconds", type=float, default=900)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--trace-jsonl", default="outputs/webshop_200_traces.jsonl")
    parser.add_argument("--summary-json", default="outputs/webshop_200_summary.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.count <= 0 or args.concurrency <= 0:
        parser.error("count and concurrency must be positive")
    indices = list(range(args.start_index, args.start_index + args.count))
    trace_path = Path(args.trace_jsonl)
    summary_path = Path(args.summary_json)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_rows(trace_path) if args.resume else []
    completed_indices = {int(row["instance_id"]) for row in existing}
    pending = [index for index in indices if index not in completed_indices]
    rows_by_index = {int(row["instance_id"]): row for row in existing}

    prewarmed = None
    if pending:
        prewarmed = _create_episode(args, pending[0])

    started = time.time()
    with trace_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        if args.concurrency == 1:
            for position, index in enumerate(pending):
                environment = prewarmed if position == 0 else None
                row = _run_one(args, index, environment=environment)
                _record_row(handle, row, rows_by_index, summary_path, indices, started)
        else:
            with ThreadPoolExecutor(
                max_workers=args.concurrency,
                thread_name_prefix="webshop-episode",
            ) as executor:
                futures: dict[Future[dict[str, Any]], int] = {}
                for position, index in enumerate(pending):
                    environment = prewarmed if position == 0 else None
                    futures[executor.submit(_run_one, args, index, environment)] = index
                for future in as_completed(futures):
                    row = future.result()
                    _record_row(handle, row, rows_by_index, summary_path, indices, started)

    ordered = [rows_by_index[index] for index in indices if index in rows_by_index]
    summary = _build_summary(ordered, indices, started)
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _create_episode(args: argparse.Namespace, index: int):
    return create_environment(
        "webshop",
        recode_root=args.recode_root,
        split=args.split,
        instance_id=index,
        max_steps=args.env_max_steps,
        seed=args.seed,
    )


def _run_one(
    args: argparse.Namespace,
    index: int,
    environment: Any | None = None,
) -> dict[str, Any]:
    episode_started = time.time()
    try:
        environment = environment or _create_episode(args, index)
        run = run_environment(
            environment,
            model_config=args.config,
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
            "metrics": analyze_environment_trace(trace),
            "run": trace,
        }
    except Exception as exc:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass
        return {
            "instance_id": index,
            "ok": False,
            "duration_seconds": time.time() - episode_started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _record_row(
    handle: Any,
    row: dict[str, Any],
    rows_by_index: dict[int, dict[str, Any]],
    summary_path: Path,
    indices: list[int],
    started: float,
) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    handle.flush()
    rows_by_index[int(row["instance_id"])] = row
    ordered = [rows_by_index[index] for index in indices if index in rows_by_index]
    summary = _build_summary(ordered, indices, started)
    _write_json(summary_path, summary)
    print(
        f"[{len(ordered)}/{len(indices)}] index={row['instance_id']} "
        f"ok={row['ok']} duration={row['duration_seconds']:.1f}s",
        flush=True,
    )


def _build_summary(
    rows: list[dict[str, Any]],
    indices: list[int],
    started: float,
) -> dict[str, Any]:
    summary = aggregate_trace_metrics(rows)
    summary.update(
        {
            "requested_indices": indices,
            "recorded_rows": len(rows),
            "elapsed_seconds": time.time() - started,
        }
    )
    return summary


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
