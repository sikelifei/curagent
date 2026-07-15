"""Run a resumable, episode-concurrent Oolong-real JSONL evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Iterator

from recursive_agent.envs import run_registered_environment


DEFAULT_DATA_URL = (
    "https://hf-mirror.com/datasets/oolongbench/oolong-real/resolve/main/"
    "dnd/test.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL)
    parser.add_argument("--output-dir", default="outputs/oolong/full_test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--episode-workers", type=int, default=8)
    parser.add_argument("--agent-max-steps", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-concurrent-subagents", type=int, default=8)
    parser.add_argument("--max-run-seconds", type=float, default=900)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("start-index must be non-negative")
    if args.count is not None and args.count <= 0:
        parser.error("count must be positive when supplied")
    if args.episode_workers <= 0:
        parser.error("episode-workers must be positive")

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    issues_path = output_dir / "call_issues.json"
    manifest_path = output_dir / "manifest.json"

    existing = _load_rows(trace_path) if args.resume else []
    rows_by_index = {int(row["instance_id"]): row for row in existing}
    manifest = {
        "environment": "oolong",
        "split": "test",
        "config": str(Path(args.config).resolve()),
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "data_url": None if args.data_path else args.data_url,
        "output_dir": str(output_dir.resolve()),
        "start_index": args.start_index,
        "count": args.count,
        "episode_workers": args.episode_workers,
        "agent_max_steps": args.agent_max_steps,
        "max_depth": args.max_depth,
        "max_concurrent_subagents": args.max_concurrent_subagents,
        "max_run_seconds": args.max_run_seconds,
        "max_observation_chars": args.max_observation_chars,
        "resumed": bool(args.resume),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(manifest_path, manifest)

    pending = _iter_pending_rows(
        _iter_rows(args.data_path, args.data_url),
        start_index=args.start_index,
        count=args.count,
        completed=set(rows_by_index),
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.episode_workers) as executor:
        futures: set[Future[dict[str, Any]]] = set()
        with trace_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
            for index, row in pending:
                futures.add(executor.submit(_run_one, args, index, row))
                if len(futures) >= args.episode_workers * 2:
                    _drain_completed(futures, handle, rows_by_index, summary_path, started, args)
            while futures:
                _drain_completed(futures, handle, rows_by_index, summary_path, started, args)

    summary = _build_summary(rows_by_index, started)
    _write_json(summary_path, summary)
    _write_json(issues_path, _build_issue_report(rows_by_index))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def _iter_rows(data_path: str | None, data_url: str) -> Iterator[dict[str, Any]]:
    if data_path:
        handle = Path(data_path).open("r", encoding="utf-8")
    else:
        request = urllib.request.Request(data_url, headers={"Accept-Encoding": "identity"})
        handle = urllib.request.urlopen(request, timeout=120)
    try:
        for line_number, raw_line in enumerate(handle, 1):
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8")
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield row
    finally:
        handle.close()


def _iter_pending_rows(
    rows: Iterable[dict[str, Any]],
    *,
    start_index: int,
    count: int | None,
    completed: set[int],
) -> Iterator[tuple[int, dict[str, Any]]]:
    stop_index = start_index + count if count is not None else None
    for index, row in enumerate(rows):
        if index < start_index:
            continue
        if stop_index is not None and index >= stop_index:
            break
        if index not in completed:
            yield index, row


def _run_one(args: argparse.Namespace, index: int, row: dict[str, Any]) -> dict[str, Any]:
    episode_started = time.time()
    try:
        run = run_registered_environment(
            "oolong",
            model_config=args.config,
            environment_kwargs={"samples": [row], "instance_id": 0},
            agent_kwargs={
                "max_steps": args.agent_max_steps,
                "max_depth": args.max_depth,
                "max_concurrent_subagents": args.max_concurrent_subagents,
                "max_run_seconds": args.max_run_seconds,
                "max_observation_chars": args.max_observation_chars,
            },
        )
        return {
            "instance_id": index,
            "ok": True,
            "duration_seconds": time.time() - episode_started,
            "score": run.environment_report.get("score", 0.0),
            "run": run.to_trace_dict(),
        }
    except Exception as exc:
        return {
            "instance_id": index,
            "ok": False,
            "duration_seconds": time.time() - episode_started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _drain_completed(
    futures: set[Future[dict[str, Any]]],
    handle: Any,
    rows_by_index: dict[int, dict[str, Any]],
    summary_path: Path,
    started: float,
    args: argparse.Namespace,
) -> None:
    done, _ = wait(futures, return_when="FIRST_COMPLETED")
    for future in done:
        futures.remove(future)
        row = future.result()
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        rows_by_index[int(row["instance_id"])] = row
        _write_json(summary_path, _build_summary(rows_by_index, started))
        print(
            f"completed index={row['instance_id']} ok={row['ok']} "
            f"duration={row['duration_seconds']:.1f}s",
            flush=True,
        )


def _build_summary(rows_by_index: dict[int, dict[str, Any]], started: float) -> dict[str, Any]:
    rows = list(rows_by_index.values())
    completed = [row for row in rows if row.get("ok")]
    scores = [float(row.get("score", 0.0) or 0.0) for row in completed]
    return {
        "environment": "oolong",
        "split": "test",
        "recorded_rows": len(rows),
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


def _build_issue_report(rows_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    report = {"rows": len(rows_by_index), "agent_statuses": {}, "errors": [], "tool_counts": {}}
    for row in rows_by_index.values():
        run = row.get("run") or {}
        result = run.get("agent_result") or {}
        status = result.get("status", "failed")
        report["agent_statuses"][status] = report["agent_statuses"].get(status, 0) + 1
        trace = result.get("trace") or {}
        stack = [trace]
        while stack:
            agent = stack.pop()
            stack.extend(agent.get("children") or [])
            for step in agent.get("steps", []):
                for execution in step.get("code_executions", []):
                    if execution.get("error"):
                        report["errors"].append({
                            "instance_id": row.get("instance_id"),
                            "depth": agent.get("depth"),
                            "error": execution["error"],
                        })
                    code = execution.get("code", "")
                    for tool in ("spawn_subagent", "spawn_subagents", "submit_answer"):
                        if re.search(rf"\b{tool}\s*\(", code):
                            report["tool_counts"][tool] = report["tool_counts"].get(tool, 0) + 1
    report["error_count"] = len(report["errors"])
    return report


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
