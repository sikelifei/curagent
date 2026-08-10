"""Run TextCraft-Synth episodes with the recursive agent."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from recursive_agent.envs import run_registered_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--data-path")
    parser.add_argument("--textcraft-root")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Episode index for a single run; otherwise use --start-index.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--generated-count", type=int, default=1)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-file")
    parser.add_argument("--model-name")
    parser.add_argument("--agent-max-steps", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-concurrent-subagents", type=int, default=4)
    parser.add_argument("--max-subagents-per-agent", type=int, default=6)
    parser.add_argument("--max-run-seconds", type=float, default=900.0)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="OpenAI-compatible client retries for transient API failures.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--trace-json")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--summary-json")
    args = parser.parse_args()
    _validate_args(parser, args)

    prompt = None
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for position in range(args.count):
        base_index = (
            args.instance_id if args.instance_id is not None else args.start_index
        )
        instance_id = base_index + position
        started = time.monotonic()
        row = _run_one(args, instance_id, prompt, started)
        rows.append(row)
        print(
            f"instance={instance_id} success={row.get('success', False)} "
            f"score={float(row.get('score', 0.0) or 0.0):.3f} "
            f"steps={row.get('steps', 0)} "
            f"children={row.get('recursive_children', 0)} "
            f"depth={row.get('max_trace_depth', 0)} "
            f"duration={row['duration_seconds']:.1f}s",
            flush=True,
        )
        if args.trace_jsonl:
            _append_jsonl(Path(args.trace_jsonl), row)

    summary = _build_summary(rows, requested=args.count)
    if args.summary_json:
        _write_json(Path(args.summary_json), summary)
    if args.trace_json:
        if len(rows) != 1:
            raise ValueError("--trace-json is only valid when --count=1")
        _write_json(Path(args.trace_json), rows[0].get("trace", rows[0]))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def _run_one(
    args: argparse.Namespace,
    instance_id: int,
    prompt: str | None,
    started: float,
) -> dict[str, Any]:
    environment_kwargs: dict[str, Any] = {
        "split": args.split,
        "instance_id": instance_id,
        "data_path": args.data_path,
        "generated_count": args.generated_count,
        "generated_difficulty": args.difficulty,
        "generated_seed": args.seed,
    }
    if args.textcraft_root:
        environment_kwargs["textcraft_root"] = args.textcraft_root
    if prompt is not None:
        environment_kwargs["agent_prompt"] = prompt

    try:
        model_overrides = {
            "timeout": args.request_timeout,
            "max_retries": args.max_retries,
            "sampling_args": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        }
        model_name = getattr(args, "model_name", None)
        if model_name:
            model_overrides["model_name"] = model_name
        run = run_registered_environment(
            "textcraft_synth",
            model_config=args.config,
            environment_kwargs=environment_kwargs,
            agent_kwargs={
                "max_steps": args.agent_max_steps,
                "max_depth": args.max_depth,
                "max_concurrent_subagents": args.max_concurrent_subagents,
                "max_subagents_per_agent": args.max_subagents_per_agent,
                "max_run_seconds": args.max_run_seconds,
                "max_observation_chars": args.max_observation_chars,
            },
            model_overrides=model_overrides,
        )
        report = run.environment_report
        trace = run.to_trace_dict()
        metrics = _trace_metrics(trace.get("agent_result", {}).get("trace"))
        usage = run.agent_result.usage.to_dict()
        return {
            "instance_id": instance_id,
            "ok": True,
            "duration_seconds": time.monotonic() - started,
            "id": report.get("id"),
            "difficulty": report.get("difficulty"),
            "crafting_depth": report.get("crafting_depth"),
            "success": bool(report.get("success", False)),
            "score": float(report.get("score", 0.0) or 0.0),
            "finished": bool(report.get("finished", False)),
            "craft_calls": int(report.get("craft_calls", 0) or 0),
            "missing": report.get("missing", {}),
            "agent_status": run.agent_result.status,
            "steps": run.agent_result.steps,
            "recursive_children": metrics["children"],
            "max_trace_depth": metrics["max_depth"],
            "models": sorted((usage.get("model_usage_summaries") or {}).keys()),
            "trace": trace,
        }
    except Exception as exc:
        trace = getattr(exc, "partial_trace", None)
        report = (trace or {}).get("environment_report") or {}
        root_trace = ((trace or {}).get("agent_result") or {}).get("trace")
        metrics = _trace_metrics(root_trace)
        usage = ((trace or {}).get("agent_result") or {}).get("usage") or {}
        row = {
            "instance_id": instance_id,
            "ok": False,
            "duration_seconds": time.monotonic() - started,
            "id": report.get("id"),
            "difficulty": report.get("difficulty"),
            "crafting_depth": report.get("crafting_depth"),
            "success": bool(report.get("success", False)),
            "score": float(report.get("score", 0.0) or 0.0),
            "finished": bool(report.get("finished", False)),
            "craft_calls": int(report.get("craft_calls", 0) or 0),
            "missing": report.get("missing", {}),
            "steps": len((root_trace or {}).get("steps") or []),
            "recursive_children": metrics["children"],
            "max_trace_depth": metrics["max_depth"],
            "models": sorted((usage.get("model_usage_summaries") or {}).keys()),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if trace is not None:
            row["trace"] = trace
        return row


def _trace_metrics(trace: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(trace, dict):
        return {"children": 0, "max_depth": 0}
    children = trace.get("children") or []
    nested = [_trace_metrics(child) for child in children]
    return {
        "children": len(children) + sum(item["children"] for item in nested),
        "max_depth": max(
            [int(trace.get("depth", 0))]
            + [item["max_depth"] for item in nested]
        ),
    }


def _build_summary(rows: list[dict[str, Any]], *, requested: int) -> dict[str, Any]:
    scores = [float(row.get("score", 0.0) or 0.0) for row in rows]
    models = sorted(
        {
            str(model)
            for row in rows
            for model in (row.get("models") or [])
            if str(model).strip()
        }
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("difficulty", "unknown"))].append(
            float(row.get("score", 0.0) or 0.0)
        )
    return {
        "environment": "textcraft_synth",
        "model": models[0] if len(models) == 1 else (models or "unknown"),
        "models": models,
        "requested_rows": requested,
        "recorded_rows": len(rows),
        "successful_runs": sum(bool(row.get("ok")) for row in rows),
        "task_successes": sum(bool(row.get("success")) for row in rows),
        "score": sum(scores) / len(scores) if scores else 0.0,
        "score_percent": 100.0 * sum(scores) / len(scores) if scores else 0.0,
        "by_difficulty": {
            name: {"n": len(values), "score": sum(values) / len(values)}
            for name, values in sorted(grouped.items())
        },
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key != "trace"
            }
            for row in rows
        ],
    }


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "count",
        "generated_count",
        "agent_max_steps",
        "max_depth",
        "max_concurrent_subagents",
        "max_subagents_per_agent",
        "max_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.start_index < 0:
        parser.error("start-index must be non-negative")
    if args.max_retries < 0:
        parser.error("max-retries must be non-negative")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
