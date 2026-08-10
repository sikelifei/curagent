"""Resumable curagent inference runner over the Platoon DeepDive harness."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ...config import load_model_config
from ..runner import run_environment
from ..trace_analysis import analyze_environment_trace
from .environment import DeepDiveEnvironment
from .harness import DEFAULT_PLATOON_ROOT, PlatoonDeepDiveHarness, make_task_ids
from .scoring import judge_deepdive_answer


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    runs_dir = output_dir / "runs"
    trajectories_dir = output_dir / "trajectories"
    logs_dir = output_dir / "logs"
    for directory in (runs_dir, trajectories_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    harness = PlatoonDeepDiveHarness(args.platoon_root)
    task_ids = (
        _parse_task_ids(args.task_id)
        if args.task_id
        else make_task_ids(args.split, args.start_index, args.limit)
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(task_ids)
    samples = [harness.load_sample(task_id) for task_id in task_ids]
    pending = []
    skipped = 0
    for sample in samples:
        path = runs_dir / _run_filename(sample.task_id)
        if args.resume and _is_completed(path):
            skipped += 1
        else:
            pending.append(sample)

    started = time.time()
    if args.concurrency == 1:
        for sample in pending:
            _run_and_persist(
                args, sample, harness, runs_dir, trajectories_dir, logs_dir
            )
    else:
        with ThreadPoolExecutor(
            max_workers=args.concurrency,
            thread_name_prefix="deepdive-episode",
        ) as executor:
            futures: dict[Future[dict[str, Any]], str] = {
                executor.submit(
                    _run_and_persist,
                    args,
                    sample,
                    harness,
                    runs_dir,
                    trajectories_dir,
                    logs_dir,
                ): sample.task_id
                for sample in pending
            }
            for future in as_completed(futures):
                future.result()

    records = _load_records(runs_dir, set(task_ids))
    summary = _build_summary(
        records,
        requested=len(task_ids),
        skipped=skipped,
        elapsed_seconds=time.time() - started,
    )
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        "--config",
        dest="model_config",
        default="configs/model_api.local.yaml",
    )
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--platoon-root", default=str(DEFAULT_PLATOON_ROOT))
    parser.add_argument("--split", choices=("qa_rl", "qa_sft"), default="qa_rl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/deepdive_curagent")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--agent-max-steps", type=int, default=25)
    parser.add_argument("--max-recursion-depth", type=int, default=4)
    parser.add_argument("--max-concurrent-subagents", type=int, default=4)
    parser.add_argument("--max-subagents-per-agent", type=int, default=None)
    parser.add_argument("--max-run-seconds", type=float, default=7200.0)
    parser.add_argument("--max-observation-chars", type=int, default=32000)
    parser.add_argument("--max-search-calls", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--skip-evaluator", action="store_true")
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-timeout", type=float, default=None)
    parser.add_argument("--judge-attempts", type=int, default=3)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "limit": args.limit,
        "concurrency": args.concurrency,
        "agent-max-steps": args.agent_max_steps,
        "max-concurrent-subagents": args.max_concurrent_subagents,
        "max-run-seconds": args.max_run_seconds,
        "max-observation-chars": args.max_observation_chars,
        "judge-max-tokens": args.judge_max_tokens,
        "judge-attempts": args.judge_attempts,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    optional_positive = {
        "max-subagents-per-agent": args.max_subagents_per_agent,
        "max-search-calls": args.max_search_calls,
        "request-timeout": args.request_timeout,
        "judge-timeout": args.judge_timeout,
    }
    for name, value in optional_positive.items():
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive when supplied")
    if args.start_index < 0:
        parser.error("start-index must be non-negative")
    if args.max_recursion_depth < 0:
        parser.error("max-recursion-depth must be non-negative")


def _run_and_persist(
    args: argparse.Namespace,
    sample: Any,
    harness: PlatoonDeepDiveHarness,
    runs_dir: Path,
    trajectories_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    stem = Path(_run_filename(sample.task_id)).stem
    run_path = runs_dir / f"{stem}.json"
    trajectory_path = trajectories_dir / f"{stem}_trajectory.json"
    steps_path = logs_dir / f"{stem}_steps.jsonl"
    steps_path.write_text("", encoding="utf-8")
    step_lock = threading.Lock()

    def record_step(trace: Any, step: Any) -> None:
        event = {
            "timestamp": time.time(),
            "task_id": sample.task_id,
            "agent_id": trace.agent_id,
            "parent_id": trace.parent_id,
            "depth": trace.depth,
            "task": trace.task,
            "step": step.number,
            "response": step.response,
            "model_observation": step.model_observation,
            "observation_truncated": step.observation_truncated,
            "code_executions": [
                {
                    "code": execution.code,
                    "stdout": execution.output,
                    "error": execution.error,
                    "variables": list(execution.variables),
                    "duration_seconds": execution.duration_seconds,
                }
                for execution in step.code_executions
            ],
            "duration_seconds": step.duration_seconds,
        }
        with step_lock:
            with steps_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        calls = _call_names(event["code_executions"])
        role = "root" if trace.parent_id is None else f"depth={trace.depth}"
        print(
            f"task_id={sample.task_id} {role} step={step.number} "
            f"calls={','.join(calls) or 'none'}",
            flush=True,
        )

    environment = DeepDiveEnvironment(
        sample=sample,
        harness=harness,
        max_search_calls=args.max_search_calls,
    )
    trace_payload: dict[str, Any] | None = None
    try:
        run = run_environment(
            environment,
            model_config=args.model_config,
            agent_kwargs={
                "max_steps": args.agent_max_steps,
                "max_depth": args.max_recursion_depth,
                "max_concurrent_subagents": args.max_concurrent_subagents,
                "max_subagents_per_agent": args.max_subagents_per_agent,
                "max_run_seconds": args.max_run_seconds,
                "max_observation_chars": args.max_observation_chars,
                "step_callback": record_step,
            },
            model_overrides=_model_overrides(args),
        )
        answer = str(run.agent_result.answer or "").strip()
        if not answer:
            raise ValueError("Agent completed without an answer")
        trace_payload = run.to_trace_dict()
        judgment = None
        if not args.skip_evaluator:
            judgment = judge_deepdive_answer(
                model_config=args.judge_config or args.model_config,
                ground_truth=sample.answer,
                agent_answer=answer,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
                timeout=args.judge_timeout,
                max_attempts=args.judge_attempts,
            ).to_dict()
        trace_payload["evaluation"] = judgment
        _write_json(trajectory_path, trace_payload)
        metrics = analyze_environment_trace(trace_payload)
        record = {
            "task_id": sample.task_id,
            "split": sample.split,
            "index": sample.index,
            "question": sample.question,
            "status": "completed",
            "agent_status": run.agent_result.status,
            "final_answer": answer,
            "success": judgment.get("success") if judgment else None,
            "reward": float(bool(judgment and judgment.get("success"))),
            "evaluation": judgment,
            "metrics": metrics,
            "environment_report": run.environment_report,
            "usage": run.agent_result.usage.to_dict(),
            "trajectory_path": str(trajectory_path),
            "steps_path": str(steps_path),
            "duration_seconds": time.time() - started,
            "model": _model_metadata(args),
            "error": None,
        }
    except Exception as exc:
        partial = getattr(exc, "partial_trace", None)
        if isinstance(partial, dict):
            trace_payload = partial
            _write_json(trajectory_path, partial)
        record = {
            "task_id": sample.task_id,
            "split": sample.split,
            "index": sample.index,
            "question": sample.question,
            "status": "error",
            "final_answer": "",
            "success": False,
            "reward": 0.0,
            "evaluation": None,
            "metrics": (
                analyze_environment_trace(trace_payload)
                if isinstance(trace_payload, dict)
                else {}
            ),
            "environment_report": environment.report(),
            "usage": {},
            "trajectory_path": str(trajectory_path) if trace_payload else None,
            "steps_path": str(steps_path),
            "duration_seconds": time.time() - started,
            "model": _model_metadata(args),
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json(run_path, record)
    print(
        f"task_id={sample.task_id} status={record['status']} "
        f"success={record['success']} subagents="
        f"{(record.get('metrics') or {}).get('child_agent_count', 0)}",
        flush=True,
    )
    return record


def _build_summary(
    records: list[dict[str, Any]],
    *,
    requested: int,
    skipped: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    judged = [
        record
        for record in completed
        if isinstance(record.get("evaluation"), dict)
        and not record["evaluation"].get("error")
    ]
    successful = sum(bool(record.get("success")) for record in judged)
    return {
        "requested": requested,
        "recorded": len(records),
        "completed": len(completed),
        "failed": requested - len(completed),
        "resume_skipped": skipped,
        "judged": len(judged),
        "successful": successful,
        "success_rate": successful / len(judged) if judged else None,
        "tasks_triggering_subagents": sum(
            int((record.get("metrics") or {}).get("child_agent_count", 0)) > 0
            for record in completed
        ),
        "total_subagents": sum(
            int((record.get("metrics") or {}).get("child_agent_count", 0))
            for record in completed
        ),
        "max_depth": max(
            (
                int((record.get("metrics") or {}).get("max_depth", 0))
                for record in completed
            ),
            default=0,
        ),
        "total_agent_steps": sum(
            int((record.get("metrics") or {}).get("agent_steps", 0))
            for record in completed
        ),
        "total_search_calls": sum(
            int(
                (record.get("environment_report") or {})
                .get("tool_call_counts", {})
                .get("search_web", 0)
            )
            for record in completed
        ),
        "elapsed_seconds": elapsed_seconds,
        "per_task": [
            {
                "task_id": record.get("task_id"),
                "status": record.get("status"),
                "success": record.get("success"),
                "subagents": (record.get("metrics") or {}).get("child_agent_count", 0),
                "max_depth": (record.get("metrics") or {}).get("max_depth", 0),
                "agent_steps": (record.get("metrics") or {}).get("agent_steps", 0),
                "search_calls": (
                    (record.get("environment_report") or {})
                    .get("tool_call_counts", {})
                    .get("search_web", 0)
                ),
                "trajectory_path": record.get("trajectory_path"),
                "error": record.get("error"),
            }
            for record in sorted(records, key=lambda item: int(item.get("index", 0)))
        ],
    }


def _model_overrides(args: argparse.Namespace) -> dict[str, Any]:
    sampling = {
        key: value
        for key, value in {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }.items()
        if value is not None
    }
    overrides: dict[str, Any] = {}
    if args.model_name is not None:
        overrides["model_name"] = args.model_name
    if args.request_timeout is not None:
        overrides["timeout"] = args.request_timeout
    if sampling:
        overrides["sampling_args"] = sampling
    return overrides


def _model_metadata(args: argparse.Namespace) -> dict[str, Any]:
    backend, kwargs = load_model_config(args.model_config)
    sampling = dict(kwargs.get("sampling_args") or {})
    for key, value in {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }.items():
        if value is not None:
            sampling[key] = value
    return {
        "backend": backend,
        "model": args.model_name or kwargs.get("model_name"),
        "base_url": kwargs.get("base_url"),
        "sampling": sampling,
    }


def _call_names(executions: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for execution in executions:
        try:
            tree = ast.parse(str(execution.get("code", "")))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in names:
                    names.append(node.func.id)
    return names


def _parse_task_ids(values: list[str]) -> list[str]:
    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]


def _run_filename(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id)).strip("._")
    return f"run_{safe or 'deepdive'}.json"


def _is_completed(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream).get("status") == "completed"
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _load_records(runs_dir: Path, task_ids: set[str]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(runs_dir.glob("run_*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
        except (json.JSONDecodeError, OSError):
            continue
        if str(value.get("task_id")) in task_ids:
            records.append(value)
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()


__all__ = ["main"]
