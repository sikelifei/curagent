"""Run the 199-sample curagent Oolong-Synthetic evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

from recursive_agent.envs import run_registered_environment
from recursive_agent.envs.oolong_synth import (
    CHUNK_CHAR_LIMIT,
    OolongSynthDataset,
    evaluate_synth_response,
    parse_gold_answer,
    select_protocol_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dataset-name", default="oolongbench/oolong-synth")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="outputs/oolong_synth/deepseek_v4_flash")
    parser.add_argument("--sample-count", type=int, default=199)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--dataset-filter", default=None)
    parser.add_argument("--min-context-len", type=int, default=None)
    parser.add_argument("--max-context-len", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--episode-workers", type=int, default=4)
    parser.add_argument("--agent-max-steps", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-concurrent-subagents", type=int, default=16)
    parser.add_argument("--max-run-seconds", type=float, default=3600)
    parser.add_argument("--max-observation-chars", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write selection/prompt manifests without calling the model",
    )
    args = parser.parse_args()
    _validate_args(parser, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    prompt_path = output_dir / "prompts.json"

    dataset = OolongSynthDataset(
        split=args.split,
        dataset_name=args.dataset_name,
        data_path=args.data_path,
    )
    metadata = dataset.selection_metadata()
    selected_indices = select_protocol_indices(
        metadata,
        sample_count=args.sample_count,
        seed=args.selection_seed,
        dataset_filter=args.dataset_filter,
        min_context_len=args.min_context_len,
        max_context_len=args.max_context_len,
    )
    selected = [dataset.raw_row(index) for index in selected_indices]
    selected = selected[args.start_index :]
    if args.count is not None:
        selected = selected[: args.count]
    args.requested_run_count = len(selected)

    prompt_preview = _build_prompt_preview(args, selected[0])
    manifest = _build_manifest(args, dataset, selected_indices, selected)
    _write_json(prompt_path, prompt_preview)
    _write_json(manifest_path, manifest)
    if args.prepare_only:
        print(json.dumps({
            "prepared": True,
            "selected_rows": len(selected),
            "manifest": str(manifest_path),
            "prompts": str(prompt_path),
        }, indent=2))
        return

    existing = _load_rows(trace_path) if args.resume else []
    rows_by_position = {int(row["protocol_position"]): row for row in existing}
    pending = [
        (position, row)
        for position, row in enumerate(selected)
        if position not in rows_by_position
    ]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.episode_workers) as executor:
        futures: set[Future[dict[str, Any]]] = set()
        with trace_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
            for position, row in pending:
                futures.add(executor.submit(_run_one, args, position, row))
                if len(futures) >= args.episode_workers * 2:
                    _drain(futures, handle, rows_by_position, summary_path, args, started)
            while futures:
                _drain(futures, handle, rows_by_position, summary_path, args, started)

    summary = _build_summary(
        rows_by_position,
        requested=len(selected),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.selection_seed,
        started=started,
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_one(args: argparse.Namespace, position: int, row: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    try:
        model_overrides = {
            "timeout": args.request_timeout,
            "sampling_args": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        }
        if args.model_name:
            model_overrides["model_name"] = args.model_name
        run = run_registered_environment(
            "oolong_synth",
            model_config=args.config,
            environment_kwargs={
                "samples": [row],
                "instance_id": 0,
            },
            agent_kwargs={
                "max_steps": args.agent_max_steps,
                "max_depth": args.max_depth,
                "max_concurrent_subagents": args.max_concurrent_subagents,
                "max_run_seconds": args.max_run_seconds,
                "max_observation_chars": args.max_observation_chars,
            },
            model_overrides=model_overrides,
        )
        report = run.environment_report
        if report.get("submitted"):
            score = float(report.get("score", 0.0) or 0.0)
            attempted_parse = report.get("attempted_parse")
            submitted_answer = report.get("submitted_answer")
            submission_source = "submit_answer_tool"
        else:
            fallback = evaluate_synth_response(
                row["answer"],
                run.agent_result.answer,
                str(row["answer_type"]),
            )
            score = fallback.score
            attempted_parse = fallback.candidate
            submitted_answer = run.agent_result.answer
            submission_source = "agent_final_fallback"
        trace = _compact_trace(run.to_trace_dict())
        return {
            "protocol_position": position,
            "source_index": int(row["_source_index"]),
            "id": str(row.get("id")),
            "ok": True,
            "duration_seconds": time.time() - started,
            "dataset": str(row.get("dataset")),
            "context_len": int(row["context_len"]),
            "answer_type": str(row["answer_type"]),
            "task_group": str(row.get("task_group")),
            "task": str(row.get("task")),
            "score": score,
            "gold_answer": str(parse_gold_answer(row["answer"])),
            "attempted_parse": attempted_parse,
            "submitted_answer": submitted_answer,
            "submission_source": submission_source,
            "trace": trace,
        }
    except Exception as exc:
        return {
            "protocol_position": position,
            "source_index": int(row["_source_index"]),
            "id": str(row.get("id")),
            "ok": False,
            "duration_seconds": time.time() - started,
            "dataset": str(row.get("dataset")),
            "context_len": int(row["context_len"]),
            "answer_type": str(row["answer_type"]),
            "task_group": str(row.get("task_group")),
            "task": str(row.get("task")),
            "score": 0.0,
            "gold_answer": str(parse_gold_answer(row["answer"])),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _drain(
    futures: set[Future[dict[str, Any]]],
    handle: Any,
    rows: dict[int, dict[str, Any]],
    summary_path: Path,
    args: argparse.Namespace,
    started: float,
) -> None:
    done, _ = wait(futures, return_when="FIRST_COMPLETED")
    for future in done:
        futures.remove(future)
        row = future.result()
        rows[int(row["protocol_position"])] = row
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        _write_json(
            summary_path,
            _build_summary(
                rows,
                requested=args.requested_run_count,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.selection_seed,
                started=started,
            ),
        )
        print(
            f"completed position={row['protocol_position']} ok={row['ok']} "
            f"score={row['score']:.4f} duration={row['duration_seconds']:.1f}s",
            flush=True,
        )


def _build_summary(
    rows_by_position: dict[int, dict[str, Any]],
    *,
    requested: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    started: float,
) -> dict[str, Any]:
    rows = [rows_by_position[index] for index in sorted(rows_by_position)]
    scores = [float(row.get("score", 0.0) or 0.0) for row in rows]
    summary = {
        "environment": "oolong_synth",
        "protocol": "curagent_harness_comparable_stratified_199",
        "requested_rows": requested,
        "recorded_rows": len(rows),
        "completed_rows": sum(bool(row.get("ok")) for row in rows),
        "failed_rows": sum(not row.get("ok") for row in rows),
        "oolong_score": sum(scores) / len(scores) if scores else 0.0,
        "score_percent": 100.0 * sum(scores) / len(scores) if scores else 0.0,
        "bootstrap_95_ci": _bootstrap_ci(scores, bootstrap_samples, bootstrap_seed),
        "by_answer_type": _group_scores(rows, "answer_type"),
        "by_context_len": _group_scores(rows, "context_len"),
        "by_dataset": _group_scores(rows, "dataset"),
        "elapsed_seconds": time.time() - started,
    }
    return summary


def _group_scores(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(float(row.get("score", 0.0) or 0.0))
    return {
        name: {"n": len(values), "score": sum(values) / len(values)}
        for name, values in sorted(grouped.items())
    }


def _bootstrap_ci(scores: list[float], samples: int, seed: int) -> list[float] | None:
    if not scores or samples <= 0:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(scores, k=len(scores))) / len(scores)
        for _ in range(samples)
    )
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return [lower, upper]


def _compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    prompts = trace.get("prompts") or {}
    initial = prompts.get("initial_context")
    if isinstance(initial, dict) and isinstance(initial.get("context_window_text"), str):
        text = initial["context_window_text"]
        initial["context_window_text"] = {
            "omitted": True,
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return trace


def _build_manifest(
    args: argparse.Namespace,
    dataset: OolongSynthDataset,
    selected_indices: list[int],
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bucket_counts = Counter(int(row["context_len"]) for row in selected_rows)
    dataset_counts = Counter(str(row["dataset"]) for row in selected_rows)
    answer_type_counts = Counter(str(row["answer_type"]) for row in selected_rows)
    task_group_counts = Counter(str(row["task_group"]) for row in selected_rows)
    return {
        "protocol": "curagent_harness_comparable_stratified_199",
        "comparable_to": "Recursive Agent Harnesses Oolong-Synthetic protocol",
        "exact_reproduction": False,
        "exact_reproduction_note": (
            "The paper does not publish its 199 source IDs; this run uses a "
            "deterministic context-length-stratified sample from the same split."
        ),
        "model": "deepseek-v4-flash",
        "config": str(Path(args.config).resolve()),
        "dataset": dataset.metadata(),
        "sample_count": args.sample_count,
        "selected_run_count": len(selected_rows),
        "selection_seed": args.selection_seed,
        "protocol_selected_source_indices": selected_indices,
        "run_selected_source_indices": [
            int(row["_source_index"]) for row in selected_rows
        ],
        "run_selected_ids": [str(row.get("id")) for row in selected_rows],
        "context_bucket_counts": dict(sorted(bucket_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "task_group_counts": dict(sorted(task_group_counts.items())),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_depth": args.max_depth,
        "max_concurrent_subagents": args.max_concurrent_subagents,
        "chunk_char_limit": CHUNK_CHAR_LIMIT,
        "bootstrap_samples": args.bootstrap_samples,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _build_prompt_preview(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    from recursive_agent.envs.oolong_synth import OolongSynthEnvironment
    from recursive_agent.prompts import FORCED_FINAL_USER, build_initial_user, build_system_prompt
    from recursive_agent.tools import format_tools_for_prompt, parse_tools

    environment = OolongSynthEnvironment(samples=[row])
    formatted_tools = format_tools_for_prompt(parse_tools(environment.tools()))
    return {
        "root_system_prompt": build_system_prompt(
            formatted_tools,
            prompt_addendum=environment.agent_prompt,
        ),
        "root_initial_user_prompt": build_initial_user(environment.task, delegated=False),
        "delegated_initial_user_wrapper_example": build_initial_user(
            "Process the assigned Oolong-Synthetic chunk and return its JSON report.",
            delegated=True,
        ),
        "child_private_context_fields": [
            "oolong_role",
            "chunk_id",
            "expected_rows",
            "context_window_text",
            "dataset_intro",
            "question",
            "dataset",
        ],
        "forced_final_user_prompt": FORCED_FINAL_USER,
    }


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "sample-count": args.sample_count,
        "episode-workers": args.episode_workers,
        "agent-max-steps": args.agent_max_steps,
        "max-concurrent-subagents": args.max_concurrent_subagents,
        "max-tokens": args.max_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.max_depth < 1:
        parser.error("max-depth must be at least 1 for root-to-subagent evaluation")
    if args.start_index < 0:
        parser.error("start-index must be non-negative")
    if args.count is not None and args.count <= 0:
        parser.error("count must be positive when supplied")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict) and "protocol_position" in row:
                rows.append(row)
            else:
                malformed += 1
    if malformed:
        print(f"warning: skipped {malformed} malformed rows while resuming {path}", flush=True)
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
