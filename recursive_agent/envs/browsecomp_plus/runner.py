"""Resumable official-BM25 smoke runner for recursive BrowseComp-Plus."""

from __future__ import annotations

import argparse
import ast
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from ...config import load_model_config
from ...exceptions import TimeoutExceededError
from ..runner import run_environment
from .dataset import (
    DEFAULT_DATA_PATH,
    DEFAULT_QUERIES_PATH,
    BrowseCompQuery,
    load_gold_answers,
    load_queries,
)
from .environment import BrowseCompPlusEnvironment
from .scoring import extract_final_answer, judge_answer, parse_final_output


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    runs_dir = output_dir / "runs"
    trajectories_dir = output_dir / "trajectories"
    logs_dir = output_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = _parse_query_ids(args.query_ids)
    queries = load_queries(
        args.queries,
        query_ids=selected_ids,
        start_index=args.start_index,
        limit=args.limit,
    )
    if not queries:
        parser.error("selected query slice is empty")

    answers = (
        load_gold_answers(
            args.gold_data,
            query_ids=(item.query_id for item in queries),
        )
        if not args.skip_local_evaluator
        else {}
    )
    model_metadata = _model_metadata(args.model_config)
    started = time.time()
    pending: list[BrowseCompQuery] = []
    skipped = 0
    for sample in queries:
        path = runs_dir / _run_filename(sample.query_id)
        if args.resume and _is_completed(path):
            skipped += 1
        else:
            pending.append(sample)

    if args.concurrency == 1:
        for sample in pending:
            _run_and_persist(
                args,
                sample,
                answers.get(sample.query_id),
                model_metadata,
                runs_dir,
                trajectories_dir,
                logs_dir,
            )
    else:
        with ThreadPoolExecutor(
            max_workers=args.concurrency,
            thread_name_prefix="browsecomp-episode",
        ) as executor:
            futures: dict[Future[dict[str, Any]], BrowseCompQuery] = {
                executor.submit(
                    _run_and_persist,
                    args,
                    sample,
                    answers.get(sample.query_id),
                    model_metadata,
                    runs_dir,
                    trajectories_dir,
                    logs_dir,
                ): sample
                for sample in pending
            }
            for future in as_completed(futures):
                future.result()

    records = _load_run_records(runs_dir, {item.query_id for item in queries})
    summary = build_summary(
        records,
        requested=len(queries),
        skipped=skipped,
        elapsed_seconds=time.time() - started,
    )
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES_PATH))
    parser.add_argument(
        "--model-config",
        "--config",
        dest="model_config",
        default="configs/model_api.local.yaml",
    )
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--gold-data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--bm25-url", default="http://127.0.0.1:8080/mcp")
    parser.add_argument(
        "--output-dir",
        default="outputs/browsecomp_plus_smoke",
    )
    parser.add_argument("--query-ids", nargs="*", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-search-calls", type=int, default=20)
    parser.add_argument("--max-recursion-depth", type=int, default=2)
    parser.add_argument("--max-concurrent-subagents", type=int, default=4)
    parser.add_argument("--max-subagents-per-agent", type=int, default=4)
    parser.add_argument("--agent-max-steps", type=int, default=20)
    parser.add_argument("--max-run-seconds", type=float, default=1800.0)
    parser.add_argument("--max-observation-chars", type=int, default=16000)
    parser.add_argument("--bm25-timeout", type=float, default=60.0)
    parser.add_argument("--snippet-max-chars", type=int, default=1000)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-timeout", type=float, default=None)
    parser.add_argument("--judge-attempts", type=int, default=3)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the API model name from the selected config.",
    )
    parser.add_argument("--skip-local-evaluator", action="store_true")
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    positive = {
        "limit": args.limit,
        "concurrency": args.concurrency,
        "max-search-calls": args.max_search_calls,
        "max-recursion-depth": args.max_recursion_depth,
        "max-concurrent-subagents": args.max_concurrent_subagents,
        "max-subagents-per-agent": args.max_subagents_per_agent,
        "agent-max-steps": args.agent_max_steps,
        "max-run-seconds": args.max_run_seconds,
        "max-observation-chars": args.max_observation_chars,
        "bm25-timeout": args.bm25_timeout,
        "snippet-max-chars": args.snippet_max_chars,
        "judge-attempts": args.judge_attempts,
    }
    for name, value in positive.items():
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if args.start_index < 0:
        parser.error("start-index must be non-negative")


def _run_and_persist(
    args: argparse.Namespace,
    sample: BrowseCompQuery,
    gold_answer: str | None,
    model_metadata: dict[str, Any],
    runs_dir: Path,
    trajectories_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    run_path = runs_dir / _run_filename(sample.query_id)
    trajectory_path = trajectories_dir / (
        Path(_run_filename(sample.query_id)).stem + "_trajectory.json"
    )
    steps_path = logs_dir / (
        Path(_run_filename(sample.query_id)).stem + "_steps.jsonl"
    )
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.write_text("", encoding="utf-8")
    step_log_lock = threading.Lock()

    def record_step(trace: Any, step: Any) -> None:
        _append_live_step(steps_path, step_log_lock, sample.query_id, trace, step)

    trace_payload: dict[str, Any] | None = None
    environment = BrowseCompPlusEnvironment(
        sample=sample,
        bm25_url=args.bm25_url,
        max_search_calls=args.max_search_calls,
        bm25_timeout=args.bm25_timeout,
        snippet_max_chars=args.snippet_max_chars,
    )
    stats = {
        "subagent_count": 0,
        "max_depth_reached": 0,
        "root_used_search": False,
        "subagent_used_search": False,
        "recursion_chain": [],
    }
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
        output = str(run.agent_result.answer or "").strip()
        full_trace = run.to_trace_dict()
        trace_payload = full_trace
        stats = analyze_recursive_trace(
            (full_trace.get("agent_result") or {}).get("trace")
        )
        search_report = run.environment_report
        _write_json(trajectory_path, full_trace)
        if not output:
            raise ValueError("Agent completed without a final output")
        if parse_final_output(output) is None:
            raise ValueError("Agent final output does not match the required format")
        local_judge = None
        if not args.skip_local_evaluator:
            if gold_answer is None:
                raise ValueError(
                    f"Gold answer unavailable to evaluator for {sample.query_id}"
                )
            try:
                local_judge = judge_answer(
                    model_config=args.judge_config or args.model_config,
                    question=sample.query,
                    correct_answer=gold_answer,
                    response=output,
                    temperature=args.judge_temperature,
                    top_p=args.judge_top_p,
                    max_tokens=args.judge_max_tokens,
                    timeout=args.judge_timeout,
                    max_attempts=args.judge_attempts,
                ).to_dict()
            except Exception as exc:
                local_judge = {
                    "correct": False,
                    "score": 0,
                    "reason": "Local evaluator call failed.",
                    "response": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": "unknown",
                    "attempts": 0,
                }
        record = _completed_record(
            args=args,
            sample=sample,
            output=output,
            model_metadata=model_metadata,
            search_report=search_report,
            stats=stats,
            trajectory_path=trajectory_path,
            duration_seconds=time.time() - started,
            agent_status=run.agent_result.status,
            usage=run.agent_result.usage.to_dict(),
            local_judge=local_judge,
        )
    except Exception as exc:
        partial_trace = getattr(exc, "partial_trace", None)
        if isinstance(partial_trace, dict):
            trace_payload = partial_trace
            partial_agent_trace = (
                partial_trace.get("agent_result", {}).get("trace")
            )
            stats = analyze_recursive_trace(partial_agent_trace)
            _write_json(trajectory_path, partial_trace)
        snapshot = environment.trace.snapshot()
        record = _error_record(
            args=args,
            sample=sample,
            model_metadata=model_metadata,
            search_report=snapshot,
            duration_seconds=time.time() - started,
            error=exc,
            trajectory_path=trajectory_path,
            stats=stats,
        )
    _write_json(run_path, record)
    _write_simple_run_log(
        logs_dir / (Path(_run_filename(sample.query_id)).stem + ".log"),
        record,
        trace_payload,
    )
    parsed = parse_final_output(record.get("final_answer", ""))
    print(
        f"query_id={sample.query_id} status={record['status']} "
        f"exact_answer={parsed['exact_answer'] if parsed else ''!r} "
        f"search_calls={record['tool_call_counts']['search']} "
        f"subagents={record['debug']['subagent_count']} "
        f"max_depth={record['debug']['max_depth_reached']}",
        flush=True,
    )
    return record


def _append_live_step(
    path: Path,
    lock: threading.Lock,
    query_id: str,
    trace: Any,
    step: Any,
) -> None:
    """Append one completed root or delegated step while an episode is running."""
    executions = []
    call_names: list[str] = []
    for execution in step.code_executions:
        code = str(execution.code)
        executions.append(
            {
                "code": code,
                "stdout": execution.output,
                "error": execution.error,
                "variables": list(execution.variables),
                "duration_seconds": execution.duration_seconds,
            }
        )
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in call_names:
                    call_names.append(node.func.id)
    event = {
        "timestamp": time.time(),
        "query_id": query_id,
        "agent_id": trace.agent_id,
        "parent_id": trace.parent_id,
        "depth": trace.depth,
        "task": trace.task,
        "step": step.number,
        "response": step.response,
        "model_observation": step.model_observation,
        "observation_truncated": step.observation_truncated,
        "code_executions": executions,
        "duration_seconds": step.duration_seconds,
    }
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    role = "root" if trace.parent_id is None else f"depth={trace.depth}"
    calls = ",".join(call_names) if call_names else "no_repl_call"
    errors = sum(bool(item["error"]) for item in executions)
    print(
        f"query_id={query_id} {role} step={step.number} "
        f"calls={calls} errors={errors} duration={step.duration_seconds:.2f}s",
        flush=True,
    )


def _write_simple_run_log(
    path: Path,
    record: dict[str, Any],
    trace_payload: dict[str, Any] | None,
) -> None:
    """Write a compact human-readable summary for one completed episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = trace_payload.get("environment_report", {}) if trace_payload else {}
    events = report.get("events") or report.get("search_events") or []
    trace = (trace_payload or {}).get("agent_result", {}).get("trace")
    agents = list(_walk_trace_agents(trace))
    searches = [str(item.get("query", "")).strip() for item in events]
    searches = [item for item in searches if item]
    lines = [
        f"Query: {record.get('query_id', '')}",
        f"Status: {record.get('status', '')}",
        f"Question: {_one_line(record.get('query', ''), 500)}",
        f"Answer: {_one_line(record.get('final_answer', ''), 300)}",
        (
            "Stats: "
            f"searches={record.get('search_calls', 0)} "
            f"subagents={record.get('subagent_count', 0)} "
            f"max_depth={record.get('max_depth_reached', 0)}"
        ),
        "Search queries:",
    ]
    if searches:
        lines.extend(f"- {index}. {_one_line(query, 260)}" for index, query in enumerate(searches, 1))
    else:
        lines.append("- none")
    lines.append("Operations:")
    if agents:
        for agent in agents:
            role = "root" if agent.get("parent_id") is None else f"worker depth={agent.get('depth', 0)}"
            calls = _trace_call_names(agent)
            first = _first_model_action(agent)
            action = ", ".join(calls) if calls else "no REPL calls"
            if first:
                action += f"; first response: {_one_line(first, 220)}"
            lines.append(f"- {role}: {action}")
    else:
        lines.append("- trace unavailable")
    lines.append(f"Summary: {_run_summary(record, agents)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _walk_trace_agents(trace: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(trace, dict):
        return
    yield trace
    for child in trace.get("children") or []:
        yield from _walk_trace_agents(child)


def _trace_call_names(agent: dict[str, Any]) -> list[str]:
    calls: list[str] = []
    for step in agent.get("steps") or []:
        for execution in step.get("code_executions") or []:
            try:
                tree = ast.parse(str(execution.get("code", "")))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id not in calls:
                        calls.append(node.func.id)
    return calls


def _first_model_action(agent: dict[str, Any]) -> str:
    for step in agent.get("steps") or []:
        response = str(step.get("response", "")).strip()
        if response:
            return response.split("```", 1)[0].strip()
    return ""


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _run_summary(record: dict[str, Any], agents: list[dict[str, Any]]) -> str:
    status = str(record.get("status", "unknown"))
    searches = int(record.get("search_calls", 0) or 0)
    children = int(record.get("subagent_count", 0) or 0)
    if status != "completed":
        return f"episode {status}; searches={searches}, subagents={children}."
    if children:
        return (
            f"root used {searches} corpus searches and delegated to {children} "
            f"agent(s); inspect worker reports for evidence coverage."
        )
    return f"root solved directly with {searches} corpus searches and no delegation."


def _completed_record(
    *,
    args: argparse.Namespace,
    sample: BrowseCompQuery,
    output: str,
    model_metadata: dict[str, Any],
    search_report: dict[str, Any],
    stats: dict[str, Any],
    trajectory_path: Path,
    duration_seconds: float,
    agent_status: str,
    usage: dict[str, Any],
    local_judge: dict[str, Any] | None,
) -> dict[str, Any]:
    search_events = search_report.get("events") or []
    result = [
        {
            "type": "tool_call",
            "tool_name": "search",
            "arguments": {"query": event.get("query")},
            "output": event.get("results", []),
        }
        for event in search_events
    ]
    result.append({"type": "output_text", "output": output})
    return {
        "metadata": _run_metadata(args, model_metadata),
        "query_id": sample.query_id,
        "query": sample.query,
        "status": "completed",
        "final_answer": output,
        "search_calls": int(search_report.get("search_calls", 0)),
        "retrieved_docids": [
            str(value) for value in search_report.get("retrieved_docids", [])
        ],
        "tool_call_counts": {
            "search": int(search_report.get("search_calls", 0)),
            "recursive": int(stats["subagent_count"]),
        },
        "recursive_calls": int(stats["subagent_count"]),
        "max_depth_reached": int(stats["max_depth_reached"]),
        "subagent_count": int(stats["subagent_count"]),
        "trajectory": {
            "path": str(trajectory_path),
            "recursion_chain": stats["recursion_chain"],
            "search_events": search_events,
        },
        "result": result,
        "debug": {
            "agent_status": agent_status,
            "subagent_count": int(stats["subagent_count"]),
            "max_depth_reached": int(stats["max_depth_reached"]),
            "root_used_search": bool(stats["root_used_search"]),
            "subagent_used_search": bool(stats["subagent_used_search"]),
            "trajectory_path": str(trajectory_path),
            "duration_seconds": duration_seconds,
            "output_format_valid": parse_final_output(output) is not None,
        },
        "usage": usage,
        "local_evaluator": local_judge,
        "error": None,
    }


def _error_record(
    *,
    args: argparse.Namespace,
    sample: BrowseCompQuery,
    model_metadata: dict[str, Any],
    search_report: dict[str, Any],
    duration_seconds: float,
    error: Exception,
    trajectory_path: Path,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = "timeout" if isinstance(error, TimeoutExceededError) else "error"
    search_calls = int(search_report.get("search_calls", 0))
    recursion = stats or {
        "subagent_count": 0,
        "max_depth_reached": 0,
        "root_used_search": search_calls > 0,
        "subagent_used_search": False,
        "recursion_chain": [],
    }
    return {
        "metadata": _run_metadata(args, model_metadata),
        "query_id": sample.query_id,
        "query": sample.query,
        "status": status,
        "final_answer": "",
        "search_calls": search_calls,
        "retrieved_docids": [
            str(value) for value in search_report.get("retrieved_docids", [])
        ],
        "tool_call_counts": {
            "search": search_calls,
            "recursive": int(recursion["subagent_count"]),
        },
        "recursive_calls": int(recursion["subagent_count"]),
        "max_depth_reached": int(recursion["max_depth_reached"]),
        "subagent_count": int(recursion["subagent_count"]),
        "trajectory": {
            "path": str(trajectory_path),
            "recursion_chain": recursion["recursion_chain"],
            "search_events": search_report.get("events", []),
        },
        "result": [{"type": "output_text", "output": ""}],
        "debug": {
            "subagent_count": int(recursion["subagent_count"]),
            "max_depth_reached": int(recursion["max_depth_reached"]),
            "root_used_search": bool(recursion["root_used_search"]),
            "subagent_used_search": bool(recursion["subagent_used_search"]),
            "trajectory_path": str(trajectory_path),
            "duration_seconds": duration_seconds,
            "output_format_valid": False,
        },
        "local_evaluator": None,
        "error": f"{type(error).__name__}: {error}",
    }


def analyze_recursive_trace(root: Any) -> dict[str, Any]:
    if not isinstance(root, dict):
        return {
            "subagent_count": 0,
            "max_depth_reached": 0,
            "root_used_search": False,
            "subagent_used_search": False,
            "recursion_chain": [],
        }
    agents = list(_walk_agents(root))
    chain = []
    for agent in agents:
        search_queries = _search_queries(agent)
        usage = agent.get("usage") or {}
        chain.append(
            {
                "agent_id": agent.get("agent_id"),
                "parent_id": agent.get("parent_id"),
                "depth": int(agent.get("depth", 0)),
                "task": str(agent.get("task", "")),
                "model_calls": int(usage.get("total_calls", 0)),
                "search_queries": search_queries,
                "returned": str(agent.get("answer") or "")[:500],
                "status": agent.get("status"),
            }
        )
    return {
        "subagent_count": max(0, len(agents) - 1),
        "max_depth_reached": max(
            (int(agent.get("depth", 0)) for agent in agents),
            default=0,
        ),
        "root_used_search": bool(chain and chain[0]["search_queries"]),
        "subagent_used_search": any(
            item["depth"] > 0 and item["search_queries"] for item in chain
        ),
        "recursion_chain": chain,
    }


def _walk_agents(root: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield root
    for child in root.get("children") or []:
        if isinstance(child, dict):
            yield from _walk_agents(child)


def _search_queries(agent: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for step in agent.get("steps") or []:
        for execution in step.get("code_executions") or []:
            try:
                tree = ast.parse(str(execution.get("code", "")))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "search"
                ):
                    continue
                if node.args and isinstance(node.args[0], ast.Constant):
                    queries.append(str(node.args[0].value))
                else:
                    queries.append("<dynamic query>")
    return queries


def build_summary(
    records: list[dict[str, Any]],
    *,
    requested: int,
    skipped: int = 0,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    judged = [
        row
        for row in completed
        if isinstance(row.get("local_evaluator"), dict)
        and not row["local_evaluator"].get("error")
    ]
    correct = sum(bool(row["local_evaluator"].get("correct")) for row in judged)
    def average(key: str) -> float:
        return (
            sum(float(row.get(key, 0)) for row in completed) / len(completed)
            if completed
            else 0.0
        )
    per_query = []
    for row in records:
        parsed = parse_final_output(str(row.get("final_answer", "")))
        judge = row.get("local_evaluator") or {}
        per_query.append(
            {
                "query_id": row.get("query_id"),
                "status": row.get("status"),
                "exact_answer": (
                    parsed["exact_answer"] if parsed else extract_final_answer(
                        str(row.get("final_answer", ""))
                    )
                ),
                "local_correctness": judge.get("correct"),
                "search_calls": int(row.get("search_calls", 0)),
                "subagent_count": int(row.get("subagent_count", 0)),
                "max_depth": int(row.get("max_depth_reached", 0)),
                "recursion_occurred": int(row.get("subagent_count", 0)) > 0,
            }
        )
    return {
        "total_questions": requested,
        "recorded": len(records),
        "completed": len(completed),
        "failed": requested - len(completed),
        "resume_skipped": skipped,
        "locally_judged": len(judged),
        "local_evaluator_correct": correct,
        "accuracy_by_local_evaluator": correct / len(judged) if judged else None,
        "average_search_calls": average("search_calls"),
        "average_subagent_count": average("subagent_count"),
        "questions_that_triggered_recursion": sum(
            int(row.get("subagent_count", 0)) > 0 for row in completed
        ),
        "maximum_recursion_depth": max(
            (int(row.get("max_depth_reached", 0)) for row in completed),
            default=0,
        ),
        "average_unique_retrieved_documents": (
            sum(len(row.get("retrieved_docids", [])) for row in completed)
            / len(completed)
            if completed
            else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
        "per_query": per_query,
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


def _model_metadata(config_path: str | Path) -> dict[str, Any]:
    backend, kwargs = load_model_config(config_path)
    sampling = kwargs.get("sampling_args") or {}
    return {
        "backend": backend,
        "model": kwargs.get("model_name"),
        "api_base": kwargs.get("base_url"),
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "max_tokens": sampling.get("max_tokens"),
        "timeout": kwargs.get("timeout"),
        "retry_count": kwargs.get("max_retries", "provider_default"),
    }


def _run_metadata(
    args: argparse.Namespace,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    generation = dict(model_metadata)
    for key, value in {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }.items():
        if value is not None:
            generation[key] = value
    if args.seed is not None:
        generation["seed"] = args.seed
    return {
        "model_config": str(Path(args.model_config).expanduser().resolve()),
        "scaffold": "curagent_recursive",
        "retriever": "BM25",
        "bm25_url": args.bm25_url,
        "bm25_top_k": 5,
        "snippet_max_tokens": 512,
        "max_search_calls": args.max_search_calls,
        "max_recursion_depth": args.max_recursion_depth,
        "max_subagents_per_agent": args.max_subagents_per_agent,
        "generation": generation,
    }


def _parse_query_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]


def _run_filename(query_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(query_id)).strip("._")
    if not safe:
        safe = "query"
    digest = __import__("hashlib").sha256(str(query_id).encode()).hexdigest()[:10]
    return f"run_{safe}_{digest}.json"


def _is_completed(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle).get("status") == "completed"
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _load_run_records(
    runs_dir: Path,
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(runs_dir.glob("run_*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if str(value.get("query_id")) in selected_ids:
            records.append(value)
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()


__all__ = [
    "analyze_recursive_trace",
    "build_summary",
    "main",
]
