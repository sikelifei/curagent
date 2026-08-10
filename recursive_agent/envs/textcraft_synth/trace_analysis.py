"""TextCraft-specific recursion quality metrics for saved run traces."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from typing import Any, Iterable


def analyze_textcraft_result(row: dict[str, Any]) -> dict[str, Any]:
    """Extract task outcome and recursion-quality metrics from one JSONL row."""
    trace = row.get("trace") or {}
    root = ((trace.get("agent_result") or {}).get("trace"))
    report = trace.get("environment_report") or {}
    system_prompt = str((trace.get("prompts") or {}).get("system", ""))
    agents = list(_walk_agents(root)) if isinstance(root, dict) else []

    total_calls: Counter[str] = Counter()
    duplicate_get_info_queries = 0
    get_info_no_arg_calls = 0
    observation_truncations = 0
    execution_errors = 0
    multi_craft_executions = 0
    child_finish_calls = 0
    unchanged_child_tasks = 0
    contracted_child_tasks = 0

    for index, agent in enumerate(agents):
        per_agent_queries: Counter[str] = Counter()
        for step in agent.get("steps") or []:
            if step.get("observation_truncated"):
                observation_truncations += 1
            for execution in step.get("code_executions") or []:
                if execution.get("error"):
                    execution_errors += 1
                try:
                    tree = ast.parse(str(execution.get("code", "")))
                except SyntaxError:
                    continue
                execution_craft_calls = 0
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    name = node.func.id
                    total_calls[name] += 1
                    if name == "craft":
                        execution_craft_calls += 1
                    if name == "finish" and index > 0:
                        child_finish_calls += 1
                    if name == "get_info":
                        signature = _get_info_signature(node)
                        per_agent_queries[signature] += 1
                        if signature == "<no-args>":
                            get_info_no_arg_calls += 1
                if execution_craft_calls > 1:
                    multi_craft_executions += 1
        duplicate_get_info_queries += sum(
            max(0, count - 1) for count in per_agent_queries.values()
        )

    for child in agents[1:]:
        task = _normalize_task(child.get("task"))
        parent = _find_parent(root, child.get("parent_id"))
        if parent is not None and task == _normalize_task(parent.get("task")):
            unchanged_child_tasks += 1
        if _has_absolute_inventory_contract(task):
            contracted_child_tasks += 1

    difficulty = str(row.get("difficulty") or report.get("difficulty") or "unknown")
    crafting_depth = int(row.get("crafting_depth") or report.get("crafting_depth") or 0)
    child_count = max(0, len(agents) - 1)
    max_depth = max((int(agent.get("depth", 0)) for agent in agents), default=0)
    recursion_issues = _recursion_issues(
        crafting_depth=crafting_depth,
        child_count=child_count,
        max_depth=max_depth,
        parallel_spawn_calls=total_calls["spawn_subagents"],
        child_finish_calls=child_finish_calls,
        unchanged_child_tasks=unchanged_child_tasks,
        contracted_child_tasks=contracted_child_tasks,
    )
    trajectory_issues = list(recursion_issues)
    if duplicate_get_info_queries:
        trajectory_issues.append("duplicate_get_info_queries")
    if get_info_no_arg_calls:
        trajectory_issues.append("get_info_without_items")
    if multi_craft_executions:
        trajectory_issues.append("multiple_crafts_in_one_execution")
    if (
        bool(row.get("success", report.get("success", False)))
        and total_calls["finish"] - child_finish_calls == 0
    ):
        trajectory_issues.append("successful_inventory_without_root_finish")

    return {
        "instance_id": row.get("instance_id"),
        "id": row.get("id") or report.get("id"),
        "difficulty": difficulty,
        "crafting_depth": crafting_depth,
        "ok": bool(row.get("ok", False)),
        "success": bool(row.get("success", report.get("success", False))),
        "score": float(row.get("score", report.get("score", 0.0)) or 0.0),
        "agent_count": len(agents),
        "child_agent_count": child_count,
        "max_trace_depth": max_depth,
        "root_steps": len((root or {}).get("steps") or []),
        "total_agent_steps": sum(len(agent.get("steps") or []) for agent in agents),
        "spawn_subagent_calls": total_calls["spawn_subagent"],
        "spawn_subagents_calls": total_calls["spawn_subagents"],
        "root_finish_calls": total_calls["finish"] - child_finish_calls,
        "child_finish_calls": child_finish_calls,
        "child_tasks_with_absolute_contract": contracted_child_tasks,
        "unchanged_child_tasks": unchanged_child_tasks,
        "get_info_calls": total_calls["get_info"],
        "get_info_no_arg_calls": get_info_no_arg_calls,
        "duplicate_get_info_queries": duplicate_get_info_queries,
        "view_inventory_calls": total_calls["view_inventory"],
        "craft_calls_in_code": total_calls["craft"],
        "multi_craft_executions": multi_craft_executions,
        "actual_craft_calls": int(row.get("craft_calls", report.get("craft_calls", 0)) or 0),
        "tool_errors": len(report.get("tool_errors") or []),
        "execution_errors": execution_errors,
        "observation_truncations": observation_truncations,
        "recursion_reasonable": bool(agents) and not recursion_issues,
        "recursion_issues": recursion_issues,
        "trajectory_reasonable": bool(agents) and not trajectory_issues,
        "trajectory_issues": trajectory_issues,
        "error_type": row.get("error_type"),
        "error": row.get("error"),
        "system_prompt_sha256": (
            hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
            if system_prompt
            else None
        ),
        "system_prompt_chars": len(system_prompt),
    }


def aggregate_textcraft_results(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate TextCraft result metrics without hiding incomplete runs."""
    rows = list(metrics)
    traced = [row for row in rows if row["agent_count"] > 0]
    completed = [row for row in rows if row["ok"]]
    recursion_issue_counts: Counter[str] = Counter(
        issue for row in traced for issue in row["recursion_issues"]
    )
    trajectory_issue_counts: Counter[str] = Counter(
        issue for row in traced for issue in row["trajectory_issues"]
    )
    return {
        "rows": len(rows),
        "completed_runs": len(completed),
        "task_successes": sum(row["success"] for row in rows),
        "score": sum(row["score"] for row in rows) / len(rows) if rows else 0.0,
        "traces_available": len(traced),
        "reasonable_recursions": sum(row["recursion_reasonable"] for row in traced),
        "reasonable_recursion_rate": (
            sum(row["recursion_reasonable"] for row in traced) / len(traced)
            if traced
            else 0.0
        ),
        "reasonable_trajectories": sum(row["trajectory_reasonable"] for row in traced),
        "reasonable_trajectory_rate": (
            sum(row["trajectory_reasonable"] for row in traced) / len(traced)
            if traced
            else 0.0
        ),
        "total_child_agents": sum(row["child_agent_count"] for row in traced),
        "max_trace_depth": max((row["max_trace_depth"] for row in traced), default=0),
        "parallel_spawn_calls": sum(row["spawn_subagents_calls"] for row in traced),
        "child_finish_calls": sum(row["child_finish_calls"] for row in traced),
        "duplicate_get_info_queries": sum(
            row["duplicate_get_info_queries"] for row in traced
        ),
        "get_info_no_arg_calls": sum(row["get_info_no_arg_calls"] for row in traced),
        "multi_craft_executions": sum(row["multi_craft_executions"] for row in traced),
        "observation_truncations": sum(row["observation_truncations"] for row in traced),
        "recursion_issue_counts": dict(sorted(recursion_issue_counts.items())),
        "trajectory_issue_counts": dict(sorted(trajectory_issue_counts.items())),
        "system_prompt_fingerprints": sorted(
            {row["system_prompt_sha256"] for row in traced if row["system_prompt_sha256"]}
        ),
        "failed_instances": [row["instance_id"] for row in rows if not row["ok"]],
    }


def _get_info_signature(node: ast.Call) -> str:
    if not node.args:
        return "<no-args>"
    try:
        value = ast.literal_eval(node.args[0])
    except (ValueError, TypeError, SyntaxError):
        return "<dynamic>"
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return "|".join(value)
    return repr(value)


def _normalize_task(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _has_absolute_inventory_contract(task: str) -> bool:
    return (
        "shared inventory contains at least" in task
        or "required_inventory" in task
        or "absolute inventory" in task
    )


def _recursion_issues(
    *,
    crafting_depth: int,
    child_count: int,
    max_depth: int,
    parallel_spawn_calls: int,
    child_finish_calls: int,
    unchanged_child_tasks: int,
    contracted_child_tasks: int,
) -> list[str]:
    issues: list[str] = []
    if crafting_depth and crafting_depth <= 3 and child_count:
        issues.append("needless_recursion_for_shallow_task")
    if crafting_depth >= 4 and child_count == 0:
        issues.append("missing_recursion_for_deep_task")
    if crafting_depth and max_depth >= crafting_depth:
        issues.append("recursion_did_not_reduce_depth")
    if parallel_spawn_calls:
        issues.append("parallel_mutation_risk")
    if child_finish_calls:
        issues.append("child_called_finish")
    if unchanged_child_tasks:
        issues.append("unchanged_task_delegation")
    if child_count and contracted_child_tasks < child_count:
        issues.append("child_missing_absolute_inventory_contract")
    return issues


def _walk_agents(root: dict[str, Any] | None):
    if not root:
        return
    yield root
    for child in root.get("children") or []:
        yield from _walk_agents(child)


def _find_parent(root: dict[str, Any] | None, parent_id: Any) -> dict[str, Any] | None:
    if not root or parent_id is None:
        return None
    for agent in _walk_agents(root):
        if agent.get("agent_id") == parent_id:
            return agent
    return None


__all__ = ["aggregate_textcraft_results", "analyze_textcraft_result"]
