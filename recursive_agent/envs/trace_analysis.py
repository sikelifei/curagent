"""Mechanical metrics extracted from credential-free environment run traces."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

_DELEGATION_DIRECTIVES = (
    "must delegate",
    "must spawn",
    "always delegate",
    "always spawn",
    "you are a planner",
    "you are an orchestrator",
)


def analyze_environment_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize REPL, variable, tool, recursion, prompt, and env behavior."""
    tool_names = set(trace.get("tools", {}))
    root = (trace.get("agent_result") or {}).get("trace")
    agents = list(_walk_agents(root)) if root else []
    call_counts: Counter[str] = Counter()
    variables: set[str] = set()
    code_blocks = 0
    code_parse_errors = 0
    execution_errors = 0
    observation_truncations = 0
    steps = 0

    for agent in agents:
        for step in agent.get("steps", []):
            steps += 1
            if step.get("observation_truncated"):
                observation_truncations += 1
            for execution in step.get("code_executions", []):
                code_blocks += 1
                variables.update(execution.get("variables") or [])
                if execution.get("error"):
                    execution_errors += 1
                try:
                    tree = ast.parse(execution.get("code", ""))
                except SyntaxError:
                    code_parse_errors += 1
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        call_counts[node.func.id] += 1

    environment_report = trace.get("environment_report") or {}
    trajectory = environment_report.get("trajectory") or []
    environment_actions = [
        row.get("action")
        for row in trajectory
        if isinstance(row, dict) and row.get("action")
    ]
    system_prompt = str((trace.get("prompts") or {}).get("system", ""))
    lowered_system = system_prompt.lower()
    result = trace.get("agent_result") or {}

    return {
        "status": result.get("status"),
        "success": bool(environment_report.get("success", False)),
        "reward": float(environment_report.get("reward", 0.0) or 0.0),
        "environment_steps": int(environment_report.get("steps", 0) or 0),
        "environment_actions": len(environment_actions),
        "model_calls": int((result.get("usage") or {}).get("total_calls", 0) or 0),
        "agent_count": len(agents),
        "child_agent_count": max(0, len(agents) - 1),
        "max_depth": max((int(agent.get("depth", 0)) for agent in agents), default=0),
        "agent_steps": steps,
        "repl_code_blocks": code_blocks,
        "code_parse_errors": code_parse_errors,
        "execution_errors": execution_errors,
        "observation_truncations": observation_truncations,
        "variable_snapshots_present": code_blocks > 0 and bool(variables),
        "variables": sorted(variables),
        "tool_calls": {
            name: call_counts[name]
            for name in sorted(tool_names)
            if call_counts[name]
        },
        "spawn_subagent_calls": call_counts["spawn_subagent"],
        "spawn_subagents_calls": call_counts["spawn_subagents"],
        "spawn_calls_total": (
            call_counts["spawn_subagent"] + call_counts["spawn_subagents"]
        ),
        "child_trace_consistent": (
            len(agents) - 1 > 0
            if call_counts["spawn_subagent"] + call_counts["spawn_subagents"] > 0
            else len(agents) - 1 == 0
        ),
        "system_prompt_has_forced_delegation": any(
            directive in lowered_system for directive in _DELEGATION_DIRECTIVES
        ),
        "system_prompt_mentions_registered_tools": all(
            f"`{name}`" in system_prompt for name in tool_names
        ),
        "task_prompt_occurrences_in_root_task": 1 if root else 0,
    }


def aggregate_trace_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in rows if row.get("ok") and row.get("metrics")]
    errors = [row for row in rows if not row.get("ok")]
    count = len(metrics)
    tool_totals: Counter[str] = Counter()
    variable_totals: Counter[str] = Counter()
    for item in metrics:
        tool_totals.update(item.get("tool_calls") or {})
        variable_totals.update(item.get("variables") or [])
    return {
        "episodes_requested": len(rows),
        "episodes_completed": count,
        "episodes_failed": len(errors),
        "successes": sum(bool(item["success"]) for item in metrics),
        "success_rate": (
            sum(bool(item["success"]) for item in metrics) / count if count else 0.0
        ),
        "average_reward": (
            sum(float(item["reward"]) for item in metrics) / count if count else 0.0
        ),
        "total_model_calls": sum(int(item["model_calls"]) for item in metrics),
        "total_agent_steps": sum(int(item["agent_steps"]) for item in metrics),
        "total_repl_code_blocks": sum(int(item["repl_code_blocks"]) for item in metrics),
        "episodes_with_repl": sum(item["repl_code_blocks"] > 0 for item in metrics),
        "episodes_with_variables": sum(
            bool(item["variable_snapshots_present"]) for item in metrics
        ),
        "episodes_with_tool_calls": sum(bool(item["tool_calls"]) for item in metrics),
        "tool_call_totals": dict(sorted(tool_totals.items())),
        "common_variables": variable_totals.most_common(30),
        "episodes_with_spawn_calls": sum(item["spawn_calls_total"] > 0 for item in metrics),
        "total_spawn_calls": sum(int(item["spawn_calls_total"]) for item in metrics),
        "total_child_agents": sum(int(item["child_agent_count"]) for item in metrics),
        "max_observed_depth": max((int(item["max_depth"]) for item in metrics), default=0),
        "episodes_with_execution_errors": sum(item["execution_errors"] > 0 for item in metrics),
        "total_execution_errors": sum(int(item["execution_errors"]) for item in metrics),
        "episodes_with_observation_truncation": sum(
            int(item.get("observation_truncations", 0)) > 0 for item in metrics
        ),
        "total_observation_truncations": sum(
            int(item.get("observation_truncations", 0)) for item in metrics
        ),
        "total_code_parse_errors": sum(int(item["code_parse_errors"]) for item in metrics),
        "episodes_with_forced_delegation_prompt": sum(
            bool(item["system_prompt_has_forced_delegation"]) for item in metrics
        ),
        "episodes_missing_tool_descriptions": sum(
            not item["system_prompt_mentions_registered_tools"] for item in metrics
        ),
        "child_trace_inconsistencies": sum(
            not item["child_trace_consistent"] for item in metrics
        ),
        "failed_indices": [row.get("instance_id") for row in errors],
    }


def _walk_agents(root: dict[str, Any] | None):
    if root is None:
        return
    yield root
    for child in root.get("children", []):
        yield from _walk_agents(child)
