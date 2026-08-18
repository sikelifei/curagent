"""Run reproducible TextCraft-Synth sampling experiments.

This module is deliberately an experiment adapter.  It converts the official
Platoon JSONL rows into the sample shape understood by the local TextCraft
environment, then delegates execution to the existing environment runner.
It does not alter environment, evaluator, reward, or recursive-harness
semantics.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from recursive_agent.envs import run_registered_environment
from recursive_agent.envs.textcraft_synth.prompts import (
    DEFAULT_TEXTCRAFT_AGENT_PROMPT,
)
from recursive_agent.envs.textcraft_synth.trace_analysis import analyze_textcraft_result
from recursive_agent.repl import find_repl_blocks


DEFAULT_DATASET_PATH = Path(
    "/data2/zhangwenjian/agent/platoon/plugins/textcraft/platoon/textcraft/"
    "textcraft_synth_val.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("validation_results/textcraft_sampling_experiment")
DEFAULT_SPLIT = "val"
DEFAULT_DIFFICULTY = "medium"
DEFAULT_SEED = 42
DEFAULT_TASK_COUNT = 5
DEFAULT_BUDGETS = (32, 64, 96, 128)
DEFAULT_CANDIDATE_BUDGET = 96
DEFAULT_MAX_DEPTH = 12

EXPERIMENT_FILES = {
    "forced_recursion": "experiment_1_forced_recursion.jsonl",
    "budget_sweep": "experiment_2_budget_sweep.jsonl",
    "baseline_recursion": "experiment_4_baseline_recursion.jsonl",
    "prompt_variant": "experiment_5_prompt_variant.jsonl",
    "rl_sampling": "experiment_6_rl_sampling.jsonl",
    "child_return_smoke": "smoke_child_return.jsonl",
    "direct_smoke": "direct_smoke.jsonl",
    "protocol_fixed_30": "curagent_fixed_30.jsonl",
}

FORCED_RECURSION_SUFFIX = """Validation requirement for this rollout:
Before completing the root task, you must delegate one coherent intermediate
crafting subtask to exactly one subagent. Give the subagent a clear crafting
objective and return condition. After the subagent returns, inspect the shared
environment and continue the root task."""

MINIMAL_PROMPT_ADDENDUM = """\n\nCODEACT STRATEGY

- A Python block may perform multiple environment operations.
- Batch related information queries when possible.
- Keep useful recipe information and calculations in the persistent REPL.
- Once several crafting operations are known to be valid, they may be
  executed in the same Python block.

DELEGATION STRATEGY

- For deep targets with multiple intermediate dependencies, prefer assigning
  one coherent intermediate branch to a subagent instead of expanding every
  branch yourself.
- Give the child an exact objective, quantity, scope, restrictions, and return
  condition.
- All agents operate on the same shared environment.
- Use concurrent children only when their work is independent and cannot
  conflict in the shared environment. Otherwise delegate sequentially.
- After a child returns, inspect the shared state before continuing.
"""


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _counts(value: Any, label: str, *, allow_zero: bool = True) -> dict[str, int]:
    raw = _mapping(value, label)
    result: dict[str, int] = {}
    for item, count in raw.items():
        name = str(item).strip()
        if not name:
            raise ValueError(f"{label} contains an empty item name")
        number = int(count)
        if number < 0 or (number == 0 and not allow_zero):
            raise ValueError(f"{label} contains an invalid count for {name!r}")
        if number:
            result[name] = result.get(name, 0) + number
    return result


def _official_misc(row: Mapping[str, Any]) -> Mapping[str, Any]:
    misc = row.get("misc")
    if not isinstance(misc, Mapping):
        raise ValueError("official TextCraft row is missing misc mapping")
    return misc


def normalize_official_row(
    row: Mapping[str, Any],
    *,
    index: int = 0,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Normalize one official Platoon row for ``TextCraftSynthEnvironment``.

    In the official gold trajectory, ``target[1]`` is the number of recipe
    executions in that step.  Its ingredient counts and total result count are
    therefore divided to recover one environment recipe execution.
    """

    if not isinstance(row, Mapping):
        raise ValueError("official TextCraft row must be a mapping")
    misc = _official_misc(row)
    row_id = str(row.get("id", index)).strip()
    if not row_id:
        raise ValueError("official TextCraft row has an empty id")

    targets = _counts(misc.get("target_items"), "misc.target_items", allow_zero=False)
    inventory = _counts(
        misc.get("initial_inventory"),
        "misc.initial_inventory",
        allow_zero=True,
    )
    trajectory = misc.get("gold_trajectory")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise ValueError("official TextCraft row is missing gold_trajectory")

    recipes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step_index, raw_step in enumerate(trajectory):
        step = _mapping(raw_step, f"gold_trajectory[{step_index}]")
        if str(step.get("action", "craft")).strip().lower() != "craft":
            continue
        target = step.get("target")
        if not isinstance(target, Sequence) or isinstance(target, (str, bytes)):
            raise ValueError(f"gold_trajectory[{step_index}] has an invalid target")
        if len(target) != 2:
            raise ValueError(f"gold_trajectory[{step_index}] target must have two values")
        item = str(target[0]).strip()
        if not item:
            raise ValueError(f"gold_trajectory[{step_index}] target item is empty")
        executions = _positive_int(
            target[1],
            f"gold_trajectory[{step_index}] target execution count",
        )
        ingredients = _counts(
            step.get("ingredients"),
            f"gold_trajectory[{step_index}].ingredients",
            allow_zero=False,
        )
        result_count = _positive_int(
            step.get("result_count"),
            f"gold_trajectory[{step_index}].result_count",
        )
        if result_count % executions:
            raise ValueError(
                f"gold_trajectory[{step_index}] result_count is not divisible by "
                "the target execution count"
            )
        scaled_ingredients: dict[str, int] = {}
        for ingredient, count in ingredients.items():
            if count % executions:
                raise ValueError(
                    f"gold_trajectory[{step_index}] ingredient {ingredient!r} is "
                    "not divisible by the target execution count"
                )
            scaled_ingredients[ingredient] = count // executions
        recipe = {
            "ingredients": scaled_ingredients,
            "result_count": result_count // executions,
        }
        if recipe not in recipes[item]:
            recipes[item].append(recipe)

    if not recipes:
        raise ValueError("official TextCraft row has no craft steps")

    difficulty = str(misc.get("difficulty", row.get("difficulty", "unknown")))
    depth_value = misc.get("max_depth", row.get("max_depth"))
    depth = int(depth_value) if depth_value is not None else 0
    return {
        "id": row_id,
        "initial_inventory": inventory,
        "recipes": dict(recipes),
        "targets": targets,
        "difficulty": difficulty,
        "crafting_depth": depth,
        "max_depth": depth,
        "split": str(split),
        "goal": row.get("goal"),
        "max_steps": row.get("max_steps"),
        "gold_trajectory": list(trajectory),
        "metadata": {
            "source": "platoon_textcraft_synth",
            "official_id": row_id,
            "official_goal": row.get("goal"),
            "official_max_steps": row.get("max_steps"),
            "official_num_craft_steps": misc.get("num_craft_steps"),
        },
    }


def load_official_rows(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Read the required official TextCraft validation JSONL."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"DATASET_NOT_FOUND: TextCraft official dataset not found: {source}"
        )
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"official TextCraft row {line_number} must be an object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"official TextCraft dataset is empty: {source}")
    return rows


def _row_difficulty(row: Mapping[str, Any]) -> str:
    misc = row.get("misc")
    if isinstance(misc, Mapping) and misc.get("difficulty") is not None:
        return str(misc["difficulty"]).strip().lower()
    return str(row.get("difficulty", "")).strip().lower()


def select_medium_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_TASK_COUNT,
) -> list[dict[str, Any]]:
    """Filter medium rows and select a stable random subset."""

    if count <= 0:
        raise ValueError("count must be positive")
    medium = [dict(row) for row in rows if _row_difficulty(row) == "medium"]
    if len(medium) < count:
        raise ValueError(
            f"need {count} medium TextCraft rows, found {len(medium)}"
        )
    return random.Random(seed).sample(medium, count)


# Descriptive alias for callers that treat the selected rows as fixed tasks.
select_fixed_tasks = select_medium_rows
select_medium_tasks = select_medium_rows

# Descriptive alias for callers that use the environment's sample terminology.
normalize_official_sample = normalize_official_row


def _walk_agents(root: Mapping[str, Any] | None) -> Iterable[Mapping[str, Any]]:
    if not isinstance(root, Mapping):
        return
    yield root
    children = root.get("children")
    if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
        for child in children:
            if isinstance(child, Mapping):
                yield from _walk_agents(child)


def _unwrap_trace(value: Mapping[str, Any] | None) -> tuple[
    Mapping[str, Any] | None,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """Return root trace, agent-result mapping, and environment report."""

    if not isinstance(value, Mapping):
        return None, {}, {}
    full = value
    nested_trace = full.get("trace")
    if (
        "agent_result" not in full
        and isinstance(nested_trace, Mapping)
        and ("agent_result" in nested_trace or "steps" in nested_trace)
    ):
        nested = dict(nested_trace)
        if "environment_report" not in nested and isinstance(
            full.get("environment_report"), Mapping
        ):
            nested["environment_report"] = full["environment_report"]
        return _unwrap_trace(nested)
    agent_result = full.get("agent_result")
    if not isinstance(agent_result, Mapping):
        agent_result = {}
    root = agent_result.get("trace")
    if not isinstance(root, Mapping):
        # Accept a serialized root trace directly in unit tests and adapters.
        root = value if "steps" in value or "children" in value else None
    report = full.get("environment_report")
    if not isinstance(report, Mapping):
        report = {}
    return root, agent_result, report


def _count_agent_calls(agent: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    steps = agent.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return dict(counts)
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        executions = step.get("code_executions")
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            continue
        for execution in executions:
            if not isinstance(execution, Mapping):
                continue
            try:
                tree = ast.parse(str(execution.get("code", "")), mode="exec")
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    counts[node.func.id] += 1
    return dict(counts)


def _call_counts(root: Mapping[str, Any] | None) -> tuple[dict[str, int], int, int]:
    counts: dict[str, int] = defaultdict(int)
    parse_errors = 0
    runtime_errors = 0
    for agent in _walk_agents(root):
        steps = agent.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            response = step.get("response")
            executions = step.get("code_executions")
            if (
                isinstance(response, str)
                and response.strip()
                and (
                    not isinstance(executions, Sequence)
                    or isinstance(executions, (str, bytes))
                    or not executions
                )
                and not find_repl_blocks(response)
            ):
                parse_errors += 1
            if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
                continue
            for execution in executions:
                if not isinstance(execution, Mapping):
                    continue
                code = str(execution.get("code", ""))
                execution_parse_error = False
                try:
                    tree = ast.parse(code, mode="exec")
                except SyntaxError:
                    parse_errors += 1
                    execution_parse_error = True
                    tree = None
                if tree is not None:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            counts[node.func.id] += 1
                error = str(execution.get("error") or "").strip()
                if error:
                    if (
                        not execution_parse_error
                        and ("syntaxerror" in error.lower() or "parse" in error.lower())
                    ):
                        parse_errors += 1
                    elif not execution_parse_error:
                        runtime_errors += 1
    return dict(counts), parse_errors, runtime_errors


def extract_rollout_metrics(
    trace: Mapping[str, Any] | None,
    *,
    task_id: str | None = None,
    budget: int | None = None,
    error: BaseException | None = None,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the required experiment metrics from one serialized trace."""

    root, agent_result, embedded_report = _unwrap_trace(trace)
    report_data = report if isinstance(report, Mapping) else embedded_report
    agents = list(_walk_agents(root))
    calls, parse_errors, runtime_errors = _call_counts(root)
    tool_errors = report_data.get("tool_errors")
    if isinstance(tool_errors, Sequence) and not isinstance(tool_errors, (str, bytes)):
        runtime_errors += len(tool_errors)

    status = agent_result.get("status") or (root or {}).get("status")
    if error is not None:
        error_name = type(error).__name__.lower()
        if "timeout" in error_name:
            status = "timeout"
        elif "cancel" in error_name and status in {None, "error"}:
            status = "cancelled"
        elif not status:
            status = "error"
    if not status:
        status = "environment_done" if report_data.get("success") else "unknown"
    if (
        status == "error"
        and runtime_errors == 0
        and isinstance(root, Mapping)
        and root.get("error")
    ):
        runtime_errors = 1

    steps_value = agent_result.get("steps")
    if isinstance(steps_value, int) and not isinstance(steps_value, bool):
        global_steps = steps_value
    else:
        global_steps = sum(
            len(agent.get("steps") or [])
            for agent in agents
            if isinstance(agent.get("steps"), Sequence)
            and not isinstance(agent.get("steps"), (str, bytes))
        )

    root_calls = _count_agent_calls(root) if isinstance(root, Mapping) else {}
    finish_called = bool(root_calls.get("finish", 0))
    missing = report_data.get("missing", {})
    if not isinstance(missing, Mapping):
        missing = {}
    analysis_row = {
        "ok": error is None,
        "success": bool(report_data.get("success", False)),
        "score": float(report_data.get("score", 0.0) or 0.0),
        "trace": dict(trace or {}),
    }
    analyzed = analyze_textcraft_result(analysis_row)
    return {
        "task_id": task_id,
        "budget": budget,
        "success": bool(report_data.get("success", False)),
        "score": float(report_data.get("score", 0.0) or 0.0),
        "finished": bool(report_data.get("finished", False)),
        "termination_reason": str(status),
        "global_steps_used": int(global_steps),
        "number_of_agents": len(agents),
        "maximum_depth": max(
            (int(agent.get("depth", 0) or 0) for agent in agents),
            default=0,
        ),
        "spawn_subagent_count": int(calls.get("spawn_subagent", 0)),
        "spawn_subagents_count": int(calls.get("spawn_subagents", 0)),
        "get_info_count": int(calls.get("get_info", 0)),
        "craft_count": int(calls.get("craft", 0)),
        "finish_called": finish_called,
        "parse_errors": int(parse_errors),
        "runtime_errors": int(runtime_errors),
        "missing": dict(missing),
        "steps": int(global_steps),
        "root_steps": analyzed["root_steps"],
        "child_steps": analyzed["child_steps"],
        "recursive_children": analyzed["child_agent_count"],
        "max_trace_depth": analyzed["max_trace_depth"],
        "spawn_subagent_calls": analyzed["spawn_subagent_calls"],
        "spawn_subagents_calls": analyzed["spawn_subagents_calls"],
        "children_completed": analyzed["children_completed"],
        "children_budget_exhausted": analyzed["children_budget_exhausted"],
        "children_failed": analyzed["children_failed"],
        "return_to_parent_calls": analyzed["return_to_parent_calls"],
        "finish_attempts": analyzed["finish_attempts"],
        "get_info_calls": analyzed["get_info_calls"],
        "noarg_get_info_calls": analyzed["get_info_no_arg_calls"],
        "craft_calls": analyzed["actual_craft_calls"],
        "craft_errors": analyzed["tool_errors"],
        "craft_attempts": analyzed["craft_attempts"],
        "craft_successes": analyzed["craft_successes"],
        "craft_error_wrong_amount": analyzed["craft_error_wrong_amount"],
        "craft_error_missing_ingredient": analyzed["craft_error_missing_ingredient"],
        "craft_error_extra_ingredient": analyzed["craft_error_extra_ingredient"],
        "craft_error_invalid_output_multiple": analyzed["craft_error_invalid_output_multiple"],
        "craft_error_insufficient_inventory": analyzed["craft_error_insufficient_inventory"],
        "invented_or_unknown_item_attempts": analyzed["invented_or_unknown_item_attempts"],
        "same_task_child_edges": analyzed["same_task_child_edges"],
        "no_progress_repetitions": analyzed["no_progress_repetitions"],
        "repeated_action_count": analyzed["repeated_action_count"],
        "repeated_observation_count": analyzed["repeated_observation_count"],
        "no_progress_streak": analyzed["no_progress_streak"],
        "no_progress_warnings": analyzed["no_progress_warnings"],
        "no_progress_terminations": analyzed["no_progress_terminations"],
        "final_missing_targets": analyzed["final_missing_targets"],
    }


# Compatibility alias used by small analysis scripts.
extract_metrics = extract_rollout_metrics


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append and flush one JSONL row immediately."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    destination = Path(path)
    if not destination.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with destination.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def completed_rollout_ids(path: str | Path) -> set[str]:
    """Return rollout IDs already persisted in a result JSONL file."""

    return {
        str(row["rollout_id"])
        for row in read_jsonl(path)
        if row.get("rollout_id") is not None
    }


# Compatibility alias for resume-oriented callers.
completed_rollout_keys = completed_rollout_ids


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "rollout"


def _rollout_id(experiment: str, task_id: str, budget: int, replicate: int) -> str:
    return f"{experiment}:{task_id}:{budget}:{replicate}"


def _trace_for_run(run: Any) -> dict[str, Any]:
    if hasattr(run, "to_trace_dict"):
        value = run.to_trace_dict()
    elif isinstance(run, Mapping):
        value = dict(run)
    else:
        raise TypeError("environment runner returned an unsupported result")
    if not isinstance(value, Mapping):
        raise TypeError("environment runner trace must be a mapping")
    return dict(value)


def run_rollout(
    *,
    sample: Mapping[str, Any],
    task_id: str,
    experiment: str,
    budget: int,
    replicate: int,
    model_config: str | Path,
    output_dir: str | Path,
    split: str = DEFAULT_SPLIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_concurrent_subagents: int = 4,
    max_subagents_per_agent: int | None = 6,
    max_run_seconds: float | None = 900.0,
    max_observation_chars: int | None = 8000,
    request_timeout: float | None = 120.0,
    max_retries: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float | None = None,
    endpoint: str | None = None,
    model_name: str | None = None,
    prompt_template: str | None = None,
    agent_prompt: str | None = None,
) -> dict[str, Any]:
    """Run one rollout and persist its row and raw trace before returning."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / EXPERIMENT_FILES[experiment]
    rollout_id = _rollout_id(experiment, task_id, budget, replicate)
    started = time.monotonic()
    trace: dict[str, Any]
    try:
        environment_kwargs: dict[str, Any] = {
            "split": split,
            "instance_id": 0,
            "samples": [dict(sample)],
        }
        if prompt_template is not None:
            environment_kwargs["prompt_template"] = prompt_template
        if agent_prompt is not None:
            environment_kwargs["agent_prompt"] = agent_prompt
        sampling_args: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            sampling_args["top_p"] = top_p
        model_overrides: dict[str, Any] = {
            "sampling_args": sampling_args,
        }
        if endpoint:
            model_overrides["base_url"] = endpoint
        if request_timeout is not None:
            model_overrides["timeout"] = request_timeout
        if max_retries is not None:
            model_overrides["max_retries"] = max_retries
        if model_name:
            model_overrides["model_name"] = model_name
        run = run_registered_environment(
            "textcraft_synth",
            model_config=model_config,
            environment_kwargs=environment_kwargs,
            agent_kwargs={
                "max_total_steps": budget,
                "max_depth": max_depth,
                "max_concurrent_subagents": max_concurrent_subagents,
                "max_subagents_per_agent": max_subagents_per_agent,
                "max_run_seconds": max_run_seconds,
                "max_observation_chars": max_observation_chars,
            },
            model_overrides=model_overrides,
        )
        trace = _trace_for_run(run)
        row = extract_rollout_metrics(trace, task_id=task_id, budget=budget)
        row.update(
            {
                "ok": True,
                "rollout_id": rollout_id,
                "experiment": experiment,
                "replicate": replicate,
                "temperature": temperature,
                "top_p": top_p,
                "duration_seconds": time.monotonic() - started,
                "trace": trace,
            }
        )
    except Exception as exc:
        partial = getattr(exc, "partial_trace", None)
        trace = dict(partial) if isinstance(partial, Mapping) else {
            "agent_result": {"status": "error", "steps": 0, "trace": None},
            "environment_report": {},
        }
        row = extract_rollout_metrics(
            trace,
            task_id=task_id,
            budget=budget,
            error=exc,
        )
        row.update(
            {
                "ok": False,
                "rollout_id": rollout_id,
                "experiment": experiment,
                "replicate": replicate,
                "temperature": temperature,
                "top_p": top_p,
                "duration_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trace": trace,
            }
        )

    raw_dir = output_root / "raw_trajectories"
    raw_path = raw_dir / f"{_safe_name(rollout_id)}.json"
    _write_json(raw_path, trace)
    row["raw_trace_path"] = str(raw_path.relative_to(output_root))
    append_jsonl(result_path, row)
    return row


def _prompt_template(suffix: str | None = None) -> str | None:
    if suffix is None:
        return None
    return "Craft the following additional items: {targets}\n\n" + suffix.strip()


def _run_plan(
    *,
    experiment: str,
    samples: Sequence[tuple[str, Mapping[str, Any]]],
    output_dir: Path,
    model_config: str | Path,
    args: argparse.Namespace,
    budget: int,
    replicates: int,
    temperature: float,
    top_p: float | None,
    task_suffix: str | None = None,
    agent_prompt: str | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    result_path = output_dir / EXPERIMENT_FILES[experiment]
    completed = completed_rollout_ids(result_path) if resume else set()
    rows: list[dict[str, Any]] = []
    for task_id, sample in samples:
        for replicate in range(replicates):
            rollout_id = _rollout_id(experiment, task_id, budget, replicate)
            if rollout_id in completed:
                continue
            row = run_rollout(
                sample=sample,
                task_id=task_id,
                experiment=experiment,
                budget=budget,
                replicate=replicate,
                model_config=model_config,
                output_dir=output_dir,
                split=args.split,
                max_depth=args.max_depth,
                max_concurrent_subagents=args.max_concurrent_subagents,
                max_subagents_per_agent=args.max_subagents_per_agent,
                max_run_seconds=args.max_run_seconds,
                max_observation_chars=args.max_observation_chars,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=top_p,
                endpoint=args.endpoint,
                model_name=args.model_name,
                prompt_template=_prompt_template(task_suffix),
                agent_prompt=agent_prompt,
            )
            rows.append(row)
            print(
                f"experiment={experiment} task={task_id} budget={budget} "
                f"replicate={replicate} success={row['success']} "
                f"steps={row['global_steps_used']}",
                flush=True,
            )
    return rows


def infer_candidate_budget(
    rows: Sequence[Mapping[str, Any]],
    *,
    default: int = DEFAULT_CANDIDATE_BUDGET,
) -> int:
    """Choose the smallest budget attaining the best observed success rate."""

    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        try:
            budget = int(row.get("budget"))
        except (TypeError, ValueError):
            continue
        grouped[budget].append(bool(row.get("success", False)))
    if not grouped:
        return default
    rates = {
        budget: sum(values) / len(values)
        for budget, values in grouped.items()
        if values
    }
    best = max(rates.values(), default=0.0)
    return min((budget for budget, rate in rates.items() if rate == best), default=default)


def _load_selected_rows(
    dataset_path: Path,
    *,
    seed: int,
    count: int,
    task_ids_path: Path,
    resume: bool,
    split: str,
) -> list[tuple[str, dict[str, Any]]]:
    official = load_official_rows(dataset_path)
    by_id = {str(row.get("id")): row for row in official}
    selected: list[dict[str, Any]]
    if resume and task_ids_path.is_file():
        stored = json.loads(task_ids_path.read_text(encoding="utf-8"))
        if isinstance(stored, Mapping):
            stored = stored.get("task_ids")
        if not isinstance(stored, list) or not all(isinstance(item, str) for item in stored):
            raise ValueError(f"invalid task_ids.json: {task_ids_path}")
        try:
            selected = [by_id[item] for item in stored]
        except KeyError as exc:
            raise ValueError(f"persisted task id is absent from official dataset: {exc}") from exc
    else:
        selected = select_medium_rows(official, seed=seed, count=count)
        _write_json(task_ids_path, [str(row["id"]) for row in selected])
    normalized = [
        (str(row["id"]), normalize_official_row(row, index=index, split=split))
        for index, row in enumerate(selected)
    ]
    return normalized


def _repository_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _build_config(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "repository_commit": _repository_commit(),
        "model_config": str(Path(args.config).expanduser().resolve()),
        "model_name": args.model_name,
        "endpoint": args.endpoint,
        "dataset_path": str(dataset_path),
        "split": args.split,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "task_count": len(task_ids),
        "task_ids": list(task_ids),
        "budgets": list(DEFAULT_BUDGETS),
        "candidate_global_budget": args.candidate_budget,
        "max_depth": args.max_depth,
        "experiments": list(EXPERIMENT_FILES),
        "rl_sampling": {"rollouts_per_task": 8, "temperature": 1.0, "top_p": 1.0},
    }


def _rows_for_report(output_dir: Path, experiment: str) -> list[dict[str, Any]]:
    return read_jsonl(output_dir / EXPERIMENT_FILES[experiment])


def _recursion_used(row: Mapping[str, Any]) -> bool:
    return bool(
        int(row.get("number_of_agents", 0) or 0) > 1
        or int(row.get("spawn_subagent_count", 0) or 0) > 0
        or int(row.get("spawn_subagents_count", 0) or 0) > 0
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _forced_path_pass(row: Mapping[str, Any]) -> bool:
    if (
        int(row.get("number_of_agents", 0) or 0) != 2
        or int(row.get("spawn_subagent_count", 0) or 0)
        + int(row.get("spawn_subagents_count", 0) or 0)
        != 1
    ):
        return False
    trace = row.get("trace")
    root, _, _ = _unwrap_trace(trace if isinstance(trace, Mapping) else None)
    children = root.get("children") if isinstance(root, Mapping) else []
    if not isinstance(children, Sequence) or not children:
        return False
    child = children[0]
    return isinstance(child, Mapping) and child.get("status") in {
        "completed",
        "environment_done",
    }


def generate_report(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any],
    task_ids: Sequence[str],
) -> Path:
    """Generate the required report from persisted JSONL rows."""

    destination = Path(output_dir)
    forced = _rows_for_report(destination, "forced_recursion")
    sweep = _rows_for_report(destination, "budget_sweep")
    baseline = _rows_for_report(destination, "baseline_recursion")
    variant = _rows_for_report(destination, "prompt_variant")
    rl = _rows_for_report(destination, "rl_sampling")

    lines = [
        "# TextCraft Sampling Experiment",
        "",
        "## Environment",
        "",
        f"repository commit: {config.get('repository_commit', 'unknown')}",
        f"endpoint: {config.get('endpoint') or 'configured model endpoint'}",
        f"served model ID: {config.get('model_name') or 'from model config'}",
        f"real TextCraft dataset path: {config.get('dataset_path')}",
        f"split: {config.get('split', DEFAULT_SPLIT)}",
        f"difficulty: {config.get('difficulty', DEFAULT_DIFFICULTY)}",
        f"fixed task IDs: {', '.join(task_ids)}",
        "",
        "## Experiment 1",
        "",
    ]
    forced_row = forced[0] if forced else {}
    forced_pass = bool(forced_row and _forced_path_pass(forced_row))
    lines.extend(
        [
            f"forced recursion path: {'PASS' if forced_pass else 'FAIL'}",
            f"shared environment verified: {'yes' if forced_pass else 'no'}",
            f"shared global budget verified: {'yes' if forced_pass else 'no'}",
            f"child return verified: {'yes' if forced_pass else 'no'}",
            "",
            "## Experiment 2",
            "",
            "budget | success | mean steps | recursion | exhausted",
            "--- | --- | --- | --- | ---",
        ]
    )
    for budget in DEFAULT_BUDGETS:
        group = [row for row in sweep if int(row.get("budget", -1)) == budget]
        successes = sum(bool(row.get("success")) for row in group)
        mean_steps = _mean([float(row.get("global_steps_used", 0)) for row in group])
        recursion = sum(_recursion_used(row) for row in group)
        exhausted = sum(row.get("termination_reason") == "budget_exhausted" for row in group)
        lines.append(
            f"{budget} | {successes}/{len(group)} | {mean_steps:.2f} | "
            f"{recursion}/{len(group)} | {exhausted}/{len(group)}"
        )
    candidate = config.get("candidate_global_budget", DEFAULT_CANDIDATE_BUDGET)
    lines.extend(["", f"candidate_global_budget = {candidate}", ""])

    all_trace_rows = sweep or baseline or variant
    generations = sum(int(row.get("global_steps_used", 0) or 0) for row in all_trace_rows)
    env_calls = sum(
        int(row.get("get_info_count", 0) or 0)
        + int(row.get("craft_count", 0) or 0)
        for row in all_trace_rows
    )
    average_calls = env_calls / generations if generations else 0.0
    utilization = (
        "PATHOLOGICAL"
        if not all_trace_rows or average_calls == 0
        else "UNDERUTILIZED"
        if average_calls < 0.5
        else "GOOD"
    )
    lines.extend(
        [
            "## Experiment 3",
            "",
            f"CodeAct utilization: {utilization}",
            "",
            f"evidence: {env_calls} environment calls across {generations} LLM generations "
            f"({average_calls:.2f} calls/generation).",
            "",
            "## Experiment 4",
            "",
            f"baseline autonomous recursion: {sum(_recursion_used(row) for row in baseline)}/{len(baseline)} used recursion",
            f"{sum(bool(row.get('success')) for row in baseline)}/{len(baseline)} succeeded",
            "",
            "## Experiment 5",
            "",
            f"current prompt: success {sum(bool(row.get('success')) for row in baseline)}/{len(baseline)}",
            f"recursion {sum(_recursion_used(row) for row in baseline)}/{len(baseline)}",
            f"minimal variant: success {sum(bool(row.get('success')) for row in variant)}/{len(variant)}",
            f"recursion {sum(_recursion_used(row) for row in variant)}/{len(variant)}",
            "",
            "## Experiment 6",
            "",
        ]
    )
    completed = len(rl)
    successes = sum(bool(row.get("success")) for row in rl)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rl:
        by_task[str(row.get("task_id"))].append(row)
    groups = {"ALL_FAIL": 0, "MIXED": 0, "ALL_SUCCESS": 0}
    for values in by_task.values():
        count = sum(bool(row.get("success")) for row in values)
        if count == 0:
            groups["ALL_FAIL"] += 1
        elif count == len(values) and len(values) == 8:
            groups["ALL_SUCCESS"] += 1
        else:
            groups["MIXED"] += 1
    lines.extend(
        [
            f"{8 * len(task_ids)} requested",
            f"{completed} completed",
            f"overall success: {successes}/{completed}",
            f"ALL_FAIL: {groups['ALL_FAIL']}/{len(task_ids)}",
            f"MIXED: {groups['MIXED']}/{len(task_ids)}",
            f"ALL_SUCCESS: {groups['ALL_SUCCESS']}/{len(task_ids)}",
            "",
            "## Final Diagnosis",
            "",
        ]
    )
    diagnoses: list[str] = []
    if forced and not forced_pass:
        diagnoses.append("HARNESS_RECURSION_PROBLEM")
    if sweep and all(not row.get("success") for row in sweep):
        diagnoses.append("MODEL_CAPABILITY_LIMIT")
    if all_trace_rows and average_calls < 0.5:
        diagnoses.append("CODEACT_UNDERUTILIZED")
    if baseline and not any(_recursion_used(row) for row in baseline):
        diagnoses.append("DELEGATION_PROMPT_TOO_WEAK")
    if not diagnoses:
        diagnoses.append("INCONCLUSIVE")
    lines.extend(diagnoses)
    report_path = destination / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        "--mode",
        dest="experiment",
        choices=tuple(EXPERIMENT_FILES) + ("all",),
        default="budget_sweep",
    )
    parser.add_argument("--config", default="configs/model_api_qwen3_4b_instruct_2507_vllm.yaml")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--endpoint", default="http://192.168.1.134:56782/v1")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    parser.add_argument("--candidate-budget", type=int, default=DEFAULT_CANDIDATE_BUDGET)
    parser.add_argument("--forced-task-index", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-concurrent-subagents", type=int, default=4)
    parser.add_argument("--max-subagents-per-agent", type=int, default=6)
    parser.add_argument("--max-run-seconds", type=float, default=900.0)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--model-name")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> Path:
    if args.difficulty != DEFAULT_DIFFICULTY:
        raise ValueError("this experiment runner only supports difficulty=medium")
    if args.task_count != DEFAULT_TASK_COUNT:
        raise ValueError("the controlled experiment requires exactly five tasks")
    if not 0 <= args.forced_task_index < args.task_count:
        raise ValueError("forced-task-index must identify one selected task")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    task_ids_path = output_dir / "task_ids.json"
    selected = _load_selected_rows(
        dataset_path,
        seed=args.seed,
        count=args.task_count,
        task_ids_path=task_ids_path,
        resume=args.resume,
        split=args.split,
    )
    for filename in EXPERIMENT_FILES.values():
        (output_dir / filename).touch(exist_ok=True)
    task_ids = [task_id for task_id, _ in selected]
    config = _build_config(args=args, dataset_path=dataset_path, task_ids=task_ids)
    _write_json(output_dir / "config.json", config)

    modes = [args.experiment] if args.experiment != "all" else [
        "forced_recursion",
        "budget_sweep",
        "baseline_recursion",
        "prompt_variant",
        "rl_sampling",
    ]
    for mode in modes:
        if mode == "forced_recursion":
            _run_plan(
                experiment=mode,
                samples=[selected[args.forced_task_index]],
                output_dir=output_dir,
                model_config=args.config,
                args=args,
                budget=64,
                replicates=1,
                temperature=0.0,
                top_p=None,
                task_suffix=FORCED_RECURSION_SUFFIX,
                resume=args.resume,
            )
        elif mode == "budget_sweep":
            for budget in DEFAULT_BUDGETS:
                _run_plan(
                    experiment=mode,
                    samples=selected,
                    output_dir=output_dir,
                    model_config=args.config,
                    args=args,
                    budget=budget,
                    replicates=1,
                    temperature=0.0,
                    top_p=None,
                    resume=args.resume,
                )
            config["candidate_global_budget"] = infer_candidate_budget(
                _rows_for_report(output_dir, "budget_sweep"),
                default=args.candidate_budget,
            )
            _write_json(output_dir / "config.json", config)
        elif mode == "baseline_recursion":
            sweep_rows = _rows_for_report(output_dir, "budget_sweep")
            candidate = infer_candidate_budget(
                sweep_rows,
                default=args.candidate_budget,
            )
            config["candidate_global_budget"] = candidate
            _write_json(output_dir / "config.json", config)
            _run_plan(
                experiment=mode,
                samples=selected,
                output_dir=output_dir,
                model_config=args.config,
                args=args,
                budget=candidate,
                replicates=1,
                temperature=0.0,
                top_p=None,
                resume=args.resume,
            )
        elif mode == "prompt_variant":
            candidate = int(config.get("candidate_global_budget", args.candidate_budget))
            _run_plan(
                experiment=mode,
                samples=selected,
                output_dir=output_dir,
                model_config=args.config,
                args=args,
                budget=candidate,
                replicates=1,
                temperature=0.0,
                top_p=None,
                agent_prompt=DEFAULT_TEXTCRAFT_AGENT_PROMPT + MINIMAL_PROMPT_ADDENDUM,
                resume=args.resume,
            )
        elif mode == "rl_sampling":
            candidate = int(config.get("candidate_global_budget", args.candidate_budget))
            _run_plan(
                experiment=mode,
                samples=selected,
                output_dir=output_dir,
                model_config=args.config,
                args=args,
                budget=candidate,
                replicates=8,
                temperature=1.0,
                top_p=1.0,
                resume=args.resume,
            )

    return generate_report(output_dir, config=config, task_ids=task_ids)


def main() -> None:
    args = _build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORCED_RECURSION_SUFFIX",
    "MINIMAL_PROMPT_ADDENDUM",
    "append_jsonl",
    "completed_rollout_ids",
    "completed_rollout_keys",
    "extract_metrics",
    "extract_rollout_metrics",
    "generate_report",
    "infer_candidate_budget",
    "load_official_rows",
    "normalize_official_row",
    "normalize_official_sample",
    "read_jsonl",
    "run_experiment",
    "run_rollout",
    "select_fixed_tasks",
    "select_medium_tasks",
    "select_medium_rows",
]
