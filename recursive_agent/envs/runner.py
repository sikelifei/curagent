"""Helpers for running a registered environment with RecursiveAgent."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import RecursiveAgent
from ..config import load_model_config
from ..types import AgentResult
from .base import AgentEnvironment
from .registry import create_environment


@dataclass
class EnvironmentRunResult:
    agent_result: AgentResult
    environment_report: dict[str, Any]
    system_prompt: str
    task_prompt: str
    initial_context: Any
    tool_descriptions: dict[str, dict[str, Any]]

    def to_trace_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, credential-free execution trace."""
        return {
            "prompts": {
                "system": self.system_prompt,
                "task": self.task_prompt,
                "initial_context": self.initial_context,
            },
            "tools": self.tool_descriptions,
            "agent_result": {
                "answer": self.agent_result.answer,
                "status": self.agent_result.status,
                "steps": self.agent_result.steps,
                "usage": self.agent_result.usage.to_dict(),
                "trace": _agent_trace_to_dict(self.agent_result.trace),
            },
            "environment_report": self.environment_report,
        }


def run_environment(
    environment: AgentEnvironment,
    *,
    model_config: str | Path,
    agent_kwargs: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> EnvironmentRunResult:
    """Run and close one initialized environment episode."""
    kwargs = dict(agent_kwargs or {})
    conflicts = {
        "tools",
        "termination_check",
        "prompt_addendum",
        "system_prompt",
        "root_prompt",
        "child_prompt",
        "completion_prompt",
        "forced_final_prompt",
        "delegated_forced_final_prompt",
        "delegated_task_prompt",
        "delegated_prompt_addendum",
        "delegated_completion_prompt",
        "delegated_disabled_tools",
        "max_repl_blocks_per_step",
        "disabled_repl_builtins",
    } & set(kwargs)
    if conflicts:
        raise ValueError(
            f"Environment runner owns these agent arguments: {sorted(conflicts)}"
        )
    try:
        task_prompt = environment.task
        initial_context = copy.deepcopy(environment.context)
        tools = environment.tools()
        backend, backend_kwargs = load_model_config(model_config)
        backend_kwargs = _merge_nested(backend_kwargs, dict(model_overrides or {}))
        agent = RecursiveAgent(
            backend=backend,
            backend_kwargs=backend_kwargs,
            tools=tools,
            termination_check=environment.status,
            prompt_addendum=environment.agent_prompt,
            system_prompt=environment.system_prompt,
            root_prompt=environment.root_prompt,
            child_prompt=environment.child_prompt,
            completion_prompt=environment.completion_prompt,
            forced_final_prompt=environment.forced_final_prompt,
            delegated_forced_final_prompt=environment.delegated_forced_final_prompt,
            delegated_task_prompt=environment.delegated_task_prompt,
            delegated_prompt_addendum=environment.delegated_prompt_addendum,
            delegated_completion_prompt=environment.delegated_completion_prompt,
            delegated_disabled_tools=environment.delegated_disabled_tools,
            max_repl_blocks_per_step=environment.max_repl_blocks_per_step,
            disabled_repl_builtins=environment.disabled_repl_builtins,
            **kwargs,
        )
        try:
            result = agent.run(task=task_prompt, context=initial_context)
        except BaseException as exc:
            report = environment.report()
            usage = getattr(exc, "usage", None)
            exc.partial_trace = {
                "prompts": {
                    "system": agent.system_prompt,
                    "task": task_prompt,
                    "initial_context": initial_context,
                },
                "tools": _describe_tools(tools),
                "agent_result": {
                    "answer": None,
                    "status": "error",
                    "steps": len(
                        (getattr(exc, "agent_trace", None) or _EMPTY_TRACE).steps
                    ),
                    "usage": usage.to_dict() if usage is not None else {},
                    "trace": _agent_trace_to_dict(getattr(exc, "agent_trace", None)),
                },
                "environment_report": report,
            }
            raise
        report = environment.report()
        return EnvironmentRunResult(
            agent_result=result,
            environment_report=report,
            system_prompt=agent.system_prompt,
            task_prompt=task_prompt,
            initial_context=initial_context,
            tool_descriptions=_describe_tools(tools),
        )
    finally:
        environment.close()


def run_registered_environment(
    name: str,
    *,
    model_config: str | Path,
    environment_kwargs: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> EnvironmentRunResult:
    environment = create_environment(name, **dict(environment_kwargs or {}))
    return run_environment(
        environment,
        model_config=model_config,
        agent_kwargs=agent_kwargs,
        model_overrides=model_overrides,
    )


def _merge_nested(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _describe_tools(tools: dict[str, Any]) -> dict[str, dict[str, Any]]:
    descriptions = {}
    for name, entry in tools.items():
        if isinstance(entry, dict) and "tool" in entry:
            value = entry["tool"]
            description = entry.get("description")
        else:
            value = entry
            description = None
        descriptions[name] = {
            "kind": "callable" if callable(value) else type(value).__name__,
            "description": description,
        }
    return descriptions


def _agent_trace_to_dict(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "agent_id": trace.agent_id,
        "parent_id": trace.parent_id,
        "depth": trace.depth,
        "task": trace.task,
        "system_prompt": trace.system_prompt,
        "steps": [
            {
                "number": step.number,
                "response": step.response,
                "model_observation": step.model_observation,
                "observation_truncated": step.observation_truncated,
                "code_executions": [
                    {
                        "code": execution.code,
                        "stdout": execution.output,
                        "error": execution.error,
                        "variables": execution.variables,
                        "duration_seconds": execution.duration_seconds,
                    }
                    for execution in step.code_executions
                ],
                "duration_seconds": step.duration_seconds,
            }
            for step in trace.steps
        ],
        "children": [_agent_trace_to_dict(child) for child in trace.children],
        "forced_final_response": trace.forced_final_response,
        "usage": trace.usage.to_dict(),
        "status": trace.status,
        "answer": trace.answer,
        "error": trace.error,
        "duration_seconds": trace.duration_seconds,
    }


class _EmptyTrace:
    steps: tuple[()] = ()


_EMPTY_TRACE = _EmptyTrace()
