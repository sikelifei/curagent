"""Base prompt and task-module composition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from curagent.core.types import Observation, ToolSchema


BASE_PROMPT = """You are one node in a recursive agent task tree.
Choose either one exact environment action, one recursive control action, or finish.
You may delegate with spawn_agent or spawn_agents. A child receives only the explicit task and JSON context you provide. You receive only its final SubagentResult, never its private prompt, observations, or trajectory.
Return exactly one tool call. Do not combine calls. Do not describe a call in natural language.
Native parallel tool calls are invalid. To launch multiple children, make one spawn_agents call containing all specs; never emit multiple spawn_agent calls in one response.
The harness validates and executes the call exactly as supplied. It never repairs parameters, chooses a similar action, rewrites code, ranks candidates, or infers an answer for you.
Rejected calls and stale observations are reported as facts. A retry must be a new complete decision; the harness never replays the old call.
All descendants share model-call, tool-call, child-count, depth, and concurrency budgets. Stop or finish when the remaining budget cannot support more work.
Terminal statuses are ok, error, uncertain, max_steps, and budget_exhausted."""


@dataclass(frozen=True)
class TaskModule:
    instruction: str
    observation_spec: str
    environment_rules: Sequence[str]
    finish_condition: str
    environment_tools: Sequence[ToolSchema] = ()

    def to_dict(self, *, available_tool_names: set[str] | None = None) -> dict[str, Any]:
        environment_tools = self.environment_tools
        if available_tool_names is not None:
            environment_tools = [
                tool for tool in environment_tools if tool.name in available_tool_names
            ]
        return {
            "instruction": self.instruction,
            "observation_spec": self.observation_spec,
            "environment_tools": [tool.to_model_dict() for tool in environment_tools],
            "environment_rules": list(self.environment_rules),
            "finish_condition": self.finish_condition,
        }


def compose_prompt(
    *,
    task: str,
    context: Any,
    expected_output: str | None,
    task_module: TaskModule,
    trajectory: Sequence[Mapping[str, Any]],
    observation: Observation,
    tools: Sequence[ToolSchema],
    remaining_budget: Mapping[str, int],
    error_feedback: Mapping[str, Any] | None = None,
) -> str:
    sections = {
        "base_prompt": BASE_PROMPT,
        "task_module": task_module.to_dict(
            available_tool_names={tool.name for tool in tools}
        ),
        "node_input": {"task": task, "context": context, "expected_output": expected_output},
        "trajectory": list(trajectory),
        "latest_observation": observation.to_dict(),
        "available_tools": [tool.to_model_dict() for tool in tools],
        "remaining_budget": dict(remaining_budget),
        "error_feedback": error_feedback,
    }
    return json.dumps(sections, ensure_ascii=False, sort_keys=True, default=str)
