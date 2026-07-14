"""Minimal prompt composition for every recursive node."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from curagent.core.types import ToolSchema


BASE_SYSTEM_PROMPT = (
    "You are a recursive agent node.\n"
    "Use one available tool, spawn child agents when useful, or call finish.\n"
    "Tool results are returned in trajectory; decide the next step yourself."
)


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def compose_prompt(
    *,
    task: str,
    context: Any,
    trajectory: Sequence[Mapping[str, Any]],
    observation: Any,
    tools: Sequence[ToolSchema],
    remaining_steps: int,
    expected_output: str | None = None,
) -> str:
    """Return the exact node payload; expected_output is intentionally not extra state."""
    del expected_output
    value = {
        "task": task,
        "context": context,
        "trajectory": list(trajectory),
        "observation": _json_value(observation),
        "tools": [tool.to_model_dict() for tool in tools],
        "remaining_steps": remaining_steps,
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
