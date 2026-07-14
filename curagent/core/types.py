"""Small JSON-oriented value types for the recursive agent harness."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def require_jsonable(value: Any, *, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable: {exc}") from exc


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = field(default_factory=new_call_id)
    provider_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "call_id": self.call_id,
            "provider_call_id": self.provider_call_id,
        }


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: Mapping[str, Any]
    is_environment_tool: bool = False

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ModelResponse:
    """Unmodified provider output in one of the supported response protocols."""

    raw_response: Any
    tool_calls: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    protocol: Literal["native", "json"] = "native"


@dataclass(frozen=True)
class SubagentSpec:
    task: str
    context: Any
    expected_output: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SubagentSpec":
        if not isinstance(value, Mapping):
            raise ValueError("subagent spec must be an object")
        allowed = {"task", "context", "expected_output"}
        extras = set(value) - allowed
        if extras:
            raise ValueError(f"unknown subagent spec fields: {sorted(extras)}")
        task = value.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("subagent task must be a non-empty string")
        if "context" not in value:
            raise ValueError("subagent context is required")
        context = value["context"]
        require_jsonable(context, label="subagent context")
        expected_output = value.get("expected_output")
        if expected_output is not None and not isinstance(expected_output, str):
            raise ValueError("expected_output must be a string")
        return cls(task=task, context=context, expected_output=expected_output)


@dataclass(frozen=True)
class SubagentResult:
    """The only child data exposed to its parent prompt."""

    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"result": self.result, "error": self.error}


@dataclass(frozen=True)
class AgentLimits:
    """The two limits shared by every node in one task tree."""

    max_total_steps: int = 24
    max_depth: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("max_total_steps", self.max_total_steps),
            ("max_depth", self.max_depth),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
