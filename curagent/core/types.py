"""Shared values for agents, tools, environments, and traces."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence


class AccessMode(str, Enum):
    OWNER = "owner"
    READONLY = "readonly"
    CLONE = "clone"
    DELEGATED = "delegated"


class ReceiptStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    FAILED = "failed"


class Effect(str, Enum):
    NO_CHANGE = "no_change"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


class TerminalStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    UNCERTAIN = "uncertain"
    MAX_STEPS = "max_steps"
    BUDGET_EXHAUSTED = "budget_exhausted"


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
    is_write: bool = False

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
    raw_response: Any
    tool_calls: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    protocol: Literal["native", "json"] = "native"


@dataclass(frozen=True)
class Observation:
    text: str
    version: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "version": self.version, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class ExecutionReceipt:
    call_id: str
    status: ReceiptStatus
    effect: Effect
    result: Any = None
    error: str | None = None
    version_before: int | None = None
    version_after: int | None = None
    observation: Observation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "status": self.status.value,
            "effect": self.effect.value,
            "result": self.result,
            "error": self.error,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "observation": self.observation.to_dict() if self.observation else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EnvCapabilities:
    mutable: bool
    supports_clone: bool
    supports_readonly: bool
    single_writer: bool
    supports_idempotency_key: bool = False


@dataclass(frozen=True)
class SubagentSpec:
    task: str
    context: Any
    expected_output: str | None = None
    access: AccessMode = AccessMode.READONLY

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SubagentSpec":
        allowed = {"task", "context", "expected_output", "access"}
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
        expected = value.get("expected_output")
        if expected is not None and not isinstance(expected, str):
            raise ValueError("expected_output must be a string")
        try:
            access = AccessMode(value.get("access", AccessMode.READONLY.value))
        except ValueError as exc:
            raise ValueError(f"unsupported access mode: {value.get('access')!r}") from exc
        return cls(task=task, context=context, expected_output=expected, access=access)


@dataclass(frozen=True)
class SubagentResult:
    task: str
    context: Any
    status: TerminalStatus
    result: Any = None
    error: str | None = None
    agent_id: str | None = None
    parent_id: str | None = None
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True)
class AgentLimits:
    max_steps_per_agent: int = 16
    max_retries_per_step: int = 1
    max_model_calls_total: int = 64
    max_tool_calls_total: int = 64
    max_depth: int = 3
    max_children_total: int = 8
    max_concurrency: int = 4
    max_model_service_retries: int = 1

    def __post_init__(self) -> None:
        integer_fields = asdict(self)
        for name, value in integer_fields.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_retries_per_step > 1:
            raise ValueError("max_retries_per_step must be 0 or 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")


@dataclass(frozen=True)
class ErrorFeedback:
    error_type: str
    original_error: str
    failed_tool_call: Mapping[str, Any] | None
    effect: Effect
    latest_observation: Observation
    remaining_budget: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "original_error": self.original_error,
            "failed_tool_call": dict(self.failed_tool_call) if self.failed_tool_call else None,
            "effect": self.effect.value,
            "latest_observation": self.latest_observation.to_dict(),
            "remaining_budget": dict(self.remaining_budget),
        }
