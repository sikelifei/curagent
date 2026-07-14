"""External traces and compact node-local trajectory feedback."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from curagent.core.types import ToolCall


TRUNCATION_MARKER = " [truncated after approximately 1000 tokens]"
_APPROX_TOKEN_LIMIT = 1000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_for_prompt(value: Any) -> Any:
    """Bound feedback to about 1000 tokens while retaining full external traces."""
    if isinstance(value, str):
        if _approx_tokens(value) <= _APPROX_TOKEN_LIMIT:
            return value
        return _truncate(value)
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
    except Exception:
        text = str(value)
    if _approx_tokens(text) <= _APPROX_TOKEN_LIMIT:
        return value
    return _truncate(text)


def _approx_tokens(text: str) -> int:
    """Estimate tokens without a provider tokenizer; non-ASCII chars count individually."""
    total = 0
    index = 0
    while index < len(text):
        char = text[index]
        if ord(char) >= 128:
            total += 1
            index += 1
        elif char.isalnum():
            start = index
            while index < len(text) and text[index].isascii() and text[index].isalnum():
                index += 1
            total += max(1, math.ceil((index - start) / 4))
        else:
            total += 0 if char.isspace() else 1
            index += 1
    return total


def _truncate(text: str) -> str:
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _approx_tokens(text[:middle]) <= _APPROX_TOKEN_LIMIT:
            low = middle
        else:
            high = middle - 1
    return text[:low] + TRUNCATION_MARKER


def _observation_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class DecisionTrace:
    agent_id: str
    parent_id: str | None
    depth: int
    step: int
    attempt: int
    prompt: str
    observation: Any
    raw_model_output: Any
    parsed_tool_call: ToolCall | None
    execution_result: Any
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observation"] = _observation_value(self.observation)
        value["parsed_tool_call"] = (
            self.parsed_tool_call.to_dict() if self.parsed_tool_call else None
        )
        return value

    def for_prompt(self) -> dict[str, Any]:
        model_output: Any = self.raw_model_output
        if self.parsed_tool_call is not None:
            model_output = {
                "name": self.parsed_tool_call.name,
                "arguments": dict(self.parsed_tool_call.arguments),
            }
        return {
            "model_output": result_for_prompt(model_output),
            "execution_result": result_for_prompt(self.execution_result),
        }


@dataclass(frozen=True)
class RuntimeTrace:
    agent_id: str
    parent_id: str | None
    depth: int
    event: str
    error: str
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceRecorder:
    """Thread-safe task-tree trace; prompts include only the current node's decisions."""

    def __init__(self) -> None:
        self._events: list[DecisionTrace | RuntimeTrace] = []
        self._lock = threading.RLock()

    def record(self, event: DecisionTrace) -> None:
        with self._lock:
            self._events.append(event)

    def record_runtime(
        self, *, agent_id: str, parent_id: str | None, depth: int, event: str, error: str
    ) -> None:
        with self._lock:
            self._events.append(RuntimeTrace(agent_id, parent_id, depth, event, error))

    def for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_dict() for event in self._events if event.agent_id == agent_id]

    def for_prompt(self, agent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                event.for_prompt()
                for event in self._events
                if isinstance(event, DecisionTrace) and event.agent_id == agent_id
            ]

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_dict() for event in self._events]
