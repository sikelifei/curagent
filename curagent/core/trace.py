"""Lossless decision trace storage."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from curagent.core.types import ExecutionReceipt, Observation, ToolCall


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DecisionTrace:
    agent_id: str
    parent_id: str | None
    depth: int
    attempt: int
    prompt: str
    observation: Observation
    raw_model_output: Any
    parsed_tool_call: ToolCall | None
    receipt: ExecutionReceipt | None
    error: str | None
    reward: float
    budget: Mapping[str, Any]
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observation"] = self.observation.to_dict()
        value["parsed_tool_call"] = self.parsed_tool_call.to_dict() if self.parsed_tool_call else None
        value["receipt"] = self.receipt.to_dict() if self.receipt else None
        return value


class TraceRecorder:
    """Thread-safe task-tree trace; prompt reads remain node-local."""

    def __init__(self) -> None:
        self._events: list[DecisionTrace] = []
        self._lock = threading.RLock()

    def record(self, event: DecisionTrace) -> None:
        with self._lock:
            self._events.append(event)

    def for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_dict() for event in self._events if event.agent_id == agent_id]

    def for_prompt(self, agent_id: str) -> list[dict[str, Any]]:
        """Return complete node events without recursively embedding old prompts."""
        events = self.for_agent(agent_id)
        for event in events:
            event.pop("prompt", None)
        return events

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_dict() for event in self._events]
