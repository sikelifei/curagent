"""Result, trace, environment, and usage types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentStatus = Literal["completed", "forced_final", "environment_done"]


@dataclass
class ModelUsageSummary:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float | None = None

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        return sum(item.total_calls for item in self.model_usage_summaries.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(item.total_input_tokens for item in self.model_usage_summaries.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(item.total_output_tokens for item in self.model_usage_summaries.values())

    @property
    def total_cost(self) -> float | None:
        costs = [
            item.total_cost
            for item in self.model_usage_summaries.values()
            if item.total_cost is not None
        ]
        return sum(costs) if costs else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_usage_summaries": {
                model: usage.to_dict()
                for model, usage in self.model_usage_summaries.items()
            },
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result


@dataclass(frozen=True)
class ModelCallUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: ModelCallUsage


@dataclass
class EnvironmentStatus:
    done: bool
    final_answer: str | None = None
    reason: str | None = None


@dataclass
class CodeExecutionTrace:
    code: str
    output: str
    error: str | None = None
    duration_seconds: float = 0.0
    variables: list[str] = field(default_factory=list)


@dataclass
class AgentStep:
    number: int
    response: str
    code_executions: list[CodeExecutionTrace] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class AgentTrace:
    agent_id: str
    parent_id: str | None
    depth: int
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    children: list["AgentTrace"] = field(default_factory=list)
    forced_final_response: str | None = None
    usage: UsageSummary = field(default_factory=UsageSummary)
    status: AgentStatus | None = None
    answer: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class AgentResult:
    answer: str
    status: AgentStatus
    steps: int
    usage: UsageSummary
    trace: AgentTrace | None = None
