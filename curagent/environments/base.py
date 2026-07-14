"""Common environment contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from curagent.core.types import (
    AccessMode,
    EnvCapabilities,
    ExecutionReceipt,
    Observation,
    ToolCall,
    ToolSchema,
)


class Environment(ABC):
    @abstractmethod
    async def reset(self, instance: Any) -> Observation:
        raise NotImplementedError

    @abstractmethod
    async def observe(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def tools(self, access: AccessMode) -> Sequence[ToolSchema]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, tool_call: ToolCall, expected_version: int) -> ExecutionReceipt:
        raise NotImplementedError

    @abstractmethod
    async def reconcile(self, call_id: str) -> ExecutionReceipt | None:
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> EnvCapabilities:
        raise NotImplementedError

    def clone(self) -> "Environment | None":
        return None

    async def close(self) -> None:
        return None
