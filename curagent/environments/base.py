"""Minimal optional shared-environment contract used by the generic harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from curagent.core.types import ToolCall, ToolSchema


class Environment(ABC):
    """A stateful resource shared by root, child, and grandchild nodes."""

    @abstractmethod
    async def observe(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def tools(self) -> Sequence[ToolSchema]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    async def close(self) -> None:
        return None
