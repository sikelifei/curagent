"""Atomic task-tree shared budget accounting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from curagent.core.errors import BudgetExceeded
from curagent.core.types import AgentLimits


@dataclass(frozen=True)
class BudgetSnapshot:
    model_calls_used: int
    tool_calls_used: int
    children_used: int
    model_calls_remaining: int
    tool_calls_remaining: int
    children_remaining: int

    def remaining_dict(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls_remaining,
            "tool_calls": self.tool_calls_remaining,
            "children": self.children_remaining,
        }


class SharedBudget:
    """One atomic budget shared by the root and every descendant."""

    def __init__(self, limits: AgentLimits) -> None:
        self.limits = limits
        self._model_calls = 0
        self._tool_calls = 0
        self._children = 0
        self._lock = asyncio.Lock()

    async def consume_model_call(self) -> None:
        async with self._lock:
            if self._model_calls >= self.limits.max_model_calls_total:
                raise BudgetExceeded("model_calls")
            self._model_calls += 1

    async def consume_tool_call(self) -> None:
        async with self._lock:
            if self._tool_calls >= self.limits.max_tool_calls_total:
                raise BudgetExceeded("tool_calls")
            self._tool_calls += 1

    async def reserve_children(self, count: int) -> None:
        if count < 0:
            raise ValueError("child reservation must be non-negative")
        async with self._lock:
            if self._children + count > self.limits.max_children_total:
                raise BudgetExceeded("children")
            self._children += count

    async def snapshot(self) -> BudgetSnapshot:
        async with self._lock:
            return BudgetSnapshot(
                model_calls_used=self._model_calls,
                tool_calls_used=self._tool_calls,
                children_used=self._children,
                model_calls_remaining=max(0, self.limits.max_model_calls_total - self._model_calls),
                tool_calls_remaining=max(0, self.limits.max_tool_calls_total - self._tool_calls),
                children_remaining=max(0, self.limits.max_children_total - self._children),
            )
