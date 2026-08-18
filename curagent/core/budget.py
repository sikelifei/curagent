"""Atomic shared model-output accounting for one complete task tree."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from curagent.core.errors import BudgetExceeded
from curagent.core.types import AgentLimits


@dataclass(frozen=True)
class BudgetSnapshot:
    total_steps_used: int
    remaining_steps: int


class SharedBudget:
    """Reserve one shared step before a model call and release it if no output arrives."""

    def __init__(self, limits: AgentLimits) -> None:
        self.limits = limits
        self._total_steps = 0
        self._next_reservation = 0
        self._pending: set[int] = set()
        self._lock = asyncio.Lock()

    async def reserve_step(self) -> int:
        async with self._lock:
            if self._total_steps >= self.limits.max_total_steps:
                raise BudgetExceeded("max_total_steps")
            self._total_steps += 1
            self._next_reservation += 1
            reservation = self._next_reservation
            self._pending.add(reservation)
            return reservation

    async def commit_step(self, reservation: int) -> None:
        async with self._lock:
            if reservation not in self._pending:
                raise RuntimeError(f"unknown step reservation: {reservation}")
            self._pending.remove(reservation)

    async def release_step(self, reservation: int) -> None:
        async with self._lock:
            if reservation not in self._pending:
                raise RuntimeError(f"unknown step reservation: {reservation}")
            self._pending.remove(reservation)
            self._total_steps -= 1

    async def snapshot(self) -> BudgetSnapshot:
        async with self._lock:
            return BudgetSnapshot(
                total_steps_used=self._total_steps,
                remaining_steps=max(0, self.limits.max_total_steps - self._total_steps),
            )
