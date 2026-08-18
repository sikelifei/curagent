"""Thread-safe global model-generation budgets."""

from __future__ import annotations

import threading
from typing import Final

from .exceptions import ConfigurationError


class SharedBudget:
    """Reserve and commit one global successful-model-generation budget.

    Reservations count against available capacity immediately, which prevents
    concurrent callers from oversubscribing the budget. A caller must
    explicitly commit a reservation after a successful model generation or
    release it when the generation fails.
    """

    def __init__(self, max_total_steps: int) -> None:
        if (
            not isinstance(max_total_steps, int)
            or isinstance(max_total_steps, bool)
            or max_total_steps <= 0
        ):
            raise ConfigurationError("max_total_steps must be a positive integer")
        self._max_total_steps = max_total_steps
        self._consumed_steps = 0
        self._reserved_steps = 0
        self._lock = threading.Lock()

    @property
    def max_total_steps(self) -> int:
        """The maximum number of successful model generations."""
        return self._max_total_steps

    @property
    def consumed_steps(self) -> int:
        """The number of reservations committed successfully."""
        with self._lock:
            return self._consumed_steps

    @property
    def used_steps(self) -> int:
        """Compatibility alias for :attr:`consumed_steps`."""
        return self.consumed_steps

    @property
    def used(self) -> int:
        """Short alias for the committed step count."""
        return self.consumed_steps

    @property
    def reserved_steps(self) -> int:
        """The number of model generations currently in flight."""
        with self._lock:
            return self._reserved_steps

    @property
    def available_steps(self) -> int:
        """The number of new reservations that can be made immediately."""
        with self._lock:
            return self._max_total_steps - self._consumed_steps - self._reserved_steps

    @property
    def remaining_steps(self) -> int:
        """The currently unallocated portion of the global budget."""
        return self.available_steps

    @property
    def remaining(self) -> int:
        """Short alias for :attr:`remaining_steps`."""
        return self.remaining_steps

    def reserve(self) -> "BudgetReservation | None":
        """Atomically reserve capacity for one model generation.

        Returns ``None`` when all capacity is committed or already reserved.
        The returned reservation must be committed or released exactly once.
        """
        with self._lock:
            if self._consumed_steps + self._reserved_steps >= self._max_total_steps:
                return None
            self._reserved_steps += 1
            return BudgetReservation(self)

    def try_reserve(self) -> "BudgetReservation | None":
        """Compatibility alias for :meth:`reserve`."""
        return self.reserve()

    def acquire(self) -> "BudgetReservation | None":
        """Compatibility alias for :meth:`reserve`."""
        return self.reserve()

    def _commit(self, reservation: "BudgetReservation") -> None:
        with self._lock:
            if reservation._budget is not self or reservation._state != _ACTIVE:
                raise RuntimeError("Budget reservation is no longer active")
            self._reserved_steps -= 1
            self._consumed_steps += 1
            reservation._state = _COMMITTED

    def _release(self, reservation: "BudgetReservation") -> None:
        with self._lock:
            if reservation._budget is not self or reservation._state != _ACTIVE:
                raise RuntimeError("Budget reservation is no longer active")
            self._reserved_steps -= 1
            reservation._state = _RELEASED

    def commit(self, reservation: "BudgetReservation") -> None:
        """Commit an active reservation through the budget object."""
        if reservation._budget is not self:
            raise RuntimeError("Budget reservation belongs to a different budget")
        reservation.commit()

    def release(self, reservation: "BudgetReservation") -> None:
        """Release an active reservation through the budget object."""
        if reservation._budget is not self:
            raise RuntimeError("Budget reservation belongs to a different budget")
        reservation.release()


_ACTIVE: Final = "active"
_COMMITTED: Final = "committed"
_RELEASED: Final = "released"


class BudgetReservation:
    """A single in-flight generation reservation owned by a SharedBudget."""

    __slots__ = ("_budget", "_state")

    def __init__(self, budget: SharedBudget) -> None:
        self._budget = budget
        self._state = _ACTIVE

    @property
    def active(self) -> bool:
        return self._state == _ACTIVE

    @property
    def committed(self) -> bool:
        return self._state == _COMMITTED

    @property
    def released(self) -> bool:
        return self._state == _RELEASED

    def commit(self) -> None:
        """Mark the model generation as successful and consume one step."""
        self._budget._commit(self)

    def release(self) -> None:
        """Return capacity after a failed or cancelled model generation."""
        self._budget._release(self)

    def __enter__(self) -> "BudgetReservation":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.active:
            self.release()


__all__ = ["BudgetReservation", "SharedBudget"]
