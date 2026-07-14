"""Typed errors raised by the recursive harness."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for errors with harness-level semantics."""


class StrictToolCallError(HarnessError):
    """A model response is not exactly one valid tool call."""


class ToolSchemaError(HarnessError):
    """A tool call does not conform to its advertised JSON schema."""


class BudgetExceeded(HarnessError):
    """A task-tree shared budget has been exhausted."""

    def __init__(self, resource: str) -> None:
        self.resource = resource
        super().__init__(f"shared budget exhausted: {resource}")


class SchedulerError(HarnessError):
    """A child request cannot be started as specified."""


class ModelServiceError(HarnessError):
    """An upstream model request failed with explicit retry semantics."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        self.retryable = retryable
        super().__init__(message)
