"""Typed errors raised by the recursive harness."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for errors with harness-level semantics."""


class StrictToolCallError(HarnessError):
    """A model response is not exactly one valid tool call."""


class ToolSchemaError(HarnessError):
    """A tool call does not conform to its advertised JSON schema."""


class BudgetExceeded(HarnessError):
    """The task tree has no shared step slots left."""

    def __init__(self, resource: str) -> None:
        self.resource = resource
        super().__init__(f"shared budget exhausted: {resource}")


class SchedulerError(HarnessError):
    """A child request cannot be started as specified."""


class ModelServiceError(HarnessError):
    """An upstream model request failed before producing a model output."""

    def __init__(
        self,
        message: str,
        *,
        has_output: bool = False,
        raw_response: object | None = None,
        protocol: str = "json",
        tool_calls: object = (),
    ) -> None:
        self.has_output = has_output
        self.raw_response = raw_response
        self.protocol = protocol
        self.tool_calls = tool_calls
        super().__init__(message)
