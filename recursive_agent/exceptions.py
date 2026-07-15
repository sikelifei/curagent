"""Public exceptions raised by the recursive agent runtime."""

from __future__ import annotations


class RecursiveAgentError(Exception):
    """Base class for runtime errors."""


class ConfigurationError(RecursiveAgentError, ValueError):
    """Raised when an agent or tool configuration is invalid."""


class ModelCallError(RecursiveAgentError):
    """Raised when a model completion fails."""

    def __init__(self, message: str, *, last_response: str | None = None) -> None:
        self.last_response = last_response
        super().__init__(message)


class TimeoutExceededError(RecursiveAgentError):
    """Raised when the shared run deadline has elapsed."""

    def __init__(self, elapsed: float, timeout: float) -> None:
        self.elapsed = elapsed
        self.timeout = timeout
        super().__init__(f"Run timed out after {elapsed:.2f}s (limit: {timeout:.2f}s)")


class CancellationError(RecursiveAgentError):
    """Raised after a run is cancelled cooperatively."""


