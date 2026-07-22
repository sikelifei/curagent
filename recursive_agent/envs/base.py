"""Common contract implemented by environment integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import EnvironmentStatus


class AgentEnvironment(ABC):
    """One initialized environment episode ready for a RecursiveAgent run."""

    name: str

    @property
    @abstractmethod
    def task(self) -> str:
        """Return the dataset task prompt for this episode."""

    @property
    @abstractmethod
    def context(self) -> Any:
        """Return the private initial REPL context snapshot."""

    @abstractmethod
    def tools(self) -> dict[str, Any]:
        """Return custom tools to register on the agent."""

    @property
    def agent_prompt(self) -> str:
        """Return environment guidance for the root agent."""
        return ""

    @property
    def delegated_agent_prompt(self) -> str:
        """Return environment guidance for delegated agents."""
        return self.agent_prompt

    @property
    def system_prompt(self) -> str | None:
        """Return the optional root system prompt."""
        return None

    @property
    def delegated_system_prompt(self) -> str | None:
        """Return the optional delegated-agent system prompt."""
        return self.system_prompt

    @property
    def forced_final_prompt(self) -> str | None:
        """Return an optional environment-specific forced-final prompt."""
        return None

    @property
    def delegated_forced_final_prompt(self) -> str | None:
        """Return an optional forced-final prompt for delegated agents."""
        return None

    @property
    def disabled_repl_builtins(self) -> frozenset[str]:
        """Return REPL built-ins unavailable to every agent in this episode."""
        return frozenset()

    @abstractmethod
    def status(self) -> EnvironmentStatus:
        """Return the current environment termination state."""

    @abstractmethod
    def report(self) -> dict[str, Any]:
        """Return evaluation metrics and trajectory data."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Implementations must be idempotent."""

    def __enter__(self) -> "AgentEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class EnvironmentDependencyError(RuntimeError):
    """Raised when an external environment or its dependencies are unavailable."""
