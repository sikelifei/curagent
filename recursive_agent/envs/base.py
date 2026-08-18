"""Common contract implemented by environment integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..tools import CapabilityCollection
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
    def use_recursive_codeact_harness(self) -> bool:
        """Opt into the generic recursive CodeAct scheduler incrementally."""
        return False

    def capabilities(self) -> CapabilityCollection:
        """Return validated environment capabilities.

        The default adapts the legacy :meth:`tools` mapping. Environments
        with role-dependent action spaces may override this method or
        :meth:`codeact_capabilities` while retaining one collection as the
        source for runtime binding and prompt rendering.
        """
        return CapabilityCollection.from_tools(self.tools())

    def codeact_capabilities(
        self,
        is_root: bool,
        depth: int,
    ) -> CapabilityCollection:
        """Return capabilities visible to a node at the given role/depth."""
        if not isinstance(is_root, bool):
            raise TypeError("is_root must be a bool")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        capabilities = self.capabilities()
        if is_root:
            return capabilities
        return capabilities.without(self.delegated_disabled_tools)

    def codeact_namespace(self, is_root: bool, depth: int) -> dict[str, Any]:
        """Return runtime bindings for the current node's action space."""
        return self.codeact_capabilities(is_root=is_root, depth=depth).bind()

    def codeact_descriptions(self, is_root: bool, depth: int) -> str | None:
        """Render the current node's action space from the same bindings."""
        return self.codeact_capabilities(is_root=is_root, depth=depth).format_for_prompt()

    @property
    def agent_prompt(self) -> str:
        """Return environment guidance shared by root and delegated agents."""
        return ""

    @property
    def root_prompt(self) -> str | None:
        """Return the complete prompt for the root agent, when customized."""
        return None

    @property
    def child_prompt(self) -> str | None:
        """Return the complete prompt for delegated agents, when customized."""
        return None

    @property
    def system_prompt(self) -> str | None:
        """Return an optional environment-specific base system prompt."""
        return None

    @property
    def environment_system_prompt(self) -> str | None:
        """Explicit alias for the environment-owned system prompt hook."""
        return self.system_prompt

    def observe(self) -> Any:
        """Return the current observation, adapting legacy context by default.

        Implementations may override this with either a synchronous or an
        asynchronous method; the generic harness is responsible for awaiting
        awaitable observations when it consumes this hook.
        """
        return self.context

    def finalize(self, result: Any = None) -> Any:
        """Finalize a root result. The compatibility default is a no-op."""
        return result

    def finalize_root(self, result: Any = None) -> Any:
        """Compatibility alias for the optional root finalization hook."""
        return self.finalize(result)

    @property
    def completion_prompt(self) -> str | None:
        """Return optional completion instructions for the environment run."""
        return None

    @property
    def delegated_completion_prompt(self) -> str | None:
        """Return optional completion instructions for assigned work."""
        return None

    @property
    def forced_final_prompt(self) -> str | None:
        """Return an optional environment-specific forced-final prompt."""
        return None

    @property
    def delegated_forced_final_prompt(self) -> str | None:
        """Return an optional forced-final prompt for delegated agents."""
        return None

    @property
    def delegated_task_prompt(self) -> str | None:
        """Return optional environment guidance appended only to delegated tasks."""
        return None

    @property
    def delegated_prompt_addendum(self) -> str | None:
        """Return the prompt addendum visible only to delegated agents."""
        return None

    @property
    def delegated_disabled_tools(self) -> frozenset[str]:
        """Return custom tools hidden from delegated agents."""
        return frozenset()

    @property
    def max_repl_blocks_per_step(self) -> int | None:
        """Return an optional per-response REPL execution limit."""
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
