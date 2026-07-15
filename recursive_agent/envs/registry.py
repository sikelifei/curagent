"""Small registry for environment plugin factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import AgentEnvironment

EnvironmentFactory = Callable[..., AgentEnvironment]
_ENVIRONMENTS: dict[str, EnvironmentFactory] = {}


def register_environment(
    name: str,
    factory: EnvironmentFactory | None = None,
) -> EnvironmentFactory | Callable[[EnvironmentFactory], EnvironmentFactory]:
    """Register an environment factory, directly or as a decorator."""
    normalized = _normalize_name(name)

    def register(candidate: EnvironmentFactory) -> EnvironmentFactory:
        existing = _ENVIRONMENTS.get(normalized)
        if existing is not None and existing is not candidate:
            raise ValueError(f"Environment {normalized!r} is already registered")
        _ENVIRONMENTS[normalized] = candidate
        return candidate

    return register(factory) if factory is not None else register


def create_environment(name: str, **kwargs: Any) -> AgentEnvironment:
    normalized = _normalize_name(name)
    try:
        factory = _ENVIRONMENTS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown environment {name!r}; available: {available_environments()}"
        ) from exc
    return factory(**kwargs)


def available_environments() -> tuple[str, ...]:
    return tuple(sorted(_ENVIRONMENTS))


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Environment name must be a non-empty string")
    return name.strip().lower().replace("-", "_")

