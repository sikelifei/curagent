"""Configuration validation and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError

_SAMPLING_KEYS = {
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "extra_body",
}

SUPPORTED_BACKENDS = frozenset(
    {
        "openai",
        "vllm",
        "openrouter",
        "vercel",
        "portkey",
        "azure_openai",
        "anthropic",
        "gemini",
    }
)


@dataclass(frozen=True)
class AgentLimits:
    """Public limits shared by the recursive harness contract."""

    max_total_steps: int = 20
    max_depth: int = 4

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_total_steps, int)
            or isinstance(self.max_total_steps, bool)
            or self.max_total_steps <= 0
        ):
            raise ConfigurationError("max_total_steps must be a positive integer")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth < 0
        ):
            raise ConfigurationError("max_depth must be a non-negative integer")


@dataclass(frozen=True)
class AgentConfig:
    backend: str = "openai"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 20
    max_depth: int = 4
    max_concurrent_subagents: int = 4
    max_subagents_per_agent: int | None = None
    max_run_seconds: float | None = None
    max_observation_chars: int | None = 8000

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ConfigurationError(
                f"Unsupported backend {self.backend!r}; expected one of "
                f"{sorted(SUPPORTED_BACKENDS)}"
            )
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ConfigurationError("max_steps must be a positive integer")
        if not isinstance(self.max_depth, int) or self.max_depth < 0:
            raise ConfigurationError("max_depth must be a non-negative integer")
        if (
            not isinstance(self.max_concurrent_subagents, int)
            or self.max_concurrent_subagents <= 0
        ):
            raise ConfigurationError("max_concurrent_subagents must be a positive integer")
        if self.max_subagents_per_agent is not None and (
            not isinstance(self.max_subagents_per_agent, int)
            or self.max_subagents_per_agent <= 0
        ):
            raise ConfigurationError(
                "max_subagents_per_agent must be a positive integer or None"
            )
        if self.max_run_seconds is not None and self.max_run_seconds <= 0:
            raise ConfigurationError("max_run_seconds must be positive when provided")
        if self.max_observation_chars is not None and (
            not isinstance(self.max_observation_chars, int)
            or self.max_observation_chars <= 0
        ):
            raise ConfigurationError(
                "max_observation_chars must be a positive integer or None"
            )

    @property
    def limits(self) -> AgentLimits:
        """Return the forward-looking limits view of legacy config fields."""
        return AgentLimits(max_total_steps=self.max_steps, max_depth=self.max_depth)


def load_model_config(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Load the single-model YAML format used by the example configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("model"), dict):
        raise ConfigurationError("Config must contain a top-level 'model' mapping")

    model_config = document["model"]
    if model_config.get("type", "api") != "api" or not isinstance(
        model_config.get("api"), dict
    ):
        raise ConfigurationError("model.type must be 'api' and model.api must be a mapping")

    api = dict(model_config["api"])
    model_name = api.pop("model", api.pop("model_name", None))
    if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigurationError("model.api.model must be a non-empty string")

    sampling_args = dict(api.pop("sampling_args", {}) or {})
    for key in list(api):
        if key in _SAMPLING_KEYS:
            sampling_args[key] = api.pop(key)
    api["model_name"] = model_name
    if sampling_args:
        api["sampling_args"] = sampling_args
    return "openai", api
