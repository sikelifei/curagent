"""Custom tool parsing, validation, and prompt formatting."""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from typing import Any

from .exceptions import ConfigurationError

RESERVED_NAMES = frozenset(
    {
        "__builtins__",
        "__name__",
        "print",
        "context",
        "answer",
        "SHOW_VARS",
        "spawn_subagent",
        "spawn_subagents",
    }
)


@dataclass(frozen=True)
class ToolInfo:
    name: str
    value: Any
    description: str | None = None


def parse_tools(tools: dict[str, Any] | None) -> dict[str, ToolInfo]:
    parsed: dict[str, ToolInfo] = {}
    for name, entry in (tools or {}).items():
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise ConfigurationError(f"Tool name must be a valid Python identifier: {name!r}")
        if name in RESERVED_NAMES:
            raise ConfigurationError(f"Tool name is reserved: {name!r}")

        if isinstance(entry, dict) and "tool" in entry:
            value = entry["tool"]
            description = entry.get("description")
            if description is not None and not isinstance(description, str):
                raise ConfigurationError(f"Description for tool {name!r} must be a string")
        else:
            value = entry
            description = None
        parsed[name] = ToolInfo(name, value, description)
    return parsed


def format_tools_for_prompt(tools: dict[str, ToolInfo]) -> str | None:
    if not tools:
        return None
    lines = []
    for info in tools.values():
        kind = "function" if callable(info.value) else f"{type(info.value).__name__} value"
        description = info.description or f"A custom {kind}."
        lines.append(f"- `{info.name}`: {description}")
    return "\n".join(lines)


def tool_values(tools: dict[str, ToolInfo]) -> dict[str, Any]:
    return {name: info.value for name, info in tools.items()}

