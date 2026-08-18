"""Custom tool parsing, validation, and prompt formatting."""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
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
        "finish",
        "return_to_parent",
    }
)

FRAMEWORK_NAMES = frozenset(
    {
        "spawn_subagent",
        "spawn_subagents",
        "finish",
        "return_to_parent",
    }
)

_LEGACY_ENVIRONMENT_NAMES = frozenset({"finish", "return_to_parent"})


@dataclass(frozen=True)
class ToolInfo:
    name: str
    value: Any
    description: str | None = None

    @property
    def runtime_value(self) -> Any:
        """Return the value bound into a Python namespace."""
        return self.value

    @property
    def prompt_description(self) -> str | None:
        """Return the action-space description, when supplied."""
        return self.description


# Capabilities and legacy tools have the same representation. The alias keeps
# old imports working while naming the role this type now serves.
Capability = ToolInfo


def _validate_name(
    name: str,
    *,
    trusted_framework: bool = False,
    legacy_environment: bool = False,
) -> None:
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ConfigurationError(f"Tool name must be a valid Python identifier: {name!r}")
    framework_allowed = trusted_framework and name in FRAMEWORK_NAMES
    legacy_environment_allowed = legacy_environment and name in _LEGACY_ENVIRONMENT_NAMES
    if name in RESERVED_NAMES and not (framework_allowed or legacy_environment_allowed):
        raise ConfigurationError(f"Tool name is reserved: {name!r}")


def _coerce_info(
    name: str,
    entry: Any,
    *,
    trusted_framework: bool = False,
    legacy_environment: bool = False,
) -> ToolInfo:
    _validate_name(
        name,
        trusted_framework=trusted_framework,
        legacy_environment=legacy_environment,
    )
    if isinstance(entry, ToolInfo):
        if entry.name != name:
            raise ConfigurationError(
                f"Capability mapping key {name!r} does not match its name {entry.name!r}"
            )
        return entry

    if isinstance(entry, Mapping) and "tool" in entry:
        value = entry["tool"]
        description = entry.get("description")
        if description is None:
            description = entry.get("prompt_description")
        if description is not None and not isinstance(description, str):
            raise ConfigurationError(f"Description for tool {name!r} must be a string")
    else:
        value = entry
        description = None
    return ToolInfo(name, value, description)


class CapabilityCollection(Mapping[str, ToolInfo]):
    """Validated capabilities used for both runtime binding and prompts.

    The collection stores each capability's callable/value and its optional
    action-space description together. Consumers should use :meth:`bind` and
    :meth:`format_for_prompt` rather than maintaining parallel mappings.
    """

    def __init__(
        self,
        capabilities: Mapping[str, Any] | Iterable[ToolInfo] | None = None,
    ) -> None:
        if isinstance(capabilities, CapabilityCollection):
            self._items = capabilities._items
            self._trusted_framework = capabilities._trusted_framework
            return
        if capabilities is None:
            entries: Iterable[tuple[str, Any]] = ()
        elif isinstance(capabilities, Mapping):
            entries = capabilities.items()
        else:
            entries = ((info.name, info) for info in capabilities)

        parsed: dict[str, ToolInfo] = {}
        for name, entry in entries:
            if name in parsed:
                raise ConfigurationError(f"Duplicate capability name: {name!r}")
            parsed[name] = _coerce_info(name, entry)
        self._items = MappingProxyType(parsed)
        self._trusted_framework = False

    @classmethod
    def from_tools(cls, tools: Mapping[str, Any] | None = None) -> "CapabilityCollection":
        """Build one validated collection from a legacy tool mapping."""
        return cls(tools)

    @classmethod
    def _from_validated(
        cls,
        capabilities: Mapping[str, ToolInfo],
        *,
        trusted_framework: bool = False,
    ) -> "CapabilityCollection":
        collection = cls.__new__(cls)
        collection._items = MappingProxyType(dict(capabilities))
        collection._trusted_framework = trusted_framework
        return collection

    def __getitem__(self, name: str) -> ToolInfo:
        return self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def bind(self) -> dict[str, Any]:
        """Return the runtime namespace values for this collection."""
        return {name: info.runtime_value for name, info in self._items.items()}

    runtime_values = bind

    def format_for_prompt(self) -> str | None:
        """Render this collection as a prompt action-space description."""
        if not self:
            return None
        lines = []
        for info in self._items.values():
            kind = (
                "function"
                if callable(info.runtime_value)
                else f"{type(info.runtime_value).__name__} value"
            )
            description = info.prompt_description or f"A custom {kind}."
            lines.append(f"- `{info.name}`: {description}")
        return "\n".join(lines)

    render = format_for_prompt

    def without(self, names: Iterable[str]) -> "CapabilityCollection":
        """Return a role-filtered view while preserving descriptions."""
        disabled = set(names)
        return CapabilityCollection._from_validated(
            {
                name: info
                for name, info in self._items.items()
                if name not in disabled
            }
        )

    def merge_framework(
        self,
        framework: Mapping[str, Any] | Iterable[ToolInfo],
    ) -> "CapabilityCollection":
        """Add trusted harness capabilities after validating collisions.

        Environment and user capabilities always go through the ordinary
        reserved-name checks. This narrowly scoped method is the only place
        where the framework's own reserved callable names may be composed
        into an action space.
        """
        if isinstance(framework, CapabilityCollection):
            entries = framework.items()
        elif isinstance(framework, Mapping):
            entries = framework.items()
        else:
            entries = ((info.name, info) for info in framework)

        merged = dict(self._items)
        for name, entry in entries:
            if name in merged:
                raise ConfigurationError(
                    f"Environment capability collides with framework capability: {name!r}"
                )
            if name not in FRAMEWORK_NAMES:
                raise ConfigurationError(
                    f"Trusted framework capability has unsupported name: {name!r}"
                )
            merged[name] = _coerce_info(
                name,
                entry,
                trusted_framework=True,
            )
        return CapabilityCollection._from_validated(merged, trusted_framework=True)

    # Factory-style aliases make the composition boundary discoverable while
    # retaining one implementation.
    with_framework = merge_framework
    add_framework = merge_framework


Capabilities = CapabilityCollection


def parse_tools(tools: dict[str, Any] | None) -> dict[str, ToolInfo]:
    parsed: dict[str, ToolInfo] = {}
    for name, entry in (tools or {}).items():
        parsed[name] = _coerce_info(
            name,
            entry,
            legacy_environment=isinstance(name, str)
            and name in _LEGACY_ENVIRONMENT_NAMES,
        )
    return parsed


def parse_capabilities(
    capabilities: Mapping[str, Any] | Iterable[ToolInfo] | None,
) -> CapabilityCollection:
    """Parse a capability mapping into the shared validated collection."""
    return CapabilityCollection(capabilities)


def format_tools_for_prompt(
    tools: Mapping[str, ToolInfo] | CapabilityCollection,
) -> str | None:
    if isinstance(tools, CapabilityCollection):
        return tools.format_for_prompt()
    if not tools:
        return None
    lines = []
    for info in tools.values():
        kind = (
            "function"
            if callable(info.runtime_value)
            else f"{type(info.runtime_value).__name__} value"
        )
        description = info.prompt_description or f"A custom {kind}."
        lines.append(f"- `{info.name}`: {description}")
    return "\n".join(lines)


def tool_values(tools: Mapping[str, ToolInfo] | CapabilityCollection) -> dict[str, Any]:
    if isinstance(tools, CapabilityCollection):
        return tools.bind()
    return {name: info.runtime_value for name, info in tools.items()}


def format_capabilities_for_prompt(
    capabilities: CapabilityCollection,
) -> str | None:
    """Render a validated capability collection for new harness callers."""
    return capabilities.format_for_prompt()


def capability_values(capabilities: CapabilityCollection) -> dict[str, Any]:
    """Bind a validated capability collection for new harness callers."""
    return capabilities.bind()


__all__ = [
    "Capability",
    "Capabilities",
    "CapabilityCollection",
    "FRAMEWORK_NAMES",
    "RESERVED_NAMES",
    "ToolInfo",
    "capability_values",
    "format_capabilities_for_prompt",
    "format_tools_for_prompt",
    "parse_capabilities",
    "parse_tools",
    "tool_values",
]
