"""Tool registry for a TextCraft-Synth episode."""

from __future__ import annotations

from typing import Any, Protocol

from ...tools import CapabilityCollection


class TextCraftToolTarget(Protocol):
    def craft(self, ingredients: dict[str, int], target: tuple[str, int]) -> str: ...

    def get_info(self, items: list[str] | None = None) -> list[dict[str, Any]] | str: ...

    def view_inventory(self) -> dict[str, int]: ...

    def finish(self, message: str) -> str: ...


def _textcraft_capability_entries(target: TextCraftToolTarget) -> dict[str, dict[str, Any]]:
    return {
        "craft": {
            "tool": target.craft,
            "description": (
                "Craft one output item by consuming exact ingredient counts. "
                "Arguments: craft(ingredients: dict[str, int], target: (item, "
                "output_count)). output_count must be divisible by the selected "
                "recipe result_count. Print the returned status. This synchronous "
                "environment tool must be called directly as `craft(...)`; it "
                "must not be awaited."
            ),
        },
        "get_info": {
            "tool": target.get_info,
            "description": (
                "Return current inventory counts, recipes, result counts, "
                "crafting depth, and whether each requested item can be crafted "
                "from the current inventory. Always pass an explicit list of item "
                "names; with no argument it returns an actionable error instead of "
                "falling back to root targets. Print the result. This "
                "synchronous environment tool must be called directly as "
                "`get_info(...)`; it must not be awaited."
            ),
        },
        "view_inventory": {
            "tool": target.view_inventory,
            "description": (
                "Return the current shared inventory. Print the result. This "
                "synchronous environment tool must be called directly as "
                "`view_inventory(...)`; it must not be awaited."
            ),
        },
    }


def build_textcraft_capabilities(target: TextCraftToolTarget) -> CapabilityCollection:
    """Build the environment-owned CodeAct action space.

    Root/child termination and recursive delegation are supplied by the generic
    scheduler. The legacy ``finish`` tool intentionally remains outside this
    collection for direct ``RecursiveAgent`` compatibility.
    """

    return CapabilityCollection(_textcraft_capability_entries(target))


def build_textcraft_tools(target: TextCraftToolTarget) -> dict[str, Any]:
    """Return the legacy tool mapping, including its environment-owned finish."""
    tools = _textcraft_capability_entries(target)
    tools["finish"] = {
        "tool": target.finish,
        "description": (
            "Submit a short completion message. The episode terminates only "
            "when every requested target is present."
        ),
    }
    return tools


__all__ = [
    "TextCraftToolTarget",
    "build_textcraft_capabilities",
    "build_textcraft_tools",
]
