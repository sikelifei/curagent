"""Tool registry for a TextCraft-Synth episode."""

from __future__ import annotations

from typing import Any, Protocol


class TextCraftToolTarget(Protocol):
    def craft(self, ingredients: dict[str, int], target: tuple[str, int]) -> str: ...

    def get_info(self, items: list[str] | None = None) -> list[dict[str, Any]]: ...

    def view_inventory(self) -> dict[str, int]: ...

    def finish(self, message: str) -> str: ...


def build_textcraft_tools(target: TextCraftToolTarget) -> dict[str, Any]:
    return {
        "craft": {
            "tool": target.craft,
            "description": (
                "Craft one output item by consuming exact ingredient counts. "
                "Arguments: craft(ingredients: dict[str, int], target: (item, "
                "output_count)). output_count must be divisible by the selected "
                "recipe result_count. Print the returned status."
            ),
        },
        "get_info": {
            "tool": target.get_info,
            "description": (
                "Return current inventory counts, recipes, result counts, "
                "crafting depth, and whether each requested item can be crafted "
                "from the current inventory. Pass a list of item names; with no "
                "argument it reports the task targets. Print the result."
            ),
        },
        "view_inventory": {
            "tool": target.view_inventory,
            "description": "Return the current shared inventory. Print the result.",
        },
        "finish": {
            "tool": target.finish,
            "description": (
                "Submit a short completion message. The episode terminates only "
                "when every requested target is present; an intermediate child "
                "should return a report instead."
            ),
        },
    }


__all__ = ["TextCraftToolTarget", "build_textcraft_tools"]
