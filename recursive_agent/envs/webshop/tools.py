"""Tool registry exposed by a WebShop environment episode."""

from __future__ import annotations

from typing import Any, Protocol


class WebShopToolTarget(Protocol):
    instruction: str

    def observe(self) -> dict[str, Any]: ...

    def act(self, action: str) -> dict[str, Any]: ...

    def available_actions(self) -> list[str]: ...

    def report(self) -> dict[str, Any]: ...


def build_webshop_tools(target: WebShopToolTarget) -> dict[str, Any]:
    """Build the custom-tool mapping consumed by RecursiveAgent."""
    return {
        "observe": {
            "tool": target.observe,
            "description": (
                "Return the current WebShop instruction, page observation, valid actions, "
                "step count, reward, and terminal state. Print the result."
            ),
        },
        "act": {
            "tool": target.act,
            "description": (
                "Execute one WebShop search[...] or currently valid click[...] action and "
                "return the updated state. click[Buy Now] is the finish action and makes "
                "the episode terminal. Print the result."
            ),
        },
        "available_actions": {
            "tool": target.available_actions,
            "description": "Return the currently valid WebShop action strings. Print the result.",
        },
        "episode_report": {
            "tool": target.report,
            "description": "Return current WebShop reward, success, steps, and trajectory.",
        },
        "shopping_instruction": {
            "tool": target.instruction,
            "description": "The immutable shopping instruction for this episode.",
        },
    }
