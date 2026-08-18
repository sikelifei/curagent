"""Tool registry exposed by a WebShop environment episode."""

from __future__ import annotations

from typing import Any, Protocol

from ...tools import CapabilityCollection


class WebShopToolTarget(Protocol):
    instruction: str

    def observe(self) -> dict[str, Any]: ...

    def act(self, action: str) -> dict[str, Any]: ...

    def available_actions(self) -> list[str]: ...

    def report(self) -> dict[str, Any]: ...

    def search(self, query: str) -> dict[str, Any]: ...

    def click(self, label: str) -> dict[str, Any]: ...

    def purchase(self) -> dict[str, Any]: ...


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


def build_webshop_capabilities(
    target: WebShopToolTarget,
    *,
    is_root: bool,
) -> CapabilityCollection:
    """Build the role-aware CodeAct action space for one shared episode."""
    click_description = (
        "Click one exact visible non-terminal label and return the updated "
        "observation. The Buy Now action is root-only via purchase()."
        if is_root
        else "Click one exact visible non-terminal label and return the updated "
        "observation. The terminal Buy Now action is unavailable to children."
    )
    entries: dict[str, Any] = {
        "observe": {
            "tool": target.observe,
            "description": (
                "Return the current shared browser snapshot, including the page, "
                "valid actions, history, reward, and terminal state."
            ),
        },
        "search": {
            "tool": target.search,
            "description": (
                "Search the shared WebShop browser and return the updated observation."
            ),
        },
        "click": {
            "tool": target.click,
            "description": click_description,
        },
        "available_actions": {
            "tool": target.available_actions,
            "description": "Return the exact currently valid browser action labels.",
        },
        "episode_report": {
            "tool": target.report,
            "description": "Return the current WebShop reward, success, steps, and trajectory.",
        },
        "shopping_instruction": {
            "tool": target.instruction,
            "description": "The immutable shopping instruction for this episode.",
        },
    }
    if is_root:
        entries["purchase"] = {
            "tool": target.purchase,
            "description": (
                "Execute the currently valid Buy Now purchase action. This is the "
                "root-only terminal action."
            ),
        }
    return CapabilityCollection(entries)


__all__ = ["WebShopToolTarget", "build_webshop_capabilities", "build_webshop_tools"]
