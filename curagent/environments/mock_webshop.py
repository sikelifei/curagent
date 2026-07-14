"""Deterministic direct-tool WebShop environment for harness tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from curagent.core.types import ToolCall, ToolSchema
from curagent.environments.base import Environment
from curagent.tasks.webshop import WEBSHOP_ENVIRONMENT_TOOLS


@dataclass
class _State:
    page: str = "home"
    selected_item: str | None = None
    selected_options: set[str] = field(default_factory=set)
    query: str = ""
    done: bool = False
    reward: float = 0.0


class MockWebShopEnvironment(Environment):
    def __init__(self) -> None:
        self.instruction = "Buy the blue 32 oz insulated stainless steel water bottle."
        self._state = _State()
        self._version = 0

    async def reset(self, instance: Any = None) -> dict[str, Any]:
        if isinstance(instance, Mapping) and isinstance(instance.get("instruction"), str):
            self.instruction = instance["instruction"]
        self._state = _State()
        self._version = 0
        return await self.observe()

    async def observe(self) -> dict[str, Any]:
        return {
            "text": self._render(),
            "version": self._version,
            "metadata": {
                "instruction": self.instruction,
                "done": self._state.done,
                "valid_targets": self._valid_targets(),
            },
        }

    def tools(self) -> Sequence[ToolSchema]:
        return WEBSHOP_ENVIRONMENT_TOOLS

    async def execute(self, tool_call: ToolCall) -> Any:
        if self._state.done:
            return "episode is already done"
        error = self._apply(tool_call)
        if error:
            return error
        self._version += 1
        return {"tool": tool_call.name, "arguments": dict(tool_call.arguments)}

    def is_done(self) -> bool:
        return self._state.done

    def reward(self) -> float:
        return self._state.reward

    def _apply(self, call: ToolCall) -> str | None:
        if call.name == "search":
            if self._state.page != "home":
                return "search is available only on the home page"
            self._state.query = call.arguments["query"]
            self._state.page = "results"
            return None
        if call.name == "click":
            target = call.arguments["target"]
            if target not in self._valid_targets():
                return f"invalid click target: {target}; valid targets: {self._valid_targets()}"
            if self._state.page == "results":
                self._state.selected_item = target
                self._state.selected_options.clear()
                self._state.page = "product"
            elif target == "< Prev":
                self._state.page = "results"
                self._state.selected_item = None
                self._state.selected_options.clear()
            else:
                self._state.selected_options.add(target)
            return None
        if call.name == "buy":
            if self._state.page != "product":
                return "buy is available only on a product page"
            self._state.done = True
            correct = self._state.selected_item == "B001" and {"Blue", "32 oz"}.issubset(
                self._state.selected_options
            )
            self._state.reward = 1.0 if correct else 0.0
            return None
        return f"unsupported environment tool: {call.name}"

    def _valid_targets(self) -> list[str]:
        if self._state.done or self._state.page == "home":
            return []
        if self._state.page == "results":
            return ["B001", "B002", "B003"]
        return ["Blue", "32 oz", "< Prev"]

    def _render(self) -> str:
        if self._state.done:
            return f"Your score (min 0.0, max 1.0): {self._state.reward}"
        if self._state.page == "home":
            return f"Instruction: {self.instruction}\nSearch is available."
        if self._state.page == "results":
            return (
                f"Instruction: {self.instruction}\nResults for {self._state.query!r}:\n"
                "[B001] Blue 32 oz insulated stainless steel water bottle\n"
                "[B002] Green 24 oz plastic sports bottle\n"
                "[B003] Blue 32 oz glass carafe"
            )
        return (
            f"Instruction: {self.instruction}\nProduct {self._state.selected_item}\n"
            "Options: [Blue] [32 oz]\nActions: buy or [< Prev]"
        )
