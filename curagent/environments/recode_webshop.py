"""Direct-tool adapter for the ReCode WebShop environment."""

from __future__ import annotations

import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from curagent.core.types import ToolCall, ToolSchema
from curagent.environments.base import Environment
from curagent.tasks.webshop import WEBSHOP_ENVIRONMENT_TOOLS


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ReCodeWebShopEnvironment(Environment):
    _reset_lock = threading.RLock()

    def __init__(
        self,
        *,
        project_root: str = "/data2/zhangwenjian/agent/ReCode",
        split: str = "test",
        max_steps: int = 30,
    ) -> None:
        self.project_root = Path(project_root)
        self.split = split
        self.max_steps = max_steps
        self.task_id = "0"
        self._env: Any = None
        self._version = 0
        self._last_observation = ""
        self._instruction = ""
        self._terminal_error: str | None = None

    async def reset(self, instance: Any) -> dict[str, Any]:
        if isinstance(instance, Mapping):
            task_id = instance.get("task_id", instance.get("id", "0"))
            split = instance.get("split", self.split)
        else:
            task_id = instance
            split = self.split
        self.task_id = str(task_id)
        self.split = str(split)
        with self._reset_lock:
            sys.path.insert(0, str(self.project_root))
            try:
                with _pushd(self.project_root):
                    from envs.webshop.env import WebShopEnv

                    self._env = WebShopEnv(logger=None, max_steps=self.max_steps)
                    initial = self._env.reset({"split": self.split}, self.task_id)
            finally:
                try:
                    sys.path.remove(str(self.project_root))
                except ValueError:
                    pass
        observations = initial.get("observations") or [""]
        self._last_observation = str(observations[-1])
        try:
            self._instruction = str(self._env.webshop_env.get_instruction_text())
        except Exception:
            self._instruction = ""
        self._version = 0
        self._terminal_error = None
        return await self.observe()

    async def observe(self) -> dict[str, Any]:
        actions = self._backend_actions()
        return {
            "text": self._last_observation,
            "version": self._version,
            "metadata": {
                "instruction": self._instruction,
                "done": self.is_done(),
                "terminal_error": self._terminal_error,
                "valid_targets": self._valid_targets(
                    self._last_observation, actions["clickables"]
                ),
                "search_available": actions["has_search_bar"],
                "task_id": self.task_id,
                "split": self.split,
            },
        }

    def tools(self) -> Sequence[ToolSchema]:
        return WEBSHOP_ENVIRONMENT_TOOLS

    async def execute(self, tool_call: ToolCall) -> Any:
        before = await self.observe()
        if self.is_done():
            return "episode is already done"
        action, error = self._translate(tool_call, before)
        if error:
            return error
        try:
            observations = await self._env.run([action])
        except Exception as exc:
            self._refresh_from_backend()
            raise RuntimeError(f"ReCode WebShop execution failed: {exc}") from exc
        if observations:
            self._last_observation = str(observations[-1])
        else:
            self._refresh_from_backend()
        self._version += 1
        backend_error = (
            self._last_observation
            if self._last_observation.startswith("WebShop step execution failed:")
            else None
        )
        if backend_error:
            self._terminal_error = backend_error
            return backend_error
        return {
            "tool": tool_call.name,
            "arguments": dict(tool_call.arguments),
            "action": action,
        }

    def is_done(self) -> bool:
        return bool(self._env is not None and self._env.is_done())

    def reward(self) -> float:
        return (
            float(getattr(self._env, "reward", 0.0) or 0.0)
            if self._env is not None
            else 0.0
        )

    async def close(self) -> None:
        backend = getattr(self._env, "webshop_env", None)
        if backend is not None and hasattr(backend, "close"):
            result = backend.close()
            if hasattr(result, "__await__"):
                await result

    def _translate(self, call: ToolCall, observation: Mapping[str, Any]) -> tuple[str, str | None]:
        metadata = observation.get("metadata") or {}
        valid_targets = list(metadata.get("valid_targets") or [])
        valid_lower = {target.lower() for target in valid_targets}
        if call.name == "search":
            if not metadata.get("search_available"):
                return "", "search is not available on the current page"
            return f"search[{call.arguments['query']}]", None
        if call.name == "click":
            target = call.arguments["target"]
            if target.lower() not in valid_lower:
                return "", f"invalid click target: {target}; valid targets: {valid_targets}"
            return f"click[{target}]", None
        if call.name == "buy":
            if "buy now" not in valid_lower:
                return "", "Buy Now is not available on the current page"
            return "click[Buy Now]", None
        return "", f"unsupported environment tool: {call.name}"

    def _refresh_from_backend(self) -> None:
        value = getattr(self._env, "last_observation", None)
        if value:
            self._last_observation = str(value)

    def _backend_actions(self) -> dict[str, Any]:
        try:
            actions = self._env.webshop_env.get_available_actions()
            return {
                "has_search_bar": bool(actions.get("has_search_bar")),
                "clickables": [str(item) for item in actions.get("clickables") or []],
            }
        except Exception:
            return {
                "has_search_bar": "[Search]" in self._last_observation,
                "clickables": [],
            }

    @staticmethod
    def _valid_targets(observation: str, clickables: Sequence[str]) -> list[str]:
        allowed = {target.lower() for target in clickables}
        targets: list[str] = []
        for token in re.findall(r"\[([^\[\]\n]+)\]", observation):
            target = token.strip()
            if not target or target == "Search":
                continue
            if (not allowed or target.lower() in allowed) and target not in targets:
                targets.append(target)
        represented = {target.lower() for target in targets}
        for target in clickables:
            if target.lower() not in represented:
                targets.append(target)
        return targets
