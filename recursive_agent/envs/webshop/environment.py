"""Adapter from ReCode's WebShop environment to RecursiveAgent custom tools."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ...types import EnvironmentStatus
from ..base import AgentEnvironment, EnvironmentDependencyError
from ..registry import register_environment
from .dataset import WebShopDataset, WebShopSample
from .prompts import DEFAULT_WEBSHOP_TASK_TEMPLATE
from .tools import build_webshop_tools

_ACTION_PATTERN = re.compile(r"^(search|click)\[(.*)\]$", re.IGNORECASE | re.DOTALL)


@register_environment("webshop")
class ReCodeWebShopEnvironment(AgentEnvironment):
    """One ReCode WebShop episode with curagent-compatible synchronous tools."""

    name = "webshop"

    def __init__(
        self,
        *,
        recode_root: str | Path | None = None,
        split: str = "test",
        instance_id: int = 0,
        max_steps: int = 30,
        success_threshold: float = 1.0,
        seed: int = 233,
        prompt_template: str = DEFAULT_WEBSHOP_TASK_TEMPLATE,
        backend: Any | None = None,
    ) -> None:
        self.recode_root = resolve_recode_root(recode_root)
        self.dataset = WebShopDataset(self.recode_root, split=split)
        sample = self.dataset[instance_id]
        self.seed = int(seed)
        self._closed = False
        self._owns_recode_backend = backend is None
        self._backend = backend or _create_recode_backend(
            self.recode_root,
            max_steps=max_steps,
            success_threshold=success_threshold,
            seed=self.seed,
        )
        init_info = self._backend.reset({"split": split}, str(sample.index))
        observations = init_info.get("observations") if isinstance(init_info, dict) else None
        initial_observation = (
            str(observations[0])
            if isinstance(observations, list) and observations
            else str(getattr(self._backend, "last_observation", ""))
        )
        instruction = str(self._backend.get_instruction_text()).strip()
        self.sample: WebShopSample = sample.with_episode_data(
            instruction=instruction,
            observation=initial_observation,
        )
        self.instruction = instruction
        self._task = self.dataset.build_task(self.sample, template=prompt_template)
        self._context = {
            "environment": self.name,
            "split": self.sample.split,
            "instance_id": self.sample.index,
            "session_id": self.sample.session_id,
            "seed": self.seed,
            "instruction": self.instruction,
            "initial_observation": initial_observation,
            "initial_valid_actions": self.available_actions(),
        }
        self._tools = build_webshop_tools(self)

    @property
    def task(self) -> str:
        return self._task

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def tools(self) -> dict[str, Any]:
        return dict(self._tools)

    def observe(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "observation": str(getattr(self._backend, "last_observation", "")),
            "valid_actions": self.available_actions(),
            "action_history": self._action_history(),
            "steps": int(self._backend.get_step_count()),
            "reward": float(self._backend.get_reward()),
            "done": bool(self._backend.is_done()),
            "success": bool(self._backend.is_success()),
        }

    def act(self, action: str) -> dict[str, Any]:
        if self._backend.is_done():
            raise RuntimeError("WebShop episode is already terminal")
        normalized = self._normalize_action(action)
        _run_awaitable(self._backend.run(normalized))
        return self.observe()

    def available_actions(self) -> list[str]:
        if self._backend.is_done():
            return []
        available = self._backend.get_available_actions() or {}
        actions: list[str] = []
        if available.get("has_search_bar"):
            actions.append("search[keywords]")
        seen: set[str] = set()
        for raw in available.get("clickables", []) or []:
            clickable = " ".join(str(raw).strip().split())
            key = clickable.casefold()
            if not clickable or key == "search" or key in seen:
                continue
            seen.add(key)
            actions.append(f"click[{clickable}]")
        return actions

    def status(self) -> EnvironmentStatus:
        done = bool(self._backend.is_done())
        if not done:
            return EnvironmentStatus(done=False)
        report = self.report()
        answer = (
            "WebShop episode completed: "
            f"success={report['success']}, reward={report['reward']:.3f}, "
            f"steps={report['steps']}"
        )
        return EnvironmentStatus(
            done=True,
            final_answer=answer,
            reason="success" if report["success"] else "terminal_without_success",
        )

    def report(self) -> dict[str, Any]:
        raw = self._backend.report() or {}
        trajectory = self._backend.get_trajectory()
        return {
            "environment": self.name,
            "split": self.sample.split,
            "instance_id": self.sample.index,
            "session_id": self.sample.session_id,
            "seed": self.seed,
            "instruction": self.instruction,
            "success": bool(raw.get("success", self._backend.is_success())),
            "reward": float(raw.get("reward", self._backend.get_reward()) or 0.0),
            "steps": int(raw.get("step", self._backend.get_step_count())),
            "trajectory": list(trajectory or []),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # ReCode's WebShopEnv.close() clears its process-global shared server.
        # Keep that server alive across a batch of episodes; the process owns it
        # and will release it on exit. Injected test/custom backends are closed
        # normally because they have no shared ReCode server.
        if self._owns_recode_backend:
            return
        close = getattr(self._backend, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                _run_awaitable(result)

    def _normalize_action(self, action: str) -> str:
        if not isinstance(action, str):
            raise TypeError("WebShop action must be a string")
        compact = action.strip()
        match = _ACTION_PATTERN.fullmatch(compact)
        if match is None:
            raise ValueError("WebShop action must be search[keywords] or click[element]")
        kind = match.group(1).lower()
        argument = " ".join(match.group(2).strip().split())
        if not argument:
            raise ValueError("WebShop action argument cannot be empty")

        valid = self.available_actions()
        if kind == "search":
            if "search[keywords]" not in valid:
                raise ValueError(f"Search is not valid now. Valid actions: {valid}")
            return f"search[{argument}]"

        valid_clicks: dict[str, str] = {}
        for candidate in valid:
            candidate_match = _ACTION_PATTERN.fullmatch(candidate)
            if candidate_match and candidate_match.group(1).lower() == "click":
                value = candidate_match.group(2)
                valid_clicks[value.casefold()] = value
        canonical = valid_clicks.get(argument.casefold())
        if canonical is None:
            raise ValueError(f"Click target is not valid now. Valid actions: {valid}")
        return f"click[{canonical}]"

    def _action_history(self) -> list[str]:
        history = []
        for entry in self._backend.get_trajectory() or []:
            action = entry.get("action") if isinstance(entry, dict) else None
            if action:
                history.append(str(action))
        return history


def resolve_recode_root(recode_root: str | Path | None = None) -> Path:
    candidates = []
    if recode_root is not None:
        candidates.append(Path(recode_root))
    if os.getenv("RECODE_ROOT"):
        candidates.append(Path(os.environ["RECODE_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[4] / "ReCode")
    candidates.append(Path.cwd() / "ReCode")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "envs" / "webshop" / "env.py").is_file():
            return resolved
    rendered = ", ".join(str(path.expanduser()) for path in candidates)
    raise EnvironmentDependencyError(
        "Could not locate ReCode WebShop. Pass recode_root or set RECODE_ROOT. "
        f"Checked: {rendered}"
    )


def _create_recode_backend(
    recode_root: Path,
    *,
    max_steps: int,
    success_threshold: float,
    seed: int,
) -> Any:
    root_text = str(recode_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("envs.webshop.env")
    except Exception as exc:
        raise EnvironmentDependencyError(
            "Failed to import ReCode WebShop. Run curagent in the ReCode conda "
            "environment (Python 3.10) with its WebShop dependencies installed."
        ) from exc

    original_reader = module.read_json_file

    def read_from_recode(path: str, encoding: str = "utf-8") -> Any:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = recode_root / candidate
        return original_reader(str(candidate), encoding=encoding)

    module.read_json_file = read_from_recode
    random_state = random.getstate()
    try:
        random.seed(seed)
        return module.WebShopEnv(
            logger=None,
            max_steps=max_steps,
            success_threshold=success_threshold,
        )
    except Exception as exc:
        raise EnvironmentDependencyError(f"Failed to initialize ReCode WebShop: {exc}") from exc
    finally:
        random.setstate(random_state)


def _run_awaitable(awaitable: Any) -> Any:
    if not inspect.isawaitable(awaitable):
        return awaitable
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="webshop-async") as executor:
        return executor.submit(asyncio.run, awaitable).result()
