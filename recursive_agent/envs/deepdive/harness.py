"""Adapter that keeps task and search behavior in the Platoon DeepDive harness."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..base import EnvironmentDependencyError

DEFAULT_PLATOON_ROOT = Path("/data2/zhangwenjian/agent/platoon")
DEEPDIVE_SPLITS = ("qa_rl", "qa_sft")


@dataclass(frozen=True)
class DeepDiveSample:
    task_id: str
    question: str
    answer: str
    split: str
    index: int
    metadata: dict[str, Any]


class DeepDiveHarnessProtocol(Protocol):
    def load_sample(self, task_id: str) -> DeepDiveSample: ...

    def search_web(self, query: str, max_results: int = 5) -> dict[str, Any]: ...

    def view_webpage_content(self, url: str) -> str: ...


class _AsyncHarnessBridge:
    """Run Platoon's process-global async Tavily client on one stable loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="deepdive-harness-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def call(self, awaitable: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        return future.result()


_BRIDGE: _AsyncHarnessBridge | None = None
_BRIDGE_LOCK = threading.Lock()


def _bridge() -> _AsyncHarnessBridge:
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = _AsyncHarnessBridge()
        return _BRIDGE


class PlatoonDeepDiveHarness:
    """Thin facade over the user's existing ``platoon.deepdive`` plugin."""

    def __init__(self, platoon_root: str | Path = DEFAULT_PLATOON_ROOT) -> None:
        root = Path(platoon_root).expanduser().resolve()
        for candidate in (root, root / "plugins" / "deepdive"):
            text = str(candidate)
            if candidate.is_dir() and text not in sys.path:
                sys.path.insert(0, text)
        try:
            tasks = importlib.import_module("platoon.deepdive.tasks")
            search_tools = importlib.import_module("platoon.deepdive.search_tools")
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "The Platoon DeepDive harness could not be imported. Run with the "
                "rao environment and pass --platoon-root if it is not located at "
                f"{DEFAULT_PLATOON_ROOT}."
            ) from exc
        self.platoon_root = root
        self._get_task = tasks.get_task
        self._search_web = search_tools.search_web
        self._view_webpage_content = search_tools.view_webpage_content

    def load_sample(self, task_id: str) -> DeepDiveSample:
        parts = str(task_id).split(".")
        if len(parts) != 3 or parts[0] != "deepdive":
            raise ValueError(f"Invalid DeepDive task id: {task_id!r}")
        split = parts[1]
        if split not in DEEPDIVE_SPLITS:
            raise ValueError(f"Unsupported DeepDive split: {split!r}")
        index = int(parts[2])
        task = self._get_task(str(task_id))
        metadata = dict(task.misc or {})
        answer = str(metadata.get("ground_truth", metadata.get("answer", "")))
        return DeepDiveSample(
            task_id=str(task.id),
            question=str(task.goal),
            answer=answer,
            split=split,
            index=index,
            metadata=metadata,
        )

    def search_web(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        return _bridge().call(
            self._search_web(query=query, max_results=max_results)
        )

    def view_webpage_content(self, url: str) -> str:
        return _bridge().call(self._view_webpage_content(url))


def make_task_ids(split: str, start_index: int, limit: int) -> list[str]:
    if split not in DEEPDIVE_SPLITS:
        raise ValueError(f"Unsupported DeepDive split: {split!r}")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if limit <= 0:
        raise ValueError("limit must be positive")
    return [
        f"deepdive.{split}.{index}"
        for index in range(start_index, start_index + limit)
    ]


__all__ = [
    "DEFAULT_PLATOON_ROOT",
    "DEEPDIVE_SPLITS",
    "DeepDiveHarnessProtocol",
    "DeepDiveSample",
    "PlatoonDeepDiveHarness",
    "make_task_ids",
]
