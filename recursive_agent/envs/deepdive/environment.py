"""Curagent execution adapter for the existing Platoon DeepDive harness."""

from __future__ import annotations

import threading
import time
from typing import Any

from ...types import EnvironmentStatus
from ..base import AgentEnvironment
from ..registry import register_environment
from .harness import DeepDiveHarnessProtocol, DeepDiveSample, PlatoonDeepDiveHarness
from .prompts import (
    DEFAULT_DEEPDIVE_AGENT_PROMPT,
    DEFAULT_DEEPDIVE_COMPLETION_PROMPT,
    DEFAULT_DEEPDIVE_FORCED_FINAL_PROMPT,
)


@register_environment("deepdive")
class DeepDiveEnvironment(AgentEnvironment):
    """One DeepDive question with a shared Platoon web-tool trace."""

    name = "deepdive"

    def __init__(
        self,
        *,
        sample: DeepDiveSample,
        harness: DeepDiveHarnessProtocol | None = None,
        max_search_calls: int | None = None,
    ) -> None:
        if not isinstance(sample, DeepDiveSample):
            raise TypeError("sample must be a DeepDiveSample")
        if max_search_calls is not None and max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive or None")
        self.sample = sample
        self.harness = harness or PlatoonDeepDiveHarness()
        self.max_search_calls = max_search_calls
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._context = {
            "environment": self.name,
            "task_id": sample.task_id,
            "dataset": "zai-org/DeepDive",
            "split": sample.split,
            "index": sample.index,
        }

    @property
    def task(self) -> str:
        return self.sample.question

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    @property
    def agent_prompt(self) -> str:
        return DEFAULT_DEEPDIVE_AGENT_PROMPT

    @property
    def completion_prompt(self) -> str:
        return DEFAULT_DEEPDIVE_COMPLETION_PROMPT

    @property
    def delegated_completion_prompt(self) -> str:
        return DEFAULT_DEEPDIVE_COMPLETION_PROMPT

    @property
    def forced_final_prompt(self) -> str:
        return DEFAULT_DEEPDIVE_FORCED_FINAL_PROMPT

    @property
    def delegated_forced_final_prompt(self) -> str:
        return DEFAULT_DEEPDIVE_FORCED_FINAL_PROMPT

    @property
    def max_repl_blocks_per_step(self) -> int:
        return 1

    def tools(self) -> dict[str, Any]:
        return {
            "search_web": {
                "tool": self.search_web,
                "description": (
                    "Platoon DeepDive/Tavily web search. Call directly; "
                    "accepts query and max_results (1-20), and returns the original "
                    "DeepDive search dictionary."
                ),
            },
            "view_webpage_content": {
                "tool": self.view_webpage_content,
                "description": (
                    "Platoon DeepDive/Tavily webpage extraction. Call directly "
                    "with one URL; returns the extracted raw content."
                ),
            },
        }

    def search_web(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        self._ensure_open()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(max_results, int) or not 1 <= max_results <= 20:
            raise ValueError("max_results must be an integer between 1 and 20")
        event = self._begin_event(
            "search_web",
            {"query": query, "max_results": max_results},
        )
        try:
            result = self.harness.search_web(query.strip(), max_results)
        except Exception as exc:
            self._finish_event(event, error=f"{type(exc).__name__}: {exc}")
            raise
        self._finish_event(event, result=result)
        return result

    def view_webpage_content(self, url: str) -> str:
        self._ensure_open()
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        event = self._begin_event("view_webpage_content", {"url": url})
        try:
            result = self.harness.view_webpage_content(url.strip())
        except Exception as exc:
            self._finish_event(event, error=f"{type(exc).__name__}: {exc}")
            raise
        self._finish_event(event, result=result)
        return result

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=False)

    def report(self) -> dict[str, Any]:
        with self._lock:
            events = [dict(event) for event in self._events]
        return {
            "environment": self.name,
            "task_id": self.sample.task_id,
            "dataset": "zai-org/DeepDive",
            "split": self.sample.split,
            "index": self.sample.index,
            "harness": "platoon.deepdive",
            "tool_call_counts": {
                "search_web": sum(event["tool"] == "search_web" for event in events),
                "view_webpage_content": sum(
                    event["tool"] == "view_webpage_content" for event in events
                ),
            },
            "events": events,
        }

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeepDive environment is closed")

    def _begin_event(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            calls = len(self._events)
            if self.max_search_calls is not None and calls >= self.max_search_calls:
                raise RuntimeError(
                    f"DeepDive shared tool-call budget exhausted ({self.max_search_calls})"
                )
            event = {
                "id": calls + 1,
                "tool": tool,
                "arguments": arguments,
                "started_at": time.time(),
            }
            self._events.append(event)
        return event

    def _finish_event(
        self,
        event: dict[str, Any],
        *,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            event["duration_seconds"] = time.time() - float(event["started_at"])
            event["result"] = result
            event["error"] = error


__all__ = ["DeepDiveEnvironment"]
