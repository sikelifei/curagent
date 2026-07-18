"""Thread-safe global retrieval accounting shared by root and subagents."""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping, Sequence


class SearchBudgetExceeded(RuntimeError):
    """Raised before an MCP call when the per-question budget is exhausted."""


class BrowseCompTrace:
    def __init__(self, max_search_calls: int) -> None:
        if max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive")
        self.max_search_calls = int(max_search_calls)
        self._lock = threading.Lock()
        self._search_calls = 0
        self._retrieved_docids: set[str] = set()
        self._events: list[dict[str, Any]] = []

    def begin_search(self, query: str) -> tuple[int, float]:
        """Atomically reserve one call so parallel workers cannot overspend."""
        with self._lock:
            if self._search_calls >= self.max_search_calls:
                raise SearchBudgetExceeded(
                    f"Global search budget exhausted "
                    f"({self._search_calls}/{self.max_search_calls})"
                )
            self._search_calls += 1
            call_id = self._search_calls
            self._events.append(
                {
                    "call_id": call_id,
                    "query": query,
                    "thread": threading.current_thread().name,
                    "status": "started",
                }
            )
            return call_id, time.monotonic()

    def finish_search(
        self,
        call_id: int,
        started: float,
        *,
        results: Sequence[Mapping[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        normalized_results = [dict(item) for item in results or ()]
        with self._lock:
            event = self._events[call_id - 1]
            event["duration_seconds"] = time.monotonic() - started
            if error is not None:
                event["status"] = "error"
                event["error"] = error
                return
            event["status"] = "completed"
            event["results"] = normalized_results
            self._retrieved_docids.update(
                str(item["docid"])
                for item in normalized_results
                if item.get("docid") is not None
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_search_calls": self.max_search_calls,
                "search_calls": self._search_calls,
                "remaining_search_calls": self.max_search_calls - self._search_calls,
                "retrieved_docids": sorted(self._retrieved_docids),
                "events": [
                    {
                        **event,
                        "results": [
                            dict(result) for result in event.get("results", ())
                        ],
                    }
                    for event in self._events
                ],
            }


__all__ = ["BrowseCompTrace", "SearchBudgetExceeded"]
