"""BrowseComp-Plus environment backed by the official BM25 MCP search tool."""

from __future__ import annotations

from typing import Any, Protocol

from ...types import EnvironmentStatus
from ..base import AgentEnvironment
from ..registry import register_environment
from .dataset import BrowseCompQuery
from .prompts import (
    DEFAULT_BROWSECOMP_AGENT_PROMPT,
    DEFAULT_BROWSECOMP_CHILD_PROMPT,
    DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT,
    DEFAULT_BROWSECOMP_ROOT_COMPLETION_PROMPT,
    DEFAULT_BROWSECOMP_WORKER_COMPLETION_PROMPT,
    DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT,
    DEFAULT_BROWSECOMP_TASK_TEMPLATE,
    DEFAULT_BROWSECOMP_ROOT_PROMPT,
    build_browsecomp_task_prompt,
)
from .tools import MCPBM25Client, build_browsecomp_tools, normalize_search_results
from .trace import BrowseCompTrace


class SearchClient(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...


@register_environment("browsecomp_plus")
class BrowseCompPlusEnvironment(AgentEnvironment):
    """One question-only episode with shared retrieval state."""

    name = "browsecomp_plus"

    def __init__(
        self,
        *,
        sample: BrowseCompQuery,
        bm25_url: str = "http://127.0.0.1:8080/mcp",
        max_search_calls: int = 20,
        bm25_timeout: float = 60.0,
        snippet_max_chars: int = 1000,
        search_client: SearchClient | None = None,
        prompt_template: str = DEFAULT_BROWSECOMP_TASK_TEMPLATE,
        agent_prompt: str = DEFAULT_BROWSECOMP_AGENT_PROMPT,
    ) -> None:
        if not isinstance(sample, BrowseCompQuery):
            raise TypeError("sample must be a BrowseCompQuery")
        self.sample = sample
        self.bm25_url = str(bm25_url)
        if snippet_max_chars <= 0:
            raise ValueError("snippet_max_chars must be positive")
        self.snippet_max_chars = int(snippet_max_chars)
        self.trace = BrowseCompTrace(max_search_calls)
        self._search_client = search_client or MCPBM25Client(
            self.bm25_url,
            timeout=bm25_timeout,
        )
        self._task = build_browsecomp_task_prompt(sample, template=prompt_template)
        self._agent_prompt = str(agent_prompt).strip()
        self._tools = build_browsecomp_tools(self)
        self._closed = False
        self._context = {
            "environment": self.name,
            "query_id": sample.query_id,
            "query": sample.query,
        }

    @property
    def task(self) -> str:
        return self._task

    @property
    def agent_prompt(self) -> str:
        return self._agent_prompt

    @property
    def root_prompt(self) -> str:
        return DEFAULT_BROWSECOMP_ROOT_PROMPT

    @property
    def child_prompt(self) -> str:
        return DEFAULT_BROWSECOMP_CHILD_PROMPT

    @property
    def completion_prompt(self) -> str:
        return DEFAULT_BROWSECOMP_ROOT_COMPLETION_PROMPT

    @property
    def delegated_completion_prompt(self) -> str:
        return DEFAULT_BROWSECOMP_WORKER_COMPLETION_PROMPT

    @property
    def forced_final_prompt(self) -> str:
        return DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT

    @property
    def delegated_forced_final_prompt(self) -> str:
        return DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT

    @property
    def disabled_repl_builtins(self) -> frozenset[str]:
        # The search tool is the only benchmark data access path for this env.
        return frozenset({"__import__", "open"})

    @property
    def context(self) -> dict[str, str]:
        return dict(self._context)

    def tools(self) -> dict[str, Any]:
        return dict(self._tools)

    def search(self, query: str) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("BrowseComp-Plus environment is closed")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be a non-empty string")
        compact = " ".join(query.split())
        call_id, started = self.trace.begin_search(compact)
        try:
            results = normalize_search_results(
                self._search_client.search(compact),
                snippet_max_chars=self.snippet_max_chars,
            )
        except Exception as exc:
            self.trace.finish_search(
                call_id,
                started,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.trace.finish_search(call_id, started, results=results)
        return results

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=False)

    def report(self) -> dict[str, Any]:
        snapshot = self.trace.snapshot()
        return {
            "environment": self.name,
            "query_id": self.sample.query_id,
            "query": self.sample.query,
            "retriever": "BM25",
            "bm25_url": self.bm25_url,
            "snippet_max_chars": self.snippet_max_chars,
            "tool_call_counts": {"search": snapshot["search_calls"]},
            **snapshot,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._search_client, "close", None)
        if callable(close):
            close()


__all__ = ["BrowseCompPlusEnvironment", "SearchClient"]
