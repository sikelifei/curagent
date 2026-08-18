"""Official BrowseComp-Plus MCP search client and tool registration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..base import EnvironmentDependencyError


class BrowseCompToolTarget(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...


class MCPBM25Client:
    """Small synchronous facade over FastMCP's asynchronous client."""

    def __init__(self, url: str, *, timeout: float = 60.0) -> None:
        if not str(url).strip():
            raise ValueError("BM25 MCP URL cannot be empty")
        if timeout <= 0:
            raise ValueError("BM25 MCP timeout must be positive")
        normalized_url = str(url).rstrip("/")
        self.url = (
            normalized_url
            if normalized_url.endswith("/mcp")
            else normalized_url + "/mcp"
        )
        self.timeout = float(timeout)

    def search(self, query: str) -> list[dict[str, Any]]:
        return asyncio.run(self._search(query))

    async def _search(self, query: str) -> list[dict[str, Any]]:
        try:
            from fastmcp import Client
            from fastmcp.client.transports import SSETransport
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "FastMCP is required for BrowseComp-Plus. Install curagent with "
                "python -m pip install -e '.[browsecomp]'."
            ) from exc

        client = Client(
            SSETransport(url=self.url),
            timeout=self.timeout,
        )
        async with client:
            result = await client.call_tool(
                "search",
                {"query": query},
                timeout=self.timeout,
            )
        return normalize_search_results(_tool_result_payload(result))


def normalize_search_results(
    value: Any,
    *,
    snippet_max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize the official search payload to docid/score/snippet mappings."""
    if snippet_max_chars is not None and snippet_max_chars <= 0:
        raise ValueError("snippet_max_chars must be positive or None")
    if value is None:
        return []
    if isinstance(value, Mapping) and set(value) == {"result"}:
        value = value["result"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("BM25 search response must be a list")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"BM25 result {index} must be an object")
        if item.get("docid") is None:
            raise ValueError(f"BM25 result {index} is missing docid")
        raw_score = item.get("score")
        snippet = str(item.get("snippet", item.get("text", "")))
        if snippet_max_chars is not None:
            snippet = snippet[:snippet_max_chars]
        normalized.append(
            {
                "docid": str(item["docid"]),
                "score": float(raw_score) if raw_score is not None else None,
                "snippet": snippet,
            }
        )
    return normalized


def _tool_result_payload(result: Any) -> Any:
    data = getattr(result, "data", None)
    if data is not None:
        return data
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured.get("result", structured)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        content = result
    else:
        content = getattr(result, "content", None) or ()
    for block in content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise ValueError("BM25 MCP response contained no parseable result")


def build_browsecomp_tools(target: BrowseCompToolTarget) -> dict[str, Any]:
    """Expose exactly one environment tool to every recursive agent."""
    return {
        "search": {
            "tool": target.search,
            "description": (
                "Search the fixed BrowseComp-Plus BM25 corpus with one non-empty "
                "lexical query. Returns up to five dictionaries containing docid, "
                "score, and snippet. Root and subagents share one call budget."
            ),
        }
    }


__all__ = [
    "BrowseCompToolTarget",
    "MCPBM25Client",
    "build_browsecomp_tools",
    "normalize_search_results",
]
