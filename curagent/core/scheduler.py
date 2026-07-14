"""Sequential recursive child scheduling with one shared environment reference."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from curagent.core.types import AgentLimits, SubagentResult, SubagentSpec
from curagent.environments.base import Environment


ChildRunner = Callable[
    [str, SubagentSpec, Environment | None, int], Awaitable[SubagentResult]
]


class RecursiveScheduler:
    """Run children in input order; all descendants receive the same environment."""

    def __init__(self, *, limits: AgentLimits) -> None:
        self.limits = limits
        self._id_lock = asyncio.Lock()
        self._next_ids: dict[str, int] = {}

    async def spawn_agent(
        self,
        raw_spec: Mapping[str, Any],
        *,
        parent_env: Environment | None,
        parent_id: str,
        parent_depth: int,
        run_child: ChildRunner,
    ) -> SubagentResult:
        results = await self.spawn_agents(
            [raw_spec],
            parent_env=parent_env,
            parent_id=parent_id,
            parent_depth=parent_depth,
            run_child=run_child,
        )
        return results[0]

    async def spawn_agents(
        self,
        raw_specs: Sequence[Mapping[str, Any]],
        *,
        parent_env: Environment | None,
        parent_id: str,
        parent_depth: int,
        run_child: ChildRunner,
    ) -> list[SubagentResult]:
        if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)):
            return [SubagentResult(error="spawn_agents specs must be an array")]
        if not raw_specs:
            return [SubagentResult(error="spawn_agents requires at least one spec")]

        child_depth = parent_depth + 1
        ids = await self._allocate_ids(parent_id, len(raw_specs))
        results: list[SubagentResult] = []
        for index, raw in enumerate(raw_specs):
            try:
                spec = SubagentSpec.from_mapping(raw)
            except (TypeError, ValueError) as exc:
                results.append(SubagentResult(error=str(exc)))
                continue
            if child_depth > self.limits.max_depth:
                results.append(
                    SubagentResult(
                        error=f"max_depth={self.limits.max_depth} would be exceeded"
                    )
                )
                continue
            try:
                result = await run_child(ids[index], spec, parent_env, child_depth)
            except Exception as exc:
                result = SubagentResult(error=f"child runtime error: {exc}")
            results.append(result)
        return results

    async def _allocate_ids(self, parent_id: str, count: int) -> list[str]:
        async with self._id_lock:
            start = self._next_ids.get(parent_id, 0) + 1
            self._next_ids[parent_id] = start + count - 1
            return [f"{parent_id}.{index}" for index in range(start, start + count)]
