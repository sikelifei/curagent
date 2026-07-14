"""Recursive scheduling, concurrency, clones, and writer leases."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from curagent.core.budget import SharedBudget
from curagent.core.errors import BudgetExceeded, SchedulerError
from curagent.core.types import (
    AccessMode,
    AgentLimits,
    ExecutionReceipt,
    SubagentResult,
    SubagentSpec,
    TerminalStatus,
    ToolCall,
)
from curagent.environments.base import Environment


ChildRunner = Callable[[str, SubagentSpec, Environment, int], Awaitable[SubagentResult]]


class ResourceManager:
    def __init__(self) -> None:
        self._writer_lock = asyncio.Lock()
        self._writer_held = contextvars.ContextVar("curagent_writer_held", default=False)

    @asynccontextmanager
    async def writer_lease(self):
        if self._writer_held.get():
            yield
            return
        async with self._writer_lock:
            token = self._writer_held.set(True)
            try:
                yield
            finally:
                self._writer_held.reset(token)

    async def execute(self, env: Environment, call: ToolCall, expected_version: int) -> ExecutionReceipt:
        if self._writer_held.get():
            return await env.execute(call, expected_version)
        async with self._writer_lock:
            return await env.execute(call, expected_version)


class RecursiveScheduler:
    def __init__(
        self,
        *,
        root_env: Environment,
        budget: SharedBudget,
        limits: AgentLimits,
        resources: ResourceManager | None = None,
    ) -> None:
        self.root_env = root_env
        self.budget = budget
        self.limits = limits
        self.resources = resources or ResourceManager()
        self._semaphore = asyncio.Semaphore(limits.max_concurrency)
        self._child_slot_held = contextvars.ContextVar("curagent_child_slot_held", default=False)
        self._id_lock = asyncio.Lock()
        self._next_id = 0

    async def execute_environment(
        self, env: Environment, access: AccessMode, call: ToolCall, expected_version: int
    ) -> ExecutionReceipt:
        if access not in {AccessMode.OWNER, AccessMode.DELEGATED, AccessMode.CLONE}:
            raise SchedulerError(f"{access.value} access cannot execute environment tools")
        if access == AccessMode.CLONE:
            return await env.execute(call, expected_version)
        return await self.resources.execute(env, call, expected_version)

    async def spawn_agent(
        self,
        raw_spec: Mapping[str, Any],
        *,
        parent_env: Environment,
        parent_id: str,
        parent_depth: int,
        parent_access: AccessMode,
        run_child: ChildRunner,
    ) -> SubagentResult:
        results = await self.spawn_agents(
            [raw_spec],
            parent_env=parent_env,
            parent_id=parent_id,
            parent_depth=parent_depth,
            parent_access=parent_access,
            run_child=run_child,
        )
        return results[0]

    async def spawn_agents(
        self,
        raw_specs: Sequence[Mapping[str, Any]],
        *,
        parent_env: Environment,
        parent_id: str,
        parent_depth: int,
        parent_access: AccessMode,
        run_child: ChildRunner,
    ) -> list[SubagentResult]:
        specs: list[SubagentSpec] = []
        try:
            if not raw_specs:
                raise SchedulerError("spawn_agents requires at least one spec")
            specs = [SubagentSpec.from_mapping(value) for value in raw_specs]
            self._validate_all(
                specs,
                parent_env=parent_env,
                parent_depth=parent_depth,
                parent_access=parent_access,
            )
            envs = await self._prepare_environments(specs, parent_env=parent_env)
            try:
                await self.budget.reserve_children(len(specs))
            except Exception:
                for spec, env in zip(specs, envs):
                    if spec.access == AccessMode.CLONE:
                        await env.close()
                raise
        except (ValueError, SchedulerError, BudgetExceeded) as exc:
            return self._preflight_failures(raw_specs, specs, parent_id, parent_depth + 1, str(exc))

        ids = await self._allocate_ids(parent_id, len(specs))

        async def launch(index: int) -> SubagentResult:
            spec = specs[index]
            env = envs[index]
            try:
                async with self._child_slot():
                    if spec.access == AccessMode.DELEGATED:
                        async with self.resources.writer_lease():
                            return await run_child(ids[index], spec, env, parent_depth + 1)
                    return await run_child(ids[index], spec, env, parent_depth + 1)
            except Exception as exc:
                return SubagentResult(
                    task=spec.task,
                    context=spec.context,
                    status=TerminalStatus.ERROR,
                    error=f"child runtime error: {exc}",
                    agent_id=ids[index],
                    parent_id=parent_id,
                    depth=parent_depth + 1,
                )
            finally:
                if spec.access == AccessMode.CLONE:
                    await env.close()

        async with self._yield_parent_slot():
            return list(await asyncio.gather(*(launch(index) for index in range(len(specs)))))

    @asynccontextmanager
    async def _child_slot(self):
        await self._semaphore.acquire()
        token = self._child_slot_held.set(True)
        try:
            yield
        finally:
            self._child_slot_held.reset(token)
            self._semaphore.release()

    @asynccontextmanager
    async def _yield_parent_slot(self):
        if not self._child_slot_held.get():
            yield
            return
        token = self._child_slot_held.set(False)
        self._semaphore.release()
        try:
            yield
        finally:
            await self._semaphore.acquire()
            self._child_slot_held.reset(token)

    def _validate_all(
        self,
        specs: Sequence[SubagentSpec],
        *,
        parent_env: Environment,
        parent_depth: int,
        parent_access: AccessMode,
    ) -> None:
        if parent_depth + 1 > self.limits.max_depth:
            raise SchedulerError(f"max_depth={self.limits.max_depth} would be exceeded")
        capabilities = parent_env.capabilities()
        for spec in specs:
            if spec.access == AccessMode.OWNER:
                raise SchedulerError("owner access is reserved for the root")
            if spec.access == AccessMode.READONLY and not capabilities.supports_readonly:
                raise SchedulerError("environment does not support readonly access")
            if spec.access == AccessMode.CLONE and not capabilities.supports_clone:
                raise SchedulerError("environment does not support clone access")
            if spec.access == AccessMode.DELEGATED:
                if not capabilities.mutable:
                    raise SchedulerError("environment does not support delegated writes")
                if parent_access == AccessMode.DELEGATED:
                    raise SchedulerError("a delegated child cannot re-delegate its writer lease")

    async def _prepare_environments(
        self, specs: Sequence[SubagentSpec], *, parent_env: Environment
    ) -> list[Environment]:
        environments: list[Environment] = []
        try:
            for spec in specs:
                if spec.access == AccessMode.CLONE:
                    clone = parent_env.clone()
                    if clone is None:
                        raise SchedulerError("environment advertised clone support but clone() returned None")
                    environments.append(clone)
                else:
                    environments.append(parent_env)
        except Exception:
            for spec, env in zip(specs, environments):
                if spec.access == AccessMode.CLONE:
                    await env.close()
            raise
        return environments

    async def _allocate_ids(self, parent_id: str, count: int) -> list[str]:
        async with self._id_lock:
            start = self._next_id + 1
            self._next_id += count
            return [f"{parent_id}.{index}" for index in range(start, start + count)]

    @staticmethod
    def _preflight_failures(
        raw_specs: Sequence[Mapping[str, Any]],
        parsed: Sequence[SubagentSpec],
        parent_id: str,
        depth: int,
        error: str,
    ) -> list[SubagentResult]:
        count = max(1, len(raw_specs))
        results = []
        for index in range(count):
            spec = parsed[index] if index < len(parsed) else None
            raw = raw_specs[index] if index < len(raw_specs) else {}
            task = spec.task if spec else str(raw.get("task") or "")
            context = spec.context if spec else raw.get("context")
            results.append(
                SubagentResult(
                    task=task,
                    context=context,
                    status=TerminalStatus.ERROR,
                    error=f"spawn preflight failed; no child started: {error}",
                    parent_id=parent_id,
                    depth=depth,
                )
            )
        return results
