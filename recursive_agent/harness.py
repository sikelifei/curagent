"""Environment-agnostic recursive CodeAct execution.

This module deliberately owns only the mechanics of recursive execution. An
environment supplies its prompt, observation, and capabilities; parent agents
choose the wording and context of delegated work.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Final

from .budget import SharedBudget
from .config import AgentLimits
from .exceptions import CancellationError, ConfigurationError, ModelCallError, TimeoutExceededError
from .repl import NodeTermination, ReplSession, find_repl_blocks
from .tools import CapabilityCollection, ToolInfo
from .types import (
    AgentResult,
    AgentStep,
    AgentTrace,
    EnvironmentStatus,
    ModelCallUsage,
    ModelResponse,
    ModelUsageSummary,
    UsageSummary,
)

_UNSET: Final = object()

MINIMAL_CODEACT_SYSTEM_PROMPT = """You are a generic CodeAct agent.
Use the current task, context, observation, and action space to make progress.
Only call capabilities listed in the action space.
""".strip()

_FRAMEWORK_DESCRIPTIONS = {
    "spawn_subagent": (
        "spawn_subagent(task: str, context=None) -> str. Run one "
        "delegated task sequentially and return its local result. Put objective, "
        "quantity, scope, restrictions, and return condition inside task. "
        "Call it directly; do not add await merely to make the recursion decision."
    ),
    "spawn_subagents": (
        "spawn_subagents(requests: list[dict]) -> list[str]. Run "
        "independent delegated tasks concurrently; results preserve request order. "
        "Put each request's objective, quantity, scope, restrictions, and return "
        "condition inside its task string. Call it directly; use sequential "
        "spawn_subagent calls when one branch depends on another."
    ),
    "finish": "Submit the root result and terminate the root agent.",
    "return_to_parent": "Return a local result to the direct parent and terminate this child.",
}


class _UsageAccumulator:
    """Thread-safe usage accounting shared by all nodes in one scheduler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelUsageSummary] = {}

    def add(self, usage: ModelCallUsage) -> None:
        with self._lock:
            current = self._models.setdefault(usage.model, ModelUsageSummary())
            current.total_calls += 1
            current.total_input_tokens += usage.input_tokens
            current.total_output_tokens += usage.output_tokens
            if usage.cost is not None:
                current.total_cost = (current.total_cost or 0.0) + usage.cost

    def snapshot(self) -> UsageSummary:
        with self._lock:
            return UsageSummary(
                model_usage_summaries={
                    name: ModelUsageSummary(
                        total_calls=value.total_calls,
                        total_input_tokens=value.total_input_tokens,
                        total_output_tokens=value.total_output_tokens,
                        total_cost=value.total_cost,
                    )
                    for name, value in self._models.items()
                }
            )


def _resolve_awaitable(value: Any) -> Any:
    """Resolve an optional awaitable from synchronous harness code."""
    if not inspect.isawaitable(value):
        return value

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(value))

    # AgentNode and model clients are synchronous. A private worker loop keeps
    # sync scheduler calls usable from tests or adapters already in an event loop.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, _await_value(value)).result()


async def _await_value(value: Any) -> Any:
    return await value


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    return str(value)


class _AwaitableString(str):
    """Keep legacy ``await spawn_subagent(...)`` calls source-compatible."""

    def __await__(self):
        async def resolve() -> str:
            return str(self)

        return resolve().__await__()


class _AwaitableList(list[str]):
    """Keep legacy ``await spawn_subagents(...)`` calls source-compatible."""

    def __await__(self):
        async def resolve() -> list[str]:
            return list(self)

        return resolve().__await__()


def _fit_head_and_tail(value: str, limit: int) -> tuple[str, bool]:
    if limit is None or len(value) <= limit:
        return value, False
    marker = f"\n... [truncated by harness: original_chars={len(value)}, limit_chars={limit}]\n"
    if limit <= len(marker):
        return value[:limit], True
    available = limit - len(marker)
    head = (available + 1) // 2
    tail = available - head
    return value[:head] + marker + (value[-tail:] if tail else ""), True


def compose_dynamic_prompt(
    *,
    task: str,
    context: Any,
    observation: Any,
    remaining_steps: int,
    capabilities: CapabilityCollection,
    execution_output: str | None = None,
) -> str:
    """Compose one node-local dynamic user turn from one capability collection."""
    sections: list[str] = []
    if execution_output is not None:
        sections.extend(["# Execution Output", execution_output])
    sections.extend(
        [
            "# Task",
            task,
            "",
            "# Context",
            _display_value(context),
            "",
            "# Observation",
            _display_value(observation),
            "",
            "# Remaining Steps",
            str(remaining_steps),
            "",
            "# Action Space",
            capabilities.format_for_prompt() or "(no capabilities available)",
            "",
            "Output exactly one <python>...</python> block. Do not include a thought or reasoning section.",
        ]
    )
    return "\n".join(sections)


# A second name is useful to callers that think of this as a prompt builder.
build_dynamic_prompt = compose_dynamic_prompt


class RecursiveScheduler:
    """Own the shared recursive runtime for one environment episode."""

    def __init__(
        self,
        environment: Any,
        model_client: Any | None = None,
        budget: SharedBudget | None = None,
        *,
        client: Any | None = None,
        backend: Any | None = None,
        limits: AgentLimits | None = None,
        max_total_steps: int | None = None,
        max_steps: int | None = None,
        max_depth: int | None = None,
        max_concurrent_subagents: int = 4,
        max_subagents_per_agent: int | None = None,
        max_run_seconds: float | None = None,
        max_observation_chars: int | None = 8000,
        max_repl_blocks_per_step: int | None = None,
        disabled_repl_builtins: frozenset[str] | set[str] | None = None,
        owns_model_client: bool = False,
    ) -> None:
        # Accept both natural positional orderings used by small adapters:
        # (environment, model_client, budget) and (environment, budget, client).
        if isinstance(model_client, SharedBudget) and budget is not None and not isinstance(
            budget, SharedBudget
        ):
            model_client, budget = budget, model_client
        supplied_clients = [
            value for value in (model_client, client, backend) if value is not None
        ]
        if len(supplied_clients) > 1:
            raise ConfigurationError("Pass only one of model_client, client, or backend")
        self.environment = environment
        self.model_client = supplied_clients[0] if supplied_clients else None
        if self.model_client is None:
            raise ConfigurationError("A shared model_client is required")

        if limits is not None and not isinstance(limits, AgentLimits):
            raise ConfigurationError("limits must be an AgentLimits instance")
        limit_steps = limits.max_total_steps if limits is not None else None
        limit_depth = limits.max_depth if limits is not None else None
        if max_total_steps is not None and max_steps is not None and max_total_steps != max_steps:
            raise ConfigurationError("max_total_steps and max_steps must agree")
        requested_steps = max_total_steps if max_total_steps is not None else max_steps
        if requested_steps is None:
            requested_steps = limit_steps if limit_steps is not None else 20
        elif limit_steps is not None and requested_steps != limit_steps:
            raise ConfigurationError("explicit step limit disagrees with limits")
        requested_depth = max_depth if max_depth is not None else (
            limit_depth if limit_depth is not None else 4
        )
        if max_depth is not None and limit_depth is not None and max_depth != limit_depth:
            raise ConfigurationError("explicit depth limit disagrees with limits")

        if budget is None:
            budget = SharedBudget(requested_steps)
        elif not isinstance(budget, SharedBudget):
            raise ConfigurationError("budget must be a SharedBudget instance")
        elif (
            (max_total_steps is not None or max_steps is not None)
            and budget.max_total_steps != requested_steps
        ):
            raise ConfigurationError("budget limit disagrees with max_total_steps")
        self.budget = budget
        self.shared_budget = budget
        self.max_depth = self._positive_or_zero_int(requested_depth, "max_depth")
        self.max_concurrent_subagents = self._positive_int(
            max_concurrent_subagents,
            "max_concurrent_subagents",
        )
        if max_subagents_per_agent is not None:
            max_subagents_per_agent = self._positive_int(
                max_subagents_per_agent,
                "max_subagents_per_agent",
            )
        self.max_subagents_per_agent = max_subagents_per_agent
        if max_run_seconds is not None and (
            isinstance(max_run_seconds, bool) or not isinstance(max_run_seconds, (int, float))
            or max_run_seconds <= 0
        ):
            raise ConfigurationError("max_run_seconds must be positive when provided")
        self.max_run_seconds = float(max_run_seconds) if max_run_seconds is not None else None
        if max_observation_chars is not None:
            max_observation_chars = self._positive_int(
                max_observation_chars,
                "max_observation_chars",
            )
        self.max_observation_chars = max_observation_chars
        if max_repl_blocks_per_step is not None:
            max_repl_blocks_per_step = self._positive_int(
                max_repl_blocks_per_step,
                "max_repl_blocks_per_step",
            )
        self.max_repl_blocks_per_step = max_repl_blocks_per_step
        self.disabled_repl_builtins = frozenset(disabled_repl_builtins or ())
        self.owns_model_client = owns_model_client
        self.client = self.model_client
        self.model = self.model_client

        self._usage = _UsageAccumulator()
        self._trace_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._started_at: float | None = None
        self._deadline: float | None = None
        self._active = False
        self.root: AgentNode | None = None
        self.nodes: list[AgentNode] = []
        self._closed = False

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _positive_or_zero_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"{name} must be a non-negative integer")
        return value

    @property
    def backend(self) -> Any:
        """Compatibility name for the one supplied model client."""
        return self.model_client

    @property
    def shared_environment(self) -> Any:
        return self.environment

    @property
    def root_node(self) -> "AgentNode | None":
        return self.root

    @property
    def usage(self) -> UsageSummary:
        return self._usage.snapshot()

    @property
    def global_usage(self) -> UsageSummary:
        return self.usage

    @property
    def trace_lock(self) -> threading.Lock:
        return self._trace_lock

    @property
    def consumed_steps(self) -> int:
        return self.budget.consumed_steps

    @property
    def steps(self) -> int:
        return self.consumed_steps

    @property
    def remaining_steps(self) -> int:
        return self.budget.remaining_steps

    @property
    def deadline(self) -> float | None:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation of the active tree."""
        self._cancel_event.set()

    request_cancel = cancel

    def close(self) -> None:
        """Close the shared model client only when the scheduler owns it."""
        if not self.owns_model_client or self._closed:
            return
        close = getattr(self.model_client, "close", None)
        if callable(close):
            close()
        self._closed = True

    def __enter__(self) -> "RecursiveScheduler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def run(
        self,
        task: str | None = None,
        context: Any = _UNSET,
        *,
        root_task: str | None = None,
        root_context: Any = _UNSET,
    ) -> AgentResult:
        """Create and run one root node using the shared environment."""
        if root_task is not None:
            if task is not None and task != root_task:
                raise ConfigurationError("task and root_task must agree")
            task = root_task
        if root_context is not _UNSET:
            if context is not _UNSET:
                raise ConfigurationError("Pass either context or root_context, not both")
            context = root_context

        if task is None:
            task = self._environment_value("task", default=None)
        if task is None or not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if context is _UNSET:
            context = self._environment_value("context", default=None)
        try:
            private_context = self._copy_context(context)
        except Exception as exc:
            raise ConfigurationError("Root context must support copy.deepcopy") from exc

        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("RecursiveScheduler is already running")
        with self._state_lock:
            self._active = True
            self._started_at = time.monotonic()
            self._deadline = (
                self._started_at + self.max_run_seconds
                if self.max_run_seconds is not None
                else None
            )
            self._cancel_event.clear()
            self.root = None
            self.nodes = []
        try:
            self._check_run()
            node = AgentNode(
                scheduler=self,
                task=task,
                context=private_context,
                depth=0,
                parent_id=None,
            )
            self.root = node
            with self._trace_lock:
                self.nodes.append(node)
            return node.run()
        finally:
            with self._state_lock:
                self._active = False
            self._run_lock.release()

    def create_node(
        self,
        *,
        task: str,
        context: Any = None,
        depth: int = 0,
        parent_id: str | None = None,
    ) -> "AgentNode":
        """Create a node with the scheduler's exact shared object identities."""
        self._validate_task(task)
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if depth > 0:
            try:
                context = self._copy_context(context)
            except Exception as exc:
                raise ConfigurationError("Child context must support copy.deepcopy") from exc
        return AgentNode(
            scheduler=self,
            task=task,
            context=context,
            depth=depth,
            parent_id=parent_id,
        )

    def _environment_value(self, name: str, *, default: Any) -> Any:
        try:
            value = getattr(self.environment, name)
        except AttributeError:
            return default
        return value() if callable(value) else value

    def _copy_context(self, context: Any) -> Any:
        # An explicit context may refer back to shared runtime objects. Preserve
        # those identities while still copying ordinary context data for the child.
        memo = {
            id(self.environment): self.environment,
            id(self): self,
            id(self.budget): self.budget,
            id(self.model_client): self.model_client,
        }
        return copy.deepcopy(context, memo)

    def _check_run(self) -> None:
        if self._cancel_event.is_set():
            raise CancellationError("Run was cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutExceededError(
                elapsed=time.monotonic() - (self._started_at or time.monotonic()),
                timeout=self.max_run_seconds or 0.0,
            )

    def _remaining_seconds(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def _system_prompt(self, *, is_root: bool = True) -> str:
        role_prompt = "root_prompt" if is_root else "child_prompt"
        use_role_specific_prompts = getattr(
            self.environment,
            "use_role_specific_prompts",
            False,
        )
        if callable(use_role_specific_prompts):
            use_role_specific_prompts = use_role_specific_prompts()
        prompt_names = (
            (role_prompt, "environment_system_prompt", "system_prompt")
            if use_role_specific_prompts
            else ("environment_system_prompt", "system_prompt")
        )
        for name in prompt_names:
            try:
                value = getattr(self.environment, name)
            except AttributeError:
                continue
            value = value() if callable(value) else value
            if value is not None:
                return str(value)
        return MINIMAL_CODEACT_SYSTEM_PROMPT

    def _observe(self) -> tuple[str, bool]:
        observe = getattr(self.environment, "observe", None)
        raw = observe() if callable(observe) else None
        raw = _resolve_awaitable(raw)
        value, truncated = _fit_head_and_tail(
            _display_value(raw),
            self.max_observation_chars,
        )
        return value, truncated

    def _environment_capabilities(self, node: "AgentNode") -> CapabilityCollection:
        result: Any = None
        hook = getattr(self.environment, "codeact_capabilities", None)
        if callable(hook):
            result = self._call_role_hook(hook, node)
        elif hook is not None:
            result = hook
        else:
            hook = getattr(self.environment, "capabilities", None)
            if callable(hook):
                result = hook()
            elif hook is not None:
                result = hook
            else:
                namespace = getattr(self.environment, "codeact_namespace", None)
                if callable(namespace):
                    result = self._call_role_hook(namespace, node)
                elif namespace is not None:
                    result = namespace
                else:
                    tools = getattr(self.environment, "tools", None)
                    result = tools() if callable(tools) else tools
        result = _resolve_awaitable(result)
        if result is None:
            return CapabilityCollection()
        if isinstance(result, CapabilityCollection):
            return result
        return CapabilityCollection(result)

    @staticmethod
    def _call_role_hook(hook: Any, node: "AgentNode") -> Any:
        try:
            parameters = inspect.signature(hook).parameters.values()
        except (TypeError, ValueError):
            return hook(is_root=node.is_root, depth=node.depth)
        if any(
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            for parameter in parameters
            if parameter.name in {"is_root", "depth"}
        ):
            return hook(node.is_root, node.depth)
        return hook(is_root=node.is_root, depth=node.depth)

    def _capabilities_for(self, node: "AgentNode") -> CapabilityCollection:
        environment_capabilities = self._environment_capabilities(node)
        framework: dict[str, ToolInfo] = {}
        if node.depth < self.max_depth:
            def spawn_subagent_capability(*args: Any, **kwargs: Any) -> Any:
                unsupported = sorted(set(kwargs) - {"task", "context"})
                if unsupported:
                    raise TypeError(
                        "spawn_subagent() valid signature is "
                        "spawn_subagent(task: str, context=None); put objective, "
                        "quantity, scope, restrictions, and return condition "
                        "inside task, not in keyword arguments"
                    )
                return node.spawn_subagent(*args, **kwargs)

            def spawn_subagents_capability(*args: Any, **kwargs: Any) -> Any:
                unsupported = sorted(set(kwargs) - {"requests"})
                if unsupported:
                    raise TypeError(
                        "spawn_subagents() valid signature is "
                        "spawn_subagents(requests: list[dict]); put objective, "
                        "quantity, scope, restrictions, and return condition "
                        "inside each request's task string, not in keyword arguments"
                    )
                return node.spawn_subagents(*args, **kwargs)

            framework["spawn_subagent"] = ToolInfo(
                "spawn_subagent",
                spawn_subagent_capability,
                _FRAMEWORK_DESCRIPTIONS["spawn_subagent"],
            )
            framework["spawn_subagents"] = ToolInfo(
                "spawn_subagents",
                spawn_subagents_capability,
                _FRAMEWORK_DESCRIPTIONS["spawn_subagents"],
            )
        if node.is_root:
            framework["finish"] = ToolInfo(
                "finish",
                node.finish,
                _FRAMEWORK_DESCRIPTIONS["finish"],
            )
        else:
            framework["return_to_parent"] = ToolInfo(
                "return_to_parent",
                node.return_to_parent,
                _FRAMEWORK_DESCRIPTIONS["return_to_parent"],
            )
        return environment_capabilities.merge_framework(framework)

    def _status(self) -> EnvironmentStatus:
        status = getattr(self.environment, "status", None)
        raw = status() if callable(status) else status
        raw = _resolve_awaitable(raw)
        if raw is None:
            return EnvironmentStatus(done=False)
        if isinstance(raw, EnvironmentStatus):
            return raw
        if isinstance(raw, Mapping):
            return EnvironmentStatus(
                done=bool(raw.get("done", False)),
                final_answer=raw.get("final_answer"),
                reason=raw.get("reason"),
            )
        if hasattr(raw, "done"):
            return EnvironmentStatus(
                done=bool(raw.done),
                final_answer=getattr(raw, "final_answer", None),
                reason=getattr(raw, "reason", None),
            )
        raise ConfigurationError(
            "environment status must be EnvironmentStatus, a mapping, or an object with done"
        )

    def _finalize_root(self, result: Any) -> Any:
        finalize = getattr(self.environment, "finalize_root", None)
        if not callable(finalize):
            finalize = getattr(self.environment, "finalize", None)
        if not callable(finalize):
            return result
        return _resolve_awaitable(finalize(result))

    def _generate(self, messages: list[dict[str, Any]]) -> ModelResponse:
        self._check_run()
        try:
            completion = self.model_client.completion
            try:
                signature = inspect.signature(completion)
                accepts_timeout = "timeout" in signature.parameters or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_timeout = True
            if accepts_timeout:
                raw = completion(messages, timeout=self._remaining_seconds())
            else:
                raw = completion(messages)
        except (CancellationError, TimeoutExceededError, ModelCallError):
            raise
        except Exception as exc:
            # A provider may report a timeout as an ordinary exception. Let
            # the scheduler's deadline/cancellation state win in that case.
            self._check_run()
            raise ModelCallError(f"Model call failed: {type(exc).__name__}: {exc}") from exc
        self._check_run()
        if isinstance(raw, ModelResponse):
            if not isinstance(raw.content, str) or not isinstance(raw.usage, ModelCallUsage):
                raise ModelCallError("Model client returned an invalid ModelResponse")
            return raw
        if isinstance(raw, str):
            return ModelResponse(
                content=raw,
                usage=ModelCallUsage(
                    model=str(getattr(self.model_client, "model_name", "unknown")),
                ),
            )
        raise ModelCallError(
            f"Model client returned unsupported response type: {type(raw).__name__}"
        )

    def _record_usage(self, usage: ModelCallUsage) -> None:
        self._usage.add(usage)

    @staticmethod
    def _validate_task(task: Any) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

    @classmethod
    def _normalize_requests(cls, requests: Any) -> list[dict[str, Any]]:
        if not isinstance(requests, list):
            raise TypeError("spawn_subagents requests must be a list")
        normalized: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError(f"spawn_subagents request {index} must be a dict")
            unknown = set(request) - {"task", "context"}
            if unknown:
                raise ValueError(
                    f"spawn_subagents request {index} has unsupported keys: {sorted(unknown)}"
                )
            try:
                cls._validate_task(request.get("task"))
            except ValueError as exc:
                raise ValueError(
                    f"spawn_subagents request {index} must have a non-empty task"
                ) from exc
            normalized.append(dict(request))
        return normalized

    def _register_child(self, parent: "AgentNode", child: "AgentNode") -> None:
        with self._trace_lock:
            parent.children.append(child)
            parent.trace.children.append(child.trace)

    def _run_child(
        self,
        parent: "AgentNode",
        task: str,
        context: Any,
    ) -> str:
        self._validate_task(task)
        if parent.depth >= self.max_depth:
            return f"Error: maximum recursion depth ({self.max_depth}) reached"
        try:
            child_context = self._copy_context(context)
        except Exception:
            return "Error: subagent context could not be copied"
        self._check_run()
        try:
            child = AgentNode(
                scheduler=self,
                task=task,
                context=child_context,
                depth=parent.depth + 1,
                parent_id=parent.agent_id,
            )
            self._register_child(parent, child)
            with self._trace_lock:
                self.nodes.append(child)
            result = child.run()
            return result.answer
        except (CancellationError, TimeoutExceededError):
            raise
        except ModelCallError:
            return "Error: subagent model call failed"
        except Exception:
            return "Error: subagent failed"

    def _spawn_one(self, parent: "AgentNode", task: str, context: Any) -> str:
        self._validate_task(task)
        if not parent._claim_spawn_slots(1):
            return (
                "Error: maximum direct subagents per agent "
                f"({self.max_subagents_per_agent}) reached"
            )
        return self._run_child(parent, task, context)

    def _spawn_many(self, parent: "AgentNode", requests: Any) -> list[str]:
        normalized = self._normalize_requests(requests)
        if not normalized:
            return []
        allowed = parent._claim_spawn_slots(len(normalized))
        selected = normalized[:allowed]
        results: list[str]
        if not selected:
            results = []
        elif len(selected) == 1:
            results = [
                self._run_child(
                    parent,
                    selected[0]["task"],
                    selected[0].get("context"),
                )
            ]
        else:
            executor = ThreadPoolExecutor(
                max_workers=min(self.max_concurrent_subagents, len(selected)),
                thread_name_prefix="recursive-agent",
            )
            futures: list[Future[str]] = []
            try:
                for request in selected:
                    self._check_run()
                    futures.append(
                        executor.submit(
                            self._run_child,
                            parent,
                            request["task"],
                            request.get("context"),
                        )
                    )
                # Reading futures in submission order preserves ordered results.
                results = [future.result() for future in futures]
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        if allowed < len(normalized):
            results.extend(
                [
                    "Error: maximum direct subagents per agent "
                    f"({self.max_subagents_per_agent}) reached"
                ]
                * (len(normalized) - allowed)
            )
        return results


class AgentNode:
    """One node-local task, history, REPL, and trace in a shared scheduler."""

    def __init__(
        self,
        *,
        scheduler: RecursiveScheduler,
        task: str,
        context: Any,
        depth: int,
        parent_id: str | None,
        agent_id: str | None = None,
    ) -> None:
        RecursiveScheduler._validate_task(task)
        self.scheduler = scheduler
        self.environment = scheduler.environment
        self.budget = scheduler.budget
        self.shared_budget = scheduler.budget
        self.model_client = scheduler.model_client
        self.client = scheduler.model_client
        self.model = scheduler.model_client
        self.agent_id = agent_id or uuid.uuid4().hex
        self.parent_id = parent_id
        self.depth = depth
        self.task = task
        self.context = context
        self.is_root = parent_id is None
        self.trace = AgentTrace(
            agent_id=self.agent_id,
            parent_id=parent_id,
            depth=depth,
            task=task,
        )
        self.children: list[AgentNode] = []
        self.messages: list[dict[str, Any]] = []
        self.history = self.messages
        self._local_usage = _UsageAccumulator()
        self._spawn_lock = threading.Lock()
        self._direct_spawned = 0
        self._last_response: str | None = None
        self._system_prompt = scheduler._system_prompt(is_root=self.is_root)
        self.trace.system_prompt = self._system_prompt
        self.capabilities = scheduler._capabilities_for(self)
        self.repl = ReplSession(
            context=context,
            capabilities=self.capabilities,
            disabled_builtins=scheduler.disabled_repl_builtins,
        )
        self.session = self.repl
        self.repl_session = self.repl

    @property
    def namespace(self) -> dict[str, Any]:
        return self.repl.namespace

    @property
    def local_usage(self) -> UsageSummary:
        return self._local_usage.snapshot()

    def _refresh_capabilities(self) -> None:
        self.capabilities = self.scheduler._capabilities_for(self)
        self.repl.bind_capabilities(self.capabilities)

    def _claim_spawn_slots(self, requested: int) -> int:
        with self._spawn_lock:
            if self.scheduler.max_subagents_per_agent is None:
                self._direct_spawned += requested
                return requested
            remaining = max(
                0,
                self.scheduler.max_subagents_per_agent - self._direct_spawned,
            )
            allowed = min(requested, remaining)
            self._direct_spawned += allowed
            return allowed

    def spawn_subagent(self, task: str, context: Any = None) -> str:
        """Run one child sequentially and return its local result."""
        return _AwaitableString(self.scheduler._spawn_one(self, task, context))

    def spawn_subagents(self, requests: list[dict[str, Any]]) -> list[str]:
        """Run independent children concurrently and preserve request order."""
        return _AwaitableList(self.scheduler._spawn_many(self, requests))

    def finish(self, result: Any = None) -> None:
        if not self.is_root:
            raise ValueError("finish is only available to the root node")
        finalized = self.scheduler._finalize_root(result)
        raise NodeTermination("finish", finalized)

    def return_to_parent(self, result: Any = None) -> None:
        if self.is_root:
            raise ValueError("return_to_parent is only available to child nodes")
        raise NodeTermination("return_to_parent", result)

    def _complete(
        self,
        *,
        answer: Any,
        status: str,
        started: float,
    ) -> AgentResult:
        answer_text = "" if answer is None else _display_value(answer)
        self.trace.status = status  # type: ignore[assignment]
        self.trace.answer = answer_text
        self.trace.usage = self._local_usage.snapshot()
        self.trace.duration_seconds = time.perf_counter() - started
        return AgentResult(
            answer=answer_text,
            status=status,  # type: ignore[arg-type]
            steps=self.scheduler.budget.consumed_steps,
            usage=self.scheduler.usage,
            trace=self.trace,
        )

    def _record_failure(self, exc: BaseException, started: float) -> None:
        self.trace.error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, CancellationError):
            self.trace.status = "cancelled"
        elif isinstance(exc, TimeoutExceededError):
            self.trace.status = "timeout"
        else:
            self.trace.status = "error"
        self.trace.usage = self._local_usage.snapshot()
        self.trace.duration_seconds = time.perf_counter() - started

    def run(self) -> AgentResult:
        """Run the generic observe/prompt/generate/execute loop."""
        started = time.perf_counter()
        self.messages.append({"role": "system", "content": self._system_prompt})
        first_turn = True
        execution_output: str | None = None
        try:
            while True:
                self.scheduler._check_run()
                observation, observation_truncated = self.scheduler._observe()
                self._refresh_capabilities()

                status = self.scheduler._status()
                if status.done:
                    answer = status.final_answer
                    if answer is None:
                        answer = self._last_response or ""
                    return self._complete(
                        answer=answer,
                        status="environment_done",
                        started=started,
                    )

                user_turn = compose_dynamic_prompt(
                    task=self.task,
                    context=self.context,
                    observation=observation,
                    remaining_steps=self.scheduler.budget.remaining_steps,
                    capabilities=self.capabilities,
                    execution_output=None if first_turn else execution_output,
                )
                self.messages.append({"role": "user", "content": user_turn})
                first_turn = False

                reservation = self.budget.reserve()
                if reservation is None:
                    return self._complete(
                        answer=self._last_response or "",
                        status="budget_exhausted",
                        started=started,
                    )
                try:
                    response = self.scheduler._generate(self.messages)
                    self.scheduler._check_run()
                except BaseException:
                    if reservation.active:
                        reservation.release()
                    raise
                reservation.commit()
                self.scheduler._record_usage(response.usage)
                self._local_usage.add(response.usage)
                self._last_response = response.content
                self.messages.append({"role": "assistant", "content": response.content})

                step = AgentStep(
                    number=len(self.trace.steps) + 1,
                    response=response.content,
                    model_observation=observation,
                    observation_truncated=observation_truncated,
                )
                self.trace.steps.append(step)

                blocks = find_repl_blocks(response.content)
                if self.scheduler.max_repl_blocks_per_step is not None:
                    blocks = blocks[: self.scheduler.max_repl_blocks_per_step]
                if not blocks:
                    execution_output = (
                        "Format error: the response contained no executable Python block. "
                        "Continue with exactly one <python>...</python> block."
                    )
                    step.model_observation = execution_output
                    self.scheduler._check_run()
                    continue

                outputs: list[str] = []
                errors: list[str] = []
                terminated: NodeTermination | None = None
                for code in blocks:
                    execution = self.repl.execute(code)
                    step.code_executions.append(execution.trace)
                    outputs.append(execution.trace.output or "(no output)")
                    if execution.trace.error:
                        errors.append(execution.trace.error)
                    if execution.termination is not None:
                        terminated = execution.termination
                        break
                    self.scheduler._check_run()

                    status = self.scheduler._status()
                    if status.done:
                        answer = status.final_answer
                        if answer is None:
                            answer = self._last_response or ""
                        return self._complete(
                            answer=answer,
                            status="environment_done",
                            started=started,
                        )

                execution_output, execution_truncated = _fit_head_and_tail(
                    self._execution_feedback(outputs, errors),
                    self.scheduler.max_observation_chars,
                )
                step.model_observation = execution_output
                step.observation_truncated = (
                    step.observation_truncated or execution_truncated
                )
                if terminated is not None:
                    if terminated.kind in {"finish", "return_to_parent"}:
                        return self._complete(
                            answer=terminated.result,
                            status="completed",
                            started=started,
                        )
                    return self._complete(
                        answer=terminated.result,
                        status="completed",
                        started=started,
                    )
                self.scheduler._check_run()
        except BaseException as exc:
            self._record_failure(exc, started)
            raise

    @staticmethod
    def _execution_feedback(outputs: list[str], errors: list[str]) -> str:
        body = "\n\n".join(outputs or ["(no output)"])
        if errors:
            body += "\n\nExecution errors:\n" + "\n".join(dict.fromkeys(errors))
        return body


__all__ = [
    "AgentNode",
    "MINIMAL_CODEACT_SYSTEM_PROMPT",
    "RecursiveScheduler",
    "build_dynamic_prompt",
    "compose_dynamic_prompt",
]
