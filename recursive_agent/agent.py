"""Unified recursive-agent lifecycle and child execution."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from . import clients
from .clients.base import ModelClient
from .config import AgentConfig, load_model_config
from .exceptions import (
    CancellationError,
    ConfigurationError,
    ModelCallError,
    TimeoutExceededError,
)
from .prompts import (
    DEFAULT_ANSWER_COMPLETION_PROMPT,
    FORCED_FINAL_USER,
    build_initial_user,
    build_system_prompt,
)
from .repl import ReplSession, find_repl_blocks
from .tools import ToolInfo, format_tools_for_prompt, parse_tools, tool_values
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

ClientFactory = Callable[[str, dict[str, Any]], ModelClient]
TerminationCheck = Callable[[], EnvironmentStatus | dict[str, Any] | Any]


class _UsageAccumulator:
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


@dataclass
class _RunContext:
    run_id: str
    started_at: float
    max_run_seconds: float | None
    cancellation: threading.Event = field(default_factory=threading.Event)
    usage: _UsageAccumulator = field(default_factory=_UsageAccumulator)
    trace_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def deadline(self) -> float | None:
        if self.max_run_seconds is None:
            return None
        return self.started_at + self.max_run_seconds

    def remaining_seconds(self) -> float | None:
        deadline = self.deadline
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def check(self) -> None:
        if self.cancellation.is_set():
            raise CancellationError("Run was cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutExceededError(
                elapsed=time.monotonic() - self.started_at,
                timeout=self.max_run_seconds or 0.0,
            )


class RecursiveAgent:
    """A general agent whose parent and children run the same REPL loop."""

    def __init__(
        self,
        backend: str = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        tools: dict[str, Any] | None = None,
        max_steps: int = 20,
        max_depth: int = 4,
        max_concurrent_subagents: int = 4,
        max_subagents_per_agent: int | None = None,
        max_run_seconds: float | None = None,
        max_observation_chars: int | None = 8000,
        max_repl_blocks_per_step: int | None = None,
        termination_check: TerminationCheck | None = None,
        prompt_addendum: str | None = None,
        system_prompt: str | None = None,
        completion_prompt: str | None = None,
        forced_final_prompt: str | None = None,
        delegated_forced_final_prompt: str | None = None,
        delegated_task_prompt: str | None = None,
        delegated_prompt_addendum: str | None = None,
        delegated_completion_prompt: str | None = None,
        delegated_disabled_tools: frozenset[str] | set[str] | None = None,
        disabled_repl_builtins: frozenset[str] | set[str] | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.config = AgentConfig(
            backend=backend,
            backend_kwargs=dict(backend_kwargs or {}),
            max_steps=max_steps,
            max_depth=max_depth,
            max_concurrent_subagents=max_concurrent_subagents,
            max_subagents_per_agent=max_subagents_per_agent,
            max_run_seconds=max_run_seconds,
            max_observation_chars=max_observation_chars,
        )
        self._tools: dict[str, ToolInfo] = parse_tools(tools)
        self._tool_values = tool_values(self._tools)
        self._formatted_tools = format_tools_for_prompt(self._tools)
        self._prompt_addendum = str(prompt_addendum).strip() if prompt_addendum else None
        self._system_prompt_override = (
            str(system_prompt).strip() if system_prompt else None
        )
        self._forced_final_prompt = (
            str(forced_final_prompt).strip() if forced_final_prompt else None
        )
        self._completion_prompt = (
            str(completion_prompt).strip()
            if completion_prompt is not None
            else DEFAULT_ANSWER_COMPLETION_PROMPT
        )
        self._delegated_forced_final_prompt = (
            str(delegated_forced_final_prompt).strip()
            if delegated_forced_final_prompt
            else None
        )
        self._delegated_task_prompt = (
            str(delegated_task_prompt).strip() if delegated_task_prompt else None
        )
        self._delegated_prompt_addendum = (
            str(delegated_prompt_addendum).strip()
            if delegated_prompt_addendum
            else None
        )
        self._delegated_completion_prompt = (
            str(delegated_completion_prompt).strip()
            if delegated_completion_prompt is not None
            else DEFAULT_ANSWER_COMPLETION_PROMPT
        )
        self._delegated_disabled_tools = frozenset(delegated_disabled_tools or ())
        if max_repl_blocks_per_step is not None and (
            not isinstance(max_repl_blocks_per_step, int)
            or max_repl_blocks_per_step <= 0
        ):
            raise ConfigurationError(
                "max_repl_blocks_per_step must be a positive integer or None"
            )
        self._max_repl_blocks_per_step = max_repl_blocks_per_step
        self._system_prompt = build_system_prompt(
            self._formatted_tools,
            prompt_addendum=self._prompt_addendum,
            base_prompt=self._system_prompt_override,
            completion_prompt=self._completion_prompt,
        )
        self._termination_check = termination_check
        self._disabled_repl_builtins = frozenset(disabled_repl_builtins or ())
        self._client_factory = client_factory or clients.get_client
        self._active_lock = threading.Lock()
        self._active_runs: dict[str, _RunContext] = {}

    @classmethod
    def from_config(cls, path: str, **kwargs: Any) -> "RecursiveAgent":
        backend, backend_kwargs = load_model_config(path)
        return cls(backend=backend, backend_kwargs=backend_kwargs, **kwargs)

    @property
    def system_prompt(self) -> str:
        """Return the exact system prompt used for the top-level run."""
        return self._system_prompt

    def cancel(self) -> None:
        """Cooperatively cancel all runs currently active on this instance."""
        with self._active_lock:
            contexts = list(self._active_runs.values())
        for context in contexts:
            context.cancellation.set()

    def run(self, task: str, context: Any | None = None) -> AgentResult:
        self._validate_task(task)
        try:
            private_context = copy.deepcopy(context)
        except Exception as exc:
            raise ConfigurationError("Root context must support copy.deepcopy") from exc

        run_context = _RunContext(
            run_id=uuid.uuid4().hex,
            started_at=time.monotonic(),
            max_run_seconds=self.config.max_run_seconds,
        )
        with self._active_lock:
            self._active_runs[run_context.run_id] = run_context
        try:
            return self._run_agent(
                task=task,
                context=private_context,
                depth=0,
                parent_trace=None,
                run_context=run_context,
            )
        except KeyboardInterrupt:
            run_context.cancellation.set()
            raise CancellationError("Run interrupted by user") from None
        finally:
            with self._active_lock:
                self._active_runs.pop(run_context.run_id, None)

    def _run_agent(
        self,
        *,
        task: str,
        context: Any,
        depth: int,
        parent_trace: AgentTrace | None,
        run_context: _RunContext,
    ) -> AgentResult:
        run_context.check()
        trace = AgentTrace(
            agent_id=uuid.uuid4().hex,
            parent_id=parent_trace.agent_id if parent_trace else None,
            depth=depth,
            task=task,
        )
        if parent_trace is not None:
            with run_context.trace_lock:
                parent_trace.children.append(trace)

        started = time.perf_counter()
        client = self._make_client()
        delegated = parent_trace is not None
        active_tools = {
            name: info
            for name, info in self._tools.items()
            if not (delegated and name in self._delegated_disabled_tools)
        }
        system_prompt = build_system_prompt(
            format_tools_for_prompt(active_tools),
            prompt_addendum=(
                self._delegated_prompt_addendum
                if delegated and self._delegated_prompt_addendum is not None
                else self._prompt_addendum
            ),
            base_prompt=self._system_prompt_override,
            completion_prompt=(
                self._delegated_completion_prompt
                if delegated
                else self._completion_prompt
            ),
        )
        trace.system_prompt = system_prompt
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_initial_user(
                    task,
                    delegated=parent_trace is not None,
                    delegated_guidance=(
                        self._delegated_task_prompt
                        if delegated
                        else None
                    ),
                ),
            },
        ]
        latest_response: str | None = None
        local_usage = _UsageAccumulator()
        direct_children_spawned = 0

        def reserve_child_slots(requested: int) -> int:
            nonlocal direct_children_spawned
            limit = self.config.max_subagents_per_agent
            if limit is None:
                direct_children_spawned += requested
                return requested
            allowed = min(requested, max(0, limit - direct_children_spawned))
            direct_children_spawned += allowed
            return allowed

        def spawn_one(task: str, context: Any | None = None) -> str:
            self._validate_task(task)
            if depth < self.config.max_depth and reserve_child_slots(1) == 0:
                return (
                    "Error: maximum direct subagents per agent "
                    f"({self.config.max_subagents_per_agent}) reached"
                )
            return self._spawn_child(
                task=task,
                context=context,
                depth=depth,
                parent_trace=trace,
                run_context=run_context,
            )

        def spawn_many(requests: list[dict[str, Any]]) -> list[str]:
            normalized = self._validate_requests(requests)
            if depth >= self.config.max_depth:
                return self._spawn_children(
                    requests=normalized,
                    depth=depth,
                    parent_trace=trace,
                    run_context=run_context,
                )
            allowed = reserve_child_slots(len(normalized))
            results = self._spawn_children(
                requests=normalized[:allowed],
                depth=depth,
                parent_trace=trace,
                run_context=run_context,
            )
            error = (
                "Error: maximum direct subagents per agent "
                f"({self.config.max_subagents_per_agent}) reached"
            )
            return results + [error] * (len(normalized) - allowed)

        repl = ReplSession(
            context=context,
            tools=tool_values(active_tools),
            spawn_subagent=spawn_one,
            spawn_subagents=spawn_many,
            disabled_builtins=self._disabled_repl_builtins,
        )

        try:
            for step_number in range(1, self.config.max_steps + 1):
                step_started = time.perf_counter()
                response = self._call_model(
                    client,
                    messages,
                    run_context,
                    latest_response,
                    local_usage,
                )
                latest_response = response
                messages.append({"role": "assistant", "content": response})
                step_trace = AgentStep(number=step_number, response=response)
                trace.steps.append(step_trace)

                code_blocks = find_repl_blocks(response)
                if self._max_repl_blocks_per_step is not None:
                    code_blocks = code_blocks[: self._max_repl_blocks_per_step]
                if not code_blocks:
                    messages.append({"role": "user", "content": "Continue."})
                    step_trace.duration_seconds = time.perf_counter() - step_started
                    continue

                observations: list[str] = []
                observation_errors: list[str] = []
                for code in code_blocks:
                    execution = repl.execute(code)
                    step_trace.code_executions.append(execution.trace)
                    observations.append(execution.trace.output or "(no output)")
                    if execution.trace.error:
                        observation_errors.append(execution.trace.error)
                    run_context.check()

                    if execution.trace.error:
                        break

                    if execution.answer_ready:
                        step_trace.duration_seconds = time.perf_counter() - step_started
                        return self._finish(
                            answer=execution.answer_content or "",
                            status="completed",
                            steps=step_number,
                            trace=trace,
                            started=started,
                            run_context=run_context,
                            local_usage=local_usage,
                        )

                    environment = self._environment_status()
                    run_context.check()
                    if environment.done:
                        step_trace.duration_seconds = time.perf_counter() - step_started
                        if environment.final_answer is not None:
                            return self._finish(
                                answer=str(environment.final_answer),
                                status="environment_done",
                                steps=step_number,
                                trace=trace,
                                started=started,
                                run_context=run_context,
                                local_usage=local_usage,
                            )
                        model_observation, was_truncated = _format_observations(
                            observations,
                            errors=observation_errors,
                            max_chars=self.config.max_observation_chars,
                        )
                        step_trace.model_observation = model_observation
                        step_trace.observation_truncated = was_truncated
                        messages.append(
                            {"role": "user", "content": model_observation}
                        )
                        return self._forced_final(
                            client=client,
                            messages=messages,
                            steps=step_number,
                            trace=trace,
                            started=started,
                            run_context=run_context,
                            latest_response=latest_response,
                            local_usage=local_usage,
                        )

                model_observation, was_truncated = _format_observations(
                    observations,
                    errors=observation_errors,
                    max_chars=self.config.max_observation_chars,
                )
                step_trace.model_observation = model_observation
                step_trace.observation_truncated = was_truncated
                messages.append({"role": "user", "content": model_observation})
                step_trace.duration_seconds = time.perf_counter() - step_started

            return self._forced_final(
                client=client,
                messages=messages,
                steps=self.config.max_steps,
                trace=trace,
                started=started,
                run_context=run_context,
                latest_response=latest_response,
                local_usage=local_usage,
            )
        except BaseException as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.duration_seconds = time.perf_counter() - started
            trace.usage = local_usage.snapshot()
            if depth == 0 or getattr(exc, "agent_trace", None) is None:
                exc.agent_trace = trace
                exc.usage = run_context.usage.snapshot()
            raise
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _forced_final(
        self,
        *,
        client: ModelClient,
        messages: list[dict[str, str]],
        steps: int,
        trace: AgentTrace,
        started: float,
        run_context: _RunContext,
        latest_response: str | None,
        local_usage: _UsageAccumulator,
    ) -> AgentResult:
        forced_final_prompt = self._forced_final_prompt
        if trace.parent_id is not None and self._delegated_forced_final_prompt:
            forced_final_prompt = self._delegated_forced_final_prompt
        messages.append(
            {
                "role": "user",
                "content": forced_final_prompt or FORCED_FINAL_USER,
            }
        )
        answer = self._call_model(
            client,
            messages,
            run_context,
            latest_response,
            local_usage,
        )
        trace.forced_final_response = answer
        return self._finish(
            answer=answer,
            status="forced_final",
            steps=steps,
            trace=trace,
            started=started,
            run_context=run_context,
            local_usage=local_usage,
        )

    def _call_model(
        self,
        client: ModelClient,
        messages: list[dict[str, str]],
        run_context: _RunContext,
        latest_response: str | None,
        local_usage: _UsageAccumulator,
    ) -> str:
        run_context.check()
        try:
            raw_response = client.completion(
                messages,
                timeout=run_context.remaining_seconds(),
            )
        except (CancellationError, TimeoutExceededError, ModelCallError):
            raise
        except Exception as exc:
            run_context.check()
            raise ModelCallError(
                f"Model call failed: {type(exc).__name__}: {exc}",
                last_response=latest_response,
            ) from exc
        run_context.check()

        if isinstance(raw_response, ModelResponse):
            response = raw_response
        elif isinstance(raw_response, str):
            response = ModelResponse(
                content=raw_response,
                usage=ModelCallUsage(model=getattr(client, "model_name", "unknown")),
            )
        else:
            raise ModelCallError(
                f"Model client returned unsupported response type: {type(raw_response).__name__}",
                last_response=latest_response,
            )
        run_context.usage.add(response.usage)
        local_usage.add(response.usage)
        return response.content

    def _spawn_child(
        self,
        *,
        task: str,
        context: Any,
        depth: int,
        parent_trace: AgentTrace,
        run_context: _RunContext,
    ) -> str:
        self._validate_task(task)
        if depth >= self.config.max_depth:
            return f"Error: maximum recursion depth ({self.config.max_depth}) reached"
        try:
            child_context = copy.deepcopy(context)
        except Exception:
            return "Error: subagent context could not be copied"

        try:
            run_context.check()
            result = self._run_agent(
                task=task,
                context=child_context,
                depth=depth + 1,
                parent_trace=parent_trace,
                run_context=run_context,
            )
            return result.answer
        except TimeoutExceededError:
            return "Error: subagent timed out"
        except CancellationError:
            return "Error: subagent cancelled"
        except ModelCallError:
            return "Error: subagent model call failed"
        except Exception:
            return "Error: subagent failed"

    def _spawn_children(
        self,
        *,
        requests: list[dict[str, Any]],
        depth: int,
        parent_trace: AgentTrace,
        run_context: _RunContext,
    ) -> list[str]:
        normalized = self._validate_requests(requests)
        if not normalized:
            return []
        if depth >= self.config.max_depth:
            error = f"Error: maximum recursion depth ({self.config.max_depth}) reached"
            return [error for _ in normalized]

        executor = ThreadPoolExecutor(
            max_workers=min(self.config.max_concurrent_subagents, len(normalized)),
            thread_name_prefix="recursive-agent",
        )
        futures: list[Future[str]] = []
        try:
            for request in normalized:
                run_context.check()
                futures.append(
                    executor.submit(
                        self._spawn_child,
                        task=request["task"],
                        context=request.get("context"),
                        depth=depth,
                        parent_trace=parent_trace,
                        run_context=run_context,
                    )
                )
            results = [future.result() for future in futures]
            run_context.check()
            return results
        finally:
            if run_context.cancellation.is_set() or (
                run_context.deadline is not None and time.monotonic() >= run_context.deadline
            ):
                for future in futures:
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _environment_status(self) -> EnvironmentStatus:
        if self._termination_check is None:
            return EnvironmentStatus(done=False)
        raw = self._termination_check()
        if raw is None:
            return EnvironmentStatus(done=False)
        if isinstance(raw, EnvironmentStatus):
            return raw
        if isinstance(raw, dict):
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
            "termination_check must return EnvironmentStatus, a mapping, an object with done, or None"
        )

    def _finish(
        self,
        *,
        answer: str,
        status: str,
        steps: int,
        trace: AgentTrace,
        started: float,
        run_context: _RunContext,
        local_usage: _UsageAccumulator,
    ) -> AgentResult:
        trace.status = status  # type: ignore[assignment]
        trace.answer = answer
        trace.usage = local_usage.snapshot()
        trace.duration_seconds = time.perf_counter() - started
        return AgentResult(
            answer=answer,
            status=status,  # type: ignore[arg-type]
            steps=steps,
            usage=run_context.usage.snapshot(),
            trace=trace,
        )

    def _make_client(self) -> ModelClient:
        return self._client_factory(self.config.backend, dict(self.config.backend_kwargs))

    @staticmethod
    def _validate_task(task: Any) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

    @classmethod
    def _validate_requests(cls, requests: Any) -> list[dict[str, Any]]:
        if not isinstance(requests, list):
            raise TypeError("spawn_subagents requests must be a list")
        normalized = []
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
            normalized.append(request)
        return normalized


def _format_observations(
    observations: list[str],
    *,
    errors: list[str] | None = None,
    max_chars: int | None = 8000,
) -> tuple[str, bool]:
    prefix = "REPL output:\n"
    body = "\n\n".join(observations or ["(no output)"])
    message = prefix + body
    if max_chars is None or len(message) <= max_chars:
        return message, False

    marker = (
        f"[truncated by harness: original_chars={len(message)}, "
        f"limit_chars={max_chars}]\n"
    )
    fixed = prefix + marker
    available = max(0, max_chars - len(fixed))
    unique_errors = list(dict.fromkeys(error for error in (errors or []) if error))
    visible_source = body
    for error in unique_errors:
        visible_source = visible_source.replace(error, "")
    visible_source = visible_source.rstrip()

    error_section = ""
    if unique_errors and available:
        raw_errors = "\n\nPreserved execution errors:\n" + "\n".join(unique_errors)
        error_budget = min(len(raw_errors), max(1, available // 2))
        error_section = _fit_head_and_tail(raw_errors, error_budget)

    body_budget = max(0, available - len(error_section))
    visible_body = _fit_head_and_tail(visible_source, body_budget)
    result = fixed + visible_body + error_section
    return result[:max_chars], True


def _fit_head_and_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    separator = "\n... [truncated] ...\n"
    if max_chars <= len(separator):
        return text[:max_chars]
    content_chars = max_chars - len(separator)
    head_chars = (content_chars + 1) // 2
    tail_chars = content_chars - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + separator + tail
