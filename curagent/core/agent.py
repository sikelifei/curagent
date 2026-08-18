"""One thin execution loop shared by root and every child node."""

from __future__ import annotations

from typing import Any

from curagent.core.budget import SharedBudget
from curagent.core.errors import BudgetExceeded
from curagent.core.model import ToolCallingModel
from curagent.core.prompt import compose_prompt
from curagent.core.scheduler import RecursiveScheduler
from curagent.core.tools import framework_tool_schemas, strict_parse_response, validate_tool_call
from curagent.core.trace import DecisionTrace, TraceRecorder
from curagent.core.types import (
    AgentLimits,
    ModelResponse,
    SubagentResult,
    SubagentSpec,
    ToolCall,
    ToolSchema,
    require_jsonable,
)
from curagent.environments.base import Environment
from curagent.executors.python import PythonExecutor


class AgentNode:
    """Run decisions until finish, environment termination, or shared exhaustion."""

    def __init__(
        self,
        *,
        agent_id: str,
        task: str,
        context: Any,
        model: ToolCallingModel,
        environment: Environment | None = None,
        expected_output: str | None = None,
        limits: AgentLimits | None = None,
        budget: SharedBudget | None = None,
        trace: TraceRecorder | None = None,
        scheduler: RecursiveScheduler | None = None,
        parent_id: str | None = None,
        depth: int = 0,
        python_executor: PythonExecutor | None = None,
    ) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        require_jsonable(context, label="node context")
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.depth = depth
        self.task = task
        self.context = context
        self.expected_output = expected_output
        self.environment = environment
        self.model = model
        self.limits = limits or AgentLimits()
        self.budget = budget or SharedBudget(self.limits)
        self.trace = trace or TraceRecorder()
        self.scheduler = scheduler or RecursiveScheduler(limits=self.limits)
        self.python_executor = python_executor
        self._attempt = 0

    async def run(self) -> SubagentResult:
        while True:
            if await self._is_done():
                return await self._environment_result()

            try:
                reservation = await self.budget.reserve_step()
            except BudgetExceeded:
                return SubagentResult(
                    error=f"task tree reached max_total_steps={self.limits.max_total_steps}"
                )

            try:
                observation = await self.environment.observe() if self.environment else None
                schemas = self._available_tools()
                snapshot = await self.budget.snapshot()
                prompt = compose_prompt(
                    task=self.task,
                    context=self.context,
                    trajectory=self.trace.for_prompt(self.agent_id),
                    observation=observation,
                    tools=schemas,
                    remaining_steps=snapshot.remaining_steps,
                )
            except Exception as exc:
                await self.budget.release_step(reservation)
                self._runtime_error("prompt", exc)
                return SubagentResult(error=self._message(exc))

            response: ModelResponse
            try:
                generated = await self.model.generate(prompt, schemas)
                if isinstance(generated, ModelResponse):
                    response = generated
                else:
                    # A provider returned content outside the adapter contract. It is
                    # still a model output and therefore consumes this shared step.
                    response = ModelResponse(raw_response=generated, protocol="json")
            except Exception as exc:
                if not self._exception_has_output(exc):
                    await self.budget.release_step(reservation)
                    self._runtime_error("model_no_response", exc)
                    return SubagentResult(error=self._message(exc))
                raw_response = getattr(exc, "raw_response", None)
                if raw_response is None and bool(getattr(exc, "has_output", False)):
                    raw_response = self._message(exc)
                response = ModelResponse(
                    raw_response=raw_response,
                    tool_calls=getattr(exc, "tool_calls", ()),
                    protocol=getattr(exc, "protocol", "json"),
                )

            if not self._response_has_output(response):
                await self.budget.release_step(reservation)
                self._runtime_error("model_empty_output", RuntimeError("model returned no output"))
                return SubagentResult()

            await self.budget.commit_step(reservation)
            self._attempt += 1
            call: ToolCall | None = None
            execution_result: Any
            terminal = False
            try:
                call = strict_parse_response(response)
                schema = validate_tool_call(call, schemas)
                execution_result, terminal = await self._execute(call, schema)
            except Exception as exc:
                # Parsing, schema, and environment failures are ordinary feedback.
                execution_result = self._message(exc)

            self.trace.record(
                DecisionTrace(
                    agent_id=self.agent_id,
                    parent_id=self.parent_id,
                    depth=self.depth,
                    step=reservation,
                    attempt=self._attempt,
                    prompt=prompt,
                    observation=observation,
                    raw_model_output=response.raw_response,
                    parsed_tool_call=call,
                    execution_result=execution_result,
                )
            )
            if terminal:
                return SubagentResult(result=execution_result)

    def _available_tools(self) -> list[ToolSchema]:
        schemas = framework_tool_schemas(python_enabled=self.python_executor is not None)
        if self.environment is not None:
            schemas.extend(self.environment.tools())
        names = [schema.name for schema in schemas]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"tool names must be unique: {duplicates}")
        return schemas

    async def _execute(self, call: ToolCall, schema: ToolSchema) -> tuple[Any, bool]:
        if call.name == "finish":
            result = call.arguments["result"]
            require_jsonable(result, label="finish result")
            return result, True
        if call.name == "spawn_agent":
            child = await self.scheduler.spawn_agent(
                call.arguments,
                parent_env=self.environment,
                parent_id=self.agent_id,
                parent_depth=self.depth,
                run_child=self._run_child,
            )
            return child.to_dict(), False
        if call.name == "spawn_agents":
            children = await self.scheduler.spawn_agents(
                call.arguments["specs"],
                parent_env=self.environment,
                parent_id=self.agent_id,
                parent_depth=self.depth,
                run_child=self._run_child,
            )
            return [child.to_dict() for child in children], False
        if call.name == "python_exec":
            if self.python_executor is None:
                return "python_exec is not enabled", False
            return await self.python_executor.execute(call), False
        if schema.is_environment_tool:
            if self.environment is None:
                return "environment is not available", False
            return await self.environment.execute(call), False
        return f"no executor for tool {call.name}", False

    async def _run_child(
        self,
        agent_id: str,
        spec: SubagentSpec,
        environment: Environment | None,
        depth: int,
    ) -> SubagentResult:
        child = self.__class__(
            agent_id=agent_id,
            parent_id=self.agent_id,
            depth=depth,
            task=spec.task,
            context=spec.context,
            expected_output=spec.expected_output,
            environment=environment,
            model=self.model,
            limits=self.limits,
            budget=self.budget,
            trace=self.trace,
            scheduler=self.scheduler,
            python_executor=self.python_executor,
        )
        return await child.run()

    async def _is_done(self) -> bool:
        if self.environment is None:
            return False
        try:
            return bool(self.environment.is_done())
        except Exception as exc:
            self._runtime_error("environment_done", exc)
            return True

    async def _environment_result(self) -> SubagentResult:
        if self.environment is None:
            return SubagentResult()
        try:
            observation = await self.environment.observe()
            if hasattr(observation, "to_dict"):
                observation = observation.to_dict()
            return SubagentResult(result=observation)
        except Exception as exc:
            self._runtime_error("environment_observe", exc)
            return SubagentResult(error=self._message(exc))

    def _runtime_error(self, event: str, exc: Exception) -> None:
        self.trace.record_runtime(
            agent_id=self.agent_id,
            parent_id=self.parent_id,
            depth=self.depth,
            event=event,
            error=self._message(exc),
        )

    @staticmethod
    def _exception_has_output(exc: Exception) -> bool:
        if bool(getattr(exc, "has_output", False)):
            return True
        return getattr(exc, "raw_response", None) is not None

    @staticmethod
    def _response_has_output(response: ModelResponse) -> bool:
        if response.tool_calls:
            return True
        raw = response.raw_response
        if raw is None:
            return False
        return not (isinstance(raw, str) and not raw.strip())

    @staticmethod
    def _message(exc: Exception) -> str:
        return str(exc) or exc.__class__.__name__
