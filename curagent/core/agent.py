"""One unified loop used by the root and every child node."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from curagent.core.budget import SharedBudget
from curagent.core.errors import BudgetExceeded, StrictToolCallError, ToolSchemaError
from curagent.core.model import ToolCallingModel
from curagent.core.prompt import TaskModule, compose_prompt
from curagent.core.scheduler import RecursiveScheduler
from curagent.core.tools import framework_tool_schemas, strict_parse_response, validate_tool_call
from curagent.core.trace import DecisionTrace, TraceRecorder
from curagent.core.types import (
    AccessMode,
    AgentLimits,
    Effect,
    ErrorFeedback,
    ExecutionReceipt,
    ModelResponse,
    Observation,
    ReceiptStatus,
    SubagentResult,
    SubagentSpec,
    TerminalStatus,
    ToolCall,
    ToolSchema,
    require_jsonable,
)
from curagent.environments.base import Environment
from curagent.executors.python import PythonExecutor


class AgentNode:
    """A recursive node whose children run this exact same loop."""

    def __init__(
        self,
        *,
        agent_id: str,
        task: str,
        context: Any,
        expected_output: str | None = None,
        environment: Environment,
        model: ToolCallingModel,
        task_module: TaskModule,
        limits: AgentLimits | None = None,
        budget: SharedBudget | None = None,
        trace: TraceRecorder | None = None,
        scheduler: RecursiveScheduler | None = None,
        access: AccessMode = AccessMode.OWNER,
        parent_id: str | None = None,
        depth: int = 0,
        python_executor: PythonExecutor | None = None,
    ) -> None:
        require_jsonable(context, label="node context")
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.depth = depth
        self.task = task
        self.context = context
        self.expected_output = expected_output
        self.environment = environment
        self.model = model
        self.task_module = task_module
        self.limits = limits or AgentLimits()
        self.budget = budget or SharedBudget(self.limits)
        self.trace = trace or TraceRecorder()
        self.scheduler = scheduler or RecursiveScheduler(
            root_env=environment,
            budget=self.budget,
            limits=self.limits,
        )
        self.access = access
        self.python_executor = python_executor
        self._attempt = 0

    async def run(self) -> SubagentResult:
        observation = await self.environment.observe()
        executed_steps = 0
        repair_used = False
        feedback: ErrorFeedback | None = None

        while True:
            if self.environment.is_done():
                terminal_error = observation.metadata.get("terminal_error")
                if terminal_error:
                    return self._result(TerminalStatus.ERROR, error=str(terminal_error))
                return self._result(
                    TerminalStatus.OK,
                    {"reward": self.environment.reward(), "observation": observation.to_dict()},
                )
            if executed_steps >= self.limits.max_steps_per_agent:
                return self._result(
                    TerminalStatus.MAX_STEPS,
                    {"reward": self.environment.reward(), "observation": observation.to_dict()},
                    error=f"agent reached max_steps_per_agent={self.limits.max_steps_per_agent}",
                )

            try:
                schemas = self._available_tools()
            except ValueError as exc:
                return self._result(TerminalStatus.ERROR, error=str(exc))
            snapshot = await self.budget.snapshot()
            prompt = compose_prompt(
                task=self.task,
                context=self.context,
                expected_output=self.expected_output,
                task_module=self.task_module,
                trajectory=self.trace.for_prompt(self.agent_id),
                observation=observation,
                tools=schemas,
                remaining_budget=snapshot.remaining_dict(),
                error_feedback=feedback.to_dict() if feedback else None,
            )

            try:
                response, service_error = await self._generate(prompt, observation)
            except BudgetExceeded as exc:
                return self._result(TerminalStatus.BUDGET_EXHAUSTED, error=str(exc))
            if response is None:
                return self._result(TerminalStatus.ERROR, error=service_error or "model service failed")

            call: ToolCall | None = None
            try:
                call = strict_parse_response(response)
                schema = validate_tool_call(call, schemas)
            except (StrictToolCallError, ToolSchemaError) as exc:
                await self._record(
                    prompt=prompt,
                    observation=observation,
                    response=response,
                    call=call,
                    receipt=None,
                    error=str(exc),
                )
                if repair_used or self.limits.max_retries_per_step == 0:
                    return self._result(TerminalStatus.ERROR, error=str(exc))
                feedback = await self._feedback(
                    error_type="invalid_tool_call",
                    error=str(exc),
                    call=call,
                    effect=Effect.NO_CHANGE,
                    observation=observation,
                )
                repair_used = True
                continue

            try:
                await self.budget.consume_tool_call()
            except BudgetExceeded as exc:
                await self._record(
                    prompt=prompt,
                    observation=observation,
                    response=response,
                    call=call,
                    receipt=None,
                    error=str(exc),
                )
                return self._result(TerminalStatus.BUDGET_EXHAUSTED, error=str(exc))

            executed_steps += 1
            receipt = await self._execute(call, schema, observation)
            await self._record(
                prompt=prompt,
                observation=observation,
                response=response,
                call=call,
                receipt=receipt,
                error=receipt.error,
            )

            if receipt.metadata.get("terminal") and receipt.status == ReceiptStatus.SUCCESS:
                return self._result(TerminalStatus.OK, receipt.result)

            if receipt.effect == Effect.UNKNOWN:
                reconciled = await self.environment.reconcile(call.call_id) if schema.is_environment_tool else None
                if reconciled is not None and reconciled.effect != Effect.UNKNOWN:
                    await self._record_reconciliation(call, observation, reconciled)
                    receipt = reconciled
                else:
                    observation = await self.environment.observe()
                    return self._result(
                        TerminalStatus.UNCERTAIN,
                        {"call_id": call.call_id, "observation": observation.to_dict()},
                        error=receipt.error or "environment effect remains unknown",
                    )

            if receipt.status == ReceiptStatus.SUCCESS or receipt.effect == Effect.COMMITTED:
                observation = receipt.observation or await self.environment.observe()
                repair_used = False
                feedback = None
                continue

            observation = receipt.observation or await self.environment.observe()
            if receipt.status == ReceiptStatus.REJECTED and receipt.effect == Effect.NO_CHANGE:
                if repair_used or self.limits.max_retries_per_step == 0:
                    return self._result(TerminalStatus.ERROR, error=receipt.error or "tool call rejected")
                feedback = await self._feedback(
                    error_type=str(receipt.metadata.get("error_type") or "environment_rejected"),
                    error=receipt.error or "tool call rejected",
                    call=call,
                    effect=receipt.effect,
                    observation=observation,
                )
                repair_used = True
                continue
            return self._result(TerminalStatus.ERROR, error=receipt.error or "tool execution failed")

    def _available_tools(self) -> list[ToolSchema]:
        schemas = framework_tool_schemas(python_enabled=self.python_executor is not None)
        environment_schemas = list(self.environment.tools(self.access))
        registered = {schema.name: schema for schema in self.task_module.environment_tools}
        for schema in environment_schemas:
            if registered and schema.name not in registered:
                raise ValueError(
                    f"environment tool {schema.name!r} is not registered by the task module"
                )
            if registered and schema.parameters != registered[schema.name].parameters:
                raise ValueError(
                    f"environment tool schema mismatch for {schema.name!r}"
                )
        schemas.extend(environment_schemas)
        names = [schema.name for schema in schemas]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"tool names must be unique: {duplicates}")
        return schemas

    async def _generate(
        self, prompt: str, observation: Observation
    ) -> tuple[ModelResponse | None, str | None]:
        last_error: str | None = None
        for infrastructure_retry in range(self.limits.max_model_service_retries + 1):
            await self.budget.consume_model_call()
            self._attempt += 1
            try:
                response = await self.model.generate(prompt, self._available_tools())
                if not isinstance(response, ModelResponse):
                    raise TypeError("model.generate must return ModelResponse")
                return response, None
            except Exception as exc:
                last_error = f"model service error: {exc}"
                await self._record(
                    prompt=prompt,
                    observation=observation,
                    response=None,
                    call=None,
                    receipt=None,
                    error=last_error,
                    increment_attempt=False,
                )
                retryable = bool(getattr(exc, "retryable", True))
                if not retryable or infrastructure_retry >= self.limits.max_model_service_retries:
                    break
        return None, last_error

    async def _execute(
        self, call: ToolCall, schema: ToolSchema, observation: Observation
    ) -> ExecutionReceipt:
        if call.name == "finish":
            try:
                require_jsonable(call.arguments["result"], label="finish result")
            except ValueError as exc:
                return self._rejected(call, observation, str(exc), "invalid_result")
            return ExecutionReceipt(
                call_id=call.call_id,
                status=ReceiptStatus.SUCCESS,
                effect=Effect.NO_CHANGE,
                result=call.arguments["result"],
                version_before=observation.version,
                version_after=observation.version,
                observation=observation,
                metadata={"terminal": True},
            )
        if call.name in {"spawn_agent", "spawn_agents"}:
            return await self._execute_spawn(call, observation)
        if call.name == "python_exec":
            if self.python_executor is None:
                return self._rejected(call, observation, "python_exec is not enabled", "permission")
            return await self.python_executor.execute(call)
        if schema.is_environment_tool:
            try:
                return await self.scheduler.execute_environment(
                    self.environment, self.access, call, observation.version
                )
            except Exception as exc:
                return ExecutionReceipt(
                    call_id=call.call_id,
                    status=ReceiptStatus.FAILED,
                    effect=Effect.UNKNOWN,
                    error=f"unrecoverable environment runtime error: {exc}",
                    version_before=observation.version,
                    observation=await self.environment.observe(),
                    metadata={"error_type": "environment_runtime"},
                )
        return self._rejected(call, observation, f"no executor for tool {call.name}", "configuration")

    async def _execute_spawn(self, call: ToolCall, observation: Observation) -> ExecutionReceipt:
        if call.name == "spawn_agent":
            child = await self.scheduler.spawn_agent(
                call.arguments,
                parent_env=self.environment,
                parent_id=self.agent_id,
                parent_depth=self.depth,
                parent_access=self.access,
                run_child=self._run_child,
            )
            result: Any = child.to_dict()
        else:
            children = await self.scheduler.spawn_agents(
                call.arguments["specs"],
                parent_env=self.environment,
                parent_id=self.agent_id,
                parent_depth=self.depth,
                parent_access=self.access,
                run_child=self._run_child,
            )
            result = [child.to_dict() for child in children]
        latest = await self.environment.observe()
        effect = Effect.COMMITTED if latest.version != observation.version else Effect.NO_CHANGE
        return ExecutionReceipt(
            call_id=call.call_id,
            status=ReceiptStatus.SUCCESS,
            effect=effect,
            result=result,
            version_before=observation.version,
            version_after=latest.version,
            observation=latest,
            metadata={"environment_changed_during_children": effect == Effect.COMMITTED},
        )

    async def _run_child(
        self, agent_id: str, spec: SubagentSpec, environment: Environment, depth: int
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
            task_module=self.task_module,
            limits=self.limits,
            budget=self.budget,
            trace=self.trace,
            scheduler=self.scheduler,
            access=spec.access,
            python_executor=self.python_executor,
        )
        return await child.run()

    async def _feedback(
        self,
        *,
        error_type: str,
        error: str,
        call: ToolCall | None,
        effect: Effect,
        observation: Observation,
    ) -> ErrorFeedback:
        snapshot = await self.budget.snapshot()
        return ErrorFeedback(
            error_type=error_type,
            original_error=error,
            failed_tool_call=call.to_dict() if call else None,
            effect=effect,
            latest_observation=observation,
            remaining_budget=snapshot.remaining_dict(),
        )

    async def _record(
        self,
        *,
        prompt: str,
        observation: Observation,
        response: ModelResponse | None,
        call: ToolCall | None,
        receipt: ExecutionReceipt | None,
        error: str | None,
        increment_attempt: bool = False,
    ) -> None:
        if increment_attempt:
            self._attempt += 1
        snapshot = await self.budget.snapshot()
        self.trace.record(
            DecisionTrace(
                agent_id=self.agent_id,
                parent_id=self.parent_id,
                depth=self.depth,
                attempt=self._attempt,
                prompt=prompt,
                observation=observation,
                raw_model_output=response.raw_response if response else None,
                parsed_tool_call=call,
                receipt=receipt,
                error=error,
                reward=self.environment.reward(),
                budget={
                    "model_calls_used": snapshot.model_calls_used,
                    "tool_calls_used": snapshot.tool_calls_used,
                    "children_used": snapshot.children_used,
                    **snapshot.remaining_dict(),
                },
            )
        )

    async def _record_reconciliation(
        self, call: ToolCall, observation: Observation, receipt: ExecutionReceipt
    ) -> None:
        await self._record(
            prompt="<environment reconcile>",
            observation=observation,
            response=None,
            call=call,
            receipt=receipt,
            error=receipt.error,
        )

    @staticmethod
    def _rejected(
        call: ToolCall, observation: Observation, error: str, error_type: str
    ) -> ExecutionReceipt:
        return ExecutionReceipt(
            call_id=call.call_id,
            status=ReceiptStatus.REJECTED,
            effect=Effect.NO_CHANGE,
            error=error,
            version_before=observation.version,
            version_after=observation.version,
            observation=observation,
            metadata={"error_type": error_type},
        )

    def _result(
        self, status: TerminalStatus, result: Any = None, *, error: str | None = None
    ) -> SubagentResult:
        return SubagentResult(
            task=self.task,
            context=self.context,
            status=status,
            result=result,
            error=error,
            agent_id=self.agent_id,
            parent_id=self.parent_id,
            depth=self.depth,
        )
