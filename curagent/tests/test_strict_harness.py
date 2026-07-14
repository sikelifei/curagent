from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from curagent.core.agent import AgentNode
from curagent.core.budget import SharedBudget
from curagent.core.errors import ModelServiceError, StrictToolCallError, ToolSchemaError
from curagent.core.model import ToolCallingModel
from curagent.core.prompt import TaskModule
from curagent.core.scheduler import RecursiveScheduler
from curagent.core.tools import framework_tool_schemas, strict_parse_response, validate_tool_call
from curagent.core.trace import TraceRecorder
from curagent.core.types import (
    AccessMode,
    AgentLimits,
    Effect,
    ExecutionReceipt,
    ModelResponse,
    Observation,
    ReceiptStatus,
    SubagentResult,
    TerminalStatus,
    ToolCall,
    ToolSchema,
)
from curagent.environments.mock_webshop import MockWebShopEnvironment
from curagent.environments.recode_webshop import ReCodeWebShopEnvironment
from curagent.executors.python import PythonExecutor
from curagent.models.openai_compatible import OpenAICompatibleModel
from curagent.tasks.webshop import WEBSHOP_TASK_MODULE


def native(name: str, arguments: Mapping[str, Any]) -> ModelResponse:
    call = {
        "id": f"provider-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return ModelResponse(raw_response={"tool_calls": [call]}, tool_calls=[call])


class SequenceModel(ToolCallingModel):
    def __init__(self, responses: Sequence[ModelResponse | Exception]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.tool_names: list[list[str]] = []

    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        self.prompts.append(prompt)
        self.tool_names.append([tool.name for tool in tools])
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class MockSolverModel(ToolCallingModel):
    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        del tools
        state = json.loads(prompt)
        observation = state["latest_observation"]
        text = observation["text"]
        if "Search is available" in text:
            return native("search", {"query": "blue 32 oz insulated stainless steel bottle"})
        if "Results for" in text:
            return native("click", {"target": "B001"})
        selected = set()
        for event in state["trajectory"]:
            call = event.get("parsed_tool_call") or {}
            if call.get("name") == "click":
                selected.add(call.get("arguments", {}).get("target"))
        if "Blue" not in selected:
            return native("click", {"target": "Blue"})
        if "32 oz" not in selected:
            return native("click", {"target": "32 oz"})
        return native("buy", {})


class RecursiveInspectionModel(ToolCallingModel):
    def __init__(self) -> None:
        self.child_prompt: dict[str, Any] | None = None

    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        del tools
        state = json.loads(prompt)
        node = state["node_input"]
        if node["task"] == "root task":
            if not state["trajectory"]:
                return native(
                    "spawn_agent",
                    {
                        "task": "child task",
                        "context": {"visible": 7},
                        "expected_output": "one object",
                    },
                )
            child_result = state["trajectory"][-1]["receipt"]["result"]
            return native("finish", {"result": child_result})
        self.child_prompt = state
        return native("finish", {"result": {"seen": node["context"]}})


class DelegatedWriterModel(ToolCallingModel):
    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        del tools
        state = json.loads(prompt)
        node = state["node_input"]
        if node["task"] == "root delegated task":
            if not state["trajectory"]:
                return native(
                    "spawn_agent",
                    {"task": "write child", "context": {}, "access": "delegated"},
                )
            return native("finish", {"result": "parent observed child"})
        if not state["trajectory"]:
            return native("search", {"query": "blue bottle"})
        return native("finish", {"result": "child committed"})


class DeepRecursiveModel(ToolCallingModel):
    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        del tools
        state = json.loads(prompt)
        task = state["node_input"]["task"]
        if state["trajectory"]:
            return native("finish", {"result": task})
        if task == "root deep":
            return native("spawn_agent", {"task": "child deep", "context": {}})
        if task == "child deep":
            return native("spawn_agent", {"task": "grandchild deep", "context": {}})
        return native("finish", {"result": "leaf"})


class UnknownEffectEnvironment(MockWebShopEnvironment):
    async def execute(self, tool_call: ToolCall, expected_version: int) -> ExecutionReceipt:
        observation = await self.observe()
        return ExecutionReceipt(
            call_id=tool_call.call_id,
            status=ReceiptStatus.FAILED,
            effect=Effect.UNKNOWN,
            error="transport timed out",
            version_before=expected_version,
            observation=observation,
            metadata={"error_type": "timeout"},
        )

    async def reconcile(self, call_id: str) -> ExecutionReceipt | None:
        del call_id
        return None


class CommittedFailureEnvironment(MockWebShopEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def execute(self, tool_call: ToolCall, expected_version: int) -> ExecutionReceipt:
        self.calls += 1
        observation = await self.observe()
        return ExecutionReceipt(
            call_id=tool_call.call_id,
            status=ReceiptStatus.FAILED,
            effect=Effect.COMMITTED,
            error="post-processing failed after commit",
            version_before=expected_version,
            version_after=expected_version + 1,
            observation=Observation(observation.text, expected_version + 1, observation.metadata),
        )


class StrictParserTests(unittest.TestCase):
    def test_native_response_requires_exactly_one_call(self) -> None:
        with self.assertRaisesRegex(StrictToolCallError, "exactly one"):
            strict_parse_response(ModelResponse(raw_response={}, tool_calls=[]))
        with self.assertRaisesRegex(StrictToolCallError, "exactly one"):
            strict_parse_response(
                ModelResponse(
                    raw_response={},
                    tool_calls=[{"name": "finish", "arguments": {}}, {"name": "finish", "arguments": {}}],
                )
            )

    def test_json_fallback_does_not_extract_from_prose_or_fences(self) -> None:
        for raw in [
            'use this {"name":"finish","arguments":{"result":1}}',
            '```json\n{"name":"finish","arguments":{"result":1}}\n```',
        ]:
            with self.assertRaises(StrictToolCallError):
                strict_parse_response(ModelResponse(raw_response=raw, protocol="json"))

    def test_json_fallback_rejects_duplicate_keys_and_non_finite_numbers(self) -> None:
        for raw in [
            '{"name":"finish","name":"search","arguments":{}}',
            '{"name":"finish","arguments":{"result":NaN}}',
        ]:
            with self.assertRaises(StrictToolCallError):
                strict_parse_response(ModelResponse(raw_response=raw, protocol="json"))

    def test_schema_rejects_unknown_and_missing_arguments_without_coercion(self) -> None:
        finish = framework_tool_schemas(python_enabled=False)[2]
        with self.assertRaises(ToolSchemaError):
            validate_tool_call(ToolCall("finish", {}), [finish])
        with self.assertRaises(ToolSchemaError):
            validate_tool_call(ToolCall("finish", {"result": 1, "answer": 1}), [finish])

    def test_native_arguments_must_be_strict_json(self) -> None:
        response = ModelResponse(
            raw_response={},
            tool_calls=[{"type": "function", "function": {"name": "finish", "arguments": "{'result': 1}"}}],
        )
        with self.assertRaises(StrictToolCallError):
            strict_parse_response(response)


class AdapterTests(unittest.TestCase):
    def test_recode_targets_are_filtered_by_official_clickables(self) -> None:
        targets = ReCodeWebShopEnvironment._valid_targets(
            "[B001]\n[4 Pack] title text\n[Buy Now]", ["b001", "buy now"]
        )
        self.assertEqual(targets, ["B001", "Buy Now"])

    def test_openai_adapter_requests_non_parallel_native_calls(self) -> None:
        class CaptureModel(OpenAICompatibleModel):
            def __init__(self) -> None:
                super().__init__(
                    base_url="http://localhost/v1",
                    model="test",
                    chat_template_kwargs={"enable_thinking": False},
                )
                self.payload = None

            def _request(self, payload):
                self.payload = payload
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {"name": "finish", "arguments": '{"result": 1}'},
                                    }
                                ],
                            }
                        }
                    ]
                }

        model = CaptureModel()
        payload = model._build_payload("prompt", framework_tool_schemas(python_enabled=False))
        self.assertEqual(payload["parallel_tool_calls"], False)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})


class SharedBudgetTests(unittest.TestCase):
    def test_model_and_tool_limits_are_atomic_under_concurrency(self) -> None:
        async def scenario() -> None:
            budget = SharedBudget(AgentLimits(max_model_calls_total=3, max_tool_calls_total=2))

            async def consume_model() -> bool:
                try:
                    await budget.consume_model_call()
                    return True
                except Exception:
                    return False

            async def consume_tool() -> bool:
                try:
                    await budget.consume_tool_call()
                    return True
                except Exception:
                    return False

            self.assertEqual(sum(await asyncio.gather(*(consume_model() for _ in range(20)))), 3)
            self.assertEqual(sum(await asyncio.gather(*(consume_tool() for _ in range(20)))), 2)

        asyncio.run(scenario())

    def test_child_reservation_is_all_or_nothing(self) -> None:
        async def scenario() -> None:
            limits = AgentLimits(max_children_total=2)
            budget = SharedBudget(limits)
            await budget.reserve_children(2)
            with self.assertRaises(Exception):
                await budget.reserve_children(1)
            self.assertEqual((await budget.snapshot()).children_used, 2)

        asyncio.run(scenario())


class SchedulerTests(unittest.TestCase):
    def test_batch_preflight_starts_no_child_when_one_access_is_unsupported(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            limits = AgentLimits(max_children_total=5)
            budget = SharedBudget(limits)
            scheduler = RecursiveScheduler(root_env=env, budget=budget, limits=limits)
            started: list[str] = []

            async def runner(agent_id, spec, child_env, depth):
                del child_env
                started.append(agent_id)
                return SubagentResult(spec.task, spec.context, TerminalStatus.OK, depth=depth)

            results = await scheduler.spawn_agents(
                [
                    {"task": "valid", "context": {}},
                    {"task": "invalid clone", "context": {}, "access": "clone"},
                ],
                parent_env=env,
                parent_id="root",
                parent_depth=0,
                parent_access=AccessMode.OWNER,
                run_child=runner,
            )
            self.assertEqual(started, [])
            self.assertEqual([item.status for item in results], [TerminalStatus.ERROR, TerminalStatus.ERROR])
            self.assertEqual((await budget.snapshot()).children_used, 0)

        asyncio.run(scenario())

    def test_batch_waits_for_failures_and_preserves_input_order(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            limits = AgentLimits(max_children_total=5, max_concurrency=3)
            budget = SharedBudget(limits)
            scheduler = RecursiveScheduler(root_env=env, budget=budget, limits=limits)

            async def runner(agent_id, spec, child_env, depth):
                del child_env
                await asyncio.sleep({"first": 0.02, "second": 0.0, "third": 0.01}[spec.task])
                if spec.task == "second":
                    raise RuntimeError("child failed")
                return SubagentResult(
                    spec.task,
                    spec.context,
                    TerminalStatus.OK,
                    result=spec.task,
                    agent_id=agent_id,
                    depth=depth,
                )

            results = await scheduler.spawn_agents(
                [{"task": name, "context": {}} for name in ["first", "second", "third"]],
                parent_env=env,
                parent_id="root",
                parent_depth=0,
                parent_access=AccessMode.OWNER,
                run_child=runner,
            )
            self.assertEqual([item.task for item in results], ["first", "second", "third"])
            self.assertEqual([item.status for item in results], [TerminalStatus.OK, TerminalStatus.ERROR, TerminalStatus.OK])

        asyncio.run(scenario())

    def test_depth_limit_returns_explicit_failure(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            limits = AgentLimits(max_depth=0, max_children_total=1)
            budget = SharedBudget(limits)
            scheduler = RecursiveScheduler(root_env=env, budget=budget, limits=limits)

            async def runner(*args):
                raise AssertionError("must not run")

            result = await scheduler.spawn_agent(
                {"task": "child", "context": {}},
                parent_env=env,
                parent_id="root",
                parent_depth=0,
                parent_access=AccessMode.OWNER,
                run_child=runner,
            )
            self.assertEqual(result.status, TerminalStatus.ERROR)
            self.assertIn("max_depth", result.error or "")

        asyncio.run(scenario())

    def test_single_concurrency_slot_allows_recursive_handoff_without_deadlock(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            await env.reset()
            result = await asyncio.wait_for(
                AgentNode(
                    agent_id="root",
                    task="root deep",
                    context={},
                    environment=env,
                    model=DeepRecursiveModel(),
                    task_module=WEBSHOP_TASK_MODULE,
                    limits=AgentLimits(
                        max_depth=2,
                        max_children_total=2,
                        max_concurrency=1,
                        max_model_calls_total=8,
                    ),
                ).run(),
                timeout=2,
            )
            self.assertEqual(result.status, TerminalStatus.OK)

        asyncio.run(scenario())


class AgentLoopTests(unittest.TestCase):
    def test_direct_environment_tools_solve_mock_without_string_action_protocol(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            await env.reset()
            trace = TraceRecorder()
            result = await AgentNode(
                agent_id="root",
                task="buy requested item",
                context={},
                environment=env,
                model=MockSolverModel(),
                task_module=WEBSHOP_TASK_MODULE,
                trace=trace,
                limits=AgentLimits(max_steps_per_agent=8),
            ).run()
            self.assertEqual(result.status, TerminalStatus.OK)
            self.assertEqual(env.reward(), 1.0)
            names = [item["parsed_tool_call"]["name"] for item in trace.all()]
            self.assertEqual(names, ["search", "click", "click", "click", "buy"])
            self.assertNotIn("act", names)

        asyncio.run(scenario())

    def test_invalid_format_gets_one_new_model_decision(self) -> None:
        async def scenario() -> None:
            model = SequenceModel(
                [
                    ModelResponse(raw_response="not json", protocol="json"),
                    native("finish", {"result": "repaired"}),
                ]
            )
            env = MockWebShopEnvironment()
            await env.reset()
            agent = AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
                limits=AgentLimits(max_model_calls_total=2),
            )
            result = await agent.run()
            self.assertEqual(result.status, TerminalStatus.OK)
            self.assertEqual(result.result, "repaired")
            second_prompt = json.loads(model.prompts[1])
            self.assertEqual(second_prompt["error_feedback"]["error_type"], "invalid_tool_call")
            self.assertEqual(len(model.prompts), 2)

        asyncio.run(scenario())

    def test_second_invalid_format_ends_node(self) -> None:
        async def scenario() -> None:
            model = SequenceModel(
                [
                    ModelResponse(raw_response="bad one", protocol="json"),
                    ModelResponse(raw_response="bad two", protocol="json"),
                ]
            )
            env = MockWebShopEnvironment()
            await env.reset()
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
                limits=AgentLimits(max_model_calls_total=2),
            ).run()
            self.assertEqual(result.status, TerminalStatus.ERROR)
            self.assertEqual(len(model.prompts), 2)

        asyncio.run(scenario())

    def test_model_service_retry_is_separate_and_consumes_model_budget(self) -> None:
        async def scenario() -> None:
            model = SequenceModel([ConnectionError("offline"), native("finish", {"result": 1})])
            env = MockWebShopEnvironment()
            await env.reset()
            budget = SharedBudget(AgentLimits(max_model_calls_total=2, max_model_service_retries=1))
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
                budget=budget,
                limits=budget.limits,
            ).run()
            self.assertEqual(result.status, TerminalStatus.OK)
            self.assertEqual((await budget.snapshot()).model_calls_used, 2)

        asyncio.run(scenario())

    def test_non_retryable_model_error_ends_without_infrastructure_retry(self) -> None:
        async def scenario() -> None:
            model = SequenceModel(
                [
                    ModelServiceError("bad request", retryable=False),
                    native("finish", {"result": "must not run"}),
                ]
            )
            env = MockWebShopEnvironment()
            await env.reset()
            budget = SharedBudget(AgentLimits(max_model_calls_total=3, max_model_service_retries=1))
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
                budget=budget,
                limits=budget.limits,
            ).run()
            self.assertEqual(result.status, TerminalStatus.ERROR)
            self.assertEqual(len(model.prompts), 1)
            self.assertEqual((await budget.snapshot()).model_calls_used, 1)

        asyncio.run(scenario())

    def test_child_receives_only_explicit_context_and_parent_gets_final_result(self) -> None:
        async def scenario() -> None:
            model = RecursiveInspectionModel()
            env = MockWebShopEnvironment()
            await env.reset()
            trace = TraceRecorder()
            result = await AgentNode(
                agent_id="root",
                task="root task",
                context={"private_root": True},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
                trace=trace,
                limits=AgentLimits(max_children_total=1, max_model_calls_total=4),
            ).run()
            self.assertEqual(result.status, TerminalStatus.OK)
            self.assertEqual(result.result["result"], {"seen": {"visible": 7}})
            self.assertIsNotNone(model.child_prompt)
            child_prompt = model.child_prompt or {}
            self.assertEqual(child_prompt["node_input"]["context"], {"visible": 7})
            self.assertEqual(child_prompt["trajectory"], [])
            self.assertEqual(len(trace.for_agent("root.1")), 1)
            self.assertEqual(len(trace.for_agent("root")), 2)

        asyncio.run(scenario())

    def test_python_capability_is_absent_by_default_and_exact_when_enabled(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            await env.reset()
            disabled = SequenceModel([native("finish", {"result": 1})])
            await AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=env,
                model=disabled,
                task_module=WEBSHOP_TASK_MODULE,
            ).run()
            self.assertNotIn("python_exec", disabled.tool_names[0])

            env2 = MockWebShopEnvironment()
            await env2.reset()
            enabled = SequenceModel(
                [native("python_exec", {"code": "print(sum([1, 2, 3]))"}), native("finish", {"result": 6})]
            )
            result = await AgentNode(
                agent_id="root",
                task="calculate",
                context={},
                environment=env2,
                model=enabled,
                task_module=WEBSHOP_TASK_MODULE,
                python_executor=PythonExecutor(),
            ).run()
            self.assertEqual(result.result, 6)
            self.assertIn("python_exec", enabled.tool_names[0])

        asyncio.run(scenario())

    def test_readonly_child_has_no_environment_tools(self) -> None:
        async def scenario() -> None:
            model = RecursiveInspectionModel()
            env = MockWebShopEnvironment()
            await env.reset()
            await AgentNode(
                agent_id="root",
                task="root task",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
                limits=AgentLimits(max_children_total=1),
            ).run()
            child_tools = {item["function"]["name"] for item in (model.child_prompt or {})["available_tools"]}
            self.assertTrue({"spawn_agent", "spawn_agents", "finish"}.issubset(child_tools))
            self.assertTrue({"search", "click", "buy"}.isdisjoint(child_tools))

        asyncio.run(scenario())

    def test_delegated_child_commit_refreshes_parent_observation(self) -> None:
        async def scenario() -> None:
            env = MockWebShopEnvironment()
            await env.reset()
            trace = TraceRecorder()
            result = await AgentNode(
                agent_id="root",
                task="root delegated task",
                context={},
                environment=env,
                model=DelegatedWriterModel(),
                task_module=WEBSHOP_TASK_MODULE,
                trace=trace,
                limits=AgentLimits(max_children_total=1, max_model_calls_total=6),
            ).run()
            self.assertEqual(result.status, TerminalStatus.OK)
            spawn_receipt = trace.for_agent("root")[0]["receipt"]
            self.assertEqual(spawn_receipt["effect"], Effect.COMMITTED.value)
            self.assertEqual(spawn_receipt["version_before"], 0)
            self.assertEqual(spawn_receipt["version_after"], 1)
            second_prompt = trace.for_agent("root")[1]["prompt"]
            self.assertIn("Results for", json.loads(second_prompt)["latest_observation"]["text"])

        asyncio.run(scenario())

    def test_unknown_effect_is_reconciled_once_then_ends_uncertain(self) -> None:
        async def scenario() -> None:
            env = UnknownEffectEnvironment()
            await env.reset()
            model = SequenceModel([native("search", {"query": "x"})])
            result = await AgentNode(
                agent_id="root",
                task="test uncertainty",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
            ).run()
            self.assertEqual(result.status, TerminalStatus.UNCERTAIN)
            self.assertEqual(len(model.prompts), 1)

        asyncio.run(scenario())

    def test_committed_failure_is_not_retried(self) -> None:
        async def scenario() -> None:
            env = CommittedFailureEnvironment()
            await env.reset()
            model = SequenceModel(
                [native("search", {"query": "x"}), native("finish", {"result": "after commit"})]
            )
            result = await AgentNode(
                agent_id="root",
                task="test commit handling",
                context={},
                environment=env,
                model=model,
                task_module=WEBSHOP_TASK_MODULE,
            ).run()
            self.assertEqual(result.status, TerminalStatus.OK)
            self.assertEqual(result.result, "after commit")
            self.assertEqual(env.calls, 1)
            self.assertIsNone(json.loads(model.prompts[1])["error_feedback"])

        asyncio.run(scenario())


class PythonExecutorTests(unittest.TestCase):
    def test_pure_computation_runs_and_filesystem_access_is_rejected(self) -> None:
        async def scenario() -> None:
            executor = PythonExecutor()
            success = await executor.execute(ToolCall("python_exec", {"code": "print(6 * 7)"}))
            denied = await executor.execute(ToolCall("python_exec", {"code": "open('x', 'w')"}))
            self.assertEqual(success.status, ReceiptStatus.SUCCESS)
            self.assertEqual(success.result["stdout"], "42\n")
            self.assertEqual(denied.status, ReceiptStatus.REJECTED)
            self.assertEqual(denied.effect, Effect.NO_CHANGE)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
