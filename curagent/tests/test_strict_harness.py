from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from curagent.core.agent import AgentNode
from curagent.core.budget import SharedBudget
from curagent.core.errors import ModelServiceError, StrictToolCallError
from curagent.core.model import ToolCallingModel
from curagent.core.scheduler import RecursiveScheduler
from curagent.core.tools import framework_tool_schemas, strict_parse_response
from curagent.core.trace import TRUNCATION_MARKER, TraceRecorder
from curagent.core.types import AgentLimits, ModelResponse, SubagentResult, ToolCall, ToolSchema
from curagent.environments.base import Environment
from curagent.executors.python import PythonExecutor


def native(name: str, arguments: Mapping[str, Any]) -> ModelResponse:
    call = {
        "id": f"provider-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return ModelResponse(raw_response={"tool_calls": [call]}, tool_calls=[call])


class SequenceModel(ToolCallingModel):
    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = list(responses)
        self.prompts: list[dict[str, Any]] = []
        self.tools: list[list[str]] = []

    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        self.prompts.append(json.loads(prompt))
        self.tools.append([tool.name for tool in tools])
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class SharedTestEnvironment(Environment):
    def __init__(self) -> None:
        self.value = "home"
        self.calls: list[str] = []

    async def observe(self) -> dict[str, Any]:
        return {"value": self.value, "calls": list(self.calls)}

    def tools(self) -> Sequence[ToolSchema]:
        return [
            ToolSchema(
                "set_value",
                "Set shared state",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                is_environment_tool=True,
            )
        ]

    async def execute(self, tool_call: ToolCall) -> Any:
        self.calls.append(tool_call.name)
        self.value = str(tool_call.arguments["value"])
        return {"value": self.value}

    def is_done(self) -> bool:
        return False


class RaisingEnvironment(SharedTestEnvironment):
    async def execute(self, tool_call: ToolCall) -> Any:
        del tool_call
        raise RuntimeError("environment failed")


class RecursiveModel(ToolCallingModel):
    def __init__(self) -> None:
        self.prompts: list[dict[str, Any]] = []

    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        del tools
        state = json.loads(prompt)
        self.prompts.append(state)
        if state["task"] == "root" and not state["trajectory"]:
            return native(
                "spawn_agent",
                {"task": "child", "context": {"requested": "set"}},
            )
        if state["task"] == "child" and not state["trajectory"]:
            return native(
                "set_value", {"value": "child"}
            )
        if state["task"] == "child" and len(state["trajectory"]) == 1:
            return native("spawn_agent", {"task": "grandchild", "context": {}})
        if state["task"] == "grandchild" and not state["trajectory"]:
            return native("set_value", {"value": "grandchild"})
        if state["task"] == "grandchild":
            return native("finish", {"result": state["observation"]})
        return native("finish", {"result": state["observation"]})


class RecursiveAnalysisModel(ToolCallingModel):
    def __init__(self) -> None:
        self.prompts: list[dict[str, Any]] = []

    async def generate(self, prompt: str, tools: Sequence[ToolSchema]) -> ModelResponse:
        del tools
        state = json.loads(prompt)
        self.prompts.append(state)
        if state["task"] == "root" and not state["trajectory"]:
            return native("spawn_agent", {"task": "child", "context": {"answer": 7}})
        if state["task"] == "root":
            return native("finish", {"result": state["trajectory"][-1]["execution_result"]})
        if state["task"] == "child":
            return native("finish", {"result": state["context"]})
        return native("finish", {"result": "done"})


class HarnessTests(unittest.TestCase):
    def test_spawn_schema_has_no_access_and_result_is_small(self) -> None:
        schemas = framework_tool_schemas(python_enabled=False)
        spawn = schemas[0].parameters["properties"]
        self.assertNotIn("access", spawn)
        self.assertEqual(set(schemas[1].parameters["properties"]), {"specs"})

    def test_strict_parser_rejects_non_tool_output(self) -> None:
        with self.assertRaises(StrictToolCallError):
            strict_parse_response(ModelResponse(raw_response="plain text", protocol="json"))

    def test_pure_analysis_needs_no_environment(self) -> None:
        async def scenario() -> None:
            model = RecursiveAnalysisModel()
            result = await AgentNode(
                agent_id="root",
                task="root",
                context={},
                model=model,
                limits=AgentLimits(max_total_steps=4, max_depth=2),
            ).run()
            self.assertEqual(result.to_dict(), {"result": {"error": None, "result": {"answer": 7}}, "error": None})
            self.assertIsNone(model.prompts[0]["observation"])
            self.assertEqual(
                set(model.prompts[0]),
                {"task", "context", "trajectory", "observation", "tools", "remaining_steps"},
            )

        asyncio.run(scenario())

    def test_child_and_grandchild_share_environment_and_tools(self) -> None:
        async def scenario() -> None:
            environment = SharedTestEnvironment()
            model = RecursiveModel()
            limits = AgentLimits(max_total_steps=8, max_depth=2)
            budget = SharedBudget(limits)
            result = await AgentNode(
                agent_id="root",
                task="root",
                context={},
                environment=environment,
                model=model,
                limits=limits,
                budget=budget,
            ).run()
            self.assertIsNone(result.error)
            self.assertEqual(environment.value, "grandchild")
            self.assertEqual(environment.calls, ["set_value", "set_value"])
            child_prompt = next(item for item in model.prompts if item["task"] == "child")
            self.assertIn("set_value", {item["function"]["name"] for item in child_prompt["tools"]})
            root_followup = [item for item in model.prompts if item["task"] == "root"][1]
            self.assertEqual(root_followup["observation"]["value"], "grandchild")
            self.assertEqual((await budget.snapshot()).total_steps_used, 7)

        asyncio.run(scenario())

    def test_spawn_agents_runs_in_input_order(self) -> None:
        async def scenario() -> None:
            scheduler = RecursiveScheduler(limits=AgentLimits(max_depth=2))
            seen: list[str] = []

            async def run_child(agent_id, spec, environment, depth):
                del agent_id, environment, depth
                seen.append(spec.task)
                return SubagentResult(result=spec.task)

            results = await scheduler.spawn_agents(
                [{"task": "a", "context": {}}, {"task": "b", "context": {}}],
                parent_env=None,
                parent_id="root",
                parent_depth=0,
                run_child=run_child,
            )
            self.assertEqual(seen, ["a", "b"])
            self.assertEqual(len(results), 2)

        asyncio.run(scenario())

    def test_depth_limit_returns_plain_failure_without_starting_child(self) -> None:
        async def scenario() -> None:
            scheduler = RecursiveScheduler(limits=AgentLimits(max_depth=0))
            started = False

            async def run_child(*args):
                nonlocal started
                started = True
                return SubagentResult(result="unexpected")

            result = await scheduler.spawn_agent(
                {"task": "child", "context": {}},
                parent_env=None,
                parent_id="root",
                parent_depth=0,
                run_child=run_child,
            )
            self.assertFalse(started)
            self.assertEqual(result.result, None)
            self.assertIn("max_depth=0", result.error or "")

        asyncio.run(scenario())

    def test_invalid_output_consumes_step_and_loop_continues(self) -> None:
        async def scenario() -> None:
            model = SequenceModel(
                [
                    ModelResponse(raw_response="not a call", protocol="json"),
                    native("finish", {"result": "done"}),
                ]
            )
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                model=model,
                limits=AgentLimits(max_total_steps=2),
            ).run()
            self.assertEqual(result.result, "done")
            self.assertEqual(len(model.prompts), 2)
            self.assertIn("not one strict JSON object", model.prompts[1]["trajectory"][0]["execution_result"])

        asyncio.run(scenario())

    def test_model_no_response_releases_reserved_step(self) -> None:
        async def scenario() -> None:
            budget = SharedBudget(AgentLimits(max_total_steps=1))
            model = SequenceModel([ConnectionError("offline")])
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                model=model,
                budget=budget,
                limits=budget.limits,
            ).run()
            self.assertEqual(result.error, "offline")
            self.assertEqual((await budget.snapshot()).total_steps_used, 0)

        asyncio.run(scenario())

    def test_received_malformed_provider_response_consumes_step(self) -> None:
        async def scenario() -> None:
            budget = SharedBudget(AgentLimits(max_total_steps=2))
            model = SequenceModel(
                [
                    ModelServiceError(
                        "provider response was malformed",
                        has_output=True,
                        raw_response={"choices": "bad"},
                    ),
                    native("finish", {"result": "continued"}),
                ]
            )
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                model=model,
                limits=budget.limits,
                budget=budget,
            ).run()
            self.assertEqual(result.result, "continued")
            self.assertEqual((await budget.snapshot()).total_steps_used, 2)
            self.assertIn("must be a string", model.prompts[1]["trajectory"][0]["execution_result"])

        asyncio.run(scenario())

    def test_environment_exception_is_plain_feedback(self) -> None:
        async def scenario() -> None:
            model = SequenceModel(
                [native("set_value", {"value": "x"}), native("finish", {"result": "handled"})]
            )
            result = await AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=RaisingEnvironment(),
                model=model,
                limits=AgentLimits(max_total_steps=2),
            ).run()
            self.assertEqual(result.result, "handled")
            self.assertEqual(
                model.prompts[1]["trajectory"][0]["execution_result"],
                "environment failed",
            )

        asyncio.run(scenario())

    def test_long_result_and_raw_output_are_truncated_for_prompt(self) -> None:
        async def scenario() -> None:
            environment = SharedTestEnvironment()
            model = SequenceModel(
                [native("set_value", {"value": "x" * 5000}), native("finish", {"result": "done"})]
            )
            await AgentNode(
                agent_id="root",
                task="test",
                context={},
                environment=environment,
                model=model,
                trace=TraceRecorder(),
                limits=AgentLimits(max_total_steps=2),
            ).run()
            feedback = model.prompts[1]["trajectory"][0]["execution_result"]
            self.assertTrue(isinstance(feedback, str))
            self.assertIn(TRUNCATION_MARKER, feedback)

        asyncio.run(scenario())

    def test_python_executor_does_not_block_and_returns_result(self) -> None:
        async def scenario() -> None:
            result = await PythonExecutor().execute(ToolCall("python_exec", {"code": "print(6 * 7)"}))
            self.assertEqual(result["stdout"], "42\n")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
