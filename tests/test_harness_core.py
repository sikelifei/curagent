from __future__ import annotations

import copy
import threading
import time
import unittest
from typing import Any

from recursive_agent import AgentNode, AgentResult, RecursiveScheduler, SharedBudget
from recursive_agent.exceptions import ConfigurationError, ModelCallError
from recursive_agent.tools import CapabilityCollection, ToolInfo
from recursive_agent.types import EnvironmentStatus, ModelCallUsage, ModelResponse


def _python(code: str) -> str:
    return f"<python>\n{code}\n</python>"


def _task(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message["content"])
        marker = "# Task\n"
        if marker in content:
            value = content.split(marker, 1)[1]
            return value.split("\n\n# Context", 1)[0]
    raise AssertionError(f"No task in messages: {messages!r}")


class _Client:
    model_name = "test-model"

    def __init__(self, handler: Any, *, delay: float = 0.0) -> None:
        self.handler = handler
        self.delay = delay
        self.calls: list[list[dict[str, Any]]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.closed = 0

    def completion(
        self,
        messages: list[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        snapshot = copy.deepcopy(messages)
        with self.lock:
            self.calls.append(snapshot)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            result = self.handler(snapshot)
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, ModelResponse):
                return result
            return ModelResponse(
                content=str(result),
                usage=ModelCallUsage(model=self.model_name, input_tokens=1, output_tokens=1),
            )
        finally:
            with self.lock:
                self.active -= 1

    def close(self) -> None:
        self.closed += 1


class _Environment:
    def __init__(self, *, dynamic: bool = False) -> None:
        self.dynamic = dynamic
        self.observe_count = 0
        self.finalized: list[Any] = []

    @property
    def environment_system_prompt(self) -> str:
        return "Environment-owned guidance."

    def observe(self) -> str:
        self.observe_count += 1
        return f"observation-{self.observe_count}"

    def codeact_capabilities(self, *, is_root: bool, depth: int) -> CapabilityCollection:
        suffix = f"{self.observe_count}" if self.dynamic else "shared"
        values: dict[str, Any] = {
            "shared_tool": ToolInfo(
                "shared_tool",
                lambda: "shared",
                "shared environment action",
            ),
            f"dynamic_{suffix}": ToolInfo(
                f"dynamic_{suffix}",
                lambda: suffix,
                "current dynamic action",
            ),
        }
        if is_root:
            values["root_tool"] = ToolInfo("root_tool", lambda: "root", "root action")
        else:
            values["child_tool"] = ToolInfo("child_tool", lambda: "child", "child action")
        return CapabilityCollection(values)

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=False)

    async def finalize_root(self, result: Any = None) -> str:
        self.finalized.append(result)
        return f"finalized:{result}"


class _AsyncTerminalEnvironment(_Environment):
    async def observe(self) -> str:
        return "async observation"

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=True, final_answer="terminal answer")


class _RolePromptEnvironment(_Environment):
    @property
    def use_role_specific_prompts(self) -> bool:
        return True

    @property
    def root_prompt(self) -> str:
        return "Root-only system guidance."

    @property
    def child_prompt(self) -> str:
        return "Child-only system guidance."


class _LegacyRolePromptEnvironment(_Environment):
    @property
    def root_prompt(self) -> str:
        return "Legacy root-only guidance."

    @property
    def child_prompt(self) -> str:
        return "Legacy child-only guidance."


class HarnessCoreTests(unittest.TestCase):
    def test_root_and_child_use_role_specific_system_prompts(self) -> None:
        scripts = {
            "root": _python("result = spawn_subagent('child')\nfinish(result)"),
            "child": _python("return_to_parent('done')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)])
        scheduler = RecursiveScheduler(
            _RolePromptEnvironment(), client, max_total_steps=2, max_depth=1
        )

        scheduler.run("root")

        self.assertEqual(client.calls[0][0]["content"], "Root-only system guidance.")
        self.assertEqual(client.calls[1][0]["content"], "Child-only system guidance.")

    def test_default_prompt_routing_keeps_environment_guidance_shared(self) -> None:
        scripts = {
            "root": _python("result = spawn_subagent('child')\nfinish(result)"),
            "child": _python("return_to_parent('done')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)])
        scheduler = RecursiveScheduler(
            _LegacyRolePromptEnvironment(), client, max_total_steps=2, max_depth=1
        )

        scheduler.run("root")

        self.assertEqual(client.calls[0][0]["content"], "Environment-owned guidance.")
        self.assertEqual(client.calls[1][0]["content"], "Environment-owned guidance.")

    def test_spawn_subagent_runs_directly_and_legacy_await_remains_compatible(self) -> None:
        scripts = {
            "root": _python("child = spawn_subagent('child')\nfinish(child)"),
            "child": _python("return_to_parent('prepared')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)])
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=2, max_depth=1)

        result = scheduler.run("root")

        self.assertEqual(result.answer, "finalized:prepared")
        assert scheduler.root is not None
        description = scheduler.root.capabilities["spawn_subagent"].prompt_description
        self.assertIsNotNone(description)
        assert description is not None
        self.assertIn("Call it directly", description)
        self.assertNotIn("always write await", description)

    def test_async_observation_and_terminal_status_stop_without_generation(self) -> None:
        client = _Client(lambda messages: _python("finish('unexpected')"))
        scheduler = RecursiveScheduler(_AsyncTerminalEnvironment(), client, max_total_steps=2)
        result = scheduler.run("root")
        self.assertEqual(result.status, "environment_done")
        self.assertEqual(result.answer, "terminal answer")
        self.assertEqual(scheduler.budget.consumed_steps, 0)
        self.assertEqual(client.calls, [])

    def test_identity_private_namespaces_and_context_assignment(self) -> None:
        scripts = {
            "root task": _python(
                "payload = {'values': ['before']}\n"
                "child_result = await spawn_subagent('child task', payload)\n"
                "payload['values'].append('after')\n"
                "finish(child_result)"
            ),
            "child task": _python(
                "child_only = True\n"
                "grand_result = await spawn_subagent('grandchild task', {'part': 1})\n"
                "return_to_parent(repr(context) + ':' + grand_result)"
            ),
            "grandchild task": _python("grand_only = True\nreturn_to_parent('grandchild result')"),
        }

        def handler(messages: list[dict[str, Any]]) -> str:
            return scripts[_task(messages)]

        environment = _Environment()
        client = _Client(handler)
        scheduler = RecursiveScheduler(
            environment,
            client,
            max_total_steps=3,
            max_depth=2,
        )
        result = scheduler.run("root task", {"root": [1]})

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "finalized:{'values': ['before']}:grandchild result")
        root = scheduler.root
        assert root is not None
        child = root.children[0]
        grandchild = child.children[0]
        for node in (root, child, grandchild):
            self.assertIs(node.environment, environment)
            self.assertIs(node.scheduler, scheduler)
            self.assertIs(node.budget, scheduler.budget)
            self.assertIs(node.model_client, client)
        self.assertIsNot(root.repl, child.repl)
        self.assertIsNot(child.repl, grandchild.repl)
        self.assertIn("payload", root.namespace)
        self.assertNotIn("child_only", root.namespace)
        self.assertNotIn("root_only", child.namespace)
        self.assertNotIn("root_only", grandchild.namespace)
        self.assertEqual(child.context, {"values": ["before"]})
        self.assertEqual(root.namespace["payload"], {"values": ["before", "after"]})
        child_user = client.calls[1][1]["content"]
        self.assertIn("# Task\nchild task", child_user)
        self.assertNotIn("root task", child_user)
        self.assertNotIn("root", child_user.split("# Context", 1)[-1])
        self.assertEqual(
            environment.finalized,
            ["{'values': ['before']}:grandchild result"],
        )

    def test_role_and_depth_capabilities_are_prompted_and_bound(self) -> None:
        scripts = {
            "root": _python("child = await spawn_subagent('child')\nfinish('done')"),
            "child": _python("grand = await spawn_subagent('grand')\nreturn_to_parent('child')"),
            "grand": _python("return_to_parent('grand')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)])
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=3, max_depth=2)
        scheduler.run("root")
        root = scheduler.root
        assert root is not None
        child = root.children[0]
        grand = child.children[0]

        self.assertIn("spawn_subagent", root.namespace)
        self.assertIn("spawn_subagents", root.namespace)
        self.assertIn("finish", root.namespace)
        self.assertNotIn("return_to_parent", root.namespace)
        self.assertIn("spawn_subagent", child.namespace)
        self.assertIn("return_to_parent", child.namespace)
        self.assertNotIn("finish", child.namespace)
        self.assertNotIn("spawn_subagent", grand.namespace)
        self.assertNotIn("spawn_subagents", grand.namespace)
        self.assertIn("return_to_parent", grand.namespace)
        self.assertIn("root_tool", root.capabilities)
        self.assertNotIn("root_tool", child.capabilities)
        self.assertIn("child_tool", child.capabilities)
        self.assertNotIn("child_tool", root.capabilities)

        root_prompt = client.calls[0][1]["content"]
        child_prompt = client.calls[1][1]["content"]
        grand_prompt = client.calls[2][1]["content"]
        self.assertIn("`finish`", root_prompt)
        self.assertNotIn("`return_to_parent`", root_prompt)
        self.assertIn("`return_to_parent`", child_prompt)
        self.assertNotIn("`finish`", child_prompt)
        self.assertNotIn("`spawn_subagent`", grand_prompt)
        self.assertNotIn("`spawn_subagents`", grand_prompt)

    def test_invalid_spawn_keywords_report_generic_contract(self) -> None:
        responses = iter(
            [
                _python(
                    "spawn_subagent(objective='craft ore', "
                    "return_condition='done')"
                ),
                _python("finish('recovered')"),
            ]
        )
        client = _Client(lambda messages: next(responses))
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=2, max_depth=1)

        result = scheduler.run("root")

        self.assertEqual(result.answer, "finalized:recovered")
        root = scheduler.root
        assert root is not None
        error = root.trace.steps[0].code_executions[0].error
        assert error is not None
        self.assertIn("spawn_subagent(task: str, context=None)", error)
        self.assertIn("objective, quantity, scope, restrictions, and return condition", error)
        self.assertIn("not in keyword arguments", error)
        self.assertIn("spawn_subagent(task: str, context=None)", client.calls[1][-1]["content"])

    def test_framework_spawn_descriptions_teach_direct_calls(self) -> None:
        client = _Client(lambda messages: _python("finish('done')"))
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=1, max_depth=1)
        scheduler.run("root")
        prompt = client.calls[0][1]["content"]
        self.assertIn("Call it directly", prompt)
        self.assertIn("use sequential spawn_subagent calls", prompt)
        self.assertNotIn("always write await", prompt)

    def test_global_budget_and_ordered_concurrent_children(self) -> None:
        scripts = {
            "root": _python(
                "results = await spawn_subagents([{'task': 'a'}, {'task': 'b'}])\n"
                "finish(results)"
            ),
            "a": _python("return_to_parent('A')"),
            "b": _python("return_to_parent('B')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)], delay=0.02)
        scheduler = RecursiveScheduler(
            _Environment(),
            client,
            SharedBudget(3),
            max_depth=1,
            max_concurrent_subagents=2,
        )
        result = scheduler.run("root")

        self.assertEqual(result.answer, "finalized:['A', 'B']")
        self.assertEqual(scheduler.budget.consumed_steps, 3)
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.max_active, 2)
        self.assertEqual(result.usage.total_calls, 3)
        self.assertEqual(result.trace.children[0].usage.total_calls, 1)
        self.assertEqual(result.trace.children[1].usage.total_calls, 1)

    def test_budget_exhaustion_has_no_forced_generation_or_overspend(self) -> None:
        scripts = {
            "root": _python(
                "results = await spawn_subagents([{'task': 'a'}, {'task': 'b'}])\n"
                "finish(results)"
            ),
            "a": _python("return_to_parent('A')"),
            "b": _python("return_to_parent('B')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)])
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=2, max_depth=1)
        result = scheduler.run("root")

        self.assertEqual(scheduler.budget.consumed_steps, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.answer, "finalized:['A', '']")
        self.assertEqual(result.status, "completed")

    def test_top_level_await_sequential_and_ordered_batch(self) -> None:
        scripts = {
            "root": _python(
                "first = await spawn_subagent('first')\n"
                "rest = await spawn_subagents([{'task': 'second'}, {'task': 'third'}])\n"
                "finish([first] + rest)"
            ),
            "first": _python("return_to_parent('1')"),
            "second": _python("return_to_parent('2')"),
            "third": _python("return_to_parent('3')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)], delay=0.01)
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=4, max_depth=1)
        result = scheduler.run("root")
        self.assertEqual(result.answer, "finalized:['1', '2', '3']")
        task_order = [_task(call) for call in client.calls]
        self.assertEqual(task_order[0:2], ["root", "first"])
        self.assertCountEqual(task_order[2:], ["second", "third"])

    def test_max_direct_subagent_limit_returns_stable_error_for_excess_requests(self) -> None:
        scripts = {
            "root": _python(
                "results = await spawn_subagents([{'task': 'a'}, {'task': 'b'}])\n"
                "finish(results)"
            ),
            "a": _python("return_to_parent('A')"),
            "b": _python("return_to_parent('B')"),
        }
        client = _Client(lambda messages: scripts[_task(messages)])
        scheduler = RecursiveScheduler(
            _Environment(),
            client,
            max_total_steps=2,
            max_depth=1,
            max_subagents_per_agent=1,
        )
        result = scheduler.run("root")
        self.assertEqual(
            result.answer,
            "finalized:['A', 'Error: maximum direct subagents per agent (1) reached']",
        )
        self.assertEqual(len(client.calls), 2)

    def test_dynamic_observation_and_action_space_share_one_collection(self) -> None:
        scripts = {
            "root": _python("value = shared_tool()"),
        }
        responses = iter([scripts["root"], _python("finish('done')")])
        client = _Client(lambda messages: next(responses))
        environment = _Environment(dynamic=True)
        scheduler = RecursiveScheduler(environment, client, max_total_steps=2, max_depth=0)
        result = scheduler.run("root")
        root = scheduler.root
        assert root is not None

        first_prompt = client.calls[0][1]["content"]
        second_prompt = client.calls[1][3]["content"]
        self.assertIn("observation-1", first_prompt)
        self.assertIn("`dynamic_1`", first_prompt)
        self.assertIn("observation-2", second_prompt)
        self.assertIn("`dynamic_2`", second_prompt)
        self.assertIn("# Execution Output", second_prompt)
        self.assertIn("dynamic_2", root.capabilities)
        self.assertNotIn("dynamic_1", root.namespace)
        self.assertIs(root.repl.capabilities, root.capabilities)
        self.assertEqual(result.status, "completed")

    def test_child_model_failure_is_stable_and_releases_reservation(self) -> None:
        def handler(messages: list[dict[str, Any]]) -> Any:
            if _task(messages) == "bad":
                return RuntimeError("boom")
            return _python("bad_result = await spawn_subagent('bad')\nfinish(bad_result)")

        client = _Client(handler)
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=2, max_depth=1)
        result = scheduler.run("root")
        self.assertEqual(result.answer, "finalized:Error: subagent model call failed")
        self.assertEqual(scheduler.budget.consumed_steps, 1)
        self.assertEqual(scheduler.budget.reserved_steps, 0)
        self.assertEqual(result.usage.total_calls, 1)

    def test_failed_root_model_call_releases_budget(self) -> None:
        client = _Client(lambda messages: RuntimeError("failure"))
        scheduler = RecursiveScheduler(_Environment(), client, max_total_steps=1)
        with self.assertRaises(ModelCallError):
            scheduler.run("root")
        self.assertEqual(scheduler.budget.consumed_steps, 0)
        self.assertEqual(scheduler.budget.reserved_steps, 0)
        self.assertEqual(scheduler.budget.remaining_steps, 1)

    def test_framework_merge_rejects_environment_collisions(self) -> None:
        with self.assertRaises(ConfigurationError):
            CapabilityCollection({"finish": lambda result=None: result})
        with self.assertRaises(ConfigurationError):
            CapabilityCollection({"spawn_subagent": lambda task: task})


if __name__ == "__main__":
    unittest.main()
