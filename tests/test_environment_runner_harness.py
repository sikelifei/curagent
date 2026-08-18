from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from recursive_agent import ModelCallError
from recursive_agent.envs.base import AgentEnvironment
from recursive_agent.envs.runner import run_environment
from recursive_agent.tools import CapabilityCollection
from recursive_agent.types import EnvironmentStatus

from .fakes import FakeFactory


def _model_config(directory: str) -> Path:
    path = Path(directory) / "model.yaml"
    path.write_text(
        "model:\n"
        "  type: api\n"
        "  api:\n"
        "    model: synthetic\n",
        encoding="utf-8",
    )
    return path


class _CodeActEnvironment(AgentEnvironment):
    name = "synthetic-codeact"

    def __init__(self) -> None:
        self._context = {"seed": [1]}
        self.events: list[tuple[str, int]] = []
        self.finalized: list[Any] = []
        self.close_calls = 0

    @property
    def use_recursive_codeact_harness(self) -> bool:
        return True

    @property
    def task(self) -> str:
        return "assemble the result"

    @property
    def context(self) -> dict[str, Any]:
        return self._context

    def tools(self) -> dict[str, Any]:
        return {}

    @property
    def environment_system_prompt(self) -> str:
        return "Synthetic CodeAct environment guidance."

    def observe(self) -> str:
        return "synthetic observation"

    def codeact_capabilities(self, *, is_root: bool, depth: int) -> CapabilityCollection:
        del is_root, depth
        return CapabilityCollection(
            {
                "record": {
                    "tool": self.record,
                    "description": "Record a shared environment event.",
                }
            }
        )

    def record(self, label: str) -> str:
        self.events.append((label, id(self)))
        return f"recorded:{label}"

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=False)

    def finalize_root(self, result: Any = None) -> str:
        self.finalized.append(result)
        return f"final:{result}"

    def report(self) -> dict[str, Any]:
        return {
            "events": [label for label, _ in self.events],
            "finalized": list(self.finalized),
        }

    def close(self) -> None:
        self.close_calls += 1


class _LegacyEnvironment(AgentEnvironment):
    name = "synthetic-legacy"

    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def task(self) -> str:
        return "legacy task"

    @property
    def context(self) -> dict[str, int]:
        return {"value": 1}

    def tools(self) -> dict[str, Any]:
        return {"lookup": lambda: "legacy"}

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=False)

    def report(self) -> dict[str, Any]:
        return {"legacy": True}

    def close(self) -> None:
        self.close_calls += 1


class EnvironmentRunnerHarnessTests(unittest.TestCase):
    def test_opt_in_codeact_uses_one_shared_client_and_environment(self) -> None:
        environment = _CodeActEnvironment()

        def handler(messages: list[dict[str, Any]], _timeout: float | None) -> str:
            self.assertEqual(messages[0]["content"], environment.environment_system_prompt)
            task_prompt = messages[1]["content"]
            if "# Task\nchild task" in task_prompt:
                return (
                    "<python>\n"
                    "record('child')\n"
                    "return_to_parent('prepared')\n"
                    "</python>"
                )
            self.assertIn("# Action Space", task_prompt)
            return (
                "<python>\n"
                "record('root')\n"
                "child = await spawn_subagent('child task')\n"
                "finish({'child': child})\n"
                "</python>"
            )

        factory = FakeFactory(handler)
        with tempfile.TemporaryDirectory() as directory:
            run = run_environment(
                environment,
                model_config=_model_config(directory),
                agent_kwargs={
                    "max_total_steps": 2,
                    "max_depth": 1,
                    "max_concurrent_subagents": 1,
                    "max_observation_chars": 100,
                    "client_factory": factory,
                },
            )

        self.assertEqual(factory.created, 1)
        self.assertEqual(factory.closed_clients, 1)
        self.assertEqual(environment.close_calls, 1)
        self.assertEqual([label for label, _ in environment.events], ["root", "child"])
        self.assertEqual({identity for _, identity in environment.events}, {id(environment)})
        self.assertEqual(len(run.agent_result.trace.children), 1)
        self.assertEqual(run.agent_result.answer, "final:{'child': 'prepared'}")
        self.assertEqual(run.agent_result.steps, 2)
        self.assertEqual(run.agent_result.usage.total_calls, 2)
        self.assertIsNot(run.initial_context, environment.context)
        self.assertEqual(run.initial_context, {"seed": [1]})
        self.assertEqual(run.system_prompt, environment.environment_system_prompt)
        self.assertEqual(run.tool_descriptions["record"]["kind"], "callable")
        self.assertEqual(
            run.tool_descriptions["record"]["description"],
            "Record a shared environment event.",
        )
        exported = run.to_trace_dict()
        self.assertEqual(exported["agent_result"]["usage"]["total_calls"], 2)
        self.assertEqual(len(exported["agent_result"]["trace"]["children"]), 1)
        json.dumps(exported)

    def test_opt_in_model_failure_attaches_partial_trace_and_closes_once(self) -> None:
        environment = _CodeActEnvironment()

        def fail(_messages: list[dict[str, Any]], _timeout: float | None) -> str:
            raise RuntimeError("synthetic model failure")

        factory = FakeFactory(fail)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelCallError) as raised:
                run_environment(
                    environment,
                    model_config=_model_config(directory),
                    agent_kwargs={"max_steps": 2, "client_factory": factory},
                )

        self.assertEqual(factory.created, 1)
        self.assertEqual(factory.closed_clients, 1)
        self.assertEqual(environment.close_calls, 1)
        partial = raised.exception.partial_trace
        self.assertEqual(partial["prompts"]["system"], environment.environment_system_prompt)
        self.assertEqual(partial["prompts"]["task"], environment.task)
        self.assertEqual(partial["prompts"]["initial_context"], {"seed": [1]})
        self.assertEqual(partial["agent_result"]["steps"], 0)
        self.assertEqual(partial["agent_result"]["usage"]["total_calls"], 0)
        self.assertEqual(partial["agent_result"]["trace"]["status"], "error")
        self.assertIn("record", partial["tools"])
        json.dumps(partial)

    def test_opt_in_rejects_runner_owned_and_unsupported_agent_arguments(self) -> None:
        for argument in ("tools", "system_prompt", "step_callback"):
            environment = _CodeActEnvironment()
            with tempfile.TemporaryDirectory() as directory:
                with self.subTest(argument=argument), self.assertRaises(ValueError) as raised:
                    run_environment(
                        environment,
                        model_config=_model_config(directory),
                        agent_kwargs={argument: object()},
                    )
            self.assertIn(argument, str(raised.exception))
            self.assertEqual(environment.close_calls, 1)

    def test_default_environment_keeps_legacy_runner_path(self) -> None:
        environment = _LegacyEnvironment()

        def handler(_messages: list[dict[str, Any]], _timeout: float | None) -> str:
            return "<repl>answer['content'] = 'legacy answer'\nanswer['ready'] = True</repl>"

        factory = FakeFactory(handler)
        with tempfile.TemporaryDirectory() as directory:
            run = run_environment(
                environment,
                model_config=_model_config(directory),
                agent_kwargs={"max_steps": 1, "client_factory": factory},
            )

        self.assertEqual(run.agent_result.answer, "legacy answer")
        self.assertEqual(factory.created, 1)
        self.assertEqual(factory.closed_clients, 1)
        self.assertEqual(environment.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
