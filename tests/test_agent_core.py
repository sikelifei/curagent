from __future__ import annotations

import threading
import time
import unittest

from recursive_agent import (
    CancellationError,
    EnvironmentStatus,
    ModelCallError,
    RecursiveAgent,
    TimeoutExceededError,
)

from .fakes import FakeFactory, initial_task


class AgentCoreTests(unittest.TestCase):
    def test_complete_root_and_child_prompts_bypass_generated_tool_text(self) -> None:
        def handler(messages, timeout):
            del timeout
            task = initial_task(messages)
            if task == "root-only benchmark task":
                return (
                    "```repl\n"
                    "result = spawn_subagent('child assignment', {'value': 1})\n"
                    "answer['content'] = result\n"
                    "answer['ready'] = True\n"
                    "```"
                )
            return (
                "```repl\n"
                "answer['content'] = 'child done'\n"
                "answer['ready'] = True\n"
                "```"
            )

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            tools={
                "lookup": {
                    "tool": lambda: "value",
                    "description": "SCHEMA_DESCRIPTION_MUST_NOT_BE_APPENDED",
                }
            },
            root_prompt="ROOT_PROMPT\nHANDWRITTEN_TOOLS",
            child_prompt="CHILD_PROMPT\nHANDWRITTEN_TOOLS",
            max_depth=1,
            client_factory=factory,
        ).run("root-only benchmark task")

        self.assertEqual(result.answer, "child done")
        root_messages, child_messages = factory.calls
        self.assertEqual(root_messages[0]["content"], "ROOT_PROMPT\nHANDWRITTEN_TOOLS")
        self.assertEqual(child_messages[0]["content"], "CHILD_PROMPT\nHANDWRITTEN_TOOLS")
        self.assertNotIn("SCHEMA_DESCRIPTION", str(factory.calls))
        self.assertEqual(root_messages[1]["content"], "Task:\nroot-only benchmark task")
        self.assertEqual(child_messages[1]["content"], "Delegated task:\nchild assignment")
        self.assertNotIn("root-only benchmark task", str(child_messages))

    def test_prompt_history_tools_and_normal_ready(self) -> None:
        def handler(messages, timeout):
            if len(messages) == 2:
                return "I will think first."
            return '```repl\nanswer["content"] = "done"\nanswer["ready"] = True\n```'

        factory = FakeFactory(handler)
        agent = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            tools={
                "lookup": {
                    "tool": lambda: "TOP_SECRET_VALUE",
                    "description": "Look up a test value.",
                },
                "private_data": "TOP_SECRET_VALUE",
            },
            max_steps=3,
            client_factory=factory,
        )
        result = agent.run("unique original task")

        self.assertEqual(result.answer, "done")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.usage.total_calls, 2)
        self.assertEqual(result.usage.total_input_tokens, 6)
        self.assertEqual(len(factory.calls), 2)
        system = factory.calls[0][0]["content"]
        self.assertIn("lookup", system)
        self.assertIn("Look up a test value.", system)
        self.assertIn("private_data", system)
        self.assertNotIn("TOP_SECRET_VALUE", system)
        self.assertNotIn("planner", system.lower())
        self.assertNotIn("orchestrator", system.lower())
        self.assertIn("Decide whether to:", system)
        self.assertIn("solve the task directly", system)
        self.assertIn("delegate one subtask", system)
        self.assertNotIn("Classify the task before solving", system)
        self.assertNotIn("DECOMPOSABLE", system)
        self.assertEqual(
            sum(
                message["content"].count("unique original task")
                for message in factory.calls[-1]
            ),
            1,
        )
        self.assertEqual(factory.closed_clients, 1)

    def test_persistent_namespace_and_stop_after_ready_block(self) -> None:
        forbidden_calls = []

        def forbidden():
            forbidden_calls.append(True)

        def handler(messages, timeout):
            if len(messages) == 2:
                return "```repl\nx = 40\nprint(x)\n```"
            return (
                "```repl\n"
                'answer["content"] = str(x + 2)\n'
                'answer["ready"] = True\n'
                "```\n"
                "```repl\nforbidden()\n```"
            )

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            tools={"forbidden": forbidden},
            client_factory=factory,
        ).run("calculate")
        self.assertEqual(result.answer, "42")
        self.assertEqual(forbidden_calls, [])
        self.assertEqual(len(result.trace.steps[-1].code_executions), 1)

    def test_python_error_becomes_observation(self) -> None:
        def handler(messages, timeout):
            if len(messages) == 2:
                return "```repl\n1 / 0\n```"
            self.assertIn("Error: ZeroDivisionError", messages[-1]["content"])
            return '```repl\nanswer["content"] = "recovered"\nanswer["ready"] = True\n```'

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            client_factory=FakeFactory(handler),
        ).run("recover")
        self.assertEqual(result.answer, "recovered")

    def test_python_error_stops_later_blocks_in_same_response(self) -> None:
        forbidden_calls = []

        def forbidden():
            forbidden_calls.append(True)

        def handler(messages, timeout):
            if len(messages) == 2:
                return "```repl\n1 / 0\n```\n```repl\nforbidden()\n```"
            self.assertIn("Error: ZeroDivisionError", messages[-1]["content"])
            return '```repl\nanswer["content"] = "recovered"\nanswer["ready"] = True\n```'

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            tools={"forbidden": forbidden},
            client_factory=FakeFactory(handler),
        ).run("recover")
        self.assertEqual(result.answer, "recovered")
        self.assertEqual(forbidden_calls, [])
        self.assertEqual(len(result.trace.steps[0].code_executions), 1)

    def test_optional_repl_block_limit_executes_only_first_block(self) -> None:
        def handler(messages, timeout):
            if len(messages) == 2:
                return "```repl\nx = 1\n```\n```repl\nx = 2\n```"
            return '```repl\nanswer["content"] = str(x)\nanswer["ready"] = True\n```'

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_repl_blocks_per_step=1,
            client_factory=FakeFactory(handler),
        ).run("limit blocks")
        self.assertEqual(result.answer, "1")
        self.assertEqual(len(result.trace.steps[0].code_executions), 1)

    def test_long_observation_is_truncated_for_model_but_preserved_in_trace(self) -> None:
        def handler(messages, timeout):
            if len(messages) == 2:
                return (
                    "```repl\n"
                    "print('HEAD-' + 'x' * 600 + '-TAIL')\n"
                    "raise ValueError('important-error')\n```"
                )
            observation = messages[-1]["content"]
            self.assertLessEqual(len(observation), 240)
            self.assertIn("[truncated by harness:", observation)
            self.assertIn("HEAD-", observation)
            self.assertIn("Error: ValueError: important-error", observation)
            return (
                '```repl\nanswer["content"] = "handled"\n'
                'answer["ready"] = True\n```'
            )

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_observation_chars=240,
            client_factory=FakeFactory(handler),
        ).run("truncate feedback")

        first_step = result.trace.steps[0]
        self.assertTrue(first_step.observation_truncated)
        self.assertLessEqual(len(first_step.model_observation or ""), 240)
        self.assertGreater(len(first_step.code_executions[0].output), 600)
        self.assertIn("-TAIL", first_step.code_executions[0].output)

    def test_forced_final_is_one_extra_call_with_user_instruction(self) -> None:
        def handler(messages, timeout):
            if "No working steps remain" in messages[-1]["content"]:
                return "best final"
            return "Still reasoning without code."

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_steps=2,
            client_factory=factory,
        ).run("eventually finish")
        self.assertEqual(result.answer, "best final")
        self.assertEqual(result.status, "forced_final")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(factory.calls), 3)
        self.assertEqual(factory.calls[-1][-1]["role"], "user")
        self.assertEqual(result.usage.total_calls, 3)
        self.assertEqual(result.trace.forced_final_response, "best final")
        self.assertEqual(result.trace.usage.total_calls, 3)

    def test_environment_done_answer_and_ready_priority(self) -> None:
        checks = []

        def status():
            checks.append(True)
            return EnvironmentStatus(done=True, final_answer="environment")

        ready_factory = FakeFactory(
            lambda messages, timeout: (
                '```repl\nanswer["content"] = "model"\nanswer["ready"] = True\n```'
            )
        )
        ready_result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            termination_check=status,
            client_factory=ready_factory,
        ).run("ready wins")
        self.assertEqual(ready_result.answer, "model")
        self.assertEqual(checks, [])

        env_factory = FakeFactory(lambda messages, timeout: "```repl\nprint('acted')\n```")
        env_result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            termination_check=status,
            client_factory=env_factory,
        ).run("environment wins")
        self.assertEqual(env_result.answer, "environment")
        self.assertEqual(env_result.status, "environment_done")
        self.assertEqual(len(env_factory.calls), 1)

    def test_environment_done_without_answer_forces_once(self) -> None:
        def handler(messages, timeout):
            if "No working steps remain" in messages[-1]["content"]:
                return "summarized terminal state"
            return "```repl\nprint('terminal observation')\n```"

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            termination_check=lambda: EnvironmentStatus(done=True),
            client_factory=factory,
        ).run("terminal")
        self.assertEqual(result.answer, "summarized terminal state")
        self.assertEqual(result.status, "forced_final")
        self.assertEqual(len(factory.calls), 2)

    def test_model_error_carries_latest_response(self) -> None:
        def handler(messages, timeout):
            if len(messages) == 2:
                return "partial reasoning"
            raise RuntimeError("provider unavailable")

        with self.assertRaises(ModelCallError) as raised:
            RecursiveAgent(
                backend_kwargs={"model_name": "fake"},
                client_factory=FakeFactory(handler),
            ).run("fail")
        self.assertEqual(raised.exception.last_response, "partial reasoning")

    def test_timeout_is_shared_and_checked_after_sync_call(self) -> None:
        def handler(messages, timeout):
            time.sleep(0.05)
            return "late"

        with self.assertRaises(TimeoutExceededError):
            RecursiveAgent(
                backend_kwargs={"model_name": "fake"},
                max_run_seconds=0.01,
                client_factory=FakeFactory(handler),
            ).run("timeout")

    def test_cancel_is_cooperative(self) -> None:
        entered = threading.Event()

        def handler(messages, timeout):
            entered.set()
            time.sleep(0.05)
            return "late"

        agent = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            client_factory=FakeFactory(handler),
        )
        caught = []

        def run():
            try:
                agent.run("cancel")
            except Exception as exc:
                caught.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(entered.wait(1))
        agent.cancel()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(caught[0], CancellationError)

    def test_keyboard_interrupt_from_tool_is_cancellation(self) -> None:
        def interrupt():
            raise KeyboardInterrupt

        factory = FakeFactory(
            lambda messages, timeout: "```repl\ninterrupt()\n```"
        )
        with self.assertRaises(CancellationError):
            RecursiveAgent(
                backend_kwargs={"model_name": "fake"},
                tools={"interrupt": interrupt},
                client_factory=factory,
            ).run("interrupt")


if __name__ == "__main__":
    unittest.main()
