from __future__ import annotations

import threading
import time
import unittest

from recursive_agent import RecursiveAgent

from .fakes import FakeFactory, initial_task


class RecursionTests(unittest.TestCase):
    def test_child_gets_private_context_without_parent_variables(self) -> None:
        original = [1]

        def handler(messages, timeout):
            task = initial_task(messages)
            if task == "root":
                if len(messages) == 2:
                    return (
                        "```repl\n"
                        "parent_only = 'secret'\n"
                        "child = spawn_subagent('child', context)\n"
                        "print(child, context)\n"
                        "```"
                    )
                return (
                    '```repl\nanswer["content"] = child + "|" + str(context)\n'
                    'answer["ready"] = True\n```'
                )
            return (
                "```repl\n"
                "context.append(9)\n"
                "try:\n    parent_only\n    isolated = False\n"
                "except NameError:\n    isolated = True\n"
                'answer["content"] = str(context) + ":" + str(isolated)\n'
                'answer["ready"] = True\n```'
            )

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_depth=2,
            client_factory=factory,
        ).run("root", original)
        self.assertEqual(result.answer, "[1, 9]:True|[1]")
        self.assertEqual(original, [1])
        self.assertEqual(factory.created, 2)
        self.assertEqual(result.usage.total_calls, 3)
        self.assertEqual(len(result.trace.children), 1)
        self.assertEqual(result.trace.children[0].task, "child")
        self.assertEqual(result.trace.children[0].status, "completed")
        system_prompts = [call[0]["content"] for call in factory.calls]
        self.assertEqual(system_prompts[0], system_prompts[1])
        self.assertIn("Every agent has the same capabilities", system_prompts[0])
        self.assertTrue(factory.calls[0][1]["content"].startswith("Task:\nroot"))
        self.assertTrue(
            factory.calls[1][1]["content"].startswith("Delegated task:\nchild")
        )
        for messages in factory.calls:
            self.assertEqual(
                sum(message["role"] == "system" for message in messages),
                1,
            )

    def test_child_keeps_only_its_own_message_history(self) -> None:
        def handler(messages, timeout):
            task = initial_task(messages)
            if task == "root-history-task":
                if len(messages) == 2:
                    return (
                        "```repl\n"
                        "child_result = spawn_subagent('child-history-task', {'value': 7})\n"
                        "print(child_result)\n```"
                    )
                return (
                    '```repl\nanswer["content"] = child_result\n'
                    'answer["ready"] = True\n```'
                )
            if len(messages) == 2:
                return "```repl\nprint(context['value'])\n```"
            return (
                '```repl\nanswer["content"] = "child-done"\n'
                'answer["ready"] = True\n```'
            )

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_depth=1,
            client_factory=factory,
        ).run("root-history-task")

        self.assertEqual(result.answer, "child-done")
        child_second_call = factory.calls[2]
        self.assertEqual(len(child_second_call), 4)
        self.assertEqual(child_second_call[0]["role"], "system")
        self.assertTrue(
            child_second_call[1]["content"].startswith(
                "Delegated task:\nchild-history-task"
            )
        )
        self.assertNotIn("root-history-task", str(child_second_call))
        self.assertIn("REPL output:\n7", child_second_call[-1]["content"])
        self.assertEqual(
            sum(message["role"] == "system" for message in child_second_call),
            1,
        )

    def test_nested_child_has_same_spawn_capability(self) -> None:
        def handler(messages, timeout):
            task = initial_task(messages)
            if task == "root":
                return (
                    "```repl\nr = spawn_subagent('middle')\n"
                    'answer["content"] = r\nanswer["ready"] = True\n```'
                )
            if task == "middle":
                return (
                    "```repl\nr = spawn_subagent('leaf')\n"
                    'answer["content"] = "middle:" + r\nanswer["ready"] = True\n```'
                )
            return '```repl\nanswer["content"] = "leaf"\nanswer["ready"] = True\n```'

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_depth=2,
            client_factory=FakeFactory(handler),
        ).run("root")
        self.assertEqual(result.answer, "middle:leaf")
        self.assertEqual(result.trace.children[0].children[0].depth, 2)

    def test_depth_limit_returns_error_without_fallback_call(self) -> None:
        def handler(messages, timeout):
            return (
                "```repl\nr = spawn_subagent('blocked')\n"
                'answer["content"] = r\nanswer["ready"] = True\n```'
            )

        factory = FakeFactory(handler)
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_depth=0,
            client_factory=factory,
        ).run("root")
        self.assertEqual(result.answer, "Error: maximum recursion depth (0) reached")
        self.assertEqual(factory.created, 1)
        self.assertEqual(len(factory.calls), 1)

    def test_batch_is_concurrent_ordered_limited_and_keeps_failures(self) -> None:
        state_lock = threading.Lock()
        active = 0
        peak = 0

        def work(delay, value):
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(delay)
            with state_lock:
                active -= 1
            return value

        def handler(messages, timeout):
            task = initial_task(messages)
            if task == "root":
                if len(messages) == 2:
                    return (
                        "```repl\n"
                        "results = spawn_subagents(["
                        "{'task': 'slow', 'context': [0.08, 'A']}, "
                        "{'task': 'bad'}, "
                        "{'task': 'fast', 'context': [0.02, 'C']}])\n"
                        "print(results)\n```"
                    )
                return (
                    '```repl\nanswer["content"] = "|".join(results)\n'
                    'answer["ready"] = True\n```'
                )
            if task == "bad":
                raise RuntimeError("child provider failure")
            return (
                "```repl\n"
                "value = work(context[0], context[1])\n"
                "print('child-' + value)\n"
                'answer["content"] = value\nanswer["ready"] = True\n```'
            )

        factory = FakeFactory(handler)
        started = time.perf_counter()
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            tools={"work": {"tool": work, "description": "Delayed test work."}},
            max_depth=1,
            max_concurrent_subagents=2,
            client_factory=factory,
        ).run("root")
        elapsed = time.perf_counter() - started

        self.assertEqual(result.answer, "A|Error: subagent model call failed|C")
        self.assertEqual(peak, 2)
        self.assertLess(elapsed, 0.14)
        child_outputs = {
            child.task: child.steps[0].code_executions[0].output
            for child in result.trace.children
            if child.steps
        }
        self.assertEqual(child_outputs["slow"], "child-A")
        self.assertEqual(child_outputs["fast"], "child-C")

    def test_empty_batch(self) -> None:
        factory = FakeFactory(
            lambda messages, timeout: (
                "```repl\nr = spawn_subagents([])\n"
                'answer["content"] = str(r)\nanswer["ready"] = True\n```'
            )
        )
        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            client_factory=factory,
        ).run("root")
        self.assertEqual(result.answer, "[]")
        self.assertEqual(factory.created, 1)


if __name__ == "__main__":
    unittest.main()
