from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recursive_agent import ConfigurationError, RecursiveAgent, load_model_config
from recursive_agent.clients import get_client
from recursive_agent.repl import NodeTermination, ReplSession, find_repl_blocks
from recursive_agent.tools import (
    CapabilityCollection,
    format_tools_for_prompt,
    parse_tools,
    tool_values,
)


class ReplToolsConfigTests(unittest.TestCase):
    def test_repl_displays_bare_top_level_expressions(self) -> None:
        repl = ReplSession(
            context=None,
            tools={"lookup": lambda: {"value": 7}},
            spawn_subagent=lambda task, context=None: "child",
            spawn_subagents=lambda requests: [],
        )
        bare = repl.execute("lookup()")
        explicit = repl.execute("print(lookup())")
        expression = repl.execute("x = 40\nx + 2")
        self.assertEqual(bare.trace.output, "{'value': 7}")
        self.assertEqual(explicit.trace.output, "{'value': 7}")
        self.assertEqual(expression.trace.output, "42")

    def test_scaffold_and_tools_are_restored_without_clearing_valid_answer(self) -> None:
        marker = object()
        repl = ReplSession(
            context={"a": 1},
            tools={"tool": marker},
            spawn_subagent=lambda task, context=None: "child",
            spawn_subagents=lambda requests: [],
        )
        repl.execute(
            "tool = None\nSHOW_VARS = None\nspawn_subagent = None\n"
            "answer['draft'] = 3\ncontext['b'] = 2"
        )
        result = repl.execute(
            "assert tool is not None\nassert callable(SHOW_VARS)\n"
            "assert callable(spawn_subagent)\nassert answer['draft'] == 3\n"
            "assert context == {'a': 1, 'b': 2}\n"
            "answer['content'] = 'ok'\nanswer['ready'] = True"
        )
        self.assertTrue(result.answer_ready)
        self.assertEqual(result.answer_content, "ok")
        self.assertIn("context", result.trace.variables)

    def test_capability_binding_replaces_stale_action_space_entries(self) -> None:
        repl = ReplSession(
            context=None,
            capabilities=CapabilityCollection({"old_tool": lambda: "old"}),
        )
        self.assertEqual(repl.execute("old_tool()").trace.output, "'old'")

        repl.bind_capabilities({"new_tool": lambda: "new"})
        result = repl.execute("new_tool()")
        self.assertEqual(result.trace.output, "'new'")
        stale = repl.execute("old_tool()")
        self.assertIn("NameError", stale.trace.error or "")

    def test_legacy_parse_tools_preserves_environment_owned_names_only(self) -> None:
        finish = object()
        return_to_parent = object()
        parsed = parse_tools(
            {
                "finish": {"tool": finish, "description": "finish the task"},
                "return_to_parent": {
                    "tool": return_to_parent,
                    "description": "return the result",
                },
            }
        )

        self.assertIs(parsed["finish"].value, finish)
        self.assertEqual(parsed["finish"].description, "finish the task")
        self.assertIs(parsed["return_to_parent"].value, return_to_parent)
        self.assertEqual(parsed["return_to_parent"].description, "return the result")
        self.assertIs(tool_values(parsed)["finish"], finish)
        self.assertIn("`finish`: finish the task", format_tools_for_prompt(parsed) or "")

        for name in ("finish", "return_to_parent"):
            with self.subTest(capability_name=name), self.assertRaises(ConfigurationError):
                CapabilityCollection({name: object()})

        for name in (
            "spawn_subagent",
            "spawn_subagents",
            "context",
            "answer",
            "SHOW_VARS",
        ):
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                parse_tools({name: object()})

    def test_top_level_await_imports_and_persistent_values(self) -> None:
        repl = ReplSession(context=None)
        result = repl.execute(
            "import asyncio\n"
            "import math\n"
            "value = await asyncio.sleep(0, result=math.sqrt(81))\n"
            "value"
        )
        self.assertIsNone(result.trace.error)
        self.assertEqual(result.trace.output, "9.0")
        self.assertEqual(repl.execute("value + 1").trace.output, "10.0")

    def test_top_level_await_works_when_sync_execute_runs_in_event_loop(self) -> None:
        repl = ReplSession(context=None)

        async def call_execute() -> object:
            return repl.execute("import asyncio\nawait asyncio.sleep(0)\n42")

        result = asyncio.run(call_execute())
        self.assertEqual(result.trace.output, "42")  # type: ignore[union-attr]

    def test_framework_termination_is_not_reported_as_python_error(self) -> None:
        def finish(result: object = None) -> None:
            raise NodeTermination("finish", result)

        repl = ReplSession(capabilities={"finish": finish})
        execution = repl.execute("print('before')\nfinish({'ok': True})\nprint('after')")
        self.assertTrue(execution.terminated)
        self.assertEqual(execution.termination_kind, "finish")
        self.assertEqual(execution.termination_result, {"ok": True})
        self.assertIsNone(execution.trace.error)
        self.assertEqual(execution.trace.output, "before")

    def test_framework_termination_bypasses_ordinary_exception_handlers(self) -> None:
        def finish(result: object = None) -> None:
            raise NodeTermination("finish", result)

        repl = ReplSession(capabilities={"finish": finish})
        execution = repl.execute(
            "try:\n"
            "    finish('done')\n"
            "except Exception:\n"
            "    print('caught')\n"
            "print('after')"
        )

        self.assertTrue(issubclass(NodeTermination, BaseException))
        self.assertFalse(issubclass(NodeTermination, Exception))
        self.assertTrue(execution.terminated)
        self.assertEqual(execution.termination_kind, "finish")
        self.assertEqual(execution.termination_result, "done")
        self.assertIsNone(execution.trace.error)
        self.assertEqual(execution.trace.output, "")

    def test_awaited_framework_termination_is_captured_by_repl_boundary(self) -> None:
        async def finish(result: object = None) -> None:
            raise NodeTermination("finish", result)

        repl = ReplSession(capabilities={"finish": finish})
        execution = repl.execute("await finish({'ok': True})\nprint('after')")

        self.assertTrue(execution.terminated)
        self.assertEqual(execution.termination_kind, "finish")
        self.assertEqual(execution.termination_result, {"ok": True})
        self.assertIsNone(execution.trace.error)
        self.assertEqual(execution.trace.output, "")

    def test_invalid_tool_names_are_rejected(self) -> None:
        for tools in ({"answer": 1}, {"not-valid": 1}, {"class": 1}):
            with self.subTest(tools=tools), self.assertRaises(ConfigurationError):
                RecursiveAgent(backend_kwargs={"model_name": "fake"}, tools=tools)

    def test_repl_block_parser(self) -> None:
        self.assertEqual(
            find_repl_blocks("x\n```repl\nprint(1)\n```\ny\n```repl\na=2\n```"),
            ["print(1)", "a=2"],
        )

    def test_repl_block_parser_accepts_xml_style_blocks_in_source_order(self) -> None:
        self.assertEqual(
            find_repl_blocks(
                "<repl>print(1)</repl>\n"
                "```repl\nprint(2)\n```\n"
                "<REPL>print(3)</REPL>"
            ),
            ["print(1)", "print(2)", "print(3)"],
        )

    def test_repl_block_parser_accepts_python_alias(self) -> None:
        self.assertEqual(
            find_repl_blocks(
                "```python\nprint(1)\n```\n"
                "<python>print(2)</python>\n"
                "```repl\nprint(3)\n```"
            ),
            ["print(1)", "print(2)", "print(3)"],
        )

    def test_model_yaml_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(
                "model:\n"
                "  type: api\n"
                "  api:\n"
                "    api_key: secret\n"
                "    base_url: https://example.test/v1\n"
                "    model: example\n"
                "    temperature: 0.2\n",
                encoding="utf-8",
            )
            backend, kwargs = load_model_config(path)
        self.assertEqual(backend, "openai")
        self.assertEqual(kwargs["model_name"], "example")
        self.assertEqual(kwargs["sampling_args"], {"temperature": 0.2})

    def test_invalid_agent_limits(self) -> None:
        for kwargs in (
            {"max_steps": 0},
            {"max_depth": -1},
            {"max_concurrent_subagents": 0},
            {"max_run_seconds": 0},
            {"max_observation_chars": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                RecursiveAgent(backend_kwargs={"model_name": "fake"}, **kwargs)

    def test_openai_compatible_provider_routes(self) -> None:
        with patch("recursive_agent.clients.OpenAIClient") as client_class:
            for backend, expected_url in (
                ("openrouter", "https://openrouter.ai/api/v1"),
                ("vercel", "https://ai-gateway.vercel.sh/v1"),
                ("portkey", "https://api.portkey.ai/v1"),
            ):
                get_client(backend, {"model_name": "test"})
                self.assertEqual(client_class.call_args.kwargs["base_url"], expected_url)
            get_client(
                "vllm",
                {"model_name": "test", "base_url": "http://localhost:8000/v1"},
            )
            self.assertEqual(client_class.call_args.kwargs["api_key"], "not-needed")

    def test_unsupported_backend_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            RecursiveAgent(backend="unknown", backend_kwargs={"model_name": "fake"})


if __name__ == "__main__":
    unittest.main()
