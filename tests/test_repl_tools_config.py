from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recursive_agent import ConfigurationError, RecursiveAgent, load_model_config
from recursive_agent.clients import get_client
from recursive_agent.repl import ReplSession, find_repl_blocks


class ReplToolsConfigTests(unittest.TestCase):
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

    def test_invalid_tool_names_are_rejected(self) -> None:
        for tools in ({"answer": 1}, {"not-valid": 1}, {"class": 1}):
            with self.subTest(tools=tools), self.assertRaises(ConfigurationError):
                RecursiveAgent(backend_kwargs={"model_name": "fake"}, tools=tools)

    def test_repl_block_parser(self) -> None:
        self.assertEqual(
            find_repl_blocks("x\n```repl\nprint(1)\n```\ny\n```repl\na=2\n```"),
            ["print(1)", "a=2"],
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
