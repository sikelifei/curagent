from __future__ import annotations

import unittest
from pathlib import Path

from recursive_agent.envs import available_environments
from recursive_agent.envs.deepdive import DeepDiveEnvironment, DeepDiveSample
from recursive_agent.envs.deepdive.prompts import DEFAULT_DEEPDIVE_AGENT_PROMPT
from recursive_agent.envs.deepdive.scoring import parse_deepdive_judgment
from recursive_agent.envs.runner import run_environment

from .fakes import FakeFactory, initial_task


ROOT = Path(__file__).resolve().parents[1]


class _FakeHarness:
    def search_web(self, query: str, max_results: int = 5):
        return {
            "query": query,
            "results": [{"url": "https://example.test", "content": "evidence"}],
        }

    def view_webpage_content(self, url: str) -> str:
        return f"page:{url}"


class DeepDiveEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = DeepDiveSample(
            task_id="deepdive.qa_rl.0",
            question="Question?",
            answer="ground truth secret",
            split="qa_rl",
            index=0,
            metadata={},
        )

    def test_prompt_ports_strategy_but_keeps_curagent_api(self) -> None:
        prompt = DEFAULT_DEEPDIVE_AGENT_PROMPT
        self.assertIn("RESEARCH STRATEGY:", prompt)
        self.assertIn("DELEGATION STRATEGY:", prompt)
        self.assertIn("spawn_subagent(task, context=None)", prompt)
        self.assertIn("spawn_subagents(requests)", prompt)
        self.assertIn('answer["ready"] = True', prompt)
        self.assertIn("one executable `repl` block", prompt)
        self.assertNotIn("launch_subagent", prompt)
        self.assertNotIn("finish(...)", prompt)
        self.assertNotIn("<python>", prompt)
        self.assertNotIn("await", prompt.lower().replace("do not use `await`", ""))

    def test_environment_uses_sync_deepdive_tools_and_hides_answer(self) -> None:
        environment = DeepDiveEnvironment(sample=self.sample, harness=_FakeHarness())
        self.assertEqual(
            environment.task,
            "Answer this DeepDive factual question.\n\nQuestion:\nQuestion?",
        )
        self.assertNotIn("ground truth secret", str(environment.context))
        result = environment.search_web("query", 3)
        self.assertEqual(result["results"][0]["content"], "evidence")
        self.assertEqual(
            environment.view_webpage_content("https://example.test"),
            "page:https://example.test",
        )
        self.assertEqual(
            environment.report()["tool_call_counts"],
            {"search_web": 1, "view_webpage_content": 1},
        )
        environment.close()

    def test_curagent_runs_recursive_deepdive_prompt_natively(self) -> None:
        def handler(messages, timeout):
            del timeout
            system = messages[0]["content"]
            self.assertNotIn("recursive agent harness", system)
            self.assertNotIn("Custom tools:", system)
            self.assertIn("DeepDive web research", system)
            self.assertNotIn("launch_subagent", system)
            if messages[1]["content"].startswith("Task:\n"):
                return (
                    "<thought>delegate one verification</thought>\n"
                    "<repl>report = spawn_subagent('verify')\n"
                    "answer['content'] = report\n"
                    "answer['ready'] = True</repl>"
                )
            return (
                "<thought>search directly</thought>\n"
                "<repl>result = search_web('query')\n"
                "answer['content'] = result['results'][0]['content']\n"
                "answer['ready'] = True</repl>"
            )

        environment = DeepDiveEnvironment(sample=self.sample, harness=_FakeHarness())
        run = run_environment(
            environment,
            model_config=ROOT / "configs" / "model_api.local.yaml",
            agent_kwargs={
                "client_factory": FakeFactory(handler),
                "max_depth": 1,
                "max_steps": 2,
            },
        )
        self.assertEqual(run.agent_result.answer, "evidence")
        self.assertEqual(len(run.agent_result.trace.children), 1)
        self.assertEqual(run.environment_report["tool_call_counts"]["search_web"], 1)

    def test_registry_and_judgment_parser(self) -> None:
        self.assertIn("deepdive", available_environments())
        parsed = parse_deepdive_judgment(
            '```json\n{"reason": "equivalent", "success": true}\n```'
        )
        self.assertEqual(parsed, {"reason": "equivalent", "success": True})


if __name__ == "__main__":
    unittest.main()
