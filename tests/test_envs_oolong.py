from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from recursive_agent.envs import available_environments, create_environment, run_environment
from recursive_agent.envs.oolong import (
    DEFAULT_OOLONG_AGENT_PROMPT,
    OolongDataset,
    OolongEnvironment,
    parse_response,
    score_answer,
)

from .fakes import FakeFactory


def sample_row(answer: str = "2") -> dict:
    return {
        "id": "sample-1",
        "context_window_id": "window-1",
        "context_window_text": "D&D context with [START OF EPISODE] facts [END OF EPISODE]",
        "question": "How many rolls?",
        "answer": answer,
        "answer_type": "ANSWER_TYPE.NUMERIC",
        "question_type": "singledoc_rolls",
        "episodes": [1],
        "campaign": "campaign1",
        "context_len": 42,
    }


class OolongEnvironmentTests(unittest.TestCase):
    def test_dataset_loads_local_jsonl_and_normalizes_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(json.dumps(sample_row()) + "\n", encoding="utf-8")
            dataset = OolongDataset(oolong_root=directory, split="test")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].sample_id, "sample-1")
            self.assertEqual(dataset[-1].episodes, (1,))
            self.assertEqual(dataset.metadata()["source"], str(path.resolve()))

    def test_dataset_uses_published_loader_when_no_local_split_exists(self) -> None:
        calls = []

        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return [sample_row()]

        with tempfile.TemporaryDirectory() as directory:
            dataset = OolongDataset(
                oolong_root=directory,
                split="test",
                loader=loader,
            )
        self.assertEqual(calls, [(
            ("oolongbench/oolong-real", "dnd"),
            {"split": "test"},
        )])
        self.assertEqual(dataset[0].question, "How many rolls?")

    def test_official_style_parser_and_partial_scores(self) -> None:
        self.assertEqual(parse_response(r"reasoning \boxed{2}"), (2, "high"))
        self.assertEqual(score_answer("2", 3), 0.75)
        self.assertEqual(score_answer("alpha,beta", "beta,gamma"), 0.5)
        self.assertEqual(score_answer("Alpha", "alpha"), 1.0)

    def test_oolong_few_shot_repl_blocks_are_valid_python(self) -> None:
        blocks = []
        marker = "```repl"
        prompt = DEFAULT_OOLONG_AGENT_PROMPT
        start = 0
        while True:
            start = prompt.find(marker, start)
            if start < 0:
                break
            code_start = start + len(marker)
            code_end = prompt.find("```", code_start)
            self.assertGreaterEqual(code_end, 0)
            blocks.append(prompt[code_start:code_end].strip().lstrip())
            start = code_end + 3
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            ast.parse(block)

    def test_environment_submits_and_reports_score(self) -> None:
        environment = OolongEnvironment(samples=[sample_row()])
        self.assertIn("How many rolls?", environment.task)
        self.assertIn("Oolong-real environment guidance", environment.agent_prompt)
        self.assertIn("spawn_subagents", environment.agent_prompt)
        self.assertIn("chunk_text", environment.agent_prompt)
        self.assertIn("rolls", environment.agent_prompt)
        self.assertIn("by_natural_value", environment.agent_prompt)
        self.assertIn("[START OF EPISODE]", environment.agent_prompt)
        self.assertIn("rfind", environment.agent_prompt)
        self.assertEqual(
            environment.context["context_window_text"],
            sample_row()["context_window_text"],
        )
        self.assertNotIn("context_window_text", environment.observe())
        self.assertFalse(environment.status().done)
        result = environment.submit_answer(r"\boxed{2}")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["attempted_parse"], 2)
        self.assertTrue(environment.status().done)
        self.assertEqual(environment.status().final_answer, r"\boxed{2}")
        environment.close()
        environment.close()

    def test_registry_and_runner_keep_root_prompt_untouched(self) -> None:
        self.assertIn("oolong", available_environments())
        environment = create_environment("Oolong", samples=[sample_row()])
        config = Path(tempfile.mkdtemp()) / "model.yaml"
        config.write_text(
            "model:\n"
            "  type: api\n"
            "  api:\n"
            "    model: fake\n",
            encoding="utf-8",
        )

        def handler(messages, timeout):
            observations = [
                message["content"]
                for message in messages
                if message["role"] == "user"
                and message["content"].startswith("REPL output:")
            ]
            if not observations:
                return "```repl\nprint(len(context['context_window_text']))\n```"
            return "```repl\nprint(submit_answer(r'\\boxed{2}'))\n```"

        factory = FakeFactory(handler)
        run = run_environment(
            environment,
            model_config=config,
            agent_kwargs={
                "max_steps": 3,
                "max_depth": 1,
                "client_factory": factory,
            },
        )
        self.assertEqual(run.environment_report["score"], 1.0)
        self.assertEqual(run.agent_result.status, "environment_done")
        self.assertNotIn("D&D context with", run.task_prompt)
        self.assertIn("context_window_text", run.task_prompt)
        self.assertIn("Oolong-real environment guidance", run.system_prompt)
        self.assertIn("chunk_text", run.system_prompt)
        self.assertIn("spawn_subagents", run.system_prompt)
        self.assertEqual(run.system_prompt, environment.root_prompt)
        self.assertNotIn("recursive agent harness", run.system_prompt)
        self.assertNotIn("Custom tools:", run.system_prompt)
        first_execution = run.to_trace_dict()["agent_result"]["trace"]["steps"][0][
            "code_executions"
        ][0]
        self.assertEqual(
            first_execution["stdout"],
            str(len(sample_row()["context_window_text"])),
        )


if __name__ == "__main__":
    unittest.main()
