from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from recursive_agent.envs import available_environments
from recursive_agent.envs.oolong_synth import (
    CHUNK_CHAR_LIMIT,
    DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT,
    DEFAULT_OOLONG_SYNTH_PROMPT,
    DEFAULT_SYNTH_AGENT_PROMPT,
    DEFAULT_SYNTH_CHILD_PROMPT,
    DEFAULT_SYNTH_ROOT_PROMPT,
    OolongSynthDataset,
    OolongSynthEnvironment,
    evaluate_synth_response,
    parse_synth_response,
    select_protocol_indices,
)
from recursive_agent.prompts import build_initial_user


def sample_row(answer: str = "[3]", *, context_len: int = 1024) -> dict:
    return {
        "id": f"sample-{context_len}",
        "context_window_id": f"window-{context_len}",
        "context_window_text": (
            "Labels are 'entity' and 'human being'.\n"
            "Date: Jan 01, 2024 || User: 100 || Instance: Who invented radio?\n"
        ),
        "question": (
            "How many data points should be classified as label 'human being'? "
            "Give your final answer in the form 'Answer: number'."
        ),
        "answer": answer,
        "answer_type": "ANSWER_TYPE.NUMERIC",
        "context_len": context_len,
        "dataset": "trec_coarse",
        "task_group": "counting",
        "task": "TASK_TYPE.NUMERIC_ONE_CLASS",
        "input_subset": "False",
    }


class OolongSynthEnvironmentTests(unittest.TestCase):
    def test_local_dataset_and_private_context_exclude_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            path.write_text(json.dumps(sample_row()) + "\n", encoding="utf-8")
            dataset = OolongSynthDataset(data_path=path)
            self.assertEqual(dataset[0].sample_id, "sample-1024")
            environment = OolongSynthEnvironment(samples=[sample_row()])
        self.assertNotIn("answer", environment.context)
        self.assertNotIn("gold", environment.context)
        self.assertEqual(environment.context["oolong_role"], "root")
        self.assertNotIn("context_len", environment.context)
        self.assertNotIn("context_chars", environment.context)
        self.assertNotIn("prompt_flow", environment.context)

    def test_downloaded_validation_parquet_directory_loads(self) -> None:
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets optional dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            data_dir.mkdir()
            Dataset.from_list([sample_row()]).to_parquet(
                str(data_dir / "validation-00000-of-00001.parquet")
            )
            dataset = OolongSynthDataset(data_path=directory)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].dataset, "trec_coarse")

    def test_default_root_discovers_downloaded_syth_directory(self) -> None:
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets optional dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "syth" / "data"
            data_dir.mkdir(parents=True)
            Dataset.from_list([sample_row()]).to_parquet(
                str(data_dir / "validation-00000-of-00001.parquet")
            )
            dataset = OolongSynthDataset(oolong_root=root, split="validation")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].dataset, "trec_coarse")
            self.assertEqual(dataset.metadata()["source"], str(root / "syth"))

    def test_official_numeric_parser_and_score(self) -> None:
        self.assertEqual(parse_synth_response("Answer: [4]")[0], "4")
        self.assertEqual(
            parse_synth_response('```repl\nsubmit_answer("Label: ham")\n```')[0],
            "ham",
        )
        evaluation = evaluate_synth_response(
            "[3]", "Answer: 4", "ANSWER_TYPE.NUMERIC"
        )
        self.assertEqual(evaluation.gold, 3)
        self.assertEqual(evaluation.score, 0.75)
        self.assertEqual(
            evaluate_synth_response(
                "['entity']", "Label: entity", "ANSWER_TYPE.LABEL"
            ).score,
            1.0,
        )

    def test_environment_submission_has_no_gold_in_report(self) -> None:
        environment = OolongSynthEnvironment(samples=[sample_row()])
        self.assertIsNone(environment.max_repl_blocks_per_step)
        report = environment.submit_answer("Answer: 3")
        self.assertEqual(report["score"], 1.0)
        self.assertEqual(report["attempted_parse"], "3")
        self.assertNotIn("gold_answer", report)
        self.assertTrue(environment.status().done)

    def test_protocol_selection_is_deterministic_and_stratified(self) -> None:
        metadata = [
            {
                "index": bucket * 20 + offset,
                "id": f"{bucket}-{offset}",
                "context_len": 2 ** (bucket + 10),
                "dataset": "trec_coarse" if offset % 2 else "spam",
            }
            for bucket in range(13)
            for offset in range(20)
        ]
        first = select_protocol_indices(metadata, sample_count=199, seed=31)
        second = select_protocol_indices(metadata, sample_count=199, seed=31)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 199)
        lengths = Counter(metadata[index]["context_len"] for index in first)
        self.assertEqual(set(lengths.values()), {15, 16})

    def test_prompt_uses_single_32k_root_worker_flow(self) -> None:
        self.assertEqual(CHUNK_CHAR_LIMIT, 32_768)
        self.assertEqual(DEFAULT_SYNTH_AGENT_PROMPT, DEFAULT_OOLONG_SYNTH_PROMPT)
        for text in (
            'question = context["question"]',
            'dataset_intro = context["dataset_intro"]',
            'source = context["context_window_text"]',
            "First check `len(source)`",
            "If `len(source) <= 32_768`",
            "If `len(source) > 32_768`",
            "complete `Date:` record boundaries",
            "spawn_subagents",
            "read the returned results once",
            "### Example: direct processing",
            "### Example: delegated processing",
            "submit_answer(\"Answer: ham is more common than spam\")",
        ):
            self.assertIn(text, DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("submit_answer", DEFAULT_SYNTH_AGENT_PROMPT)

        environment = OolongSynthEnvironment(samples=[sample_row()])
        self.assertEqual(environment.root_prompt, DEFAULT_SYNTH_ROOT_PROMPT)
        self.assertEqual(environment.child_prompt, DEFAULT_SYNTH_CHILD_PROMPT)
        self.assertEqual(
            DEFAULT_SYNTH_ROOT_PROMPT.count(DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT),
            1,
        )
        self.assertEqual(
            DEFAULT_SYNTH_CHILD_PROMPT.count(DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT),
            1,
        )
        self.assertNotIn("Custom tools:", DEFAULT_SYNTH_ROOT_PROMPT)
        self.assertNotIn("Custom tools:", DEFAULT_SYNTH_CHILD_PROMPT)
        self.assertIn("Labels are", environment.context["dataset_intro"])
        self.assertIn("submit_answer(...)` exactly once", environment.task)
        self.assertIsNone(environment.delegated_task_prompt)
        delegated_task = (
            "Process chunk 2 and return concise mergeable text containing "
            "chunk_id, rows processed, and the required statistics."
        )
        root_user = build_initial_user(environment.task)
        child_user = build_initial_user(delegated_task, delegated=True)
        self.assertIn("Solve this Oolong-Synthetic benchmark task", root_user)
        self.assertIn("submit_answer", root_user)
        self.assertNotIn("Solve this Oolong-Synthetic benchmark task", child_user)
        self.assertNotIn("submit_answer", child_user)
        self.assertIn(delegated_task, child_user)
        self.assertIn("oolong_synth", available_environments())

    def test_prompt_chunking_example_is_executable_and_preserves_records(self) -> None:
        marker = "for chunk_id, chunk in enumerate(chunks):"
        start = DEFAULT_SYNTH_AGENT_PROMPT.index(marker)
        block_start = DEFAULT_SYNTH_AGENT_PROMPT.rfind("```repl\n", 0, start)
        chunking_code = DEFAULT_SYNTH_AGENT_PROMPT[
            block_start + len("```repl\n") :
        ].split("```", 1)[0]
        records = [
            "Date: Jan 01, 2024 || User: 1 || Instance: alpha\n",
            "Date: Jan 02, 2024 || User: 2 || Instance: beta\n",
            "Date: Jan 03, 2024 || User: 3 || Instance: gamma\n",
        ]
        captured = []

        def spawn_subagents(requests):
            captured.extend(requests)
            return ["partial result" for _ in requests]

        namespace = {
            "context": {
                "question": "Count the records.",
                "dataset_intro": "Classification instructions.",
                "context_window_text": "Dataset introduction.\n" + "".join(records),
            },
            "question": "Count the records.",
            "dataset_intro": "Classification instructions.",
            "source": "Dataset introduction.\n" + "".join(records),
            "chunks": ["".join(records)],
            "spawn_subagents": spawn_subagents,
        }
        exec(chunking_code, namespace)

        self.assertEqual(len(captured), 1)
        child_context = captured[0]["context"]
        self.assertEqual(child_context["context_window_text"], "".join(records))
        self.assertEqual(child_context["chunk_id"], 0)


if __name__ == "__main__":
    unittest.main()
