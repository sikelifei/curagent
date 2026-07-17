from __future__ import annotations

import ast
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from recursive_agent.envs import available_environments
from recursive_agent.envs.oolong_synth import (
    DEFAULT_SYNTH_AGENT_PROMPT,
    OolongSynthDataset,
    OolongSynthEnvironment,
    PROMPT_FLOWS,
    build_flow_prompt,
    evaluate_synth_response,
    parse_synth_response,
    select_protocol_indices,
)


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
        self.assertIn("child_task_template", environment.context)
        self.assertEqual(environment.context["oolong_role"], "root")
        self.assertNotIn("context_len", environment.context)
        self.assertNotIn("context_chars", environment.context)

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

    def test_prompt_uses_model_selected_decomposition_and_registry_is_loaded(self) -> None:
        self.assertIn('context["oolong_role"]', DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("Measure its character length", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("does not pre-split", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("metadata-only questions", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("expected_rows", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("NEVER call `submit_answer`", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("exactly this Markdown shape", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("Never use `python`, `<repl>`, bare `repl`", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("first response must be exactly the single block", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn('"context_chars": len(source)', DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("This is not\n   JSON or JSONL", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("records are intentionally UNLABELED", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertIn("route immediately", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertNotIn("8_000", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertNotIn("max_chars=", DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertNotIn("EXACTLY FOUR", DEFAULT_SYNTH_AGENT_PROMPT)
        blocks = DEFAULT_SYNTH_AGENT_PROMPT.split("```repl\n")[1:]
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks:
            ast.parse(block.split("```", 1)[0])
        self.assertIn("oolong_synth", available_environments())

    def test_prompt_flow_switch_preserves_default_and_selects_variants(self) -> None:
        self.assertEqual(PROMPT_FLOWS, ("adaptive_flat", "paged_flat", "hierarchical"))
        self.assertEqual(build_flow_prompt("adaptive_flat"), DEFAULT_SYNTH_AGENT_PROMPT)
        self.assertNotEqual(
            build_flow_prompt("paged_flat"), DEFAULT_SYNTH_AGENT_PROMPT
        )
        self.assertNotEqual(
            build_flow_prompt("hierarchical"), DEFAULT_SYNTH_AGENT_PROMPT
        )
        environment = OolongSynthEnvironment(
            samples=[sample_row()], prompt_flow="hierarchical"
        )
        self.assertEqual(environment.context["prompt_flow"], "hierarchical")
        self.assertIn("can_delegate", environment.context["child_task_template"])
        with self.assertRaises(ValueError):
            OolongSynthEnvironment(samples=[sample_row()], prompt_flow="unknown")


if __name__ == "__main__":
    unittest.main()
