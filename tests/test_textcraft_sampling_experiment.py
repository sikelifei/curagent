from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from recursive_agent.envs.textcraft_synth.dataset import TextCraftDataset
from scripts.run_textcraft_sampling_experiment import (
    EXPERIMENT_FILES,
    _run_plan,
    append_jsonl,
    completed_rollout_ids,
    extract_rollout_metrics,
    normalize_official_row,
    read_jsonl,
    select_medium_rows,
)


def _official_row(row_id: str, difficulty: str = "medium") -> dict:
    return {
        "id": row_id,
        "goal": "Craft 3x target",
        "misc": {
            "target_items": {"target": 3},
            "initial_inventory": {"ore": 8},
            "difficulty": difficulty,
            "max_depth": 3,
            "gold_trajectory": [
                {
                    "action": "craft",
                    "target": ["intermediate", 2],
                    "ingredients": {"ore": 4},
                    "result_count": 6,
                },
                {
                    "action": "craft",
                    "target": ["target", 1],
                    "ingredients": {"intermediate": 2},
                    "result_count": 2,
                },
            ],
        },
    }


class TextCraftSamplingExperimentTests(unittest.TestCase):
    def test_normalizes_official_execution_counts_for_environment(self) -> None:
        sample = normalize_official_row(_official_row("medium-1"), split="val")

        self.assertEqual(sample["id"], "medium-1")
        self.assertEqual(
            sample["recipes"]["intermediate"],
            [{"ingredients": {"ore": 2}, "result_count": 3}],
        )
        self.assertEqual(
            sample["recipes"]["target"],
            [{"ingredients": {"intermediate": 2}, "result_count": 2}],
        )
        dataset = TextCraftDataset(samples=[sample], split="val")
        self.assertEqual(dataset[0].targets, {"target": 3})
        self.assertEqual(dataset[0].recipes["intermediate"][0].result_count, 3)

    def test_medium_selection_is_seeded_and_filters_other_difficulties(self) -> None:
        rows = [
            _official_row("easy", "easy"),
            _official_row("medium-a"),
            _official_row("medium-b"),
            _official_row("medium-c"),
            _official_row("hard", "hard"),
        ]

        first = [row["id"] for row in select_medium_rows(rows, seed=42, count=2)]
        second = [row["id"] for row in select_medium_rows(rows, seed=42, count=2)]

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(value.startswith("medium-") for value in first))

    def test_metric_extraction_counts_recursive_codeact_calls(self) -> None:
        trace = {
            "agent_result": {
                "steps": 3,
                "status": "completed",
                "trace": {
                    "depth": 0,
                    "steps": [
                        {
                            "response": "<python>get_info(['target']); craft({}, ('target', 1)); "
                            "await spawn_subagent('prepare'); finish('done')</python>",
                            "code_executions": [
                                {
                                    "code": "get_info(['target']); craft({}, ('target', 1)); "
                                    "await spawn_subagent('prepare'); finish('done')",
                                    "error": None,
                                }
                            ],
                        }
                    ],
                    "children": [
                        {
                            "depth": 1,
                            "status": "completed",
                            "steps": [
                                {
                                    "response": "<python>finish('child')</python>",
                                    "code_executions": [
                                        {"code": "finish('child')", "error": None}
                                    ],
                                }
                            ],
                            "children": [],
                        }
                    ],
                },
            },
            "environment_report": {"success": True, "missing": {}, "tool_errors": []},
        }

        metrics = extract_rollout_metrics(trace, task_id="task-1", budget=64)

        self.assertEqual(metrics["task_id"], "task-1")
        self.assertEqual(metrics["global_steps_used"], 3)
        self.assertEqual(metrics["number_of_agents"], 2)
        self.assertEqual(metrics["maximum_depth"], 1)
        self.assertEqual(metrics["spawn_subagent_count"], 1)
        self.assertEqual(metrics["get_info_count"], 1)
        self.assertEqual(metrics["craft_count"], 1)
        self.assertTrue(metrics["finish_called"])
        self.assertEqual(metrics["parse_errors"], 0)
        self.assertEqual(metrics["runtime_errors"], 0)

    def test_append_and_resume_skip_completed_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result_path = output_dir / EXPERIMENT_FILES["budget_sweep"]
            append_jsonl(result_path, {"rollout_id": "budget_sweep:task-1:32:0"})
            append_jsonl(result_path, {"rollout_id": "budget_sweep:task-1:64:0"})

            self.assertEqual(
                completed_rollout_ids(result_path),
                {"budget_sweep:task-1:32:0", "budget_sweep:task-1:64:0"},
            )
            self.assertEqual(len(read_jsonl(result_path)), 2)

            args = Namespace(
                split="val",
                max_depth=12,
                max_concurrent_subagents=1,
                max_subagents_per_agent=1,
                max_run_seconds=10.0,
                max_observation_chars=1000,
                request_timeout=1.0,
                max_retries=0,
                max_tokens=32,
                model_name=None,
            )
            sample = normalize_official_row(_official_row("task-1"), split="val")
            with patch("scripts.run_textcraft_sampling_experiment.run_rollout") as run_mock:
                rows = _run_plan(
                    experiment="budget_sweep",
                    samples=[("task-1", sample)],
                    output_dir=output_dir,
                    model_config="unused.yaml",
                    args=args,
                    budget=32,
                    replicates=1,
                    temperature=0.0,
                    top_p=None,
                    resume=True,
                )

            self.assertEqual(rows, [])
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
