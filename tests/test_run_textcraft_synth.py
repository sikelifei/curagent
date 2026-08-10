from __future__ import annotations

import unittest

from argparse import Namespace
from unittest.mock import patch

from examples.run_textcraft_synth import _build_summary, _run_one, _trace_metrics
from recursive_agent.exceptions import TimeoutExceededError


class TextCraftSynthRunnerTests(unittest.TestCase):
    def test_summary_keeps_failed_rows_and_groups_scores(self) -> None:
        summary = _build_summary(
            [
                {
                    "ok": True,
                    "success": True,
                    "score": 1.0,
                    "difficulty": "easy",
                    "models": ["model-a"],
                },
                {"ok": False, "success": False, "score": 0.0, "difficulty": "hard"},
            ],
            requested=2,
        )
        self.assertEqual(summary["successful_runs"], 1)
        self.assertEqual(summary["task_successes"], 1)
        self.assertEqual(summary["score"], 0.5)
        self.assertEqual(summary["model"], "model-a")
        self.assertEqual(summary["models"], ["model-a"])
        self.assertEqual(summary["by_difficulty"]["hard"]["n"], 1)
        self.assertEqual(len(summary["rows"]), 2)

    def test_trace_metrics_counts_nested_children(self) -> None:
        trace = {
            "depth": 0,
            "children": [
                {"depth": 1, "children": [{"depth": 2, "children": []}]},
                {"depth": 1, "children": []},
            ],
        }
        self.assertEqual(_trace_metrics(trace), {"children": 3, "max_depth": 2})

    @patch("examples.run_textcraft_synth.run_registered_environment")
    def test_timeout_row_keeps_partial_trace(self, run_mock) -> None:
        error = TimeoutExceededError(12.0, 10.0)
        error.partial_trace = {
            "agent_result": {
                "usage": {"model_usage_summaries": {"model-a": {}}},
                "trace": {
                    "depth": 0,
                    "steps": [{"number": 1}],
                    "children": [{"depth": 1, "steps": [], "children": []}],
                },
            },
            "environment_report": {
                "id": "case",
                "difficulty": "medium",
                "crafting_depth": 4,
                "score": 0.25,
                "craft_calls": 2,
                "missing": {"target": 1},
            },
        }
        run_mock.side_effect = error
        args = Namespace(
            split="test",
            data_path=None,
            generated_count=1,
            difficulty="medium",
            seed=0,
            textcraft_root=None,
            config="config.yaml",
            agent_max_steps=25,
            max_depth=12,
            max_concurrent_subagents=4,
            max_subagents_per_agent=6,
            max_run_seconds=10,
            max_observation_chars=8000,
            request_timeout=10,
            max_retries=0,
            temperature=0.2,
            max_tokens=100,
        )

        row = _run_one(args, 0, None, 0.0)

        self.assertFalse(row["ok"])
        self.assertEqual(row["steps"], 1)
        self.assertEqual(row["recursive_children"], 1)
        self.assertEqual(row["score"], 0.25)
        self.assertEqual(row["models"], ["model-a"])
        self.assertIn("trace", row)


if __name__ == "__main__":
    unittest.main()
