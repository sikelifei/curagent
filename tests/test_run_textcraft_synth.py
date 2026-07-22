from __future__ import annotations

import unittest

from examples.run_textcraft_synth import _build_summary, _trace_metrics


class TextCraftSynthRunnerTests(unittest.TestCase):
    def test_summary_keeps_failed_rows_and_groups_scores(self) -> None:
        summary = _build_summary(
            [
                {"ok": True, "success": True, "score": 1.0, "difficulty": "easy"},
                {"ok": False, "success": False, "score": 0.0, "difficulty": "hard"},
            ],
            requested=2,
        )
        self.assertEqual(summary["successful_runs"], 1)
        self.assertEqual(summary["task_successes"], 1)
        self.assertEqual(summary["score"], 0.5)
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


if __name__ == "__main__":
    unittest.main()
