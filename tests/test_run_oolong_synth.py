from __future__ import annotations

import unittest

from examples.run_oolong_synth import _bootstrap_ci, _build_summary


class OolongSynthRunnerTests(unittest.TestCase):
    def test_summary_counts_failures_as_zero(self) -> None:
        rows = {
            0: {
                "ok": True,
                "score": 1.0,
                "answer_type": "ANSWER_TYPE.LABEL",
                "context_len": 1024,
                "dataset": "trec_coarse",
            },
            1: {
                "ok": False,
                "score": 0.0,
                "answer_type": "ANSWER_TYPE.NUMERIC",
                "context_len": 2048,
                "dataset": "spam",
            },
        }
        summary = _build_summary(
            rows,
            requested=2,
            bootstrap_samples=100,
            bootstrap_seed=7,
            started=0.0,
        )
        self.assertEqual(summary["oolong_score"], 0.5)
        self.assertEqual(summary["failed_rows"], 1)
        self.assertEqual(summary["by_context_len"]["1024"]["n"], 1)

    def test_bootstrap_is_deterministic(self) -> None:
        self.assertEqual(
            _bootstrap_ci([0.0, 0.5, 1.0], 100, 11),
            _bootstrap_ci([0.0, 0.5, 1.0], 100, 11),
        )


if __name__ == "__main__":
    unittest.main()
