from __future__ import annotations

import unittest

from examples.run_oolong_full import _build_summary, _iter_pending_rows


class OolongFullRunnerTests(unittest.TestCase):
    def test_pending_rows_honor_range_and_resume(self) -> None:
        rows = [{"id": index} for index in range(6)]
        pending = list(
            _iter_pending_rows(
                iter(rows),
                start_index=1,
                count=4,
                completed={2},
            )
        )
        self.assertEqual(pending, [(1, rows[1]), (3, rows[3]), (4, rows[4])])

    def test_summary_counts_completed_and_submitted_rows(self) -> None:
        rows = {
            0: {
                "instance_id": 0,
                "ok": True,
                "score": 1.0,
                "run": {"environment_report": {"submitted": True}},
            },
            1: {
                "instance_id": 1,
                "ok": False,
                "score": 0.0,
            },
        }
        summary = _build_summary(rows, 0.0)
        self.assertEqual(summary["recorded_rows"], 2)
        self.assertEqual(summary["episodes_completed"], 1)
        self.assertEqual(summary["episodes_failed"], 1)
        self.assertEqual(summary["submitted"], 1)
        self.assertEqual(summary["average_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
