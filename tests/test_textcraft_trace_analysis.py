from __future__ import annotations

import unittest

from recursive_agent.envs.textcraft_synth.trace_analysis import (
    aggregate_textcraft_results,
    analyze_textcraft_result,
)


def _agent(*, agent_id: str, parent_id=None, depth=0, task="root", code="", children=None):
    return {
        "agent_id": agent_id,
        "parent_id": parent_id,
        "depth": depth,
        "task": task,
        "steps": [
            {
                "number": 1,
                "observation_truncated": False,
                "code_executions": [
                    {"code": code, "error": None, "variables": []}
                ],
            }
        ],
        "children": list(children or []),
    }


def _row(root, *, difficulty="medium", crafting_depth=4, success=True):
    return {
        "instance_id": 0,
        "ok": True,
        "success": success,
        "score": float(success),
        "difficulty": difficulty,
        "crafting_depth": crafting_depth,
        "craft_calls": 1,
        "trace": {
            "agent_result": {"trace": root},
            "environment_report": {"success": success, "tool_errors": []},
        },
    }


class TextCraftTraceAnalysisTests(unittest.TestCase):
    def test_reasonable_serial_recursion(self) -> None:
        child = _agent(
            agent_id="child",
            parent_id="root",
            depth=1,
            task="Ensure the shared inventory contains at least 2 x part.",
            code="answer['ready'] = True",
        )
        root = _agent(
            agent_id="root",
            task="Craft target",
            code=(
                "get_info(['target'])\n"
                "spawn_subagent('Ensure the shared inventory contains at least 2 x part.')\n"
                "view_inventory()\nfinish('done')"
            ),
            children=[child],
        )
        metrics = analyze_textcraft_result(_row(root))
        self.assertTrue(metrics["recursion_reasonable"])
        self.assertEqual(metrics["child_agent_count"], 1)
        self.assertEqual(metrics["spawn_subagents_calls"], 0)
        self.assertEqual(metrics["child_finish_calls"], 0)

    def test_flags_repetition_parallelism_and_child_finish(self) -> None:
        child = _agent(
            agent_id="child",
            parent_id="root",
            depth=1,
            task="Craft target",
            code="finish('wrong')",
        )
        root = _agent(
            agent_id="root",
            task="Craft target",
            code=(
                "get_info()\nget_info(['target'])\nget_info(['target'])\n"
                "craft({}, ('part', 1)); craft({}, ('part', 1))\n"
                "spawn_subagents([])"
            ),
            children=[child],
        )
        metrics = analyze_textcraft_result(_row(root))
        self.assertFalse(metrics["recursion_reasonable"])
        self.assertEqual(metrics["duplicate_get_info_queries"], 1)
        self.assertEqual(metrics["get_info_no_arg_calls"], 1)
        self.assertEqual(metrics["multi_craft_executions"], 1)
        self.assertIn("parallel_mutation_risk", metrics["recursion_issues"])
        self.assertIn("child_called_finish", metrics["recursion_issues"])
        self.assertIn("unchanged_task_delegation", metrics["recursion_issues"])
        self.assertIn(
            "multiple_crafts_in_one_execution", metrics["trajectory_issues"]
        )

    def test_shallow_direct_solution_and_aggregation(self) -> None:
        root = _agent(
            agent_id="root",
            task="Craft target",
            code="get_info(['target'])\ncraft({}, ('target', 1))\nfinish('done')",
        )
        metrics = analyze_textcraft_result(
            _row(root, difficulty="easy", crafting_depth=2)
        )
        self.assertTrue(metrics["recursion_reasonable"])
        summary = aggregate_textcraft_results([metrics])
        self.assertEqual(summary["task_successes"], 1)
        self.assertEqual(summary["reasonable_recursion_rate"], 1.0)

    def test_success_without_root_finish_is_not_reasonable(self) -> None:
        root = _agent(
            agent_id="root",
            task="Craft target",
            code="get_info(['target'])\nanswer['ready'] = True",
        )
        metrics = analyze_textcraft_result(
            _row(root, difficulty="easy", crafting_depth=2)
        )
        self.assertTrue(metrics["recursion_reasonable"])
        self.assertFalse(metrics["trajectory_reasonable"])
        self.assertIn(
            "successful_inventory_without_root_finish",
            metrics["trajectory_issues"],
        )


if __name__ == "__main__":
    unittest.main()
