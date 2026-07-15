from __future__ import annotations

import unittest

from recursive_agent.envs.trace_analysis import (
    aggregate_trace_metrics,
    analyze_environment_trace,
)


class TraceAnalysisTests(unittest.TestCase):
    def test_counts_repl_tools_variables_and_children_without_policy_inference(self) -> None:
        trace = {
            "prompts": {
                "system": "Use REPL and tools. Decide yourself whether to delegate.",
                "task": "task",
            },
            "tools": {
                "observe": {"kind": "callable", "description": "observe"},
                "act": {"kind": "callable", "description": "act"},
            },
            "agent_result": {
                "status": "completed",
                "steps": 1,
                "usage": {"total_calls": 2},
                "trace": {
                    "depth": 0,
                    "steps": [
                        {
                            "observation_truncated": True,
                            "code_executions": [
                                {
                                    "code": "x = observe()\nspawn_subagent('check', x)\n",
                                    "variables": ["context", "x"],
                                    "stdout": "obs",
                                    "error": None,
                                }
                            ]
                        }
                    ],
                    "children": [
                        {
                            "depth": 1,
                            "steps": [],
                            "children": [],
                        }
                    ],
                },
            },
            "environment_report": {
                "success": False,
                "reward": 0,
                "steps": 1,
                "trajectory": [{"action": "search[x]"}],
            },
        }
        metrics = analyze_environment_trace(trace)
        self.assertEqual(metrics["child_agent_count"], 1)
        self.assertEqual(metrics["spawn_subagent_calls"], 1)
        self.assertEqual(metrics["tool_calls"]["observe"], 1)
        self.assertTrue(metrics["variable_snapshots_present"])
        self.assertEqual(metrics["observation_truncations"], 1)
        self.assertEqual(metrics["variables"], ["context", "x"])
        self.assertFalse(metrics["system_prompt_has_forced_delegation"])

    def test_aggregate_reports_batch_invariants(self) -> None:
        rows = [
            {
                "instance_id": 0,
                "ok": True,
                "metrics": {
                    "success": True,
                    "reward": 1,
                    "model_calls": 3,
                    "agent_steps": 3,
                    "repl_code_blocks": 3,
                    "variable_snapshots_present": True,
                    "tool_calls": {"act": 2},
                    "spawn_calls_total": 0,
                    "child_agent_count": 0,
                    "max_depth": 0,
                    "execution_errors": 0,
                    "code_parse_errors": 0,
                    "system_prompt_has_forced_delegation": False,
                    "system_prompt_mentions_registered_tools": True,
                    "child_trace_consistent": True,
                    "variables": ["state"],
                },
            },
            {"instance_id": 1, "ok": False, "error": "test"},
        ]
        summary = aggregate_trace_metrics(rows)
        self.assertEqual(summary["episodes_completed"], 1)
        self.assertEqual(summary["episodes_failed"], 1)
        self.assertEqual(summary["tool_call_totals"], {"act": 2})
        self.assertEqual(summary["failed_indices"], [1])


if __name__ == "__main__":
    unittest.main()
