from __future__ import annotations

import argparse
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from recursive_agent import ConfigurationError, RecursiveAgent
from recursive_agent.envs import available_environments, run_environment
from recursive_agent.envs.browsecomp_plus import (
    BrowseCompPlusEnvironment,
    BrowseCompQuery,
    SearchBudgetExceeded,
    judge_answer,
    normalize_search_results,
    parse_final_output,
    parse_judge_response,
)
from recursive_agent.envs.browsecomp_plus.runner import (
    _completed_record,
    _error_record,
    _is_completed,
    _write_json,
    analyze_recursive_trace,
)
from recursive_agent.envs.browsecomp_plus.evaluator import main as evaluator_main
from recursive_agent.envs.browsecomp_plus.prompts import (
    DEFAULT_BROWSECOMP_AGENT_PROMPT,
    DEFAULT_BROWSECOMP_TOOLS_PROMPT,
)
from recursive_agent.envs.browsecomp_plus.tools import _tool_result_payload
from recursive_agent.repl import ReplSession
from recursive_agent.types import ModelCallUsage, ModelResponse


class StubSearchClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[str] = []

    def search(self, query: str):
        with self._lock:
            self.calls.append(query)
            call_number = len(self.calls)
        return [
            {
                "docid": call_number,
                "score": call_number / 10,
                "snippet": f"evidence for {query}",
            }
        ]


class BrowseCompPlusEnvironmentTests(unittest.TestCase):
    def test_prompt_ports_deep_research_strategy_to_native_tools(self) -> None:
        prompt = DEFAULT_BROWSECOMP_AGENT_PROMPT
        tools_prompt = DEFAULT_BROWSECOMP_TOOLS_PROMPT
        self.assertIn("RESEARCH STRATEGY:", prompt)
        self.assertIn("DELEGATION STRATEGY:", prompt)
        self.assertIn("spawn_subagent(task, context=None)", prompt)
        self.assertIn("spawn_subagents(requests)", prompt)
        self.assertIn('answer["ready"] = True', prompt)
        self.assertIn("one executable `repl` block", prompt)
        self.assertIn("`search(query: str) -> list[dict]`", tools_prompt)
        self.assertIn("up to five results", tools_prompt)
        self.assertNotIn("launch_subagent", prompt + tools_prompt)
        self.assertNotIn("finish(...)", prompt + tools_prompt)
        self.assertNotIn("<python>", prompt + tools_prompt)
        self.assertNotIn(
            "await",
            (prompt + tools_prompt).lower().replace("do not use `await`", ""),
        )

    def test_bm25_payload_normalizes_docids_and_empty_results(self) -> None:
        self.assertEqual(normalize_search_results([]), [])
        self.assertEqual(
            normalize_search_results(
                {"result": [{"docid": 17, "score": None, "snippet": "text"}]}
            ),
            [{"docid": "17", "score": None, "snippet": "text"}],
        )
        self.assertEqual(
            _tool_result_payload(
                [SimpleNamespace(text='[{"docid": "9", "snippet": "x"}]')]
            ),
            [{"docid": "9", "snippet": "x"}],
        )

    def test_bm25_payload_truncates_snippets_by_characters(self) -> None:
        results = normalize_search_results(
            [{"docid": "doc", "score": 1, "snippet": "abcdefgh"}],
            snippet_max_chars=5,
        )
        self.assertEqual(results[0]["snippet"], "abcde")

    def test_empty_search_result_does_not_crash_environment(self) -> None:
        client = SimpleNamespace(search=lambda query: [])
        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("empty", "Question"),
            search_client=client,
        )
        self.assertEqual(environment.search("no matches"), [])
        self.assertEqual(environment.report()["search_calls"], 1)
        self.assertEqual(environment.report()["retrieved_docids"], [])

    def test_agent_context_and_tools_cannot_access_gold_data(self) -> None:
        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("q1", "Which entity matches both clues?"),
            search_client=StubSearchClient(),
        )
        self.assertEqual(
            set(environment.context),
            {"environment", "query_id", "query"},
        )
        self.assertEqual(set(environment.tools()), {"search"})
        self.assertEqual(
            environment.disabled_repl_builtins,
            frozenset({"__import__", "open"}),
        )
        self.assertEqual(environment.max_repl_blocks_per_step, 1)
        serialized = json.dumps(environment.context)
        for forbidden in ("answer", "gold", "evidence", "qrel", "judge"):
            self.assertNotIn(forbidden, serialized.lower())
        self.assertIn("browsecomp_plus", available_environments())

    def test_browsecomp_repl_cannot_open_files_or_import_modules(self) -> None:
        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("locked", "Question"),
            search_client=StubSearchClient(),
        )
        repl = ReplSession(
            context=environment.context,
            tools={
                name: entry["tool"]
                for name, entry in environment.tools().items()
            },
            spawn_subagent=lambda task, context=None: "",
            spawn_subagents=lambda requests: [],
            disabled_builtins=environment.disabled_repl_builtins,
        )
        opened = repl.execute("print(open('/tmp/not-allowed'))")
        imported = repl.execute("print(__import__('os'))")
        self.assertIn("NameError", opened.trace.error)
        self.assertIn("NameError", imported.trace.error)

    def test_parallel_callers_share_budget_and_retrieved_union(self) -> None:
        client = StubSearchClient()
        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("q2", "Question"),
            max_search_calls=2,
            search_client=client,
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(environment.search, f"query {index}")
                for index in range(3)
            ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except SearchBudgetExceeded:
                outcomes.append("budget")
        report = environment.report()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(report["search_calls"], 2)
        self.assertEqual(report["retrieved_docids"], ["1", "2"])
        self.assertEqual(sum(value == "budget" for value in outcomes), 1)
        with self.assertRaises(SearchBudgetExceeded):
            environment.search("one too many")

    def test_scripted_runtime_shares_search_and_honors_depth(self) -> None:
        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("q3", "Question with independent clues"),
            max_search_calls=2,
            search_client=StubSearchClient(),
        )

        class ScriptedClient:
            def __init__(self) -> None:
                self.calls = 0
                self.model_name = "scripted-runtime-test"

            def completion(self, messages, *, timeout=None):
                self.calls += 1
                delegated = str(messages[1]["content"]).startswith("Delegated task:")
                fence = chr(96) * 3
                if delegated:
                    content = (
                        f"{fence}repl\n"
                        "blocked = spawn_subagent('too deep', {'objective': 'nested'})\n"
                        "found = search('child evidence')\n"
                        "print(blocked, found)\n"
                        "answer['content'] = 'Finding: child evidence [1]'\n"
                        "answer['ready'] = True\n"
                        f"{fence}"
                    )
                elif self.calls == 1:
                    content = (
                        f"{fence}repl\n"
                        "reports = spawn_subagents([{'task': 'verify clue', "
                        "'context': {'objective': 'verify clue'}}])\n"
                        "print(reports)\n"
                        f"{fence}"
                    )
                else:
                    content = (
                        f"{fence}repl\n"
                        "found = search('root verification')\n"
                        "print(found)\n"
                        "answer['content'] = 'Explanation: verified [1]\\n"
                        "Exact Answer: Example\\nConfidence: 80%'\n"
                        "answer['ready'] = True\n"
                        f"{fence}"
                    )
                return ModelResponse(
                    content=content,
                    usage=ModelCallUsage(model=self.model_name),
                )

            def close(self):
                return None

        agent = RecursiveAgent(
            backend_kwargs={"model_name": "unused"},
            tools=environment.tools(),
            root_prompt=environment.root_prompt,
            child_prompt=environment.child_prompt,
            delegated_forced_final_prompt=environment.delegated_forced_final_prompt,
            disabled_repl_builtins=environment.disabled_repl_builtins,
            max_depth=1,
            max_steps=3,
            client_factory=lambda backend, kwargs: ScriptedClient(),
        )
        result = agent.run(environment.task, environment.context)
        stats = analyze_recursive_trace(_trace_mapping(result.trace))
        self.assertEqual(environment.report()["search_calls"], 2)
        self.assertEqual(stats["subagent_count"], 1)
        self.assertEqual(stats["max_depth_reached"], 1)
        self.assertTrue(stats["root_used_search"])
        self.assertTrue(stats["subagent_used_search"])
        system_prompt = result.trace.system_prompt
        worker_system_prompt = result.trace.children[0].system_prompt
        self.assertNotEqual(worker_system_prompt, system_prompt)
        self.assertEqual(system_prompt, environment.root_prompt)
        self.assertEqual(worker_system_prompt, environment.child_prompt)
        self.assertIn("root agent for one BrowseComp-Plus", system_prompt)
        self.assertIn("child agent for BrowseComp-Plus", worker_system_prompt)
        root_completion = environment.completion_prompt.strip()
        delegated_completion = environment.delegated_completion_prompt.strip()
        self.assertIn(root_completion, system_prompt)
        self.assertNotIn(root_completion, worker_system_prompt)
        self.assertIn(delegated_completion, worker_system_prompt)
        self.assertNotIn(delegated_completion, system_prompt)
        self.assertIn("`search(query: str) -> list[dict]`", system_prompt)
        self.assertIn("`search(query: str) -> list[dict]`", worker_system_prompt)
        self.assertNotIn("Custom tools:", system_prompt)
        self.assertNotIn("Custom tools:", worker_system_prompt)
        self.assertIsNone(environment.delegated_prompt_addendum)
        self.assertIn("BrowseComp-Plus", system_prompt)
        self.assertNotIn("You are a recursive agent harness", system_prompt)

    def test_environment_runner_uses_browsecomp_system_only(self) -> None:
        from tests.fakes import FakeFactory

        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("prompt", "Find one entity"),
            search_client=StubSearchClient(),
        )

        def handler(messages, timeout):
            return (
                "```repl\n"
                "hits = search('entity')\n"
                "answer['content'] = 'Explanation: evidence [1]\\n"
                "Exact Answer: Example\\nConfidence: 60%'\n"
                "answer['ready'] = True\n"
                "```"
            )

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.yaml"
            config.write_text(
                "model:\n"
                "  type: api\n"
                "  api:\n"
                "    model: fake\n",
                encoding="utf-8",
            )
            run = run_environment(
                environment,
                model_config=config,
                agent_kwargs={
                    "max_steps": 1,
                    "client_factory": FakeFactory(handler),
                },
            )

        self.assertEqual(run.system_prompt, environment.root_prompt)
        self.assertNotIn("recursive agent harness", run.system_prompt)
        self.assertNotIn("Custom tools:", run.system_prompt)
        self.assertIn("BrowseComp-Plus", run.system_prompt)
        self.assertEqual(run.agent_result.trace.system_prompt, run.system_prompt)

    def test_worker_uses_worker_specific_forced_final_prompt(self) -> None:
        from tests.fakes import FakeFactory, initial_task

        environment = BrowseCompPlusEnvironment(
            sample=BrowseCompQuery("worker-final", "Question"),
            search_client=StubSearchClient(),
        )

        def handler(messages, timeout):
            task = initial_task(messages)
            if task == "root":
                if "best supported answer" in messages[-1]["content"]:
                    return (
                        "Explanation: worker report collected [1]\n"
                        "Exact Answer: Example\nConfidence: 50%"
                    )
                return (
                    "```repl\n"
                    "report = spawn_subagent('worker')\n"
                    "print(report)\n"
                    "```"
                )
            if "evidence report for the parent" in messages[-1]["content"]:
                return (
                    "Findings: Example\n"
                    "Evidence: evidence [1]\n"
                    "Queries tried: one query\n"
                    "Unresolved: one clue"
                )
            return "I am still investigating."

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            system_prompt=environment.system_prompt,
            prompt_addendum=environment.agent_prompt,
            forced_final_prompt=environment.forced_final_prompt,
            delegated_forced_final_prompt=environment.delegated_forced_final_prompt,
            max_depth=1,
            max_steps=1,
            client_factory=FakeFactory(handler),
        ).run("root")

        child = result.trace.children[0]
        self.assertTrue(child.forced_final_response.startswith("Findings:"))
        self.assertNotIn("Exact Answer:", child.forced_final_response)
        self.assertIn("Exact Answer: Example", result.answer)

    def test_runtime_caps_total_direct_subagents_per_agent(self) -> None:
        from tests.fakes import FakeFactory, initial_task

        def handler(messages, timeout):
            task = initial_task(messages)
            fence = chr(96) * 3
            if task == "root":
                return (
                    f"{fence}repl\n"
                    "results = spawn_subagents(["
                    "{'task': 'child-1'}, {'task': 'child-2'}, "
                    "{'task': 'child-3'}, {'task': 'child-4'}])\n"
                    "answer['content'] = str(results)\n"
                    "answer['ready'] = True\n"
                    f"{fence}"
                )
            return (
                f"{fence}repl\n"
                f"answer['content'] = '{task}'\n"
                "answer['ready'] = True\n"
                f"{fence}"
            )

        result = RecursiveAgent(
            backend_kwargs={"model_name": "fake"},
            max_depth=1,
            max_subagents_per_agent=2,
            client_factory=FakeFactory(handler),
        ).run("root")
        self.assertEqual(len(result.trace.children), 2)
        self.assertIn("child-1", result.answer)
        self.assertIn("child-2", result.answer)
        self.assertEqual(
            result.answer.count("maximum direct subagents per agent (2) reached"),
            2,
        )
        with self.assertRaises(ConfigurationError):
            RecursiveAgent(
                backend_kwargs={"model_name": "fake"},
                max_subagents_per_agent=0,
            )

    def test_final_output_parser_accepts_common_markdown_wrappers(self) -> None:
        parsed = parse_final_output(
            "Explanation: supported by [12]\n"
            "Exact Answer: Ada Lovelace\n"
            "Confidence: 85%"
        )
        self.assertEqual(parsed["exact_answer"], "Ada Lovelace")
        self.assertEqual(parsed["confidence"], 85.0)
        wrapped = parse_final_output(
            "```text\n"
            "**Explanation:** supported by [12]\n"
            "**Exact Answer:** Ada Lovelace\n"
            "**Confidence:** 85%\n"
            "```"
        )
        self.assertEqual(wrapped["exact_answer"], "Ada Lovelace")
        with_preamble = parse_final_output(
            "I could not find more evidence.\n\n"
            "Explanation: no matching document [12]\n"
            "Exact Answer: Unknown\n"
            "Confidence: 0%"
        )
        self.assertEqual(with_preamble["exact_answer"], "Unknown")
        self.assertIsNone(parse_final_output("Ada Lovelace"))
        self.assertIsNone(
            parse_final_output(
                "Explanation: x\nExact Answer: y\nConfidence: 101%"
            )
        )

    def test_judge_parser_requires_structured_json(self) -> None:
        valid = parse_judge_response(
            '{"correct": true, "score": 1, "reason": "equivalent"}',
            model="judge",
        )
        invalid = parse_judge_response("correct: yes", model="judge")
        self.assertTrue(valid.correct)
        self.assertIsNone(valid.error)
        self.assertIsNotNone(invalid.error)

    def test_evaluator_retries_json_parse_failure(self) -> None:
        responses = iter(
            [
                "not json",
                '{"correct": true, "score": 1, "reason": "same entity"}',
            ]
        )

        class FakeJudgeClient:
            def __init__(self, **kwargs):
                pass

            def completion(self, messages, *, timeout=None):
                return SimpleNamespace(content=next(responses))

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "model.yaml"
            config.write_text(
                "model:\n  type: api\n  api:\n"
                "    model: local-judge\n    api_key: test\n",
                encoding="utf-8",
            )
            with patch(
                "recursive_agent.envs.browsecomp_plus.scoring.OpenAIClient",
                FakeJudgeClient,
            ):
                result = judge_answer(
                    model_config=config,
                    question="Who?",
                    correct_answer="Ada",
                    response="Exact Answer: Ada",
                    max_attempts=2,
                )
        self.assertTrue(result.correct)
        self.assertEqual(result.attempts, 2)

    def test_failure_record_is_saved_and_resume_skips_only_completed(self) -> None:
        args = _args()
        sample = BrowseCompQuery("q/error", "Question")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed_path = root / "failed.json"
            completed_path = root / "completed.json"
            record = _error_record(
                args=args,
                sample=sample,
                model_metadata={"model": "test"},
                search_report={
                    "search_calls": 1,
                    "retrieved_docids": [9],
                    "events": [],
                },
                duration_seconds=0.1,
                error=RuntimeError("failure"),
                trajectory_path=root / "trajectory.json",
                stats={
                    "subagent_count": 1,
                    "max_depth_reached": 1,
                    "root_used_search": True,
                    "subagent_used_search": True,
                    "recursion_chain": [{"agent_id": "root"}],
                },
            )
            _write_json(failed_path, record)
            _write_json(completed_path, {"status": "completed"})
            loaded = json.loads(failed_path.read_text(encoding="utf-8"))
            failed_done = _is_completed(failed_path)
            completed_done = _is_completed(completed_path)
        self.assertEqual(loaded["status"], "error")
        self.assertEqual(loaded["retrieved_docids"], ["9"])
        self.assertEqual(loaded["subagent_count"], 1)
        self.assertEqual(
            loaded["trajectory"]["recursion_chain"],
            [{"agent_id": "root"}],
        )
        self.assertFalse(failed_done)
        self.assertTrue(completed_done)

    def test_completed_record_is_directly_official_evaluator_compatible(self) -> None:
        args = _args()
        record = _completed_record(
            args=args,
            sample=BrowseCompQuery("official", "Question"),
            output=(
                "Explanation: evidence [9]\n"
                "Exact Answer: Example\n"
                "Confidence: 70%"
            ),
            model_metadata={"model": "test"},
            search_report={
                "search_calls": 1,
                "retrieved_docids": [9],
                "events": [
                    {
                        "query": "example",
                        "results": [
                            {"docid": "9", "score": 1.0, "snippet": "evidence"}
                        ],
                    }
                ],
            },
            stats={
                "subagent_count": 1,
                "max_depth_reached": 1,
                "root_used_search": True,
                "subagent_used_search": True,
                "recursion_chain": [],
            },
            trajectory_path=Path("/tmp/trajectory.json"),
            duration_seconds=1.0,
            agent_status="completed",
            usage={},
            local_judge=None,
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["retrieved_docids"], ["9"])
        self.assertEqual(record["tool_call_counts"], {"search": 1, "recursive": 1})
        self.assertTrue(record["result"])
        self.assertEqual(record["result"][-1]["type"], "output_text")
        self.assertEqual(record["result"][-1]["output"], record["final_answer"])

    def test_standalone_evaluator_summary_preserves_failed_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            _write_json(
                runs / "run_completed.json",
                {
                    "query_id": "completed",
                    "query": "Who?",
                    "status": "completed",
                    "final_answer": "Exact Answer: Ada",
                    "search_calls": 1,
                    "subagent_count": 0,
                    "max_depth_reached": 0,
                    "retrieved_docids": [],
                    "local_evaluator": None,
                },
            )
            _write_json(
                runs / "run_failed.json",
                {
                    "query_id": "failed",
                    "query": "Who?",
                    "status": "error",
                    "final_answer": "",
                    "search_calls": 0,
                    "subagent_count": 0,
                    "max_depth_reached": 0,
                    "retrieved_docids": [],
                    "local_evaluator": None,
                },
            )
            fake_judge = SimpleNamespace(
                to_dict=lambda: {
                    "correct": True,
                    "score": 1,
                    "reason": "match",
                    "response": "{}",
                    "error": None,
                    "model": "test",
                    "attempts": 1,
                }
            )
            argv = [
                "evaluator",
                "--output-dir",
                str(root),
                "--model-config",
                "unused.yaml",
            ]
            with (
                patch(
                    "recursive_agent.envs.browsecomp_plus.evaluator.load_gold_answers",
                    return_value={"completed": "Ada"},
                ),
                patch(
                    "recursive_agent.envs.browsecomp_plus.evaluator.judge_answer",
                    return_value=fake_judge,
                ),
                patch("sys.argv", argv),
            ):
                evaluator_main()
            summary = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(summary["total_questions"], 2)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)


def _trace_mapping(trace):
    return {
        "agent_id": trace.agent_id,
        "parent_id": trace.parent_id,
        "depth": trace.depth,
        "task": trace.task,
        "steps": [
            {
                "code_executions": [
                    {"code": execution.code}
                    for execution in step.code_executions
                ]
            }
            for step in trace.steps
        ],
        "usage": trace.usage.to_dict(),
        "answer": trace.answer,
        "status": trace.status,
        "children": [_trace_mapping(child) for child in trace.children],
    }


def _args():
    return argparse.Namespace(
        model_config="configs/model_api.local.yaml",
        bm25_url="http://127.0.0.1:8080/mcp",
        max_search_calls=20,
        max_recursion_depth=2,
        max_subagents_per_agent=4,
        temperature=None,
        top_p=None,
        max_tokens=None,
        request_timeout=None,
        seed=None,
    )


if __name__ == "__main__":
    unittest.main()
