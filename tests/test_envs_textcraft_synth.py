from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recursive_agent import RecursiveAgent
from recursive_agent.envs import available_environments
from recursive_agent.envs.runner import run_environment
from recursive_agent.envs.textcraft_synth import (
    DEFAULT_TEXTCRAFT_AGENT_PROMPT,
    DEFAULT_TEXTCRAFT_CHILD_PROMPT,
    DEFAULT_TEXTCRAFT_ROOT_PROMPT,
    TextCraftDataset,
    TextCraftSynthEnvironment,
    evaluate_inventory,
    generate_textcraft_samples,
)
from tests.fakes import FakeFactory, initial_task


def sample_row() -> dict:
    return {
        "id": "tiny-0",
        "initial_inventory": {"ore": 2, "wood": 1, "gem": 2},
        "recipes": {
            "ingot": {
                "ingredients": {"ore": 2},
                "result_count": 2,
            },
            "tool": {
                "ingredients": {"ingot": 1, "wood": 1},
                "result_count": 1,
            },
        },
        "targets": {"tool": 1},
        "difficulty": "easy",
        "crafting_depth": 2,
    }


def _model_config(directory: str) -> Path:
    path = Path(directory) / "model.yaml"
    path.write_text(
        "model:\n"
        "  type: api\n"
        "  api:\n"
        "    model: synthetic\n",
        encoding="utf-8",
    )
    return path


def _dynamic_task(messages: list[dict]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str) and content.startswith("# Task\n"):
            return content.split("\n", 2)[1]
    raise AssertionError(f"No dynamic task in messages: {messages!r}")


class TextCraftSynthEnvironmentTests(unittest.TestCase):
    def test_jsonl_loader_and_environment_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(json.dumps(sample_row()) + "\n", encoding="utf-8")
            dataset = TextCraftDataset(data_path=path)
            self.assertEqual(dataset[0].sample_id, "tiny-0")
            environment = TextCraftSynthEnvironment(data_path=path)
        self.assertIn("textcraft_synth", available_environments())
        self.assertEqual(
            environment.task,
            "Craft the following additional items: 1x tool",
        )
        self.assertIn("shared inventory", DEFAULT_TEXTCRAFT_AGENT_PROMPT)
        self.assertIn("result_count", DEFAULT_TEXTCRAFT_AGENT_PROMPT)
        self.assertNotIn("depth 4", DEFAULT_TEXTCRAFT_AGENT_PROMPT)
        self.assertNotIn("finish(message)", DEFAULT_TEXTCRAFT_AGENT_PROMPT)
        self.assertIn("finish(message)", environment.completion_prompt)
        self.assertIn("return_to_parent(result=None)", environment.delegated_completion_prompt)
        self.assertNotIn('answer["ready"]', environment.delegated_completion_prompt)

    def test_fixed_output_and_existing_target_semantics(self) -> None:
        row = sample_row()
        row["initial_inventory"]["tool"] = 2
        environment = TextCraftSynthEnvironment(samples=[row])
        self.assertEqual(environment.get_info(["tool"])[0]["item"], "tool")
        self.assertEqual(
            environment.get_info(),
            'Error: get_info requires explicit item names. Example: get_info(["m4_i1"])',
        )
        self.assertIn(
            "Error:",
            environment.craft({"ore": 1}, ("ingot", 2)),
        )
        self.assertEqual(environment.view_inventory()["ore"], 2)
        environment.craft({"ore": 2}, ("ingot", 2))
        environment.craft({"ingot": 1, "wood": 1}, ("tool", 1))
        report = environment.report()
        self.assertTrue(report["success"])
        self.assertEqual(report["required_final_inventory"]["tool"], 3)
        self.assertEqual(report["inventory"]["tool"], 3)
        environment.finish("complete")
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            environment.craft({"ore": 2}, ("ingot", 2))

    def test_finish_requires_complete_inventory_and_score_is_partial(self) -> None:
        environment = TextCraftSynthEnvironment(samples=[sample_row()])
        finished = environment.finish("stopping")
        self.assertIn("Not finished", finished)
        self.assertFalse(environment.report()["success"])
        self.assertFalse(environment.status().done)
        evaluation = evaluate_inventory(
            initial_inventory={"x": 0}, targets={"item": 4}, inventory={"item": 2}
        )
        self.assertEqual(evaluation.score, 0.5)

    def test_framework_finalize_root_incomplete_is_nonterminal_and_actionable(self) -> None:
        environment = TextCraftSynthEnvironment(samples=[sample_row()])

        with self.assertRaisesRegex(RuntimeError, "missing quantities.*tool"):
            environment.finalize_root("too early")

        self.assertEqual(environment.report()["finish_attempts"], 1)
        self.assertFalse(environment.status().done)
        environment.craft({"ore": 2}, ("ingot", 2))
        environment.craft({"ingot": 1, "wood": 1}, ("tool", 1))
        self.assertIn("success=True", environment.finalize_root("complete"))
        self.assertTrue(environment.status().done)

    def test_codeact_runner_uses_shared_inventory_and_role_action_spaces(self) -> None:
        environment = TextCraftSynthEnvironment(samples=[sample_row()])

        def handler(messages, _timeout):
            task = _dynamic_task(messages)
            if task == environment.task:
                root_calls = sum(
                    message["role"] == "assistant" for message in messages
                )
                if root_calls == 0:
                    return (
                        "<python>\n"
                        "child = await spawn_subagent("
                        "'Craft the ingot intermediate and report.'\n"
                        ")\n"
                        "print(child)\n"
                        "</python>"
                    )
                return (
                    "<python>\n"
                    "print(view_inventory())\n"
                    "print(craft({'ingot': 1, 'wood': 1}, ('tool', 1)))\n"
                    "finish('root complete')\n"
                    "</python>"
                )
            self.assertEqual(task, "Craft the ingot intermediate and report.")
            return (
                "<python>\n"
                "print(craft({'ore': 2}, ('ingot', 2)))\n"
                "print(view_inventory())\n"
                "return_to_parent('child intermediate ready')\n"
                "</python>"
            )

        factory = FakeFactory(handler)
        with tempfile.TemporaryDirectory() as directory:
            run = run_environment(
                environment,
                model_config=_model_config(directory),
                agent_kwargs={
                    "max_total_steps": 3,
                    "max_depth": 1,
                    "max_concurrent_subagents": 1,
                    "client_factory": factory,
                },
            )

        self.assertEqual(factory.created, 1)
        self.assertEqual(factory.closed_clients, 1)
        self.assertTrue(run.environment_report["success"])
        self.assertEqual(run.environment_report["craft_calls"], 2)
        self.assertEqual(run.environment_report["inventory"]["tool"], 1)
        self.assertEqual(len(run.agent_result.trace.children), 1)

        root_prompt = factory.calls[0][1]["content"]
        child_prompt = factory.calls[1][1]["content"]
        final_root_prompt = factory.calls[2][-1]["content"]
        self.assertIn("`finish`", root_prompt)
        self.assertNotIn("`return_to_parent`", root_prompt)
        self.assertIn("`return_to_parent`", child_prompt)
        self.assertNotIn("`finish`", child_prompt)
        self.assertNotIn(environment.task, child_prompt)
        self.assertIn("'ingot': 2", final_root_prompt)

    def test_codeact_prompt_and_capabilities_are_environment_owned(self) -> None:
        environment = TextCraftSynthEnvironment(samples=[sample_row()])
        prompt = environment.environment_system_prompt
        for phrase in (
            "fixed quantities",
            "scale every ingredient",
            "additional",
            "recipe and",
            "Reuse existing",
            "recoverable feedback",
            "smallest valid multiple",
            "Subagents are optional",
            "immediately visible",
            "shared-resource",
            "should run sequentially",
            "re-observe",
            "persistent Python REPL",
            "output exactly one executable block",
            "complete assignment",
            "natural-language `task` string",
            "result = spawn_subagent(",
            "current\nAction Space",
        ):
            self.assertIn(phrase, prompt)
        self.assertNotIn("Calls are synchronous", prompt)
        self.assertNotIn("highly recommended", prompt)
        self.assertNotIn("recursively delegate", prompt)
        self.assertEqual(prompt.count("result = spawn_subagent("), 1)
        self.assertNotIn('answer["ready"]', prompt)
        capability_prompt = environment.codeact_descriptions(is_root=True, depth=0)
        self.assertIsNotNone(capability_prompt)
        assert capability_prompt is not None
        for tool_name in ("craft", "get_info", "view_inventory"):
            description = environment.codeact_capabilities(is_root=True, depth=0)[
                tool_name
            ].prompt_description
            self.assertIsNotNone(description)
            assert description is not None
            self.assertIn("synchronous", description)
            self.assertIn("called directly", description)
            self.assertIn("must not be awaited", description)
            self.assertIn(f"`{tool_name}`", capability_prompt)

        for role_prompt in (DEFAULT_TEXTCRAFT_ROOT_PROMPT, DEFAULT_TEXTCRAFT_CHILD_PROMPT):
            self.assertIn(
                "`craft`, `get_info`, and `view_inventory` are synchronous environment tools",
                role_prompt,
            )
            self.assertIn("call\neach directly; they must not be awaited", role_prompt)
            self.assertIn(
                "`spawn_subagent(...)` and `spawn_subagents(...)` tools are synchronous from the\nmodel's perspective; call them directly.",
                role_prompt,
            )
        self.assertEqual(
            set(environment.codeact_capabilities(is_root=True, depth=0)),
            {"craft", "get_info", "view_inventory"},
        )
        self.assertEqual(
            set(environment.codeact_capabilities(is_root=False, depth=1)),
            {"craft", "get_info", "view_inventory"},
        )
        self.assertEqual(
            environment.agent_prompt,
            environment.environment_system_prompt,
        )

        custom = TextCraftSynthEnvironment(
            samples=[sample_row()],
            agent_prompt="Custom TextCraft system guidance.",
        )
        self.assertEqual(custom.environment_system_prompt, "Custom TextCraft system guidance.")

    def test_textcraft_prompts_match_direct_crafting_and_role_structure(self) -> None:
        prompt = DEFAULT_TEXTCRAFT_AGENT_PROMPT
        for phrase in (
            "### TextCraft-Synth",
            "Craft the requested additional target items using the shared inventory",
            "requested quantity is additional to the existing amount",
            "get_info(...)",
            "view_inventory()",
            "CRAFTING STRATEGY:",
            "Recipes produce fixed quantities per execution",
            "result_count",
            "smallest valid multiple that satisfies",
            "overproduction is correct",
            "recoverable feedback",
            "verify recipe and",
            "DELEGATION STRATEGY:",
            "Subagents are optional",
            "clearly bounded intermediate task",
            "unchanged copy of the current task",
            "complete assignment",
            "required additional quantity",
            "natural-language `task` string",
            "shared live inventory",
            "immediately visible",
            "run sequentially",
            "independent tasks",
            "After any child returns",
            "One valid delegation example:",
            "result = spawn_subagent(",
            "Craft at least 3 additional m4_i1",
        ):
            self.assertIn(phrase, prompt)

        self.assertEqual(
            prompt.count("result = spawn_subagent("),
            1,
        )
        self.assertNotIn("highly recommended", prompt)
        self.assertNotIn("recursively delegate", prompt)
        self.assertNotIn("launch_subagent", prompt)
        self.assertNotIn("answer[\"ready\"]", prompt)
        self.assertNotIn("<thought>", prompt)

        child_prompt = DEFAULT_TEXTCRAFT_CHILD_PROMPT
        self.assertIn("CRAFTING STRATEGY:", child_prompt)
        self.assertIn("DELEGATION STRATEGY:", child_prompt)
        self.assertIn("spawn_subagent(task: str, context=None)", child_prompt)
        self.assertIn("spawn_subagents(requests: list[dict])", child_prompt)
        self.assertIn("return_to_parent(result=None)", child_prompt)
        self.assertIn('return_to_parent("short result")', child_prompt)
        self.assertIn("A plain string does NOT return to the parent.", child_prompt)
        self.assertIn(
            'return_to_parent("Crafted the requested additional 3x m4_i1.")',
            child_prompt,
        )
        self.assertIn('"Crafted the requested additional 3x m4_i1."', child_prompt)
        self.assertIn("return_to_parent\n", child_prompt)
        self.assertNotIn("<thought>", child_prompt)
        self.assertNotIn("finish", child_prompt)
        self.assertNotIn('answer["ready"]', child_prompt)
        self.assertIn("`finish(message: str) -> str`", DEFAULT_TEXTCRAFT_ROOT_PROMPT)
        self.assertNotIn("return_to_parent", DEFAULT_TEXTCRAFT_ROOT_PROMPT)

    def test_generated_tasks_have_the_paper_depth_bands(self) -> None:
        rows = generate_textcraft_samples(count=3, difficulty="hard", seed=4)
        dataset = TextCraftDataset(samples=rows)
        self.assertEqual(len(dataset), 3)
        self.assertTrue(all(sample.crafting_depth >= 7 for sample in dataset._samples))
        for sample in dataset._samples:
            environment = TextCraftSynthEnvironment(samples=[sample])
            root = next(iter(sample.targets))
            info = environment.get_info([root])[0]
            self.assertEqual(info["crafting_depth"], sample.crafting_depth)
            self.assertFalse(environment.report()["success"])

    def test_recursive_child_crafts_shared_intermediate_before_root_assembly(self) -> None:
        environment = TextCraftSynthEnvironment(samples=[sample_row()])

        def handler(messages, _timeout):
            task = initial_task(messages)
            assistant_calls = sum(message["role"] == "assistant" for message in messages)
            if task.startswith("Craft the following additional items:"):
                if assistant_calls == 0:
                    return (
                        "```repl\n"
                        "print(spawn_subagent('Craft the ingot intermediate and report.'))\n"
                        "```"
                    )
                return (
                    "```repl\n"
                    "print(craft({'ingot': 1, 'wood': 1}, ('tool', 1)))\n"
                    "print(finish('root complete'))\n"
                    "```"
                )
            return (
                "```repl\n"
                "print(craft({'ore': 2}, ('ingot', 2)))\n"
                "answer['content'] = 'child complete'\n"
                "answer['ready'] = True\n"
                "```"
            )

        factory = FakeFactory(handler)
        agent = RecursiveAgent(
            backend="openai",
            backend_kwargs={"model_name": "fake-model"},
            tools=environment.tools(),
            termination_check=environment.status,
            root_prompt=environment.root_prompt,
            child_prompt=environment.child_prompt,
            delegated_disabled_tools=environment.delegated_disabled_tools,
            max_steps=4,
            max_depth=3,
            client_factory=factory,
        )
        result = agent.run(environment.task, context=environment.context)
        report = environment.report()
        self.assertEqual(result.status, "environment_done")
        self.assertTrue(report["success"])
        self.assertEqual(len(result.trace.children), 1)
        self.assertEqual(result.trace.children[0].depth, 1)
        self.assertEqual(report["craft_calls"], 2)

        root_messages = factory.calls[0]
        child_messages = factory.calls[1]
        self.assertNotEqual(root_messages[0]["content"], child_messages[0]["content"])
        self.assertIn("### TextCraft-Synth", root_messages[0]["content"])
        self.assertIn("`finish(message)`", root_messages[0]["content"])
        self.assertNotIn('answer["ready"]', root_messages[0]["content"])
        self.assertNotIn('answer["ready"]', child_messages[0]["content"])
        self.assertIn("`finish(message: str) -> str`", root_messages[0]["content"])
        self.assertNotIn("`finish(message: str) -> str`", child_messages[0]["content"])
        self.assertNotIn("finish", child_messages[0]["content"])
        self.assertIn("return_to_parent(result=None)", child_messages[0]["content"])
        self.assertNotIn("Custom tools:", root_messages[0]["content"])
        self.assertNotIn("Custom tools:", child_messages[0]["content"])
        self.assertIn(environment.task, root_messages[1]["content"])
        self.assertNotIn(environment.task, child_messages[1]["content"])
        self.assertTrue(
            child_messages[1]["content"].startswith(
                "Delegated task:\nCraft the ingot intermediate and report."
            )
        )


if __name__ == "__main__":
    unittest.main()
