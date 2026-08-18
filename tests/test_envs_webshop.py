from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recursive_agent.envs import (
    available_environments,
    create_environment,
    run_environment,
)
from recursive_agent.envs.webshop import (
    ReCodeWebShopEnvironment,
    WebShopDataset,
    build_webshop_task_prompt,
)

from .fakes import FakeFactory


class FakeWebShopBackend:
    def __init__(self) -> None:
        self.stage = 0
        self.done = False
        self.reward = 0.0
        self.actions: list[str] = []
        self.last_observation = "initial page [Search]"
        self.trajectory: list[dict] = []
        self.close_calls = 0

    def reset(self, config, instance_id):
        assert config == {"split": "test"}
        assert instance_id == "0"
        self.trajectory = [{"action": None, "observation": self.last_observation}]
        return {"observations": [self.last_observation], "env_name": "webshop", "env": self}

    def get_instruction_text(self):
        return "find a red test product under 20 dollars"

    def get_available_actions(self):
        if self.done:
            return {"has_search_bar": False, "clickables": []}
        if self.stage == 0:
            return {"has_search_bar": True, "clickables": ["search"]}
        if self.stage == 1:
            return {"has_search_bar": False, "clickables": ["item1"]}
        if self.stage == 2:
            return {"has_search_bar": False, "clickables": ["red", "buy now"]}
        return {"has_search_bar": False, "clickables": ["buy now"]}

    async def run(self, action):
        self.actions.append(action)
        self.stage += 1
        self.last_observation = f"page after {action}"
        if action == "click[buy now]":
            self.done = True
            self.reward = 1.0
        self.trajectory.append(
            {"action": action, "observation": self.last_observation, "reward": self.reward}
        )
        return [self.last_observation]

    def is_done(self):
        return self.done

    def is_success(self):
        return self.done and self.reward >= 1.0

    def get_step_count(self):
        return len(self.actions)

    def get_reward(self):
        return self.reward

    def get_trajectory(self):
        return self.trajectory

    def report(self):
        return {
            "success": self.is_success(),
            "reward": self.reward,
            "step": len(self.actions),
        }

    async def close(self):
        self.close_calls += 1


class WebShopEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        data = self.root / "envs" / "webshop" / "data"
        data.mkdir(parents=True)
        (data.parent / "env.py").write_text("# test marker\n", encoding="utf-8")
        (data / "test_indices.json").write_text("[2853, 5934]", encoding="utf-8")
        (data / "train_indices.json").write_text("[0, 3, 4]", encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def make_environment(self, backend=None, **kwargs):
        return ReCodeWebShopEnvironment(
            recode_root=self.root,
            split="test",
            instance_id=0,
            backend=backend or FakeWebShopBackend(),
            **kwargs,
        )

    def test_dataset_and_editable_prompt(self) -> None:
        dataset = WebShopDataset(self.root, "test")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].session_id, 2853)
        sample = dataset[0].with_episode_data(
            instruction="buy x",
            observation="search page",
        )
        self.assertEqual(
            dataset.build_task(
                sample,
                template='Dataset task: {instruction}\nReturn JSON: {"done": true}',
            ),
            'Dataset task: buy x\nReturn JSON: {"done": true}',
        )
        with self.assertRaises(ValueError):
            build_webshop_task_prompt("buy x", template="missing placeholder")

    def test_tools_validate_actions_and_report_terminal_state(self) -> None:
        backend = FakeWebShopBackend()
        environment = self.make_environment(backend)
        self.assertIn("find a red test product", environment.task)
        self.assertEqual(environment.context["session_id"], 2853)
        self.assertEqual(
            set(environment.tools()),
            {"observe", "act", "available_actions", "episode_report", "shopping_instruction"},
        )
        self.assertEqual(environment.available_actions(), ["search[keywords]"])
        with self.assertRaises(ValueError):
            environment.act("[FINISH]")

        environment.act("search[red product]")
        environment.act("click[ITEM1]")
        environment.act("click[red]")
        state = environment.act("click[Buy Now]")
        self.assertTrue(state["done"])
        self.assertTrue(state["success"])
        self.assertEqual(backend.actions[-1], "click[buy now]")
        status = environment.status()
        self.assertTrue(status.done)
        self.assertIn("reward=1.000", status.final_answer)
        self.assertEqual(environment.report()["steps"], 4)
        environment.close()
        environment.close()
        self.assertEqual(backend.close_calls, 1)

    def test_codeact_capabilities_are_role_safe_and_live(self) -> None:
        backend = FakeWebShopBackend()
        environment = self.make_environment(backend)
        root = environment.codeact_capabilities(is_root=True, depth=0)
        child = environment.codeact_capabilities(is_root=False, depth=1)

        self.assertIn("purchase", root)
        self.assertNotIn("return_to_parent", root)
        self.assertNotIn("purchase", child)
        self.assertIn("observe", child)
        self.assertIn("search", child)
        self.assertIn("click", child)
        self.assertIn("available_actions", child)
        self.assertIn("episode_report", child)
        self.assertIn("shopping_instruction", child)
        self.assertEqual(root["shopping_instruction"].runtime_value, environment.instruction)

        with self.assertRaisesRegex(ValueError, "purchase"):
            child["click"].runtime_value("Buy Now")
        self.assertEqual(backend.actions, [])
        self.assertFalse(backend.done)

        with self.assertRaisesRegex(RuntimeError, "purchase.*automatically") as raised:
            environment.finalize_root("premature")
        self.assertNotIn("finish", str(raised.exception).casefold())
        self.assertEqual(backend.actions, [])
        environment.close()

    def test_registry_creates_webshop_plugin(self) -> None:
        self.assertIn("webshop", available_environments())
        environment = create_environment(
            "WebShop",
            recode_root=self.root,
            split="test",
            instance_id=0,
            backend=FakeWebShopBackend(),
        )
        self.assertIsInstance(environment, ReCodeWebShopEnvironment)
        environment.close()

    def test_environment_runner_uses_shared_codeact_webshop_episode(self) -> None:
        backend = FakeWebShopBackend()
        environment = self.make_environment(backend)
        self.assertTrue(environment.use_recursive_codeact_harness)
        config = self.root / "model.yaml"
        config.write_text(
            "model:\n"
            "  type: api\n"
            "  api:\n"
            "    model: fake\n"
            "    api_key: super-secret-api-key\n",
            encoding="utf-8",
        )

        root_action_spaces: list[str] = []
        child_action_spaces: list[str] = []
        root_observed_stages: list[int] = []

        def handler(messages, timeout):
            initial_user = messages[1]["content"]
            is_child = "Navigate to the product page and search for the red product." in initial_user
            action_space = initial_user.split("# Action Space\n", 1)[1]
            if is_child:
                child_action_spaces.append(action_space)
                child_turn = len(child_action_spaces)
                if child_turn == 1:
                    return (
                        "<python>\n"
                        "state = search('red product')\n"
                        "print(state)\n"
                        "</python>"
                    )
                return (
                    "<python>\n"
                    "return_to_parent('searched red product')\n"
                    "</python>"
                )

            root_action_spaces.append(action_space)
            root_turn = len(root_action_spaces)
            if root_turn == 2:
                root_observed_stages.append(backend.stage)
                return "<python>\nstate = observe()\nprint(state)\n</python>"
            responses = {
                1: (
                    "<python>\n"
                    "child_result = await spawn_subagent("
                    "'Navigate to the product page and search for the red product.'"
                    ")\n"
                    "print(child_result)\n"
                    "</python>"
                ),
                3: "<python>\nstate = click('item1')\nprint(state)\n</python>",
                4: "<python>\nstate = click('red')\nprint(state)\n</python>",
                5: "<python>\nstate = purchase()\nprint(state)\n</python>",
            }
            return responses[root_turn]

        factory = FakeFactory(handler)
        run = run_environment(
            environment,
            model_config=config,
            agent_kwargs={
                "max_steps": 7,
                "max_depth": 1,
                "client_factory": factory,
            },
        )
        self.assertEqual(run.agent_result.status, "environment_done")
        self.assertEqual(run.environment_report["reward"], 1.0)
        self.assertEqual(run.environment_report["steps"], 4)
        self.assertEqual(backend.actions, [
            "search[red product]",
            "click[item1]",
            "click[red]",
            "click[buy now]",
        ])
        self.assertEqual(factory.created, 1)
        self.assertEqual(factory.closed_clients, 1)
        self.assertEqual(backend.close_calls, 1)
        self.assertEqual(root_observed_stages, [1])
        self.assertEqual(len(run.agent_result.trace.children), 1)
        self.assertEqual(run.agent_result.trace.children[0].status, "completed")

        system_prompt = factory.calls[0][0]["content"]
        self.assertEqual(system_prompt, environment.environment_system_prompt)
        self.assertIn("shared browser", system_prompt)
        self.assertIn("state-changing", system_prompt)
        self.assertIn("await spawn_subagent", system_prompt)
        self.assertIn("exactly one executable block", system_prompt)
        self.assertIn("<python>", system_prompt)
        self.assertIn("current Action Space", system_prompt)
        self.assertNotIn(environment.instruction, system_prompt)
        self.assertNotIn("# Task", system_prompt)
        self.assertIn("purchase", root_action_spaces[0])
        self.assertIn("finish", root_action_spaces[0])
        self.assertNotIn("return_to_parent", root_action_spaces[0])
        self.assertIn("return_to_parent", child_action_spaces[0])
        self.assertNotIn("purchase", child_action_spaces[0])
        self.assertNotIn("finish", child_action_spaces[0])
        exported = run.to_trace_dict()
        serialized = json.dumps(exported)
        self.assertNotIn("super-secret-api-key", serialized)
        self.assertEqual(exported["prompts"]["task"], environment.task)
        self.assertEqual(exported["tools"]["purchase"]["kind"], "callable")
        self.assertIn("search", exported["tools"])
        first_execution = exported["agent_result"]["trace"]["steps"][0][
            "code_executions"
        ][0]
        self.assertIn("await spawn_subagent", first_execution["code"])
        self.assertIn("searched red product", first_execution["stdout"])
        self.assertIn("child_result", first_execution["variables"])


if __name__ == "__main__":
    unittest.main()
