from __future__ import annotations

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from recursive_agent import (
    AgentConfig,
    AgentLimits,
    CapabilityCollection,
    ConfigurationError,
    SharedBudget,
)
from recursive_agent.envs.base import AgentEnvironment
from recursive_agent.tools import (
    format_tools_for_prompt,
    parse_tools,
    tool_values,
)
from recursive_agent.types import EnvironmentStatus


class _LegacyEnvironment(AgentEnvironment):
    name = "test"

    @property
    def task(self) -> str:
        return "task"

    @property
    def context(self) -> dict[str, int]:
        return {"value": 1}

    def tools(self) -> dict[str, object]:
        return {
            "root_only": {
                "tool": lambda: "root",
                "description": "root capability",
            },
            "shared": {
                "tool": lambda: "shared",
                "description": "shared capability",
            },
        }

    @property
    def system_prompt(self) -> str:
        return "environment guidance"

    @property
    def delegated_disabled_tools(self) -> frozenset[str]:
        return frozenset({"root_only"})

    def status(self) -> EnvironmentStatus:
        return EnvironmentStatus(done=False)

    def report(self) -> dict[str, object]:
        return {}

    def close(self) -> None:
        return None


class _AsyncEnvironment(_LegacyEnvironment):
    async def observe(self) -> dict[str, bool]:
        return {"async": True}


class SharedBudgetTests(unittest.TestCase):
    def test_validation_and_reservation_lifecycle(self) -> None:
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                SharedBudget(value)  # type: ignore[arg-type]

        budget = SharedBudget(2)
        self.assertEqual(budget.max_total_steps, 2)
        self.assertEqual(budget.consumed_steps, 0)
        self.assertEqual(budget.remaining_steps, 2)

        failed = budget.reserve()
        self.assertIsNotNone(failed)
        assert failed is not None
        failed.release()
        self.assertEqual(budget.consumed_steps, 0)
        self.assertEqual(budget.remaining_steps, 2)

        successful = budget.reserve()
        self.assertIsNotNone(successful)
        assert successful is not None
        successful.commit()
        self.assertEqual(budget.consumed_steps, 1)
        self.assertEqual(budget.used_steps, 1)
        self.assertEqual(budget.remaining_steps, 1)
        with self.assertRaises(RuntimeError):
            successful.commit()

    def test_concurrent_reservations_never_oversubscribe(self) -> None:
        max_steps = 5
        workers = 32
        budget = SharedBudget(max_steps)
        barrier = threading.Barrier(workers)

        def reserve_once() -> object | None:
            barrier.wait()
            return budget.reserve()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            reservations = list(executor.map(lambda _: reserve_once(), range(workers)))

        accepted = [item for item in reservations if item is not None]
        self.assertEqual(len(accepted), max_steps)
        self.assertEqual(budget.reserved_steps, max_steps)
        self.assertEqual(budget.consumed_steps, 0)
        for reservation in accepted:
            assert reservation is not None
            reservation.release()  # type: ignore[union-attr]
        self.assertEqual(budget.reserved_steps, 0)
        self.assertEqual(budget.consumed_steps, 0)


class CapabilityAndEnvironmentContractTests(unittest.TestCase):
    def test_one_capability_collection_binds_and_renders(self) -> None:
        lookup = lambda: 7
        capabilities = CapabilityCollection(
            {"lookup": {"tool": lookup, "description": "look up a value"}}
        )
        self.assertIs(capabilities.bind()["lookup"], lookup)
        self.assertEqual(
            capabilities.format_for_prompt(),
            "- `lookup`: look up a value",
        )
        self.assertIs(tool_values(capabilities)["lookup"], lookup)
        self.assertEqual(
            format_tools_for_prompt(parse_tools({"lookup": {"tool": lookup, "description": "look up a value"}})),
            "- `lookup`: look up a value",
        )

    def test_capability_names_are_validated(self) -> None:
        for name in ("answer", "not-valid", "class"):
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                CapabilityCollection({name: object()})

    def test_legacy_environment_gets_role_aware_capabilities(self) -> None:
        environment = _LegacyEnvironment()
        root = environment.codeact_capabilities(is_root=True, depth=0)
        child = environment.codeact_capabilities(is_root=False, depth=1)
        self.assertEqual(set(root), {"root_only", "shared"})
        self.assertEqual(set(child), {"shared"})
        self.assertEqual(environment.codeact_namespace(is_root=False, depth=1).keys(), {"shared"})
        self.assertIn("shared capability", environment.codeact_descriptions(is_root=False, depth=1) or "")
        self.assertEqual(environment.observe(), {"value": 1})
        self.assertEqual(environment.environment_system_prompt, "environment guidance")
        self.assertEqual(environment.finalize("result"), "result")
        self.assertEqual(environment.finalize_root("result"), "result")

    def test_async_observation_override_is_compatible(self) -> None:
        observation = asyncio.run(_AsyncEnvironment().observe())
        self.assertEqual(observation, {"async": True})

    def test_agent_limits_keep_legacy_config_compatible(self) -> None:
        limits = AgentLimits(max_total_steps=9, max_depth=2)
        self.assertEqual((limits.max_total_steps, limits.max_depth), (9, 2))
        config = AgentConfig(max_steps=9, max_depth=2)
        self.assertEqual(config.limits, limits)
        for kwargs in ({"max_total_steps": 0}, {"max_depth": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                AgentLimits(**kwargs)


if __name__ == "__main__":
    unittest.main()
