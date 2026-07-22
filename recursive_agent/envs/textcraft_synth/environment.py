"""Shared-inventory TextCraft-Synth environment."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...types import EnvironmentStatus
from ..base import AgentEnvironment
from ..registry import register_environment
from .dataset import (
    DEFAULT_SPLIT,
    DEFAULT_TEXTCRAFT_ROOT,
    TextCraftDataset,
    TextCraftRecipe,
    TextCraftSample,
)
from .prompts import (
    DEFAULT_TEXTCRAFT_AGENT_PROMPT,
    DEFAULT_TEXTCRAFT_DELEGATED_FORCED_FINAL_PROMPT,
    DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT,
    DEFAULT_TEXTCRAFT_TASK_TEMPLATE,
    build_textcraft_task_prompt,
)
from .scoring import evaluate_inventory
from .tools import build_textcraft_tools


@register_environment("textcraft_synth")
class TextCraftSynthEnvironment(AgentEnvironment):
    name = "textcraft_synth"

    def __init__(
        self,
        *,
        textcraft_root: str | Path | None = DEFAULT_TEXTCRAFT_ROOT,
        split: str = DEFAULT_SPLIT,
        instance_id: int = 0,
        data_path: str | Path | None = None,
        samples: Sequence[Mapping[str, Any] | TextCraftSample] | None = None,
        generated_count: int = 1,
        generated_difficulty: str = "medium",
        generated_seed: int = 0,
        prompt_template: str = DEFAULT_TEXTCRAFT_TASK_TEMPLATE,
        agent_prompt: str | None = None,
    ) -> None:
        self.dataset = TextCraftDataset(
            textcraft_root=textcraft_root,
            split=split,
            data_path=data_path,
            samples=samples,
            generated_count=generated_count,
            generated_difficulty=generated_difficulty,
            generated_seed=generated_seed,
        )
        self.sample = self.dataset[instance_id]
        self._task = build_textcraft_task_prompt(
            self.sample.targets,
            template=prompt_template,
        )
        self._agent_prompt = (
            str(agent_prompt).strip()
            if agent_prompt is not None
            else DEFAULT_TEXTCRAFT_AGENT_PROMPT
        )
        self._inventory = dict(self.sample.initial_inventory)
        self._finished = False
        self._finish_message: str | None = None
        self._finish_attempts = 0
        self._craft_history: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._lock = threading.RLock()
        self._tools = build_textcraft_tools(self)
        self._context = {
            "environment": self.name,
            "dataset": self.dataset.dataset_name,
            "split": self.dataset.split,
            "instance_id": self.sample.index,
            "id": self.sample.sample_id,
            "difficulty": self.sample.difficulty,
            "crafting_depth": self.sample.crafting_depth,
        }

    @property
    def task(self) -> str:
        return self._task

    @property
    def agent_prompt(self) -> str:
        return self._agent_prompt

    @property
    def forced_final_prompt(self) -> str:
        return DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT

    @property
    def delegated_forced_final_prompt(self) -> str:
        return DEFAULT_TEXTCRAFT_DELEGATED_FORCED_FINAL_PROMPT

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def tools(self) -> dict[str, Any]:
        return dict(self._tools)

    def view_inventory(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._inventory.items()))

    def get_info(self, items: list[str] | None = None) -> list[dict[str, Any]]:
        if items is None:
            items = list(self.sample.targets)
        if not isinstance(items, list):
            raise TypeError("get_info expects a list of item names")
        with self._lock:
            return [self._item_info(str(item)) for item in items]

    def craft(self, ingredients: dict[str, int], target: tuple[str, int]) -> str:
        try:
            return self._craft(ingredients, target)
        except Exception as exc:
            with self._lock:
                self._errors.append(f"{type(exc).__name__}: {exc}")
            raise

    def _craft(self, ingredients: dict[str, int], target: tuple[str, int]) -> str:
        item, output_count = _parse_target(target)
        supplied = _parse_counts(ingredients, "ingredients")
        with self._lock:
            if self._finished:
                raise RuntimeError("TextCraft episode is already finished")
            recipes = self.sample.recipes.get(item, ())
            if not recipes:
                raise ValueError(f"Item {item!r} has no crafting recipe")

            selected: TextCraftRecipe | None = None
            for recipe in recipes:
                if output_count % recipe.result_count:
                    continue
                executions = output_count // recipe.result_count
                expected = {
                    ingredient: count * executions
                    for ingredient, count in recipe.ingredients.items()
                }
                if supplied == expected:
                    selected = recipe
                    break
            if selected is None:
                raise ValueError(
                    f"Ingredients do not match a recipe for {item!r}; "
                    "use get_info() and provide exact scaled counts"
                )
            for ingredient, count in supplied.items():
                if self._inventory.get(ingredient, 0) < count:
                    raise ValueError(
                        f"Insufficient {ingredient!r}: have "
                        f"{self._inventory.get(ingredient, 0)}, need {count}"
                    )
            for ingredient, count in supplied.items():
                remaining = self._inventory[ingredient] - count
                if remaining:
                    self._inventory[ingredient] = remaining
                else:
                    self._inventory.pop(ingredient, None)
            self._inventory[item] = self._inventory.get(item, 0) + output_count
            event = {
                "item": item,
                "output_count": output_count,
                "ingredients": dict(supplied),
                "inventory_after": dict(self._inventory),
            }
            self._craft_history.append(event)
            return (
                f"Crafted {output_count}x {item}. "
                f"Inventory now has {self._inventory.get(item, 0)}x {item}."
            )

    def finish(self, message: str) -> str:
        with self._lock:
            if self._finished:
                return self._finish_message or "TextCraft episode already finished."
            evaluation = self._evaluation()
            self._finish_attempts += 1
            if not evaluation.success:
                return (
                    f"Not finished: requested targets are still missing "
                    f"{evaluation.missing}. Continue crafting and check the inventory."
                )
            self._finished = True
            self._finish_message = (
                f"{str(message).strip()} | success={evaluation.success} "
                f"score={evaluation.score:.3f}"
            )
            return self._finish_message

    def status(self) -> EnvironmentStatus:
        with self._lock:
            if not self._finished:
                return EnvironmentStatus(done=False)
            evaluation = self._evaluation()
            return EnvironmentStatus(
                done=True,
                final_answer=self._finish_message,
                reason="success" if evaluation.success else "finished_incomplete",
            )

    def report(self) -> dict[str, Any]:
        with self._lock:
            evaluation = self._evaluation()
            return {
                "environment": self.name,
                "dataset": self.dataset.dataset_name,
                "split": self.dataset.split,
                "instance_id": self.sample.index,
                "id": self.sample.sample_id,
                "difficulty": self.sample.difficulty,
                "crafting_depth": self.sample.crafting_depth,
                "initial_inventory": dict(self.sample.initial_inventory),
                "targets": dict(self.sample.targets),
                "inventory": evaluation.inventory,
                "required_final_inventory": evaluation.required,
                "missing": evaluation.missing,
                "success": evaluation.success,
                "score": evaluation.score,
                "finished": self._finished,
                "finish_attempts": self._finish_attempts,
                "finish_message": self._finish_message,
                "craft_calls": len(self._craft_history),
                "craft_history": list(self._craft_history),
                "tool_errors": list(self._errors),
                "source": self.dataset.metadata()["source"],
            }

    def close(self) -> None:
        return None

    def _evaluation(self):
        return evaluate_inventory(
            initial_inventory=self.sample.initial_inventory,
            targets=self.sample.targets,
            inventory=self._inventory,
        )

    def _item_info(self, item: str) -> dict[str, Any]:
        recipes = self.sample.recipes.get(item, ())
        return {
            "item": item,
            "can_craft": any(
                all(self._inventory.get(name, 0) >= count for name, count in recipe.ingredients.items())
                for recipe in recipes
            ),
            "is_base": not recipes,
            "in_inventory": self._inventory.get(item, 0),
            "crafting_depth": self._item_depth(item),
            "recipes": [recipe.to_dict() for recipe in recipes],
        }

    def _item_depth(self, item: str) -> int:
        memo: dict[str, int] = {}
        visiting: set[str] = set()

        def depth(current: str) -> int:
            if current in memo:
                return memo[current]
            if current in visiting:
                return 0
            recipes = self.sample.recipes.get(current, ())
            if not recipes:
                memo[current] = 0
                return 0
            visiting.add(current)
            result = 1 + max(depth(name) for name in recipes[0].ingredients)
            visiting.remove(current)
            memo[current] = result
            return result

        return depth(item)


def _parse_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a dict of item names to counts")
    result = {}
    for raw_item, raw_count in value.items():
        item = str(raw_item).strip()
        count = int(raw_count)
        if not item or count <= 0:
            raise ValueError(f"{label} counts must be positive")
        result[item] = result.get(item, 0) + count
    return result


def _parse_target(value: Any) -> tuple[str, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("target must be a (item_name, output_count) tuple")
    item = str(value[0]).strip()
    count = int(value[1])
    if not item or count <= 0:
        raise ValueError("target item must be non-empty and output_count positive")
    return item, count


__all__ = ["TextCraftSynthEnvironment"]
