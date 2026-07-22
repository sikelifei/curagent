"""TextCraft-Synth data loading and deterministic local task generation."""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..base import EnvironmentDependencyError

DEFAULT_TEXTCRAFT_ROOT = Path(
    "/data2/zhangwenjian/agent/bench/textcraft-synth"
)
DEFAULT_DATASET_NAME = "textcraft-synth"
DEFAULT_SPLIT = "test"


@dataclass(frozen=True)
class TextCraftRecipe:
    """One recipe execution, before scaling it by a requested output count."""

    ingredients: dict[str, int]
    result_count: int

    def __post_init__(self) -> None:
        if self.result_count <= 0:
            raise ValueError("recipe result_count must be positive")
        if not self.ingredients:
            raise ValueError("recipe ingredients cannot be empty")
        for item, count in self.ingredients.items():
            if not isinstance(item, str) or not item.strip():
                raise ValueError("recipe ingredient names must be non-empty strings")
            if not isinstance(count, int) or count <= 0:
                raise ValueError("recipe ingredient counts must be positive integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingredients": dict(self.ingredients),
            "result_count": self.result_count,
        }


@dataclass(frozen=True)
class TextCraftSample:
    """A single agent-visible crafting episode."""

    index: int
    sample_id: str
    initial_inventory: dict[str, int]
    recipes: dict[str, tuple[TextCraftRecipe, ...]]
    targets: dict[str, int]
    difficulty: str = "unknown"
    crafting_depth: int | None = None
    split: str = DEFAULT_SPLIT
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("TextCraft sample_id cannot be empty")
        _validate_counts(self.initial_inventory, "initial_inventory", allow_zero=True)
        _validate_counts(self.targets, "targets", allow_zero=False)
        if not self.targets:
            raise ValueError("TextCraft targets cannot be empty")
        for item, recipe_list in self.recipes.items():
            if not item.strip() or not recipe_list:
                raise ValueError("every recipe item must have at least one recipe")
            if not all(isinstance(recipe, TextCraftRecipe) for recipe in recipe_list):
                raise ValueError("recipes must contain TextCraftRecipe values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "initial_inventory": dict(self.initial_inventory),
            "recipes": {
                item: [recipe.to_dict() for recipe in recipe_list]
                for item, recipe_list in self.recipes.items()
            },
            "targets": dict(self.targets),
            "difficulty": self.difficulty,
            "crafting_depth": self.crafting_depth,
            "split": self.split,
            "metadata": dict(self.metadata or {}),
        }


class TextCraftDataset:
    """Load TextCraft rows from JSON/JSONL/parquet or generated fallback rows.

    The paper describes the environment action space but does not publish a
    canonical file format.  The loader accepts the common field names used by
    generated benchmark files and normalizes them into ``TextCraftSample``.
    """

    def __init__(
        self,
        *,
        textcraft_root: str | Path | None = DEFAULT_TEXTCRAFT_ROOT,
        split: str = DEFAULT_SPLIT,
        data_path: str | Path | None = None,
        samples: Sequence[Mapping[str, Any] | TextCraftSample] | None = None,
        generated_count: int = 1,
        generated_difficulty: str = "medium",
        generated_seed: int = 0,
    ) -> None:
        self.split = str(split).strip() or DEFAULT_SPLIT
        self.dataset_name = DEFAULT_DATASET_NAME
        self._source: str

        if samples is not None:
            raw_rows: list[Mapping[str, Any] | TextCraftSample] = list(samples)
            self._source = "in_memory"
        else:
            source = _resolve_data_path(
                data_path=data_path,
                textcraft_root=textcraft_root,
                split=self.split,
            )
            if source is None:
                if generated_count <= 0:
                    raise FileNotFoundError(
                        "No TextCraft data found. Pass --data-path or use a positive "
                        "generated_count for the local synthetic fallback."
                    )
                raw_rows = generate_textcraft_samples(
                    count=generated_count,
                    difficulty=generated_difficulty,
                    seed=generated_seed,
                    split=self.split,
                )
                self._source = "generated"
            else:
                raw_rows = list(_iter_rows(source))
                self._source = str(source)

        self._samples = tuple(
            row
            if isinstance(row, TextCraftSample)
            else _normalize_sample(row, index=index, split=self.split)
            for index, row in enumerate(raw_rows)
        )
        if not self._samples:
            raise ValueError("TextCraft dataset contains no samples")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> TextCraftSample:
        if not isinstance(index, int):
            raise TypeError("TextCraft sample index must be an integer")
        try:
            return self._samples[index]
        except IndexError as exc:
            raise IndexError(
                f"TextCraft sample index {index} is outside split {self.split!r} "
                f"with {len(self)} samples"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        difficulties: dict[str, int] = defaultdict(int)
        for sample in self._samples:
            difficulties[sample.difficulty] += 1
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "source": self._source,
            "count": len(self),
            "difficulty_counts": dict(sorted(difficulties.items())),
        }


def _resolve_data_path(
    *,
    data_path: str | Path | None,
    textcraft_root: str | Path | None,
    split: str,
) -> Path | None:
    explicit = data_path or os.getenv("TEXTCRAFT_DATA_PATH")
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"TextCraft data path not found: {source}")
        return source
    if textcraft_root is None:
        return None
    root = Path(textcraft_root).expanduser().resolve()
    if root.is_file():
        return root
    candidates = [
        root / f"{split}.jsonl",
        root / f"{split}.json",
        root / "data" / f"{split}.jsonl",
        root / "data" / f"{split}.json",
        root / f"{split}-00000-of-00001.parquet",
        root / "data" / f"{split}-00000-of-00001.parquet",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _iter_rows(source: Path) -> Iterator[Mapping[str, Any]]:
    if source.is_dir():
        candidates = sorted(
            path
            for pattern in ("*.jsonl", "*.json", "*.parquet")
            for path in source.rglob(pattern)
            if path.is_file()
        )
        if not candidates:
            raise FileNotFoundError(f"No JSON/JSONL/parquet TextCraft files in {source}")
        for candidate in candidates:
            yield from _iter_rows(candidate)
        return

    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed TextCraft JSONL row {line_number} in {source}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"TextCraft row {line_number} must be an object")
                yield row
        return

    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and "samples" in value:
            value = value["samples"]
        if not isinstance(value, list):
            raise ValueError("TextCraft JSON must contain a list or a samples list")
        for row in value:
            if not isinstance(row, Mapping):
                raise ValueError("TextCraft JSON rows must be objects")
            yield row
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "Install parquet support with python -m pip install -e '.[all-providers]'"
            ) from exc
        parquet_file = parquet.ParquetFile(source)
        for batch in parquet_file.iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return

    raise ValueError(f"Unsupported TextCraft data file: {source}")


def _normalize_sample(
    row: Mapping[str, Any],
    *,
    index: int,
    split: str,
) -> TextCraftSample:
    inventory_value = _first(row, "initial_inventory", "inventory", "initial_items")
    if inventory_value is None:
        raise ValueError(f"TextCraft row {index} is missing initial_inventory")
    targets_value = _first(row, "targets", "target", "goal", "requested_items")
    if targets_value is None and row.get("target_item") is not None:
        targets_value = {row["target_item"]: row.get("target_count", 1)}
    if targets_value is None:
        targets_value = _targets_from_text(row.get("task", row.get("prompt", "")))
    if targets_value is None:
        raise ValueError(f"TextCraft row {index} is missing targets")
    recipes_value = _first(row, "recipes", "recipe_book", "crafting_recipes")
    if recipes_value is None:
        raise ValueError(f"TextCraft row {index} is missing recipes")

    sample_id = str(_first(row, "id", "sample_id", "instance_id") or index)
    normalized_recipes = _normalize_recipes(recipes_value)
    targets = _normalize_counts(targets_value, "targets", allow_zero=False)
    inventory = _normalize_counts(inventory_value, "initial_inventory", allow_zero=True)
    depth_value = _first(row, "crafting_depth", "depth", "task_depth")
    depth = int(depth_value) if depth_value is not None else _max_target_depth(
        targets, normalized_recipes
    )
    metadata = {
        str(key): value
        for key, value in row.items()
        if key
        not in {
            "id",
            "sample_id",
            "instance_id",
            "initial_inventory",
            "inventory",
            "initial_items",
            "targets",
            "target",
            "goal",
            "requested_items",
            "target_item",
            "target_count",
            "recipes",
            "recipe_book",
            "crafting_recipes",
            "difficulty",
            "crafting_depth",
            "depth",
            "task_depth",
        }
    }
    return TextCraftSample(
        index=index,
        sample_id=sample_id,
        initial_inventory=inventory,
        recipes=normalized_recipes,
        targets=targets,
        difficulty=str(row.get("difficulty", _difficulty_for_depth(depth))),
        crafting_depth=depth,
        split=str(row.get("split", split)),
        metadata=metadata,
    )


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _normalize_recipes(value: Any) -> dict[str, tuple[TextCraftRecipe, ...]]:
    normalized: dict[str, list[TextCraftRecipe]] = defaultdict(list)
    if isinstance(value, Mapping):
        for item, raw in value.items():
            normalized[str(item)].extend(_recipe_list(raw))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for raw in value:
            if not isinstance(raw, Mapping):
                raise ValueError("recipe list entries must be objects")
            item = _first(raw, "item", "target", "output", "result")
            if item is None:
                raise ValueError("recipe list entry is missing its output item")
            normalized[str(item)].extend(_recipe_list(raw))
    else:
        raise ValueError("recipes must be a mapping or a list")
    return {item: tuple(recipes) for item, recipes in normalized.items()}


def _recipe_list(value: Any) -> list[TextCraftRecipe]:
    if isinstance(value, Mapping):
        if "recipes" in value:
            value = value["recipes"]
        else:
            return [_normalize_recipe(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_recipe(item) for item in value]
    raise ValueError("a recipe must be an object or list of objects")


def _normalize_recipe(value: Any) -> TextCraftRecipe:
    if not isinstance(value, Mapping):
        raise ValueError("recipe must be an object")
    ingredients = _first(value, "ingredients", "input", "inputs")
    if ingredients is None or not isinstance(ingredients, Mapping):
        raise ValueError("recipe is missing an ingredients mapping")
    result = _first(value, "result_count", "count", "output_count", "quantity")
    if result is None:
        result_value = value.get("result")
        if isinstance(result_value, Mapping):
            result = _first(result_value, "count", "quantity")
        elif isinstance(result_value, int):
            result = result_value
    if result is None:
        raise ValueError("recipe is missing result_count")
    return TextCraftRecipe(
        ingredients=_normalize_counts(ingredients, "recipe ingredients", allow_zero=False),
        result_count=int(result),
    )


def _normalize_counts(value: Any, label: str, *, allow_zero: bool) -> dict[str, int]:
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = []
        for item in value:
            if isinstance(item, Mapping):
                name = _first(item, "item", "name")
                count = _first(item, "count", "quantity", "amount")
                if name is None or count is None:
                    raise ValueError(f"{label} list entry needs item and count")
                pairs.append((name, count))
            elif isinstance(item, Sequence) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise ValueError(f"{label} list entries must be item/count pairs")
    else:
        raise ValueError(f"{label} must be a mapping or item/count list")
    normalized: dict[str, int] = {}
    for raw_name, raw_count in pairs:
        name = str(raw_name).strip()
        count = int(raw_count)
        if not name or count < 0 or (count == 0 and not allow_zero):
            raise ValueError(f"{label} contains invalid item/count: {raw_name!r}, {raw_count!r}")
        if count == 0 and allow_zero:
            continue
        normalized[name] = normalized.get(name, 0) + count
    return normalized


def _validate_counts(value: Mapping[str, int], label: str, *, allow_zero: bool) -> None:
    _normalize_counts(value, label, allow_zero=allow_zero)


def _targets_from_text(value: Any) -> dict[str, int] | None:
    if not isinstance(value, str):
        return None
    matches = re.findall(r"(\d+)\s*x\s*([^,;]+?)(?=\s*,|\s*;|$)", value, flags=re.I)
    if not matches:
        return None
    return {item.strip(): int(count) for count, item in matches if item.strip()}


def _max_target_depth(
    targets: Mapping[str, int], recipes: Mapping[str, Sequence[TextCraftRecipe]]
) -> int:
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(item: str) -> int:
        if item in memo:
            return memo[item]
        if item in visiting:
            raise ValueError(f"cyclic TextCraft recipe dependency at {item!r}")
        recipe_list = recipes.get(item)
        if not recipe_list:
            memo[item] = 0
            return 0
        visiting.add(item)
        value = 1 + max(
            depth(ingredient)
            for ingredient in recipe_list[0].ingredients
        )
        visiting.remove(item)
        memo[item] = value
        return value

    return max(depth(item) for item in targets)


def _difficulty_for_depth(depth: int) -> str:
    if depth <= 3:
        return "easy"
    if depth <= 6:
        return "medium"
    return "hard"


@dataclass
class _GeneratedNode:
    item: str
    depth: int
    recipe: TextCraftRecipe | None
    children: list[tuple["_GeneratedNode", int]]


def generate_textcraft_samples(
    *,
    count: int,
    difficulty: str = "medium",
    seed: int = 0,
    split: str = DEFAULT_SPLIT,
) -> list[dict[str, Any]]:
    """Generate solvable synthetic tasks with the paper's depth bands.

    This is a local fallback and a useful smoke-test corpus.  A released
    TextCraft file passed via ``data_path`` always takes precedence.
    """
    if count <= 0:
        raise ValueError("generated count must be positive")
    normalized_difficulty = str(difficulty).strip().lower()
    ranges = {"easy": (2, 3), "medium": (4, 6), "hard": (7, 9)}
    if normalized_difficulty not in ranges:
        raise ValueError("difficulty must be easy, medium, or hard")
    low, high = ranges[normalized_difficulty]
    rows = []
    for sample_index in range(count):
        rng = random.Random(seed + sample_index * 1009)
        depth = rng.randint(low, high)
        counter = [0]

        def build(level: int, path: str) -> _GeneratedNode:
            if level == 0:
                item = f"raw_{sample_index}_{path}"
                return _GeneratedNode(item, 0, None, [])
            item = f"m{level}_{sample_index}_{path}"
            child_count = 1 if level == 1 else rng.randint(2, 3)
            children: list[tuple[_GeneratedNode, int]] = []
            for child_index in range(child_count):
                child = build(level - 1, f"{path}{child_index}")
                children.append((child, rng.randint(1, 2)))
            recipe = TextCraftRecipe(
                ingredients={child.item: amount for child, amount in children},
                result_count=rng.choice((1, 2)),
            )
            counter[0] += 1
            return _GeneratedNode(item, level, recipe, children)

        root = build(depth, "r")
        target_count = root.recipe.result_count * rng.choice((1, 2)) if root.recipe else 1
        inventory: dict[str, int] = defaultdict(int)
        recipes: dict[str, dict[str, Any]] = {}

        def add_requirements(node: _GeneratedNode, quantity: int) -> None:
            if node.recipe is None:
                inventory[node.item] += quantity
                return
            executions = math.ceil(quantity / node.recipe.result_count)
            recipes[node.item] = node.recipe.to_dict()
            for child, amount in node.children:
                add_requirements(child, executions * amount)

        add_requirements(root, target_count)
        rows.append(
            {
                "id": f"generated-{normalized_difficulty}-{seed}-{sample_index}",
                "split": split,
                "initial_inventory": dict(sorted(inventory.items())),
                "recipes": recipes,
                "targets": {root.item: target_count},
                "difficulty": normalized_difficulty,
                "crafting_depth": depth,
                "generator_seed": seed + sample_index * 1009,
                "generator_nodes": counter[0],
            }
        )
    return rows


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_SPLIT",
    "DEFAULT_TEXTCRAFT_ROOT",
    "TextCraftDataset",
    "TextCraftRecipe",
    "TextCraftSample",
    "generate_textcraft_samples",
]
