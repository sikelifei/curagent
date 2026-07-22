"""Scoring helpers for TextCraft-Synth inventory outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TextCraftEvaluation:
    success: bool
    score: float
    required: dict[str, int]
    inventory: dict[str, int]
    missing: dict[str, int]


def evaluate_inventory(
    *,
    initial_inventory: Mapping[str, int],
    targets: Mapping[str, int],
    inventory: Mapping[str, int],
) -> TextCraftEvaluation:
    required = {
        item: int(initial_inventory.get(item, 0)) + int(count)
        for item, count in targets.items()
    }
    normalized_inventory = {str(item): int(count) for item, count in inventory.items()}
    missing = {
        item: max(0, count - normalized_inventory.get(item, 0))
        for item, count in required.items()
        if normalized_inventory.get(item, 0) < count
    }
    if not required:
        return TextCraftEvaluation(
            success=True,
            score=1.0,
            required=required,
            inventory=normalized_inventory,
            missing=missing,
        )
    ratios = [
        min(1.0, normalized_inventory.get(item, 0) / count)
        for item, count in required.items()
        if count > 0
    ]
    return TextCraftEvaluation(
        success=not missing,
        score=sum(ratios) / len(ratios) if ratios else 1.0,
        required=required,
        inventory=normalized_inventory,
        missing=missing,
    )


__all__ = ["TextCraftEvaluation", "evaluate_inventory"]
