"""Official-style answer parsing and scoring for Oolong-real DnD examples."""

from __future__ import annotations

import re
from typing import Any

_BOXED_TEXT = re.compile(r"\\boxed\{\\text\{([^}]*)\}\}")
_BOXED = re.compile(r"\\boxed\{+([^}]*)\}+")


def parse_answer(value: Any) -> int | str | list[str]:
    """Match Oolong's DnD parser for integer, comma-list, and text answers."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return text


def parse_response(response: str) -> tuple[int | str | list[str], str]:
    text = str(response)
    match = _BOXED_TEXT.search(text) or _BOXED.search(text)
    if match is None:
        return parse_answer(text), "low"
    return parse_answer(match.group(1)), "high"


def score_answer(gold: Any, submitted: Any) -> float:
    expected = parse_answer(gold)
    actual = parse_answer(submitted)
    if isinstance(expected, int) and isinstance(actual, int):
        return 0.75 ** abs(expected - actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return float(expected.strip().lower() == actual.strip().lower())
    if isinstance(expected, list) and isinstance(actual, list):
        overlap = set(expected) & set(actual)
        return len(overlap) / len(expected) if expected else 0.0
    return 0.0
