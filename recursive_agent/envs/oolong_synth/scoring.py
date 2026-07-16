"""Official-style parsing and scoring for Oolong-Synthetic."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

_BOXED_TEXT = re.compile(r"\\boxed\{\\text\{([^}]*)\}\}")
_BOXED = re.compile(r"\\boxed\{+([^}]*)\}+")
_DATE_OBJECT = re.compile(r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)")
_SUBMIT_CALL = re.compile(
    r"submit_answer\(\s*(?:r|u|f|b)?(?P<quote>['\"])(?P<answer>.*?)(?P=quote)\s*\)",
    re.DOTALL,
)


@dataclass(frozen=True)
class SynthEvaluation:
    candidate: str
    gold: Any
    parse_confidence: str
    score: float


def parse_gold_answer(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    text = str(value).strip()
    date_match = _DATE_OBJECT.search(text)
    if date_match:
        year, month, day = (int(part) for part in date_match.groups())
        return date(year, month, day)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, (list, tuple)):
        return parsed[0] if parsed else ""
    return parsed


def parse_synth_response(response: str) -> tuple[str, str]:
    """Follow the benchmark's last-colon parser with boxed-answer support."""
    text = str(response).strip()
    submit_call = _SUBMIT_CALL.search(text)
    if submit_call:
        text = submit_call.group("answer").strip()
    boxed = _BOXED_TEXT.search(text) or _BOXED.search(text)
    if boxed:
        text = boxed.group(1).strip()
    confidence = "low"
    if ":" in text:
        candidate = text.rsplit(":", 1)[-1].strip()
        confidence = "med"
    elif len(text) < 20:
        candidate = text
    else:
        candidate = text.split()[-1]
    candidate = candidate.replace("*", "").replace("[", "").replace("]", "")
    candidate = candidate.strip().strip("`").strip()
    if any(prefix in text for prefix in ("User:", "Answer:", "Date:", "Label:")):
        confidence = "high"
    if len(candidate) < 20:
        confidence = "vhigh" if confidence != "low" else confidence
    elif "more common" in candidate:
        candidate = "more common"
    elif "less common" in candidate:
        candidate = "less common"
    elif "same frequency" in candidate:
        candidate = "same frequency"
    return candidate, confidence


def evaluate_synth_response(
    gold_value: Any,
    response: str,
    answer_type: str,
) -> SynthEvaluation:
    gold = parse_gold_answer(gold_value)
    candidate, confidence = parse_synth_response(response)
    score = 0.0
    if str(candidate) == str(gold):
        score = 1.0
    elif candidate in {"more common", "less common", "same frequency"}:
        score = float(candidate in str(gold))
    elif str(answer_type) == "ANSWER_TYPE.NUMERIC":
        try:
            score = 0.75 ** abs(int(gold) - int(candidate))
        except (TypeError, ValueError):
            confidence = "low"
    elif str(answer_type) == "ANSWER_TYPE.DATE":
        parsed_date = _parse_date(candidate)
        score = float(parsed_date == gold)
        if parsed_date is None:
            confidence = "low"
    return SynthEvaluation(
        candidate=candidate,
        gold=gold,
        parse_confidence=confidence,
        score=float(score),
    )


def _parse_date(value: str) -> date | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


__all__ = [
    "SynthEvaluation",
    "evaluate_synth_response",
    "parse_gold_answer",
    "parse_synth_response",
]
