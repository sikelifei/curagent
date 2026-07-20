"""Smoke-test answer parsing and local-model semantic judging."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...clients.openai import OpenAIClient
from ...config import load_model_config

GRADER_TEMPLATE = """Judge semantic equivalence between a prediction and a
reference answer for the supplied question.

Question:
{question}

Reference answer:
{correct_answer}

Prediction:
{response}

Ignore capitalization, punctuation, and minor formatting differences. Do not
penalize a different explanation when the exact answer is equivalent. Be strict
about numbers, dates, and entity identity. An ambiguous answer or an answer with
multiple unsupported candidates is normally incorrect.

Return one JSON object and no markdown or extra text:
{{"correct": true, "score": 1, "reason": "brief reason"}}
The correct field must be boolean, score must be 0 or 1, and reason must be a
short string."""

_FINAL_PATTERN = re.compile(
    r"(?ims)^\s*(?:\*\*)?Explanation:\*{0,2}\s*(.*?)\s*"
    r"^\s*(?:\*\*)?Exact Answer:\*{0,2}\s*(.*?)\s*"
    r"^\s*(?:\*\*)?Confidence:\*{0,2}\s*(\d+(?:\.\d+)?)\s*%\s*$"
)


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    score: int
    reason: str
    response: str
    error: str | None
    model: str
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_final_output(response: str) -> dict[str, Any] | None:
    text = str(response).strip()
    # Models often wrap an otherwise valid answer in a Markdown code fence.
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    text = "\n".join(lines).strip()
    # A forced-final response may include a short preamble before the required
    # three-line answer. Keep the structured block and ignore that preamble.
    explanation_start = re.search(
        r"(?im)^\s*(?:\*\*)?Explanation:\*{0,2}", text
    )
    if explanation_start is not None:
        text = text[explanation_start.start() :]
    match = _FINAL_PATTERN.fullmatch(text)
    if match is None:
        return None
    confidence = float(match.group(3))
    if not 0 <= confidence <= 100:
        return None
    return {
        "explanation": match.group(1).strip(),
        "exact_answer": match.group(2).strip(),
        "confidence": confidence,
    }


def extract_final_answer(response: str) -> str:
    parsed = parse_final_output(response)
    if parsed is not None and parsed["exact_answer"]:
        return str(parsed["exact_answer"])
    return str(response).strip()


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    ).strip()


def parse_judge_response(response: str, *, model: str = "unknown") -> JudgeResult:
    compact = str(response).strip()
    fence = chr(96) * 3
    if compact.startswith(fence) and compact.endswith(fence):
        compact = compact[len(fence) : -len(fence)].strip()
        if compact.lower().startswith("json"):
            compact = compact[4:].strip()
    try:
        value = json.loads(compact)
    except (TypeError, json.JSONDecodeError) as exc:
        return JudgeResult(
            correct=False,
            score=0,
            reason="Judge output was not valid JSON.",
            response=str(response),
            error=f"{type(exc).__name__}: {exc}",
            model=model,
            attempts=1,
        )
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("correct"), bool)
        or value.get("score") not in (0, 1)
        or not isinstance(value.get("reason"), str)
    ):
        return JudgeResult(
            correct=False,
            score=0,
            reason="Judge JSON did not match the required schema.",
            response=str(response),
            error="invalid judge JSON schema",
            model=model,
            attempts=1,
        )
    correct = bool(value["correct"])
    score = int(value["score"])
    if score != int(correct):
        return JudgeResult(
            correct=False,
            score=0,
            reason="Judge correct and score fields disagreed.",
            response=str(response),
            error="inconsistent judge JSON",
            model=model,
            attempts=1,
        )
    return JudgeResult(
        correct=correct,
        score=score,
        reason=value["reason"].strip(),
        response=str(response),
        error=None,
        model=model,
        attempts=1,
    )


def judge_answer(
    *,
    model_config: str | Path,
    question: str,
    correct_answer: str,
    response: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_attempts: int = 3,
) -> JudgeResult:
    """Call the configured model separately and retry malformed JSON outputs."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if not str(response).strip():
        return JudgeResult(
            correct=False,
            score=0,
            reason="Agent response was empty.",
            response="",
            error="empty agent response",
            model="not_called",
            attempts=0,
        )

    backend, kwargs = load_model_config(model_config)
    if backend != "openai":
        raise ValueError("BrowseComp-Plus judge requires an OpenAI-compatible config")
    model = str(kwargs.get("model_name", "unknown"))
    sampling_overrides = {
        key: value
        for key, value in {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }.items()
        if value is not None
    }
    overrides: dict[str, Any] = {}
    if timeout is not None:
        overrides["timeout"] = timeout
    if sampling_overrides:
        overrides["sampling_args"] = sampling_overrides
    client = OpenAIClient(**_merge_nested(kwargs, overrides))
    prompt = create_judge_prompt(
        question,
        extract_final_answer(response),
        correct_answer,
    )
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a strict answer-equivalence judge. Return only the "
                "requested JSON object."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    last: JudgeResult | None = None
    try:
        for attempt in range(1, max_attempts + 1):
            raw = client.completion(messages, timeout=timeout)
            parsed = parse_judge_response(raw.content, model=model)
            last = JudgeResult(
                correct=parsed.correct,
                score=parsed.score,
                reason=parsed.reason,
                response=parsed.response,
                error=parsed.error,
                model=parsed.model,
                attempts=attempt,
            )
            if last.error is None:
                return last
            if attempt < max_attempts:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw.content},
                        {
                            "role": "user",
                            "content": (
                                "That output was not valid schema-compliant JSON. "
                                "Return only correct, score, and reason now."
                            ),
                        },
                    ]
                )
    finally:
        client.close()
    assert last is not None
    return last


def _merge_nested(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "GRADER_TEMPLATE",
    "JudgeResult",
    "create_judge_prompt",
    "extract_final_answer",
    "judge_answer",
    "parse_final_output",
    "parse_judge_response",
]
