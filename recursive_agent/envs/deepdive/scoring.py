"""DeepDive root-answer evaluator matching the Platoon rubric contract."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ... import clients
from ...config import load_model_config


DEEPDIVE_JUDGE_SYSTEM_PROMPT = (
    "We need to judge the performance of an deepresearch agent on a task. The task requires searching the web for information across various sources and synthesizing information together to answer a question.\n"
    "The agent may use subagents to solve parts of the task. Do not penalize the model for relying on subagents, unless the subtasks delegated to the subagents are not meaningful or useful for the task.\n"
    "You will be given the ground truth answer to the task and the agent's answer to the task.\n"
    "When comparing the agent's answer to the ground truth answer, it is acceptable to have minor formatting differences as long as the core information is equivalent."
    "Please provide a reason and success flag (boolean value) in the following format:\n"
    "```json\n"
    "{\n"
    '    "reason": "Brief reasoning for success flag here.",\n'
    '    "success": <true|false>\n'
    "}\n"
)


@dataclass(frozen=True)
class DeepDiveJudgment:
    success: bool
    reason: str
    raw_response: str
    model: str
    attempts: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def judge_deepdive_answer(
    *,
    model_config: str | Path,
    ground_truth: str,
    agent_answer: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    timeout: float | None = None,
    max_attempts: int = 3,
) -> DeepDiveJudgment:
    """Evaluate one answer with the same prompt/schema as DeepDiveEnv."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    backend, kwargs = load_model_config(model_config)
    sampling = dict(kwargs.get("sampling_args") or {})
    sampling.update({"temperature": temperature, "max_tokens": max_tokens})
    kwargs["sampling_args"] = sampling
    if timeout is not None:
        kwargs["timeout"] = timeout
    model = str(kwargs.get("model_name", "unknown"))
    messages = [
        {"role": "system", "content": DEEPDIVE_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Ground truth answer: {ground_truth}\n\n"
                f"Agent's answer: {agent_answer}"
            ),
        },
    ]
    last_response = ""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        client = clients.get_client(backend, dict(kwargs))
        try:
            last_response = client.completion(messages, timeout=timeout).content
            parsed = parse_deepdive_judgment(last_response)
            return DeepDiveJudgment(
                success=parsed["success"],
                reason=parsed["reason"],
                raw_response=last_response,
                model=model,
                attempts=attempt,
            )
        except Exception as exc:
            last_error = exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return DeepDiveJudgment(
        success=False,
        reason="DeepDive evaluator failed.",
        raw_response=last_response,
        model=model,
        attempts=max_attempts,
        error=(
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "Unknown evaluator failure"
        ),
    )


def parse_deepdive_judgment(response: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if match is None:
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    payload = match.group(1).strip() if match else response.strip()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("DeepDive evaluator response must be a JSON object")
    if not isinstance(parsed.get("success"), bool):
        raise ValueError("DeepDive evaluator success must be boolean")
    if "reason" not in parsed:
        raise ValueError("DeepDive evaluator response is missing reason")
    return {"success": parsed["success"], "reason": str(parsed["reason"])}


__all__ = [
    "DEEPDIVE_JUDGE_SYSTEM_PROMPT",
    "DeepDiveJudgment",
    "judge_deepdive_answer",
    "parse_deepdive_judgment",
]
