"""Curagent environment adapter for Oolong-real DnD evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...types import EnvironmentStatus
from ..base import AgentEnvironment
from ..registry import register_environment
from .dataset import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_DATASET_NAME,
    DEFAULT_OOLONG_ROOT,
    OolongDataset,
    OolongSample,
)
from .prompts import (
    DEFAULT_OOLONG_AGENT_PROMPT,
    DEFAULT_OOLONG_TASK_TEMPLATE,
    build_oolong_task_prompt,
)
from .scoring import parse_response, score_answer
from .tools import build_oolong_tools


@register_environment("oolong")
class OolongEnvironment(AgentEnvironment):
    """One Oolong-real example with a read-only context and scored submission."""

    name = "oolong"

    def __init__(
        self,
        *,
        oolong_root: str | Path | None = DEFAULT_OOLONG_ROOT,
        split: str = "test",
        instance_id: int = 0,
        dataset_name: str = DEFAULT_DATASET_NAME,
        config_name: str = DEFAULT_CONFIG_NAME,
        data_path: str | Path | None = None,
        samples: Sequence[Mapping[str, Any]] | None = None,
        loader: Any | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
        prompt_template: str = DEFAULT_OOLONG_TASK_TEMPLATE,
        agent_prompt: str = DEFAULT_OOLONG_AGENT_PROMPT,
    ) -> None:
        self.dataset = OolongDataset(
            oolong_root=oolong_root,
            split=split,
            dataset_name=dataset_name,
            config_name=config_name,
            data_path=data_path,
            samples=samples,
            loader=loader,
            load_kwargs=load_kwargs,
        )
        self.sample: OolongSample = self.dataset[instance_id]
        self._task = build_oolong_task_prompt(self.sample, template=prompt_template)
        self._agent_prompt = str(agent_prompt).strip()
        self._submitted_answer: str | None = None
        self._closed = False
        self._tools = build_oolong_tools(self)
        self._context = {
            "environment": self.name,
            "dataset": self.sample.dataset,
            "dataset_name": self.dataset.dataset_name,
            "config_name": self.dataset.config_name,
            "split": self.dataset.split,
            "instance_id": self.sample.index,
            "id": self.sample.sample_id,
            "context_window_id": self.sample.context_window_id,
            "question": self.sample.question,
            "answer_type": self.sample.answer_type,
            "question_type": self.sample.question_type,
            "episodes": list(self.sample.episodes),
            "campaign": self.sample.campaign,
            "context_len": self.sample.context_len,
            "context_chars": len(self.sample.context_window_text),
            "context_type": "str",
            "context_total_length": len(self.sample.context_window_text),
            "context_window_text": self.sample.context_window_text,
            "source": self.dataset.metadata()["source"],
        }

    @property
    def task(self) -> str:
        return self._task

    @property
    def agent_prompt(self) -> str:
        return self._agent_prompt

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def tools(self) -> dict[str, Any]:
        return dict(self._tools)

    def observe(self) -> dict[str, Any]:
        # Keep the full context available through the private REPL variable;
        # printing observe() must remain a small metadata operation.
        metadata = {
            key: value for key, value in self.context.items() if key != "context_window_text"
        }
        return {**metadata, "submitted": self._submitted_answer is not None}

    def submit_answer(self, answer: str) -> dict[str, Any]:
        if self._submitted_answer is not None:
            raise RuntimeError("Oolong answer has already been submitted")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Oolong submitted answer must be a non-empty string")
        self._submitted_answer = answer.strip()
        return self.report()

    def status(self) -> EnvironmentStatus:
        if self._submitted_answer is None:
            return EnvironmentStatus(done=False)
        return EnvironmentStatus(
            done=True,
            final_answer=self._submitted_answer,
            reason="answer_submitted",
        )

    def report(self) -> dict[str, Any]:
        submitted = self._submitted_answer
        parsed, confidence = (
            parse_response(submitted) if submitted is not None else (None, "missing")
        )
        return {
            "environment": self.name,
            "dataset": self.sample.dataset,
            "dataset_name": self.dataset.dataset_name,
            "config_name": self.dataset.config_name,
            "split": self.dataset.split,
            "instance_id": self.sample.index,
            "id": self.sample.sample_id,
            "context_window_id": self.sample.context_window_id,
            "context_len": self.sample.context_len,
            "context_chars": len(self.sample.context_window_text),
            "source": self.dataset.metadata()["source"],
            "question": self.sample.question,
            "answer_type": self.sample.answer_type,
            "question_type": self.sample.question_type,
            "episodes": list(self.sample.episodes),
            "campaign": self.sample.campaign,
            "submitted": submitted is not None,
            "submitted_answer": submitted,
            "attempted_parse": parsed,
            "parse_confidence": confidence,
            "score": score_answer(self.sample.answer, parsed) if parsed is not None else 0.0,
            "gold_answer": self.sample.answer,
        }

    def close(self) -> None:
        self._closed = True


__all__ = ["OolongEnvironment"]
