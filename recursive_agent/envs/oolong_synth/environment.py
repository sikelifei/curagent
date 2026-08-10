"""Curagent environment for Oolong-Synthetic evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...types import EnvironmentStatus
from ..base import AgentEnvironment
from ..registry import register_environment
from .dataset import (
    DEFAULT_DATASET_NAME,
    DEFAULT_OOLONG_ROOT,
    DEFAULT_SPLIT,
    OolongSynthDataset,
    OolongSynthSample,
)
from .prompts import (
    DEFAULT_SYNTH_AGENT_PROMPT,
    DEFAULT_SYNTH_CHILD_PROMPT,
    DEFAULT_SYNTH_FORCED_FINAL_PROMPT,
    DEFAULT_SYNTH_SUBAGENT_FORCED_FINAL_PROMPT,
    DEFAULT_OOLONG_SYNTH_ROOT_COMPLETION_PROMPT,
    DEFAULT_OOLONG_SYNTH_SUBAGENT_COMPLETION_PROMPT,
    DEFAULT_SYNTH_TASK_TEMPLATE,
    DEFAULT_SYNTH_ROOT_PROMPT,
    build_synth_agent_prompt,
    build_synth_task_prompt,
)
from .scoring import evaluate_synth_response
from .tools import build_synth_tools


@register_environment("oolong_synth")
class OolongSynthEnvironment(AgentEnvironment):
    name = "oolong_synth"

    def __init__(
        self,
        *,
        oolong_root: str | Path | None = DEFAULT_OOLONG_ROOT,
        split: str = DEFAULT_SPLIT,
        instance_id: int = 0,
        dataset_name: str = DEFAULT_DATASET_NAME,
        data_path: str | Path | None = None,
        samples: Sequence[Mapping[str, Any]] | None = None,
        loader: Any | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
        prompt_template: str = DEFAULT_SYNTH_TASK_TEMPLATE,
        agent_prompt: str | None = None,
    ) -> None:
        self.dataset = OolongSynthDataset(
            oolong_root=oolong_root,
            split=split,
            dataset_name=dataset_name,
            data_path=data_path,
            samples=samples,
            loader=loader,
            load_kwargs=load_kwargs,
        )
        self.sample: OolongSynthSample = self.dataset[instance_id]
        self._task = build_synth_task_prompt(self.sample, template=prompt_template)
        self._agent_prompt = (
            str(agent_prompt).strip()
            if agent_prompt is not None
            else DEFAULT_SYNTH_AGENT_PROMPT
        )
        self._submitted_answer: str | None = None
        self._closed = False
        self._tools = build_synth_tools(self)
        dataset_intro = "\n".join(
            line
            for line in self.sample.context_window_text.splitlines()
            if not line.startswith("Date:")
        ).strip()
        self._context = {
            "oolong_role": "root",
            "environment": self.name,
            "dataset": self.sample.dataset,
            "dataset_name": self.dataset.dataset_name,
            "split": self.dataset.split,
            "instance_id": self.sample.source_index,
            "id": self.sample.sample_id,
            "context_window_id": self.sample.context_window_id,
            "question": self.sample.question,
            "dataset_intro": dataset_intro,
            "answer_type": self.sample.answer_type,
            "task_group": self.sample.task_group,
            "task": self.sample.task,
            "input_subset": self.sample.input_subset,
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
    def root_prompt(self) -> str:
        return DEFAULT_SYNTH_ROOT_PROMPT

    @property
    def child_prompt(self) -> str:
        return DEFAULT_SYNTH_CHILD_PROMPT

    @property
    def delegated_prompt_addendum(self) -> str | None:
        return build_synth_agent_prompt(delegated=True)

    @property
    def delegated_disabled_tools(self) -> frozenset[str]:
        return frozenset({"submit_answer"})

    @property
    def forced_final_prompt(self) -> str:
        return DEFAULT_SYNTH_FORCED_FINAL_PROMPT

    @property
    def completion_prompt(self) -> str:
        return DEFAULT_OOLONG_SYNTH_ROOT_COMPLETION_PROMPT

    @property
    def delegated_completion_prompt(self) -> str:
        return DEFAULT_OOLONG_SYNTH_SUBAGENT_COMPLETION_PROMPT

    @property
    def delegated_forced_final_prompt(self) -> str:
        return DEFAULT_SYNTH_SUBAGENT_FORCED_FINAL_PROMPT

    @property
    def max_repl_blocks_per_step(self) -> int | None:
        return None

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def tools(self) -> dict[str, Any]:
        return dict(self._tools)

    def submit_answer(self, answer: str) -> dict[str, Any]:
        if self._submitted_answer is not None:
            raise RuntimeError("Oolong-Synth answer has already been submitted")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Oolong-Synth answer must be a non-empty string")
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
        evaluation = (
            evaluate_synth_response(
                self.sample.answer,
                submitted,
                self.sample.answer_type,
            )
            if submitted is not None
            else None
        )
        return {
            "environment": self.name,
            "dataset": self.sample.dataset,
            "dataset_name": self.dataset.dataset_name,
            "split": self.dataset.split,
            "instance_id": self.sample.source_index,
            "id": self.sample.sample_id,
            "context_window_id": self.sample.context_window_id,
            "context_len": self.sample.context_len,
            "context_chars": len(self.sample.context_window_text),
            "question": self.sample.question,
            "answer_type": self.sample.answer_type,
            "task_group": self.sample.task_group,
            "task": self.sample.task,
            "input_subset": self.sample.input_subset,
            "source": self.dataset.metadata()["source"],
            "submitted": submitted is not None,
            "submitted_answer": submitted,
            "attempted_parse": evaluation.candidate if evaluation else None,
            "parse_confidence": evaluation.parse_confidence if evaluation else "missing",
            "score": evaluation.score if evaluation else 0.0,
        }

    def close(self) -> None:
        self._closed = True


__all__ = ["OolongSynthEnvironment"]
