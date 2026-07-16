"""Submission tool for one Oolong-Synthetic sample."""

from __future__ import annotations

from typing import Any, Protocol


class SynthToolTarget(Protocol):
    def submit_answer(self, answer: str) -> dict[str, Any]: ...


def build_synth_tools(target: SynthToolTarget) -> dict[str, Any]:
    return {
        "submit_answer": {
            "tool": target.submit_answer,
            "description": (
                "Root-only: submit one final Oolong-Synthetic answer using the exact "
                "prefix requested by the question (Answer:, Label:, User:, or Date:). "
                "This ends and scores the sample. Delegated agents must never call it."
            ),
        }
    }


__all__ = ["build_synth_tools"]
