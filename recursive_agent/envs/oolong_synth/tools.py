"""Submission tool for one Oolong-Synthetic sample."""

from __future__ import annotations

from typing import Any, Protocol

from ...tools import CapabilityCollection


class SynthToolTarget(Protocol):
    def submit_answer(self, answer: str) -> dict[str, Any]: ...


def build_synth_tools(target: SynthToolTarget) -> dict[str, Any]:
    return {
        "submit_answer": {
            "tool": target.submit_answer,
            "description": (
                "Submit one final Oolong-Synthetic answer using the exact "
                "prefix requested by the question (Answer:, Label:, User:, or Date:). "
                "This ends and scores the sample."
            ),
        }
    }


def build_synth_capabilities(target: Any | None = None) -> CapabilityCollection:
    """Return the immutable Oolong CodeAct environment capability set.

    Oolong work is performed with each node's private Python context. Recursive
    delegation and root/child termination are supplied by the generic harness;
    the legacy ``submit_answer`` tool is intentionally not exposed here.
    """
    del target
    return CapabilityCollection()


__all__ = ["SynthToolTarget", "build_synth_capabilities", "build_synth_tools"]
