"""Custom tools exposed by one Oolong-real episode."""

from __future__ import annotations

from typing import Any, Protocol


class OolongToolTarget(Protocol):
    def observe(self) -> dict[str, Any]: ...

    def submit_answer(self, answer: str) -> dict[str, Any]: ...

    def report(self) -> dict[str, Any]: ...


def build_oolong_tools(target: OolongToolTarget) -> dict[str, Any]:
    return {
        "observe": {
            "tool": target.observe,
            "description": (
                "Return Oolong sample metadata, question, context length, and whether "
                "an answer has been submitted. The full context is in private REPL "
                "variable context['context_window_text'], not in this metadata output."
            ),
        },
        "submit_answer": {
            "tool": target.submit_answer,
            "description": (
                "Submit the final Oolong answer exactly as written, preferably in "
                "\\boxed{...}; this records the official-style score and ends the episode."
            ),
        },
        "episode_report": {
            "tool": target.report,
            "description": "Return the current parsed answer, score, and Oolong episode metadata.",
        },
    }
