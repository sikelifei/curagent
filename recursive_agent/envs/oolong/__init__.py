"""Oolong-real environment plugin."""

from .dataset import OolongDataset, OolongSample
from .environment import OolongEnvironment
from .prompts import (
    DEFAULT_OOLONG_AGENT_PROMPT,
    DEFAULT_OOLONG_CHILD_PROMPT,
    DEFAULT_OOLONG_ROOT_PROMPT,
    DEFAULT_OOLONG_TASK_TEMPLATE,
    DEFAULT_OOLONG_TOOLS_PROMPT,
    build_oolong_task_prompt,
)
from .scoring import parse_answer, parse_response, score_answer
from .tools import build_oolong_tools

__all__ = [
    "DEFAULT_OOLONG_AGENT_PROMPT",
    "DEFAULT_OOLONG_CHILD_PROMPT",
    "DEFAULT_OOLONG_ROOT_PROMPT",
    "DEFAULT_OOLONG_TASK_TEMPLATE",
    "DEFAULT_OOLONG_TOOLS_PROMPT",
    "OolongDataset",
    "OolongEnvironment",
    "OolongSample",
    "build_oolong_task_prompt",
    "build_oolong_tools",
    "parse_answer",
    "parse_response",
    "score_answer",
]
