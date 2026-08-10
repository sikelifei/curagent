"""DeepDive benchmark integration backed by the Platoon harness."""

from .environment import DeepDiveEnvironment
from .harness import (
    DEFAULT_PLATOON_ROOT,
    DEEPDIVE_SPLITS,
    DeepDiveHarnessProtocol,
    DeepDiveSample,
    PlatoonDeepDiveHarness,
    make_task_ids,
)
from .prompts import (
    DEFAULT_DEEPDIVE_AGENT_PROMPT,
    DEFAULT_DEEPDIVE_CHILD_PROMPT,
    DEFAULT_DEEPDIVE_COMPLETION_PROMPT,
    DEFAULT_DEEPDIVE_ROOT_PROMPT,
    DEFAULT_DEEPDIVE_TASK_TEMPLATE,
    DEFAULT_DEEPDIVE_TOOLS_PROMPT,
    build_deepdive_task_prompt,
)
from .scoring import DeepDiveJudgment, judge_deepdive_answer

__all__ = [
    "DEFAULT_DEEPDIVE_AGENT_PROMPT",
    "DEFAULT_DEEPDIVE_CHILD_PROMPT",
    "DEFAULT_DEEPDIVE_COMPLETION_PROMPT",
    "DEFAULT_DEEPDIVE_ROOT_PROMPT",
    "DEFAULT_DEEPDIVE_TASK_TEMPLATE",
    "DEFAULT_DEEPDIVE_TOOLS_PROMPT",
    "DEFAULT_PLATOON_ROOT",
    "DEEPDIVE_SPLITS",
    "DeepDiveEnvironment",
    "DeepDiveHarnessProtocol",
    "DeepDiveJudgment",
    "DeepDiveSample",
    "PlatoonDeepDiveHarness",
    "build_deepdive_task_prompt",
    "judge_deepdive_answer",
    "make_task_ids",
]
