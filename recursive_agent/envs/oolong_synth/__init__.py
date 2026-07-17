"""Oolong-Synthetic environment plugin."""

from .dataset import (
    OolongSynthDataset,
    OolongSynthSample,
    select_protocol_indices,
)
from .environment import OolongSynthEnvironment
from .prompts import (
    CHILD_TASK_TEMPLATE,
    DEFAULT_SYNTH_AGENT_PROMPT,
    DEFAULT_SYNTH_TASK_TEMPLATE,
    build_synth_agent_prompt,
    build_synth_task_prompt,
)
from .flow_prompts import (
    CHILD_TASK_TEMPLATES,
    DEFAULT_PROMPT_FLOW,
    PROMPT_FLOWS,
    build_flow_prompt,
    child_task_template,
)
from .scoring import (
    SynthEvaluation,
    evaluate_synth_response,
    parse_gold_answer,
    parse_synth_response,
)

__all__ = [
    "CHILD_TASK_TEMPLATE",
    "CHILD_TASK_TEMPLATES",
    "DEFAULT_PROMPT_FLOW",
    "DEFAULT_SYNTH_AGENT_PROMPT",
    "DEFAULT_SYNTH_TASK_TEMPLATE",
    "OolongSynthDataset",
    "OolongSynthEnvironment",
    "OolongSynthSample",
    "PROMPT_FLOWS",
    "SynthEvaluation",
    "build_synth_agent_prompt",
    "build_synth_task_prompt",
    "build_flow_prompt",
    "child_task_template",
    "evaluate_synth_response",
    "parse_gold_answer",
    "parse_synth_response",
    "select_protocol_indices",
]
