"""Oolong-Synthetic environment plugin."""

from .dataset import (
    OolongSynthDataset,
    OolongSynthSample,
    select_protocol_indices,
)
from .environment import OolongSynthEnvironment
from .prompts import (
    CHUNK_CHAR_LIMIT,
    DEFAULT_SYNTH_AGENT_PROMPT,
    DEFAULT_SYNTH_TASK_TEMPLATE,
    build_synth_agent_prompt,
    build_synth_task_prompt,
)
from .scoring import (
    SynthEvaluation,
    evaluate_synth_response,
    parse_gold_answer,
    parse_synth_response,
)

__all__ = [
    "CHUNK_CHAR_LIMIT",
    "DEFAULT_SYNTH_AGENT_PROMPT",
    "DEFAULT_SYNTH_TASK_TEMPLATE",
    "OolongSynthDataset",
    "OolongSynthEnvironment",
    "OolongSynthSample",
    "SynthEvaluation",
    "build_synth_agent_prompt",
    "build_synth_task_prompt",
    "evaluate_synth_response",
    "parse_gold_answer",
    "parse_synth_response",
    "select_protocol_indices",
]
