"""Alternative Oolong-Synth prompt flows.

The original adaptive prompt remains in ``prompts.py`` unchanged. This module
only adds selectable alternatives and their matching worker task templates.
"""

from __future__ import annotations

from .prompts import CHILD_TASK_TEMPLATE, build_synth_agent_prompt


DEFAULT_PROMPT_FLOW = "adaptive_flat"
PROMPT_FLOWS = ("adaptive_flat", "paged_flat", "hierarchical")


_REPL_RULE = r"""Use only ```repl blocks. Wait for the real REPL observation after
each block. Never print a fake `REPL output:` line. Rows are unlabeled: semantic
labels must come from reading each Instance, never a keyword, regex, word list,
or generated classifier. Python may parse exact Date/User metadata and add
labels that an agent has explicitly assigned after reading. Never put a literal
three-backtick fence inside executable REPL code; it terminates the code block."""


_MEASURE_BLOCK = r"""ROOT ONLY: your first response must be exactly this block, with
no prose, delegation, or submission. A delegated worker must skip this root
block and follow its worker task instead.

```repl
source = context["context_window_text"]
query = context["question"]
lines = source.splitlines()
rows = [
    line for line in lines
    if line.startswith("Date:")
    and " || User: " in line
    and " || Instance: " in line
]
first_row = next(i for i, line in enumerate(lines) if line in rows)
dataset_intro = "\n".join(lines[:first_row])
print({"context_chars": len(source), "complete_rows": len(rows),
       "question": query})
```

Wait for the observation before planning."""


_REPORT_CONTRACT = r"""A worker returns exactly one compact JSON object through
`answer["content"] = json.dumps(report)` and `answer["ready"] = True`. It must
contain `chunk_id`, `rows_seen`, `mode`, `counts`, `totals`, and `uncertain`,
with `rows_seen == expected_rows` and all requested candidate labels present.
The content must be raw JSON, never fenced JSON or REPL code. Workers never call
`submit_answer`. The root parses raw reports with `json.loads`, validates every
disjoint range, merges counts, and calls `submit_answer(...)` exactly once in
the question's requested format."""


_PAGED_CHILD = r"""[OOLONG-SYNTH PAGED WORKER]
You are a read-only worker for the complete rows in
`context["context_window_text"]`. Use `chunk_id`, `expected_rows`,
`dataset_intro`, and `question`. Inspect every row exactly once in consecutive
bounded pages. If the harness reports truncation, shrink the page and reread
only that range. Apply exact metadata filters in Python, but classify semantic
labels by reading. Do not delegate or submit globally. Return only the compact
JSON report described below."""


_HIER_CHILD = r"""[OOLONG-SYNTH HIERARCHICAL WORKER]
You own one disjoint complete-row range and must return one report for it. Read
`chunk_id`, `expected_rows`, `context_window_text`, `dataset_intro`, and
`question`.

If `context.get("can_delegate") is True`, you are a coarse aggregator. First
measure only character and complete-row counts; do not print or classify rows.
Your first executable response must then split at complete-row boundaries and
call `spawn_subagents`. The task of every leaf is exactly
`context["child_task_template"]`; never put data rows in task text. Put only its
rows in the leaf's private `context_window_text`, and also pass a unique
`chunk_id`, exact `expected_rows`, `dataset_intro`, `question`, `dataset`, the
same `child_task_template`, `oolong_role="worker"`, and `can_delegate=False`.
Verify leaf coverage, merge their raw JSON counts, and return one raw JSON report
for your own range. Do not return zero counts merely because delegation failed.

If `can_delegate` is false, you are a leaf. Read every assigned row exactly once
in bounded consecutive pages, classify its meaning, and return the report. A
leaf must not delegate. No worker may call submit_answer."""


def _child(flow: str) -> str:
    if flow == "adaptive_flat":
        return CHILD_TASK_TEMPLATE
    head = _PAGED_CHILD if flow == "paged_flat" else _HIER_CHILD
    return "\n\n".join((head, _REPL_RULE, _REPORT_CONTRACT))


CHILD_TASK_TEMPLATES = {flow: _child(flow) for flow in PROMPT_FLOWS}


_PAGED_PROMPT = "\n\n".join(
    (
        "Oolong-Synthetic PAGED-FLAT flow. The root solves the global question; "
        "a worker only reports its assigned rows and never calls submit_answer.",
        _REPL_RULE,
        _MEASURE_BLOCK,
        "Immediately after measurement, apply exact Date/User filters first; "
        "do not inspect context keys, tools, modules, or prompt text. For semantic "
        "work choose a balanced flat fan-out from the actual relevant size. "
        "Use complete-row boundaries and give each worker a range it can read "
        "through bounded pages. Do not create one child per row or send the "
        "full context to every child. Pass `oolong_role=worker`, a unique "
        "`chunk_id`, exact `expected_rows`, only the assigned rows, "
        "`dataset_intro`, `question`, `dataset`, and the selected child "
        "template. The root merges only after exact coverage validation.",
        _REPORT_CONTRACT,
    )
)


_HIERARCHICAL_PROMPT = "\n\n".join(
    (
        "Oolong-Synthetic HIERARCHICAL flow. Check the role before acting. If "
        "`context[\"oolong_role\"] == \"worker\"`, skip every ROOT instruction "
        "and obey the delegated hierarchical worker task. The root is the only "
        "global solver and the only agent allowed to call submit_answer.",
        _REPL_RULE,
        _MEASURE_BLOCK,
        "Immediately after measurement, apply exact Date/User filters first; "
        "do not inspect sample rows, context keys, tools, modules, or prompt "
        "text. The root must never semantically classify rows or write a Python "
        "classifier. Solve a small filtered set directly only when it fits in "
        "one bounded observation. For a large set, the root's next response must "
        "call `spawn_subagents` with a small number of balanced coarse complete-"
        "row ranges, not hundreds of leaf requests. Each request's task must be "
        "exactly `context[\"child_task_template\"]`; never copy data rows into "
        "task text. Its private context must contain `oolong_role=worker`, a "
        "unique `chunk_id`, exact `expected_rows`, only its rows in "
        "`context_window_text`, `dataset_intro`, `question`, `dataset`, the same "
        "`child_task_template`, and `can_delegate=True`. A coarse worker must "
        "split into disjoint leaf ranges with `can_delegate=False`; it verifies "
        "and merges those reports before returning its parent report. Every "
        "range is complete, non-overlapping, and counted exactly once. If a "
        "report fails, retry only that range; the root must never fall back to "
        "a Python/regex/keyword classifier.",
        _REPORT_CONTRACT,
    )
)


def build_flow_prompt(flow: str) -> str:
    if flow not in PROMPT_FLOWS:
        raise ValueError(
            f"unknown Oolong-Synth prompt flow {flow!r}; choose one of {PROMPT_FLOWS}"
        )
    if flow == "adaptive_flat":
        return build_synth_agent_prompt()
    return _PAGED_PROMPT if flow == "paged_flat" else _HIERARCHICAL_PROMPT


def child_task_template(flow: str) -> str:
    if flow not in PROMPT_FLOWS:
        raise ValueError(
            f"unknown Oolong-Synth prompt flow {flow!r}; choose one of {PROMPT_FLOWS}"
        )
    return CHILD_TASK_TEMPLATES[flow]


__all__ = [
    "CHILD_TASK_TEMPLATES",
    "DEFAULT_PROMPT_FLOW",
    "PROMPT_FLOWS",
    "build_flow_prompt",
    "child_task_template",
]
