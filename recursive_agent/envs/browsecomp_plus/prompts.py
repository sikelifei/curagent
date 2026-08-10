"""Prompt text for the BrowseComp-Plus recursive retrieval environment."""

from __future__ import annotations

from .dataset import BrowseCompQuery


DEFAULT_BROWSECOMP_AGENT_PROMPT = r"""### BrowseComp-Plus evidence search

You are a deep research agent. You need to answer the given question by interact
ing with a search engine, using the search tool provided. Please perform reasoning and
use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Available in the REPL:

```repl
results = search(query)

report = spawn_subagent(task, context=None)

reports = spawn_subagents([
    {"task": task1, "context": context1},
    {"task": task2, "context": context2},
])
```

"""


DEFAULT_BROWSECOMP_ROOT_COMPLETION_PROMPT = r"""### Completion

Finish only after combining the available evidence and checking the selected
answer against the original question.

Set `answer["content"]` to exactly three lines:

Explanation: <concise evidence-based explanation with useful docids>
Exact Answer: <answer>
Confidence: <0-100%>

Then set `answer["ready"] = True`.
"""


DEFAULT_BROWSECOMP_WORKER_COMPLETION_PROMPT = r"""### Completion

Return the evidence needed by the parent to continue or combine the
investigation.

Set `answer["content"]` to exactly:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT | ERROR
Objective: <assigned objective>
Findings: <relevant candidates or facts, or NONE>
Evidence: <supported claims with docids, or NONE>
Contradictions: <contradictory claims with docids, or NONE>
Unresolved: <remaining constraints or questions, or NONE>
Useful leads: <names, dates, phrases, or follow-up lexical directions, or NONE>
Queries tried: <important queries only>

Use VERIFIED only when the assigned objective is supported by evidence.
Use PARTIAL when useful evidence was found but the objective remains unresolved.
Use NOT_FOUND only after successful searches found no supporting evidence.
Use CONFLICT when retrieved evidence materially disagrees.
Use ERROR only for tool or execution failure.

Never include unobserved claims or docids.

Then set `answer["ready"] = True`.
"""


DEFAULT_BROWSECOMP_TASK_TEMPLATE = r"""Answer this evidence-seeking question
using only the fixed BrowseComp-Plus corpus.

Question:
{query}

Resolve the constraints, combine the evidence, and verify the selected answer.

Return exactly:
Explanation: <concise evidence-based explanation with useful docids>
Exact Answer: <answer>
Confidence: <0-100%>
"""


DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = r"""FINAL FORMAT OVERRIDE.

Return exactly three newline-separated lines and nothing else:

Explanation: <concise evidence-based explanation with useful docids>
Exact Answer: <answer>
Confidence: <0-100%>

Never invent evidence or citations.

If no answer is sufficiently supported, return exactly:
Explanation: No supported answer was retrieved
Exact Answer: Unable to determine
Confidence: 0%
"""


DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT = r"""Return only:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT | ERROR
Objective: <assigned objective>
Findings: <relevant candidates or facts, or NONE>
Evidence: <supported claims with docids, or NONE>
Contradictions: <contradictory claims with docids, or NONE>
Unresolved: <remaining questions, or NONE>
Useful leads: <names, dates, phrases, or follow-up lexical directions, or NONE>
Queries tried: <important queries only>

NOT_FOUND requires successful searches.
Tool or execution failure is ERROR.
Never invent claims or docids.
"""


def build_browsecomp_task_prompt(
    sample: BrowseCompQuery,
    *,
    template: str = DEFAULT_BROWSECOMP_TASK_TEMPLATE,
) -> str:
    if "{query}" not in template:
        raise ValueError(
            "BrowseComp-Plus task template must contain {query}"
        )
    return template.format(query=sample.query).strip()


__all__ = [
    "DEFAULT_BROWSECOMP_AGENT_PROMPT",
    "DEFAULT_BROWSECOMP_ROOT_COMPLETION_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_COMPLETION_PROMPT",
    "DEFAULT_BROWSECOMP_TASK_TEMPLATE",
    "DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT",
    "build_browsecomp_task_prompt",
]