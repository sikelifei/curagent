"""Prompt text for the BrowseComp-Plus recursive retrieval environment."""

from __future__ import annotations

from ...prompts import (
    build_browsecomp_system_prompt,
    build_browsecomp_worker_system_prompt,
)
from .dataset import BrowseCompQuery


DEFAULT_BROWSECOMP_SYSTEM_PROMPT = build_browsecomp_system_prompt()
DEFAULT_BROWSECOMP_WORKER_SYSTEM_PROMPT = build_browsecomp_worker_system_prompt()

DEFAULT_BROWSECOMP_AGENT_PROMPT = r"""BrowseComp root addendum.
Pass each worker the original question, one focused objective, useful leads, and
exclusions. After reports arrive, compare their claims and docids, delegate at
most one narrower retry for an unresolved branch, then synthesize the answer.
The benchmark answer uses the required Explanation / Exact Answer / Confidence
format."""

DEFAULT_BROWSECOMP_WORKER_PROMPT = r"""BrowseComp worker addendum.
Keep search results in REPL variables; use Python to filter, deduplicate, and
compare candidates, printing only bounded snippets. Return only:
WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT
Objective: <assigned search objective>
Candidates: <names or NONE>
Evidence: <claims with docids, or NONE>
Queries tried: <compact list>
Unresolved: <missing fact or NONE>
Recommended next action: <targeted retry or NONE>"""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this BrowseComp-Plus question using
the fixed corpus and the available search tool.

Question:
{query}

ROOT ACTION CONTRACT: Your next response must be exactly one `repl` code block,
not an explanation and not a final answer. In that block, create one focused
worker request for a coherent search branch, or 2-4 requests for independent
constraints, then call `spawn_subagent(...)` or `spawn_subagents(...)` and print
the returned reports. Do not call `search` in the root. Return the required
three-line format only after worker reports are available."""

DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = """FINAL FORMAT OVERRIDE. Replace your
entire previous response now. No reasoning, prose, Markdown, code, tools, or
further investigation will be executed. Return exactly three newline-separated
lines; the first character must be E in "Explanation":

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Do not add any other characters or lines."""

DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT = """No working steps remain.
You are a BrowseComp-Plus worker. Return only a compact report supported by
evidence already observed, not the root three-line answer:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT
Objective: <assigned search objective>
Candidates: <names or NONE>
Evidence: <claims with docids, or NONE>
Queries tried: <compact list>
Unresolved: <missing fact or NONE>
Recommended next action: <targeted retry or NONE>"""


def build_browsecomp_task_prompt(
    sample: BrowseCompQuery,
    *,
    template: str = DEFAULT_BROWSECOMP_TASK_TEMPLATE,
) -> str:
    if "{query}" not in template:
        raise ValueError("BrowseComp-Plus task template must contain {query}")
    return template.format(query=sample.query).strip()


__all__ = [
    "DEFAULT_BROWSECOMP_SYSTEM_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_SYSTEM_PROMPT",
    "DEFAULT_BROWSECOMP_AGENT_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_PROMPT",
    "DEFAULT_BROWSECOMP_TASK_TEMPLATE",
    "DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT",
    "build_browsecomp_task_prompt",
]
