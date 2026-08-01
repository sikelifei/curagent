"""Prompt text for the BrowseComp-Plus recursive retrieval environment."""

from __future__ import annotations

from ...prompts import BROWSECOMP_TASK_ROUTING_PROMPT, SYSTEM_PROMPT
from .dataset import BrowseCompQuery


# Root and delegated agents deliberately share the same base system prompt.
# BrowseComp-specific behavior is supplied as the environment addendum below.
DEFAULT_BROWSECOMP_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_BROWSECOMP_WORKER_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_BROWSECOMP_AGENT_PROMPT = r"""## BrowseComp fixed-corpus policy

These rules refine generic routing for this environment. Multiple clues about
one unknown entity usually form one linked evidence chain; they are not
independent merely because they are listed separately.

Until a forced-final instruction, every response must contain exactly one
executable block and no other text:

```repl
# plain Python; one action only
```

Never use <repl>, nested fences, or multiple blocks. Each turn may perform only
one action: one search call, one spawn_subagent/spawn_subagents call, or setting
answer. Never call search inside a loop or batch several searches. Observe the
result before choosing the next action.

Inspect the evidence state first. Before the first search, count the unresolved
clues. For one short linked chain, local search is allowed. For two or more
distinct discovery or verification routes, delegate those routes first so the
root can compare reports; the root should not duplicate a child's search. Pass
a narrow objective, observed leads, previous queries, docids, and exclusions;
never pass the whole question unchanged.

A child may recurse when its assigned route exposes two genuinely different
subchecks, and may repair its own partial route with a narrower new query before
returning. It must not repeat an ancestor's query or delegate the same unresolved
question. At every node, compare reports and synthesize the answer; reports are
evidence to verify, not final truth.

Use at most three distinct queries per node. Never repeat a query or use a docid
as a query. Stop after decisive evidence or two successful searches with no new
candidate or useful phrase.

Use only snippets and docids actually observed. A failed call is ERROR, never
NOT_FOUND. Workers return compact reports. The root verifies reports and sets
exactly three lines: Explanation, Exact Answer, Confidence."""


DEFAULT_BROWSECOMP_WORKER_PROMPT = r"""Return the assigned objective through
answer["content"] using exactly:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT | ERROR
Objective: <assigned objective>
Candidates: <names or NONE>
Evidence: <supported claims with docids, or NONE>
Queries tried: <compact list>
Unresolved: <missing facts or NONE>

Use NOT_FOUND only after successful searches. Use ERROR for tool or execution
failure. Never include unobserved claims or docids."""


DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using only the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Find a candidate from the strongest linked clues, verify the remaining criteria,
and return exactly:
Explanation: <brief explanation with citations>
Exact Answer: <shortest unambiguous answer>
Confidence: <0-100%>"""


DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = """FINAL FORMAT OVERRIDE. Return exactly
three newline-separated lines and nothing else:

Explanation: <brief explanation using only observed citations>
Exact Answer: <shortest supported answer>
Confidence: <0-100%>

Never invent evidence or citations. If no answer is supported, return:
Explanation: No supported answer was retrieved
Exact Answer: Unable to determine
Confidence: 0%"""


DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT = """BrowseComp-Plus worker forced final. Return only:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT | ERROR
Objective: <assigned objective>
Candidates: <names or NONE>
Evidence: <observed claims with docids, or NONE>
Queries tried: <compact list>
Unresolved: <missing facts or NONE>

NOT_FOUND requires a successful search. Tool or execution failure is ERROR."""


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
