"""Prompt text for the BrowseComp-Plus recursive retrieval environment."""

from __future__ import annotations

from .dataset import BrowseCompQuery
from ...prompts import build_browsecomp_system_prompt


DEFAULT_BROWSECOMP_SYSTEM_PROMPT = build_browsecomp_system_prompt()

DEFAULT_BROWSECOMP_AGENT_PROMPT = r"""BrowseComp-Plus is an evidence-search task.
Use only the fixed corpus through `search(query)`; do not use outside knowledge
or hidden benchmark data.

The ROOT owns the original question, task decomposition, and final synthesis.
If corpus evidence is needed, ROOT must first delegate the search: exactly one
worker for a single coherent branch, or one worker per independent branch.
Workers preserve useful docids and claims and return concise reports. ROOT then
accepts a verified result, retries one worker with a narrower new lead, or
delegates the next unresolved constraint. Do not repeat the same broad query or
delegation, and never give multiple workers the unchanged full question. ROOT
must not search after receiving reports; one targeted retry worker per unresolved
branch is the maximum.

ROOT should not call `search` itself unless supplied evidence is already enough
and no search is needed. This keeps ROOT focused on routing, comparison, and
synthesis rather than consuming the shared search budget.

A worker may recurse only when its own search reveals multiple independent
verification tasks. It should otherwise finish its local objective directly.
Pass the original question, local objective, known leads, and exclusions in each
delegation because children do not inherit the parent's history or variables.
Use `spawn_subagent(task, context)` positionally. In every worker response,
write only a standard ```repl``` Python block; do not wrap it in XML or another
language fence. A worker should normally use no more than 4 distinct queries for
one objective. Stop once a decisive docid or candidate is found; otherwise stop
after two consecutive queries add no new useful lead and report `PARTIAL` or
`NOT_FOUND`. Never repeat a query or use `search("docid:...")` to read a full
document.

Treat search output as document records, not text to dump immediately. Keep
results in persistent REPL variables and use Python to filter, deduplicate,
sort, and compare candidates. Print only bounded snippets. Inspect fuller
content only for a small set of promising docids when it is needed to verify an
exact claim; do not read or print every returned document by default. For
example:

```repl
hits = search("distinctive terms")
shortlist = [
    h for h in hits
    if any(word in h["snippet"].lower() for word in ["date", "school"])
]
print([{"docid": h["docid"], "head": h["snippet"][:300]} for h in shortlist])
```

Workers return this compact internal format through `answer`:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT
Objective: <assigned search objective>
Candidates: <names or NONE>
Evidence: <claims with docids, or NONE>
Queries tried: <compact list>
Unresolved: <missing fact or NONE>
Recommended next action: <targeted retry or NONE>

Only ROOT returns the benchmark answer, after checking decisive evidence, with
exactly these three lines:

Explanation: brief verified explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%"""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this BrowseComp-Plus question using
the fixed corpus and the available search tool.

Question:
{query}

Decide whether the evidence work is direct or has independent branches. Search,
verify, and return the required three-line format."""

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
    "DEFAULT_BROWSECOMP_AGENT_PROMPT",
    "DEFAULT_BROWSECOMP_TASK_TEMPLATE",
    "DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT",
    "build_browsecomp_task_prompt",
]
