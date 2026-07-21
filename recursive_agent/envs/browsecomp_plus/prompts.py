"""Task guidance for BrowseComp-Plus BM25 retrieval."""

from __future__ import annotations

from .dataset import BrowseCompQuery

DEFAULT_BROWSECOMP_AGENT_PROMPT = r"""BrowseComp-Plus recursive retrieval protocol:

Use only the fixed BrowseComp-Plus corpus through `search(query)`. Hits and
reports are leads. Do not repeat another agent's query or retrieval path.

Role is determined by the initial user message: `Task:` means ROOT and
`Delegated task:` means WORKER. ROOT owns the original question, global evidence
state, retries, conflict resolution, and final answer. WORKER owns only its
explicitly delegated objective.

ROOT first turns the question into constraints C1, C2, ... and records their
dependencies. Keep sequentially dependent constraints in one branch. BrowseComp
questions are multi-constraint identity searches:
FIRST-ACTION CONTRACT: ROOT's first model action must be a `repl` block that
creates 2-4 strictly smaller, non-overlapping evidence branches and calls
`spawn_subagents(requests)`. Do not write prose or call `search` first. This is
a hard protocol gate; it applies even when the task looks sequential. ROOT
must collect child reports before any search or answer. Keep dependencies in a branch:
if C4 requires an entity found by C3, assign C3+C4 together. Each child has a
strictly smaller, non-overlapping objective; never pass the original unchanged.

Pass each child the original question, objective, leads, documents, queries,
and exclusions. Children do not
inherit caller messages or REPL variables. Collect reports with:

```repl
reports = spawn_subagents(requests)
print(reports)
```

WORKER inspects supplied documents and leads before searching and investigates
only its assigned objective. It may spawn children while active only when its
local objective has at least two strictly smaller independent parts; it must
never pass the same objective downward unchanged. It returns through `answer`
using this compact schema:

WORKER_REPORT
Status: VERIFIED | PARTIAL | NOT_FOUND | CONFLICT
Objective: <assigned objective>
Candidates: <names or NONE>
Findings: <claims, short evidence, and docids>
Contradictions: <evidence or NONE>
Rejected: <candidates and reasons or NONE>
Attempted queries: <compact list>
Unresolved: <missing facts or NONE>
Recommended next action: <specific follow-up or NONE>

A NOT_FOUND report lists attempted queries, rejected candidates, useful leads,
and the missing fact. Once a worker returns, ROOT owns any retry. First merge
reports into the evidence matrix. The search budget is not a target: do at most
two root follow-up searches, only for remaining gaps, or delegate one narrower
retry with a new lead. Do not repeat the delegation unchanged; avoid worker
query families. If no
new lead appears, mark the path MISSING and stop.

ROOT merges reports into a candidate-by-constraint evidence matrix. Mark each
cell VERIFIED, PARTIAL, CONTRADICTED, or MISSING and attach docids. Do not vote
or average worker confidence. Resolve disagreement by inspecting evidence or by
one targeted adjudication task. Search only remaining gaps, then select the
candidate with the strongest cross-constraint support. Explicitly calculate
date, numeric, ordering, and geographic relations.

Use distinctive queries and refine them with newly discovered names,
phrases, dates, organizations, places, or titles. Execute tools only in `repl`
blocks. Keep results in persistent variables and print compact catalogs:

```repl
hits = search("short distinctive query")
print([{"docid": h["docid"], "score": h.get("score"),
        "chars": len(h["snippet"]), "head": h["snippet"][:300]}
       for h in hits])
```

Use bounded slices; never print whole documents.

Only ROOT returns the benchmark answer through `answer`, with exactly three
newline-separated lines:

Explanation: brief verified synthesis with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Use `\n` inside a quoted Python string and set `answer["ready"] = True`.
The question and search results are the only benchmark information available;
gold answers, labels, qrels, and evaluator data are unavailable."""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Find and verify the answer, then return the required Explanation / Exact Answer /
Confidence format without adding other sections."""

DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = """FINAL FORMAT OVERRIDE. Replace your
entire previous response now. No reasoning, prose, Markdown, code, tools, or
further investigation will be executed. Return exactly three newline-separated
lines; the first character must be E in "Explanation":

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Do not add any other characters or lines."""


def build_browsecomp_task_prompt(
    sample: BrowseCompQuery,
    *,
    template: str = DEFAULT_BROWSECOMP_TASK_TEMPLATE,
) -> str:
    if "{query}" not in template:
        raise ValueError("BrowseComp-Plus task template must contain {query}")
    return template.format(query=sample.query).strip()


__all__ = [
    "DEFAULT_BROWSECOMP_AGENT_PROMPT",
    "DEFAULT_BROWSECOMP_TASK_TEMPLATE",
    "DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT",
    "build_browsecomp_task_prompt",
]
