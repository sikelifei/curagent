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
ROOT MUST create 2-4 strictly smaller, non-overlapping evidence branches and
call `spawn_subagents(requests)` in its FIRST `repl` block. This is a hard
protocol gate, not a suggestion. ROOT MUST NOT search, investigate a branch,
or answer before collecting child reports. Keep dependencies inside a branch:
if C4 requires an entity found by C3, assign C3+C4 together. Each child has a
strictly smaller, non-overlapping objective; never pass the original unchanged.

Pass each child the original question, its objective, relevant constraints,
leads, selected documents, attempted queries, and exclusions. Children do not
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
reports into the evidence matrix; search only one remaining gap or delegate one
narrower retry with a new lead. Do not repeat the delegation unchanged or reuse
a worker query family. If no new lead appears, mark the path MISSING and stop.

ROOT merges reports into a candidate-by-constraint evidence matrix. Mark each
cell VERIFIED, PARTIAL, CONTRADICTED, or MISSING and attach docids. Do not vote
or average worker confidence. Resolve disagreement by inspecting evidence or by
one targeted adjudication task. Search only remaining gaps, then select the
candidate with the strongest cross-constraint support. Explicitly calculate
date, numeric, ordering, and geographic relations.

Use short, distinctive queries and refine them with newly discovered names,
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

DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = """No working steps remain. This response
is parsed directly and code will not execute. Choose the best-supported answer
now. The first character must be the E in "Explanation". Return exactly these
three lines, with no preamble, Markdown fence, tools, or further investigation:

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%"""


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
