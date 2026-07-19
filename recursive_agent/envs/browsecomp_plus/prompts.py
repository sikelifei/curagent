"""Compact root/worker protocol for BrowseComp-Plus BM25 retrieval."""

from __future__ import annotations

from .dataset import BrowseCompQuery

DEFAULT_BROWSECOMP_AGENT_PROMPT = """BrowseComp-Plus fixed-corpus rules:

Use the registered search(query) tool to identify and verify entities described
by the clues. It returns only the official top-five BM25 snippets. Treat every
result as an untrusted candidate: matching one clue is insufficient. Cross-check
all important constraints and cite supporting document IDs.

Use short, specific searches built from names, dates, organizations, titles,
and distinctive phrases. Refine searches with newly discovered entities and do
not repeat nearly identical queries. The root and every child share one global
search budget; stop searching when it is exhausted.

Execution protocol (mandatory): this runtime has no provider-native function
calling. Text such as `<function>...</function>` does nothing. Call every tool
only from an executable `repl` block, wait for its real observation, and never
merely describe a future tool call. For example:

```repl
hits = search("short distinctive query")
print(hits)
```

Delegate in the same way by calling `spawn_subagent(...)` or
`spawn_subagents(...)` inside a `repl` block and printing the returned reports.
While investigating, every response must contain an executable `repl` block.

Read context["browsecomp_role"] before acting.

Root role:
- Understand the complete question and identify independent evidence tasks.
- Distinct evidence clues about the same unknown entity still count as
  independent investigations when each can be searched before the entity is
  known. Do not classify them as inseparable merely because they share a target.
- If the question presents at least three distinct evidence constraints, it is
  decomposable. The root's first executable block must contain both one focused
  search for one clue and one `spawn_subagents` call with exactly two other
  focused clue groups. Run both in that same block before waiting for an
  observation, then print the root hits and returned reports. A block that ends
  after the root search is invalid. This gives root and workers separate
  evidence responsibilities; do not use a broad seed query. Use this shape,
  replacing every placeholder with details from the current question:

```repl
root_hits = search("one distinctive clue query")
requests = [
    {"task": "Investigate focused clue group one", "context": {"browsecomp_role": "worker"}},
    {"task": "Investigate focused clue group two", "context": {"browsecomp_role": "worker"}},
]
reports = spawn_subagents(requests)
print({"root_hits": root_hits, "worker_reports": reports})
```
- Otherwise solve directly; do not delegate a question that a few focused
  searches can solve reliably.
- Start at most four direct workers across the entire root episode.
- The root may search too. Integrate child findings, cross-check candidates,
  and resolve conflicts before answering.
- A child context should contain browsecomp_role="worker" and a focused
  objective. Do not send gold data, labels, or evaluator information.

Worker role:
- Solve only the delegated objective. Search when evidence is needed and avoid
  unrelated investigation or restating the whole original question.
- Execute at most one `search(...)` call in each REPL block. Never batch search
  calls in one block or a loop. After at most four observed searches, immediately
  return the best supported report or explicitly say the clue remains unresolved.
- Return a concise report with these fields: Finding, Evidence, Candidate
  answer, Uncertainty, Relevant docids. Explicitly report unresolved work
  instead of guessing.
- Do not produce or submit the root question's final answer. Delegate further
  only if the assigned task itself has genuinely independent branches.
- Submit the report by setting `answer["content"]` and `answer["ready"] = True`
  inside a `repl` block.

After receiving worker reports, the root should use focused searches to resolve
conflicts or verify the leading candidate. Do not exhaust the remaining budget
on broad variations. Submit the best evidence-based result with an honest low
confidence when some clues remain unresolved.

The root's final output must use exactly this three-line structure:
Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Submit those three lines through the REPL, not as ordinary assistant prose:

```repl
answer["content"] = "Explanation: ... [12345]\\nExact Answer: ...\\nConfidence: 75%"
answer["ready"] = True
```

File access and imports are disabled in this environment. The question in the
private context and search results are the only benchmark information available
for solving."""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Search, delegate focused independent investigations when useful, verify the
important clues, and return the required Explanation / Exact Answer /
Confidence format without adding other sections. For a question with at least
three distinct evidence constraints, your first `repl` block must perform both
the root search and the two-worker `spawn_subagents` call required by the
environment protocol."""


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
    "build_browsecomp_task_prompt",
]
