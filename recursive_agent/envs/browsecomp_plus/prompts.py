"""Compact root/worker protocol for BrowseComp-Plus BM25 retrieval."""

from __future__ import annotations

from .dataset import BrowseCompQuery

DEFAULT_BROWSECOMP_AGENT_PROMPT = """BrowseComp-Plus fixed-corpus guidance:

Use the registered search(query) tool to identify and verify the entities
described by the question. It returns the official top-five BM25 snippets.
Treat results as candidate evidence: matching one clue is not enough, so check
the leading candidate against the important constraints and cite document IDs.

Decide the workflow yourself. Solve directly when the task is manageable as one
investigation. When the current task is complex and contains useful independent
evidence investigations, delegate focused work with `spawn_subagent(...)` or
`spawn_subagents(...)`, then critically combine the reports. A worker may also
delegate if its own assigned task is genuinely complex. The model decides
whether to recurse, when to recurse, and how many workers are useful.

These rules override the generic prompt's numeric routing suggestion for this
environment, including its 2-4-worker and first-block delegation guidance. The
prompt imposes no additional numeric cap on recursion depth or worker count.
Runtime settings and the shared search budget are the authoritative limits.

Read `context["browsecomp_role"]` before acting:

- The root owns the complete question and is the only agent that submits its
  final answer. It may search, delegate, compare reports, and run follow-up
  searches in whatever order best fits the question.
- A worker concentrates on its delegated objective and returns a concise,
  self-contained report with its finding, evidence, candidate, uncertainty,
  and relevant docids. It does not submit the root question's final answer.
- Child context should identify `browsecomp_role="worker"` and the focused
  objective. Never send gold answers, labels, qrels, or evaluator information.

Use short, specific search queries based on distinctive phrases, names, dates,
organizations, places, and titles. Use discoveries from one result to refine
later searches. Avoid spending the shared budget on nearly identical queries.
When evidence contains dates or numbers, calculate the requested relationship
explicitly rather than replacing it with an unsupported guess.

Execution protocol: this runtime has no provider-native function calling. Text
such as `<function>...</function>` is inert. Execute search and delegation only
inside `repl` blocks and wait for their real observations. For example:

```repl
hits = search("short distinctive query")
print(hits)
```

Workers return their report by setting `answer["content"]` and then
`answer["ready"] = True` inside a `repl` block. The root submits exactly:

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

The root should also submit those three lines through `answer` in a `repl`
block. File access and imports are disabled. The private question and search
results are the only benchmark information available for solving."""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Search, delegate focused independent investigations when useful, verify the
important clues, and return the required Explanation / Exact Answer /
Confidence format without adding other sections. Decide for yourself whether
the question benefits from recursive delegation."""


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
