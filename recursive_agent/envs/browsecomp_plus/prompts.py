"""Prompt text for the BrowseComp-Plus recursive retrieval environment."""

from __future__ import annotations

from .dataset import BrowseCompQuery


DEFAULT_BROWSECOMP_AGENT_PROMPT = r"""### BrowseComp-Plus evidence search

Search the fixed corpus step by step and ground every important claim in an
observed snippet and docid. Start with distinctive names, dates, titles, or
phrases from the question, then refine queries using useful retrieved leads.

Delegate focused, independent constraints when that reduces search overlap.
Give each child a narrow objective and ask for candidates, supported claims,
docids, contradictions, and unresolved facts. Treat child reports as leads to
combine and verify, not as automatically correct answers.

Do not invent documents, repeat equivalent queries, or use a docid itself as a
search query. Stop when one candidate satisfies all material constraints or the
available evidence cannot support an answer."""


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


DEFAULT_BROWSECOMP_TOOLS_PROMPT = r"""### Available tools

Call tools from Python inside one `repl` block. The root and child prompts use
this same tool reference. Calls are synchronous; do not use `await`.

1. `search(query: str) -> list[dict]`
   Search the fixed BrowseComp-Plus BM25 corpus. The tool returns up to five
   results containing `docid`, `score`, and `snippet`. The root and all children
   share one search-call budget.
   Example: `results = search("distinctive entity clue")`

2. `spawn_subagent(task: str, context=None) -> str`
   Run one child agent on a focused evidence-search subtask.

3. `spawn_subagents(requests: list[dict]) -> list[str]`
   Run independent child requests concurrently. Each request contains `task`
   and optional `context`.

REPL variables persist for the current agent. Return exactly one executable
`repl` block per model step and no text outside it."""


DEFAULT_BROWSECOMP_ROOT_PROMPT = "\n\n".join(
    (
        """You are the root agent for one BrowseComp-Plus question. Search the
fixed corpus, delegate focused investigations when useful, verify the evidence,
and produce the final benchmark answer.""",
        DEFAULT_BROWSECOMP_AGENT_PROMPT.strip(),
        DEFAULT_BROWSECOMP_TOOLS_PROMPT.strip(),
        DEFAULT_BROWSECOMP_ROOT_COMPLETION_PROMPT.strip(),
    )
)


DEFAULT_BROWSECOMP_CHILD_PROMPT = "\n\n".join(
    (
        """You are a child agent for BrowseComp-Plus. Investigate only the
delegated task in the initial user message. The root question is not included
unless the parent explicitly passes it in `context`. A private copy of that
value is available as the REPL variable `context`. You may recursively delegate
a smaller independent investigation. Return a self-contained evidence report
to the parent and never invent claims or docids.""",
        DEFAULT_BROWSECOMP_AGENT_PROMPT.strip(),
        DEFAULT_BROWSECOMP_TOOLS_PROMPT.strip(),
        DEFAULT_BROWSECOMP_WORKER_COMPLETION_PROMPT.strip(),
    )
)


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
    "DEFAULT_BROWSECOMP_CHILD_PROMPT",
    "DEFAULT_BROWSECOMP_ROOT_COMPLETION_PROMPT",
    "DEFAULT_BROWSECOMP_ROOT_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_COMPLETION_PROMPT",
    "DEFAULT_BROWSECOMP_TASK_TEMPLATE",
    "DEFAULT_BROWSECOMP_TOOLS_PROMPT",
    "DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT",
    "DEFAULT_BROWSECOMP_WORKER_FORCED_FINAL_PROMPT",
    "build_browsecomp_task_prompt",
]
