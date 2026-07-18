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

Read context["browsecomp_role"] before acting.

Root role:
- Understand the complete question and identify independent evidence tasks.
- Delegate only clear, narrow tasks when separate investigations are useful.
  Do not force delegation for a question that direct search can solve reliably.
- The root may search too. Integrate child findings, cross-check candidates,
  and resolve conflicts before answering.
- A child context should contain browsecomp_role="worker" and a focused
  objective. Do not send gold data, labels, or evaluator information.

Worker role:
- Solve only the delegated objective. Search when evidence is needed and avoid
  unrelated investigation or restating the whole original question.
- Return a concise report with these fields: Finding, Evidence, Candidate
  answer, Uncertainty, Relevant docids. Explicitly report unresolved work
  instead of guessing.
- Do not produce or submit the root question's final answer. Delegate further
  only if the assigned task itself has genuinely independent branches.

The root's final output must use exactly this three-line structure:
Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Do not read benchmark files from the REPL. The question in the private context
and search results are the only benchmark information available for solving."""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Search, delegate focused independent investigations when useful, verify the
important clues, and return the required Explanation / Exact Answer /
Confidence format without adding other sections."""


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
