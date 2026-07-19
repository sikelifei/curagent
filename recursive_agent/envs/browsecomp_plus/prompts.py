"""Task guidance for BrowseComp-Plus BM25 retrieval."""

from __future__ import annotations

from .dataset import BrowseCompQuery

DEFAULT_BROWSECOMP_AGENT_PROMPT = """BrowseComp-Plus task guidance:

Answer the question using the fixed BrowseComp-Plus corpus and the registered
search(query) tool, which returns the official top-five BM25 snippets.

Use short, specific queries containing distinctive phrases, names, dates,
organizations, places, or titles. Refine later queries with entities discovered
in earlier results instead of repeating near-identical paraphrases.

Treat each result as candidate evidence. Verify the leading answer against the
important clues, calculate requested date or numeric relationships explicitly,
and cite supporting document IDs. All agents share the same search budget.

Return exactly:

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

The question and search results are the only benchmark information available;
gold answers, labels, qrels, and evaluator data are not available."""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Find and verify the answer, then return the required Explanation / Exact Answer /
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
