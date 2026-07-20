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

The runtime executes tools only inside Python-style repl blocks. When searching,
actually call the tool and wait for its result:

```repl
hits = search("short specific query")
print(hits)
```

Do not merely describe a search in ordinary text. Use the returned snippets to
choose the next query. You may use spawn_subagent or spawn_subagents inside a
repl block when independent investigation is useful.

Return exactly:

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Submit the final answer through a repl block:

```repl
answer["content"] = "Explanation: ... [12345]\nExact Answer: ...\nConfidence: 75%"
answer["ready"] = True
```

The question and search results are the only benchmark information available;
gold answers, labels, qrels, and evaluator data are not available."""

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Find and verify the answer, then return the required Explanation / Exact Answer /
Confidence format without adding other sections."""

DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = """No working steps remain. Return the best final answer now.
Use exactly these three lines, with no Markdown fence or extra text:

Explanation: brief explanation with citations such as [12345]
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Do not use tools or subagents."""


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
