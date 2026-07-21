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

BROWSECOMP_SEARCH_AND_DELEGATION_STRATEGY = r"""
Search and recursive delegation strategy:

Every agent, at every recursion depth, must first decide whether its current
task can be split before making a search call. This is an internal routing
decision; do not spend a search call explaining it and do not output a formal
constraint table.

Split the task when it contains at least two independent clues, evidence
branches, events, documents, claims, time periods, organizations, or candidate
families that can be investigated separately. Multiple constraints describing
different events or documents are usually separate branches when each branch
can produce useful evidence without knowing the result of another branch.
Constraints that form a necessary sequence, where a later search requires an
entity discovered by an earlier search, remain one linked evidence chain.

If the current task can be split:

- Create two to four non-overlapping requests before searching.
- Make the requests collectively cover the independent branches.
- Give each request one distinct local evidence output to produce.
- Call spawn_subagents in the first REPL block and collect all reports.
- Do not independently repeat searches assigned to a subagent.

If the task contains multiple independent branches, delegation should happen
before broad root searching. If no two useful independent branches exist,
solve the linked task directly. Never spawn one subagent merely to outsource or
repeat the current task.

Every delegated task must be strictly smaller than its parent task:

- It has exactly one investigation objective.
- It contains only the clues, entities, constraints, leads, and exclusions
  needed for that objective.
- It excludes constraints assigned to sibling requests.
- It states the expected local output, such as a candidate entity, event date,
  publication identity, relationship, verification result, or contradiction.
- It asks for supporting and contradicting evidence, unresolved points,
  document IDs, and the query families already tried.

A request is invalid if it copies the complete original question, asks for the
same final answer without narrowing the evidence, substantially repeats the
parent's objective or query plan, overlaps a sibling's constraint group, or
cannot explain what distinct evidence it will add. Rewrite invalid requests
before delegating.

A delegated task applies exactly the same routing decision before its first
search. If its local objective still contains multiple independent branches,
split and delegate them again. Otherwise solve the local objective directly. It
must never pass its own objective or query plan downward unchanged.

Search discipline:

- Begin with the most discriminative source-like clue: an exact phrase, unusual
  event, date, title, organization, proper name, or rare term combination.
- Make one search call at a time and inspect its result before choosing the next
  query.
- A new query must introduce a newly discovered entity, source phrase, date,
  candidate, or genuinely different retrieval path.
- Do not issue more than two near-equivalent paraphrases of the same query.
- If repeated searches produce no new candidate, document, or discriminative
  term, stop that path and return it as unresolved instead of consuming the
  shared search budget.
- Do not search for a calculated relationship literally. Retrieve the
  underlying dates or values first, then calculate the relationship.

A delegated report should answer its local objective and contain:

local result | candidates | supported claims | contradicted claims |
unresolved claims | docids | tried query families

After receiving reports, compare the distinct candidates and evidence returned
by each branch. Intersect candidate-generating branches when they should identify
the same entity, reject candidates contradicted by an important clue, and
perform only targeted searches needed to connect, verify, or reject reports.
Do not restart the original broad search or repeat a subagent's failed query
family without a new entity or phrase.

Treat every report as an evidence lead rather than final truth. A strong answer
must satisfy multiple discriminative clues, including the hardest
identity-bearing clue.

Before setting answer["ready"] = True, verify that answer["content"] contains
exactly three newline-separated lines:

Explanation: brief explanation with document citations
Exact Answer: the shortest unambiguous answer
Confidence: 0-100%

Use literal newline characters between the three fields. Never place multiple
fields on one line, and never finish with a proposed search or unexecuted REPL
block.
"""

DEFAULT_BROWSECOMP_AGENT_PROMPT = (
    DEFAULT_BROWSECOMP_AGENT_PROMPT.rstrip()
    + "\n\n"
    + BROWSECOMP_SEARCH_AND_DELEGATION_STRATEGY.strip()
)

DEFAULT_BROWSECOMP_TASK_TEMPLATE = """Answer this evidence-seeking question
using the fixed BrowseComp-Plus BM25 corpus.

Question:
{query}

Find and verify the answer, then return the required Explanation / Exact Answer /
Confidence format without adding other sections."""

DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT = """  No working steps remain. This response will be parsed directly and no code will
  be executed. Even if the evidence is incomplete, choose the best-supported
  answer now.

  Your entire response must consist of exactly these three lines. The first
  character of the response must be the E in "Explanation":

  Explanation: brief explanation with citations such as [12345]
  Exact Answer: the shortest unambiguous answer
  Confidence: 0-100%

  Do not mention or output search calls, REPL blocks, subagents, Markdown fences,
  further investigation, or additional sections."""


# """No working steps remain. Return the best final answer now.
# Use exactly these three lines, with no Markdown fence or extra text:

# Explanation: brief explanation with citations such as [12345]
# Exact Answer: the shortest unambiguous answer
# Confidence: 0-100%

# Do not use tools or subagents."""


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
    "BROWSECOMP_SEARCH_AND_DELEGATION_STRATEGY",
    "DEFAULT_BROWSECOMP_TASK_TEMPLATE",
    "DEFAULT_BROWSECOMP_FORCED_FINAL_PROMPT",
    "build_browsecomp_task_prompt",
]
