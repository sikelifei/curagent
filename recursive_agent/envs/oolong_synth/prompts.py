"""Prompts for adaptive Oolong-Synthetic direct or recursive processing."""

from __future__ import annotations

from .dataset import OolongSynthSample


CHUNK_CHAR_LIMIT = 32 * 1024


def build_synth_agent_prompt() -> str:
    """Build the single prompt shared by Oolong-Synth roots and workers."""

    prompt = r"""Oolong-Synthetic environment guidance:

The complete dataset is available only in the private REPL context. Records are
unlabeled. There is no labels file, hidden label field, or tool that reveals the
answer.

This is one bounded root/worker workflow that refines the generic autonomous
routing guidance. Every agent receives this same prompt;
`context["oolong_role"]` decides its role:

- `root`: inspect the task, choose direct processing or disjoint 32K chunks,
  merge verified results, and be the only agent allowed to call
  `submit_answer`.
- `worker`: process one assigned chunk, never delegate, never call
  `submit_answer`, and return one JSON report through `answer`.

Use only executable `repl` blocks for Python. Wait for the real observation
after each block and never invent a `REPL output:` line. Your first response
must be exactly this block:

```repl
source = context["context_window_text"]
lines = source.splitlines()
rows = [
    line for line in lines
    if line.startswith("Date:")
    and " || User: " in line
    and " || Instance: " in line
]
dataset_intro = context.get("dataset_intro")
if dataset_intro is None:
    first_row = next(i for i, line in enumerate(lines) if line in rows)
    dataset_intro = "\n".join(lines[:first_row])
print({
    "role": context["oolong_role"],
    "context_chars": len(source),
    "complete_rows": len(rows),
    "question": context["question"],
})
```

Each valid record is one complete line in the exact form
`Date: ... || User: ... || Instance: ...`. Never split a record. Text before
the first record is the dataset introduction. The line starts with `Date:`;
there is no ` || Date: ` field. Parse it from the start of the line, for
example `line[len("Date:"):].split(" || User: ", 1)[0]`. Keep `answer` reserved
for the completion dictionary and use `final_text` for an answer string.

ROOT ROUTING

1. Use the `context_chars` measured above. Inspect the question type, required
   aggregation, and record count. Process locally only when a complete scan is
   manageable in the remaining steps. For a full scan larger than __CHUNK_CHAR_LIMIT__
   characters, or with more than 80 rows, use disjoint chunks. This is a
   feasibility gate: do not ask one agent to semantically read a context that
   cannot fit in its remaining observations. Process locally in bounded pages
   when the selected data is manageable.
2. For a full scan selected for chunking, greedily split
   `rows` at record boundaries. The sum of `len(row) + 1` in each chunk must be
   at most __CHUNK_CHAR_LIMIT__ characters. For a smaller-model full scan,
   prefer roughly 8K-16K character chunks when that gives each worker a
   tractable semantic reading task. Every row must occur in exactly one chunk;
   a single longer record forms its own chunk.
3. Choose local processing, `spawn_subagent` for one useful chunk, or
   `spawn_subagents` for several independent chunks according to expected
   benefit. For a full scan selected for chunking, dispatch every chunk before
   attempting final aggregation; process a small remainder locally only when
   that remainder is recorded explicitly. Parallel reports must be additively
   mergeable. Never delegate the same rows twice, create overlapping chunks, or
   pass the full task unchanged.
4. Give a worker only `oolong_role="worker"`, unique `chunk_id`, exact
   `expected_rows`, its rows in `context_window_text`, `dataset_intro`, the
   global `question`, and `dataset`. Do not put rows in the task text.
5. Parse every worker result with `json.loads`. Verify unique chunk id,
   `rows_seen == expected_rows`, requested keys, and complete coverage. Retry
   only one failed bounded chunk with a corrected request. Merge verified
   reports, perform the final arithmetic or ranking locally, and call
   `submit_answer(final_text)` immediately exactly once in the requested
   format. Its argument is the plain answer string, not a dictionary. Once
   verified reports cover every row, do not rescan the full dataset or start an
   alternate analysis branch; submission is mandatory.

WORKER WORKFLOW

Read every assigned record exactly once, using consecutive bounded pages if the
chunk does not fit in one observation. If an observation is truncated, shrink
the page and reread only that page. Classify the meaning of each Instance; do
not classify with keywords, regexes, word lists, label-name matches, or guessed
class balance. Python may parse exact Date/User metadata and add labels that you
explicitly assigned after reading.

Apply the question's Date/User filters to your own rows. Return a compact raw
JSON object with `chunk_id`, `rows_seen`, `counts`, and `totals`. Include zero
counts for every candidate label, and make `rows_seen` equal `expected_rows`.
Use `counts` for the question's additive partial results and `totals` for any
population/denominator counts needed by before/after or ratio questions. For
User/Date/month questions, use those normalized keys with nested label counts.
Set `answer["content"] = json.dumps(report)` and then
`answer["ready"] = True`. Return no prose or Markdown. A worker owns one atomic
chunk: it must never call `submit_answer`, must never spawn another agent, and
must never reassign its rows.

SEMANTIC RULES

Labels come only from reading each Instance. For `trec_coarse`: abbreviation is
an acronym or expansion; entity is a concrete object, organization, product,
language, event, animal, or substance; description and abstract concept asks
for a definition, reason, manner, explanation, purpose, or meaning; human being
is a person or group; location is a place; numeric value is a count, amount,
date, age, distance, price, duration, percentage, or other number.

The root must not replace failed worker coverage with a keyword classifier or a
guess. Only the root submits the global answer; workers only report their own
chunks. Never classify by keyword, label-name matching, or a guessed default;
read each selected Instance and assign its semantic label explicitly."""
    return prompt.replace("__CHUNK_CHAR_LIMIT__", f"{CHUNK_CHAR_LIMIT:,}")


DEFAULT_SYNTH_TASK_TEMPLATE = """Solve this Oolong-Synthetic benchmark task.

Question:
{question}

The complete unlabeled dataset is available only in the private REPL variable
`context["context_window_text"]`. Choose direct processing or bounded disjoint
record chunks, then finish with `submit_answer(...)` in the exact requested
format."""


def build_synth_task_prompt(
    sample: OolongSynthSample,
    *,
    template: str = DEFAULT_SYNTH_TASK_TEMPLATE,
) -> str:
    if "{question}" not in template:
        raise ValueError("Oolong-Synth task template must contain {question}")
    return template.replace("{question}", sample.question).strip()


DEFAULT_SYNTH_AGENT_PROMPT = build_synth_agent_prompt()

DEFAULT_SYNTH_FORCED_FINAL_PROMPT = """No working steps remain. Return the best final answer for the Oolong-Synthetic
question as plain text using the answer format requested by the task. Do not add
analysis, Markdown fences, or unsupported alternatives. Do not use tools,
subagents, or submit_answer now."""

__all__ = [
    "CHUNK_CHAR_LIMIT",
    "DEFAULT_SYNTH_AGENT_PROMPT",
    "DEFAULT_SYNTH_TASK_TEMPLATE",
    "DEFAULT_SYNTH_FORCED_FINAL_PROMPT",
    "build_synth_agent_prompt",
    "build_synth_task_prompt",
]
