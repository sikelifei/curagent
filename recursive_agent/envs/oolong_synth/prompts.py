"""Prompts for flat 64K Oolong-Synthetic context decomposition."""

from __future__ import annotations

from .dataset import OolongSynthSample


CHUNK_CHAR_LIMIT = 64 * 1024


def build_synth_agent_prompt() -> str:
    """Build the single prompt shared by Oolong-Synth roots and workers."""

    prompt = r"""Oolong-Synthetic environment guidance:

The complete dataset is available only in the private REPL context. Records are
unlabeled. There is no labels file, hidden label field, or tool that reveals the
answer.

This is one flat root/worker workflow. It overrides the generic autonomous
task-routing guidance. Every agent receives this same prompt;
`context["oolong_role"]` decides its role:

- `root`: choose direct processing or 64K chunks, merge results, and be the only
  agent allowed to call `submit_answer`.
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
the first record is the dataset introduction.

ROOT WORKFLOW

1. Use the `context_chars` measured above. If it is at most
   __CHUNK_CHAR_LIMIT__ characters (64K), process the complete task in the root.
   Read semantic records in consecutive bounded pages so no REPL observation is
   truncated. Python may parse exact Date/User metadata and aggregate explicit
   classifications.
2. If `context_chars` is greater than __CHUNK_CHAR_LIMIT__, use the REPL to
   greedily split all `rows`, in order, at record boundaries. The sum of
   `len(row) + 1` in every chunk must be at most __CHUNK_CHAR_LIMIT__ characters.
   A single record longer than the limit forms its own chunk. Every row must
   appear in exactly one chunk.
3. Send the chunks with `spawn_subagents`, in batches if needed. Use the task
   `Process the assigned Oolong-Synthetic chunk and return its JSON report.`
   Do not put data rows in the task text. Give each worker a private context with
   only these fields: `oolong_role="worker"`, unique `chunk_id`, exact
   `expected_rows`, that chunk's rows in `context_window_text`, `dataset_intro`,
   the global `question`, and `dataset`.
4. Parse every worker result with `json.loads`. Verify each chunk id once,
   `rows_seen == expected_rows`, and all requested labels or grouping keys are
   present. Retry only a missing or malformed chunk. Merge the verified partial
   counts, perform the final ranking/comparison/arithmetic, and call
   `submit_answer(...)` exactly once in the format requested by the question.

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
`answer["ready"] = True`. Return no prose or Markdown. A worker must never call
`submit_answer` and must never spawn another agent.

SEMANTIC RULES

Labels come only from reading each Instance. For `trec_coarse`: abbreviation is
an acronym or expansion; entity is a concrete object, organization, product,
language, event, animal, or substance; description and abstract concept asks
for a definition, reason, manner, explanation, purpose, or meaning; human being
is a person or group; location is a place; numeric value is a count, amount,
date, age, distance, price, duration, percentage, or other number.

The root must not replace failed worker coverage with a keyword classifier or a
guess. Only the root submits the global answer; workers only report their own
chunks."""
    return prompt.replace("__CHUNK_CHAR_LIMIT__", f"{CHUNK_CHAR_LIMIT:,}")


DEFAULT_SYNTH_TASK_TEMPLATE = """Solve this Oolong-Synthetic benchmark task.

Question:
{question}

The complete unlabeled dataset is available only in the private REPL variable
`context["context_window_text"]`. Follow the environment's single 64K routing
workflow and finish with `submit_answer(...)` in the exact requested format."""


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
