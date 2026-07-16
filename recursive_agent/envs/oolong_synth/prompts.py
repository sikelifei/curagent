"""Prompts for the curagent Oolong-Synthetic map/reduce evaluation."""

from __future__ import annotations

from .dataset import OolongSynthSample

DEFAULT_CHUNK_CHARS = 8_000

CHILD_TASK_TEMPLATE = r"""[OOLONG-SYNTH CHUNK WORKER]
You are a read-only worker for exactly one disjoint chunk. Never call an
environment tool and never delegate. Use context["dataset_intro"] for the label
definitions, context["question"] for filters/statistics, and only
context["context_window_text"] as data.

SEMANTIC CLASSIFICATION REQUIREMENT
- Classify from the meaning of each complete Instance. Python may parse Date,
  User, and rows and may aggregate labels that you explicitly assign.
- Never write a keyword list, regex classifier, word-matching classifier,
  guessed-label function, or any other heuristic that infers a semantic label
  in code. A report produced by such code is invalid.
- Every relevant row must be inspected and counted at most once. Do not infer
  an unobserved row's label from class balance or neighboring rows.
- The candidate-label universe comes from the question. For rankings, include
  only labels explicitly allowed by phrases such as "answer is one of the
  labels"; never add another label merely because it appears in the dataset
  intro. If the question allows one label, report that label's count only.

For dataset `trec_coarse`, classify the type of the expected answer, not words
in the question or its topic:
- abbreviation: asks for an acronym/abbreviation expansion or abbreviation.
- entity: asks for an object, animal, product, language, organization, event,
  substance, or other concrete entity.
- description and abstract concept: asks for a definition, reason, manner,
  explanation, purpose, meaning, or other descriptive/abstract answer.
- human being: asks for a person or group of people.
- location: asks for a place.
- numeric value: asks for a count, amount, date, age, distance, price, duration,
  percentage, or other number.

Use at most two responses:
1. First response: execute one short REPL block that prints the dataset intro
   once and every valid row in this chunk once. The environment limits chunks
   to a bounded size, so do not abbreviate Instance text and do not generate a
   classifier. For metadata-only date/user questions, direct exact parsing is
   allowed and this inspection response may be skipped.
2. Second response: semantically assign the required labels from what you read,
   then execute one final REPL block that hard-codes the resulting compact
   report, sets answer["content"] = json.dumps(report), and sets
   answer["ready"] = True. Do not print reasoning or per-row classifications.

The report has chunk_index, mode, rows_seen, counts, totals, and uncertain.
mode is label_counts, user_counts, date_counts, period_stats,
month_label_counts, or metadata_counts. uncertain contains at most three short
row indices.

- Label totals/rankings/comparisons: counts maps every candidate label to its
  filtered count.
- Most/second user: counts maps user IDs to counts; classify first when a label
  condition exists.
- Most/second date or dates represented N times: counts maps normalized dates
  to counts; no semantic classification is needed unless the question says so.
- Before/after frequency: counts has before_target and after_target; totals has
  before_all and after_all. Boundaries are strict.
- Calendar-month comparisons: counts maps YYYY-MM to nested requested-label
  counts. Include every candidate label when choosing the most frequent label.

Return no prose outside REPL blocks."""


def build_synth_agent_prompt(chunk_chars: int = DEFAULT_CHUNK_CHARS) -> str:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    return rf"""Oolong-Synthetic environment guidance:

The initial agent is the root. An agent whose initial user message starts with
`Delegated task:` is a read-only chunk worker and must follow that task's worker
contract instead of this root workflow. Only the root may call submit_answer.
Gold labels and context_window_text_with_labels are unavailable.

ROOT: EXACTLY FOUR REPL RESPONSES
Your first response must execute the initialization block below immediately.
Do not inspect context keys, print the question, print data rows, or add a
planning/exploration response. Then execute fan-out, merge, and submission in
that order. The root never semantically classifies Instance text itself.

1. INITIALIZE: extract complete valid rows, retain the label-definition intro,
   and split rows into disjoint chunks of at most {chunk_chars} characters.
2. FAN OUT: call spawn_subagents once with one worker per chunk.
3. MERGE: parse all compact reports, verify exact chunk coverage, and add maps.
   Retry a malformed/missing report once. For a ranking question, a global tie
   between answer candidates is also a failed classification: retry the chunks
   once with an explicit request for careful semantic review. Never guess a tie.
4. SUBMIT: calculate the requested ranking/count/frequency/date from the merged
   exact statistics and call submit_answer with the requested prefix, e.g.
   `Answer: 17`, `Label: entity`, `User: 12345`, or `Date: 01/31/2024`.

Metadata Date/User filtering may use exact Python parsing. Semantic labels must
come from workers' reading of complete Instance text, never root regex/keywords.
Before/after frequency compares target/total proportions with strict boundaries.
Second-most and represented-N-times are computed only after the global merge.
For most/least rankings, consider only labels explicitly allowed by the question,
not every label found in worker reports. A single allowed label is the answer.

Initialization block (first root response):
```repl
source = context["context_window_text"]
query = context["question"]
lines = source.splitlines()
row_positions = [
    i for i, line in enumerate(lines)
    if line.startswith("Date:")
    and " || User: " in line
    and " || Instance: " in line
]
if not row_positions:
    raise ValueError("no Oolong data rows found")
dataset_intro = "\n".join(lines[:row_positions[0]])
data_rows = [lines[i] for i in row_positions]

def chunk_rows(rows, max_chars={chunk_chars}):
    chunks, current, current_chars = [], [], 0
    for row in rows:
        row_chars = len(row) + 1
        if current and current_chars + row_chars > max_chars:
            chunks.append("\n".join(current))
            current, current_chars = [], 0
        current.append(row)
        current_chars += row_chars
    if current:
        chunks.append("\n".join(current))
    return chunks

chunks = chunk_rows(data_rows)
print({{"rows": len(data_rows), "chunks": len(chunks),
       "max_chunk_chars": max(map(len, chunks), default=0)}})
```

Fan-out block (second root response):
```repl
requests = [
    {{
        "task": context["child_task_template"] + "\n\nQuestion: " + query,
        "context": {{
            "context_window_text": chunk,
            "dataset_intro": dataset_intro,
            "question": query,
            "answer_type": context.get("answer_type"),
            "dataset": context.get("dataset"),
            "chunk_index": index,
        }},
    }}
    for index, chunk in enumerate(chunks)
]
reports = spawn_subagents(requests)
print({{"requested_chunks": len(requests), "returned_reports": len(reports)}})
```

Merge block (third root response):
```repl
import json

parsed_reports = []
fence = "`" * 3
for raw in reports:
    clean = raw.replace(fence + "repl", "").replace(fence + "json", "").replace(fence, "").strip()
    report = json.loads(clean)
    if not isinstance(report, dict) or not isinstance(report.get("chunk_index"), int):
        raise ValueError("malformed child report")
    parsed_reports.append(report)
expected = set(range(len(chunks)))
observed = {{report["chunk_index"] for report in parsed_reports}}
if observed != expected:
    raise ValueError("missing or duplicate chunk reports")

def add_maps(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {{}})
            add_maps(child, value)
        else:
            target[key] = int(target.get(key, 0)) + int(value)

merged_counts, merged_totals = {{}}, {{}}
for report in parsed_reports:
    add_maps(merged_counts, report.get("counts", {{}}))
    add_maps(merged_totals, report.get("totals", {{}}))
print({{"mode": parsed_reports[0].get("mode"),
       "counts": merged_counts, "totals": merged_totals,
       "chunks": len(parsed_reports)}})
```

The fourth root response must only compute from merged_counts/merged_totals and
call submit_answer. Do not inspect variables or data again. Do not finish by
setting answer directly, and do not wrap the answer in `\\boxed{{}}`."""


DEFAULT_SYNTH_TASK_TEMPLATE = """Solve this Oolong-Synthetic benchmark task.

Question:
{question}

The complete unlabeled dataset is available only in the private REPL variable
`context["context_window_text"]`. Read it through the REPL, split its data rows
into disjoint chunks, delegate chunk processing, aggregate the returned JSON
statistics exactly, and finish by calling `submit_answer(...)` with the answer
format requested in the question. Your first response must execute the
environment's initialization block; do not inspect context first or print the
long context."""


def build_synth_task_prompt(
    sample: OolongSynthSample,
    *,
    template: str = DEFAULT_SYNTH_TASK_TEMPLATE,
) -> str:
    if "{question}" not in template:
        raise ValueError("Oolong-Synth task template must contain {question}")
    return template.replace("{question}", sample.question).strip()


DEFAULT_SYNTH_AGENT_PROMPT = build_synth_agent_prompt()

__all__ = [
    "CHILD_TASK_TEMPLATE",
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_SYNTH_AGENT_PROMPT",
    "DEFAULT_SYNTH_TASK_TEMPLATE",
    "build_synth_agent_prompt",
    "build_synth_task_prompt",
]
