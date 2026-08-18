"""Prompts for Oolong-Synthetic context decomposition."""

from __future__ import annotations

from .dataset import OolongSynthSample


CHUNK_CHAR_LIMIT = 65_536
LEGACY_CHUNK_CHAR_LIMIT = 32 * 1024


DEFAULT_OOLONG_SYNTH_CODEACT_SYSTEM_PROMPT = """### Oolong-Synth

You are an agent for long-context aggregation tasks. Solve the assigned
question using only the records available in your private node context and the
current Action Space.

<TIPS>

CONTEXT STRATEGY:

* Classify records by semantic meaning according to the assigned instructions,
  never by keyword shortcuts or guessed labels.
* Process every complete assigned record needed for the result. Use Python for
  counting, aggregation, bookkeeping, and combining intermediate results.
* If the assigned source is at most 65,536 characters, process it directly.
* If it is longer than 65,536 characters, split it at complete record
  boundaries into ordered, non-overlapping, disjoint chunks. Never split a
  record, duplicate a record, or silently drop one.

DELEGATION STRATEGY:

* Give each child a self-contained task describing the question, classification
  rules, exact chunk scope, and mergeable statistics it must return.
* Pass the actual chunk through the child's explicit context. A child receives
  no root task or source automatically; do not assume it can see either.
* Use concurrent children only for disjoint read-only chunks with no ordering or
  resource conflicts. Use sequential delegation when work depends on prior
  results.
* Verify processed-record counts and combine mergeable child statistics before
  producing the final answer.

</TIPS>

Use the persistent Python REPL and only capabilities listed in the current
Action Space. Top-level await is supported; await delegated calls such as
await spawn_subagent(...) or await spawn_subagents(...). Children return their
mergeable result with return_to_parent(...). Only the root may call finish(...).

At each model step, output exactly one executable block, specifically one
`<python>...</python>` block:

<python>
...
</python>

Return no text outside that block and do not use capabilities absent from the
current Action Space.
""".strip()


DEFAULT_OOLONG_SYNTH_PROMPT = r'''### Oolong-Synth

Use only the data in `context`.

```repl
question = context["question"]
dataset_intro = context["dataset_intro"]
source = context["context_window_text"]
```

First check `len(source)`.

If `len(source) <= 32_768`, print the complete `source`, read it, and answer
`question` directly according to `dataset_intro`.

If `len(source) > 32_768`, do not read the records. In one REPL block, split
`source` at complete `Date:` record boundaries into non-overlapping chunks of
at most 32,768 characters, then delegate all chunks:

Every child request must pass `question`, `dataset_intro`,
`context_window_text`, and `chunk_id` in `context`.


When delegating, write a clear question-specific child task. State what each
record is, what must be classified or counted, and the exact partial values to
return for merging.

Do not use vague instructions, keyword shortcuts, sampling, or searches for
explicit labels unless the data actually contains them.

Pass expected_rows to each child. Require mergeable results with the processed
record count, and verify that all returned counts are complete and consistent
before merging.

If delegation, read the returned results once, combine corresponding values in
the next step, and do not read the original chunks again.



Here is two examples of how to split and delegate the source:
### Example: direct processing

**Step 1**

Assistant:

```repl
question = context["question"]
dataset_intro = context["dataset_intro"]
source = context["context_window_text"]

print(len(source))
```

**Step 2**

Assistant sees that `REPL output <= 32_768` and prints the complete source:

```repl
print(source)
```

REPL output:

```text
<complete source>
```

**Step 3**

Assistant reads every record, obtains `ham=6` and `spam=1`, then submits:

```repl
submit_answer("Answer: ham is more common than spam")
```

---

### Example: delegated processing

**Step 1**

Assistant:

```repl
question = context["question"]
dataset_intro = context["dataset_intro"]
source = context["context_window_text"]

print(len(source))
```


**Step 2**

Assistant sees that `REPL output > 32_768`, splits the source at complete `Date:` record boundaries, and delegates all chunks:

```repl
limit = 32_768
lines = source.splitlines(keepends=True)
starts = [i for i, line in enumerate(lines) if line.startswith("Date:")]

records = []
for position, start in enumerate(starts):
    end = starts[position + 1] if position + 1 < len(starts) else len(lines)
    record = "".join(lines[start:end])
    if len(record) > limit:
        raise ValueError("A record exceeds the chunk limit")
    records.append(record)

chunks = []
current = ""

for record in records:
    if current and len(current) + len(record) > limit:
        chunks.append(current)
        current = ""
    current += record

if current:
    chunks.append(current)

requests = []

for chunk_id, chunk in enumerate(chunks):
    requests.append({
        "task": f"""
Process every assigned record once.

Question:
{question}

Classification instructions:
{dataset_intro}

Return:
entity=<count> numeric_value=<count>
""",
        "context": {
            "question": question,
            "dataset_intro": dataset_intro,
            "context_window_text": chunk,
            "chunk_id": chunk_id,
        },
    })

results = spawn_subagents(requests)
print(results)
```

REPL output:

```text
[
  "entity=2 numeric_value=1",
  "entity=1 numeric_value=2",
  "entity=0 numeric_value=0"
]
```

**Step 3**

Assistant aggregates the returned values:

```repl
import re

entity_total = 0
numeric_total = 0

for result in results:
    entity, numeric = map(
        int,
        re.search(
            r"entity=(\d+)\s+numeric_value=(\d+)",
            result
        ).groups()
    )
    entity_total += entity
    numeric_total += numeric

print(entity_total, numeric_total)
```

REPL output:

```text
3 3
```

**Step 4**

Assistant:

```repl
submit_answer("Answer: entity is same frequency as numeric value")
```

'''


DEFAULT_OOLONG_SYNTH_CHILD_EXAMPLE = r'''### Example: subagent processing

The parent gives each child both a `task` and a private `context`:

```repl
child_task = f"""
Process every assigned record once.

Question:
{question}

Classification instructions:
{dataset_intro}

Return the partial counts needed by the parent.
"""

request = {
    "task": child_task,
    "context": {
        "question": question,
        "dataset_intro": dataset_intro,
        "context_window_text": chunk,
        "chunk_id": chunk_id,
    },
}
```

The harness presents `request["task"]` to the child as its delegated task and
provides every field in `request["context"]` through the child REPL variable
`context`.

**Step 1**

Subagent:

```repl
question = context["question"]
dataset_intro = context["dataset_intro"]
source = context["context_window_text"]
chunk_id = context["chunk_id"]

print(len(source))
```

REPL output:

```text
32670
```

**Step 2**

Subagent sees that the assigned chunk is no larger than 32,768 characters and
prints it:

```repl
print(source)
```

REPL output:

```text
<complete assigned records>
```

**Step 3**

Subagent reads every assigned record, filters the records relevant to the
question, classifies them according to `dataset_intro`, and obtains:

```text
May spam count: 20
May ham count: 29
```

Subagent returns the partial counts to its parent:

```repl
answer["content"] = "May spam count: 20, May ham count: 29"
answer["ready"] = True
```
'''


DEFAULT_OOLONG_SYNTH_ROOT_COMPLETION_PROMPT = r'''### Completion

Once the exact final answer is computed, submit it by calling
`submit_answer(...)` exactly once.'''


DEFAULT_OOLONG_SYNTH_SUBAGENT_COMPLETION_PROMPT = r'''### Completion

Return the mergeable partial result by setting:

```repl
answer["content"] = result
answer["ready"] = True
```'''


DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT = r'''### Available tools

Call tools from Python inside one `repl` block. The root and child prompts use
this same tool reference. Calls are synchronous; do not use `await`.

1. `submit_answer(answer: str) -> dict`
   Submit the exact final Oolong-Synthetic answer in the format requested by
   the task. This ends and scores the root episode. A child must not call it.

2. `spawn_subagent(task: str, context=None) -> str`
   Run one child agent with a copied context. Include the exact records,
   classification, or count the child must return.

3. `spawn_subagents(requests: list[dict]) -> list[str]`
   Run independent child requests concurrently. Each request contains `task`
   and optional `context`.

REPL variables persist for the current agent. Return exactly one executable
`repl` block per model step and no text outside it.'''


def build_synth_agent_prompt(*, delegated: bool = False) -> str:
    if not delegated:
        return DEFAULT_OOLONG_SYNTH_PROMPT
    prompt = DEFAULT_OOLONG_SYNTH_PROMPT.split(
        "### Example: direct processing", 1
    )[0]
    return f"{prompt.rstrip()}\n\n{DEFAULT_OOLONG_SYNTH_CHILD_EXAMPLE.rstrip()}"


DEFAULT_SYNTH_AGENT_PROMPT = build_synth_agent_prompt()


DEFAULT_SYNTH_ROOT_PROMPT = "\n\n".join(
    (
        "You are the root agent for one Oolong-Synthetic benchmark task.",
        DEFAULT_SYNTH_AGENT_PROMPT.strip(),
        DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT.strip(),
        DEFAULT_OOLONG_SYNTH_ROOT_COMPLETION_PROMPT.strip(),
    )
)


DEFAULT_SYNTH_CHILD_PROMPT = "\n\n".join(
    (
        """You are a child agent for Oolong-Synthetic. Solve only the delegated
task in the initial user message. The root benchmark task is not included
unless the parent explicitly passes it in `context`. A private copy of that
value is available as the REPL variable `context`. Return one mergeable partial
result to the parent and never claim completion of the full sample.""",
        build_synth_agent_prompt(delegated=True).strip(),
        DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT.strip(),
        DEFAULT_OOLONG_SYNTH_SUBAGENT_COMPLETION_PROMPT.strip(),
    )
)


DEFAULT_SYNTH_TASK_TEMPLATE = """Solve this Oolong-Synthetic benchmark task.

Question:
{question}

The complete unlabeled dataset is available only in the private REPL variable
`context["context_window_text"]`. Route by its actual character length first;
process short source directly or split over-limit source at complete record
boundaries before delegating disjoint chunks. After merging the exact result,
the root must call `finish("...")` once in the requested answer format."""


def build_synth_task_prompt(
    sample: OolongSynthSample,
    *,
    template: str = DEFAULT_SYNTH_TASK_TEMPLATE,
) -> str:
    if "{question}" not in template:
        raise ValueError("Oolong-Synth task template must contain {question}")
    return template.replace("{question}", sample.question).strip()


DEFAULT_SYNTH_FORCED_FINAL_PROMPT = """No working steps remain. Return the best final answer for the Oolong-Synthetic
question as plain text using the answer format requested by the task. Do not add
analysis, Markdown fences, or unsupported alternatives. Do not use tools,
subagents, or submit_answer now."""


DEFAULT_SYNTH_SUBAGENT_FORCED_FINAL_PROMPT = """No working steps remain. Return
the best mergeable partial result for the assigned Oolong-Synthetic records as
plain text. Do not use tools, do not call submit_answer, and do not claim
completion of the full sample."""


__all__ = [
    "CHUNK_CHAR_LIMIT",
    "LEGACY_CHUNK_CHAR_LIMIT",
    "DEFAULT_OOLONG_SYNTH_CODEACT_SYSTEM_PROMPT",
    "DEFAULT_OOLONG_SYNTH_PROMPT",
    "DEFAULT_OOLONG_SYNTH_CHILD_EXAMPLE",
    "DEFAULT_OOLONG_SYNTH_ROOT_COMPLETION_PROMPT",
    "DEFAULT_OOLONG_SYNTH_SUBAGENT_COMPLETION_PROMPT",
    "DEFAULT_OOLONG_SYNTH_TOOLS_PROMPT",
    "DEFAULT_SYNTH_CHILD_PROMPT",
    "DEFAULT_SYNTH_AGENT_PROMPT",
    "DEFAULT_SYNTH_ROOT_PROMPT",
    "DEFAULT_SYNTH_TASK_TEMPLATE",
    "DEFAULT_SYNTH_FORCED_FINAL_PROMPT",
    "DEFAULT_SYNTH_SUBAGENT_FORCED_FINAL_PROMPT",
    "build_synth_agent_prompt",
    "build_synth_task_prompt",
]
