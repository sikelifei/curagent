"""Oolong-real task and environment prompt text."""

from __future__ import annotations

from .dataset import OolongSample

DEFAULT_OOLONG_AGENT_PROMPT = r"""Oolong-real environment guidance:

MODE RULES:
- The initial root agent owns the episode. A delegated agent is identified by an
  initial user message beginning with `Delegated task:`.
- A delegated agent is read-only. It must never call `observe`,
  `episode_report`, `submit_answer`, `spawn_subagent`, or `spawn_subagents`.
  It must return one compact JSON report through one final REPL block that sets
  `answer["content"]` and then `answer["ready"] = True`.
- The root agent must submit exactly once. Do not submit from a delegated agent.

PRIVATE CONTEXT:
- The private REPL variable `context` is a dict. The full transcript is in
  `context["context_window_text"]`; the question is in `context["question"]`.
  A child receives a copied chunk plus `context["mapping"]` and
  `context["chunk_index"]`.
- Never print the full transcript. The root must split it into disjoint,
  line-preserving chunks of at most about 12,000 characters.
- Only count text between `[START OF EPISODE]` and `[END OF EPISODE]`.
  The preamble itself mentions these literal marker strings, so use the last
  occurrence of each marker (`rfind`), never the first `find` or
  `split(..., 1)` occurrence.
  Ignore the question, instructions, player/character mapping prose, character
  backstories, advertisements, and all text after `[END OF EPISODE]`.

WHAT COUNTS AS A ROLL:
- Count one actual die/check result once. A result must be tied to a declared
  check, save, attack, skill, initiative, damage, or other D&D roll in the
  episode transcript.
- Do not count the word "roll" in encouragement ("roll well"), a question about
  an earlier result ("what did you roll?"), repeated narration of a result, or
  ordinary prose. Do not treat a modifier or final total as a natural die value.
- If advantage/disadvantage explicitly produces two die results, record both
  results; do not count the later check total again. If a result is not explicit,
  mark it uncertain instead of guessing.
- For a character/player question, use the mapping passed in
  `context["mapping"]`. The speaker is the player; the mapped character is the
  character. Do not infer a mapping from a chunk that does not contain it.
- For a roll-type question, use the explicit type named by the transcript, such
  as `Investigation` or `Intimidation`. For a natural-value question, count only
  explicit natural die values, not totals after modifiers.

CHILD REPORT SCHEMA:
- A valid child report is a JSON object with exactly these useful fields:
  `chunk_index`, `records`, and `uncertain`.
- `records` is a list of objects with `speaker`, `character`, `kind`,
  `roll_value`, `natural_value`, and a short `evidence` string. `roll_value` is
  the explicit result or damage total; `natural_value` is populated only when
  the transcript explicitly gives the unmodified die value. A damage total such
  as 25 is not a natural value of 25. Use JSON `null` when a field is unknown.
  Each record represents one local die/check result, not an aggregate number.
- `uncertain` is at most three short snippets, each at most 120 characters. Never
  copy a backstory or transcript paragraph into the report. Never put prose
  outside the JSON report. Malformed reports are not evidence and must be
  ignored by the aggregate child.

ROOT RUNBOOK:
- Execute exactly three logical phases: initialize chunks; fan out read-only
  child reports; aggregate and submit immediately.
- Use `spawn_subagent(task, context)` positionally. Use
  `spawn_subagents([{"task": ..., "context": ...}])` for independent chunks.
  Do not use `context=` as a keyword with `spawn_subagent`.
- After the fan-out returns, do not print, re-open, or manually debug individual
  reports. The next REPL block must aggregate and submit. The aggregate child
  must parse valid JSON reports, preserve disjoint chunk ownership, filter the
  question, and return a JSON object containing an integer or string
  `candidate`.
- The aggregate child is also read-only and must finish with
  `answer["content"] = json.dumps(result)` and `answer["ready"] = True`.
- The root parses `candidate` and immediately calls
  `submit_answer(r'\boxed{...}')`. If the candidate is unavailable, submit the
  best explicitly supported candidate rather than entering another exploratory
  loop. Never return prose-only or an unclosed REPL block.

Few-shot 1 - initialize bounded, disjoint chunks and pass the mapping to every child:
```repl
import json

source = context["context_window_text"]
query = context["question"]
start_marker = "[START OF EPISODE]"
end_marker = "[END OF EPISODE]"
start = source.rfind(start_marker)
end = source.rfind(end_marker)
mapping = source[:start] if start >= 0 else ""
episode_text = (
    source[start:end + len(end_marker)]
    if start >= 0 and end >= start
    else source
)
player_to_character = {}
for line in mapping.splitlines():
    mapping_marker = " plays the character "
    if mapping_marker in line:
        player, character = line.split(mapping_marker, 1)
        player_to_character[player.strip()] = character.rstrip(".").strip()

def chunk_text(text, max_chars=12000):
    chunks, current, current_len = [], [], 0
    for line in text.splitlines(keepends=True):
        if current and current_len + len(line) > max_chars:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks

chunks = chunk_text(episode_text)
evidence = []
print({"question": query, "chunks": len(chunks),
       "max_chunk_chars": max([len(chunk) for chunk in chunks], default=0)})
```

Few-shot 2 - one child report is structured JSON, not a free-form delta:
```repl
import json

toy_report = {
    "chunk_index": 0,
    "records": [{
        "speaker": "Marisha",
        "character": "Keyleth",
        "kind": "Perception",
        "roll_value": 14,
        "natural_value": 14,
        "evidence": "Matt: Make a perception check. Marisha: 14.",
    }],
    "uncertain": [],
}
answer["content"] = json.dumps(toy_report, ensure_ascii=False)
answer["ready"] = True
```

Few-shot 3 - fan out independent read-only chunk auditors:
```repl
requests = [
    {
        "task": (
            "[CHILD JSON CONTRACT] Inspect only context['context_window_text']; "
            "analyze only the episode portion between the START and END markers. "
            "Never call an environment tool or delegate. Count explicit D&D die "
            "results exactly once and use context['mapping'] for speaker names. "
            "Your entire response must be exactly one repl block. In that block, "
            "import json, construct a JSON object with chunk_index, records, and "
            "uncertain, set answer['content'] = json.dumps(report), and set "
            "answer['ready'] = True. Do not write prose, analysis, or another "
            "code block. Each record must include speaker, mapped character, kind, "
            "roll_value, natural_value, and one short evidence string. Return an "
            "empty records list immediately when there is no explicit result. "
            "Example: 'Matt: Make a perception check. Marisha: 14.' is one "
            "record; 'Roll well!' and 'what did you roll?' are not records; "
            "'25 points of damage' has roll_value 25 and natural_value null. "
            "Question: " + query
        ),
        "context": {
            "context_window_text": chunk,
            "question": query,
            "mapping": player_to_character,
            "chunk_index": i,
        },
    }
    for i, chunk in enumerate(chunks)
]
evidence = spawn_subagents(requests)
print({"processed_chunks": len(evidence),
       "evidence_chars": sum(len(item) for item in evidence)})
```

Few-shot 4 - aggregate and submit in the same root execution:
```repl
aggregate = spawn_subagent(
    "[AGGREGATOR JSON CONTRACT] You are read-only: do not call tools or delegate. "
    "Parse only valid JSON child reports from context['reports']. Ignore malformed "
    "reports and snippets outside the episode. The chunks are disjoint, so do not "
    "subtract reports merely because speakers or numbers repeat. Filter records "
    "using context['question'], mapping, kind, roll_value, and natural_value; count records, "
    "not prose mentions. Return exactly one repl block that imports json, sets "
    "result = {'candidate': ..., 'used_chunks': [...], 'uncertain': [...]}, then "
    "sets answer['content'] = json.dumps(result) and answer['ready'] = True. "
    "Question: " + query,
    {"question": query, "mapping": player_to_character, "reports": evidence},
)

import json
import re

fence = "`" * 3
clean = aggregate.replace(fence + "repl", "").replace(fence, "").strip()
candidate = None
try:
    parsed = json.loads(clean)
    if isinstance(parsed, dict):
        candidate = parsed.get("candidate")
except Exception:
    pass
if candidate is None:
    match = re.search(r"(?:candidate|answer|total)[^0-9-]*(\d+)", clean, re.I)
    if match:
        candidate = match.group(1)
if candidate is None:
    raise ValueError("aggregate did not contain a candidate")
submit_answer(r"\boxed{" + str(candidate) + r"}")
```

Do not expose the gold answer. Do not use `observe` to retrieve the transcript.
The only successful root termination is the `submit_answer` call in the final
combined phase above.
"""

DEFAULT_OOLONG_TASK_TEMPLATE = """Solve this Oolong-real benchmark example.

Question:
{question}

The complete transcript is available only in the private REPL variable
`context["context_window_text"]`. Follow the Oolong JSON child-report workflow:
split the transcript without printing it, fan out read-only chunk auditors, then
aggregate and submit immediately. Do not return a prose answer. The root must
execute `submit_answer(r'\\boxed{YOUR_ANSWER}')` after aggregation so the official
scorer can parse the answer."""


def build_oolong_task_prompt(
    sample: OolongSample,
    *,
    template: str = DEFAULT_OOLONG_TASK_TEMPLATE,
) -> str:
    if "{question}" not in template:
        raise ValueError("Oolong prompt template must contain {question}")
    return template.replace("{context_window_text}", sample.context_window_text).replace(
        "{question}", sample.question
    ).strip()
