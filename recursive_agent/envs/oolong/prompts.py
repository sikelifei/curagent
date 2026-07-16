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
  A child receives a copied chunk plus `context["mapping"]`,
  `context["episode_index"]`, and `context["chunk_index"]`.
- Never print the full transcript. The root must split it into disjoint,
  line-preserving chunks of at most about 12,000 characters. This keeps each
  child request far below the roughly 500K-character child context capacity
  while keeping each auditor's semantic scan small enough to finish reliably.
- Only count text inside actual episode blocks whose marker lines are exactly
  `[START OF EPISODE]` and `[END OF EPISODE]`. The preamble mentions these
  strings inline, so detect standalone marker lines with a multiline regex,
  not `rfind`, `find`, or `split(..., 1)`.
- A context may contain many episodes. For `episode N` questions select that
  episode; for cumulative-through-episode-N questions select episodes 1..N;
  for `across all`, `each episode`, or `all episodes` select every episode.
  For `this episode`, select the only block when there is one. Ignore the
  question, instructions, mapping prose, backstories, advertisements, and all
  text outside selected episode blocks.

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
- A valid child report is a JSON object with `chunk_index`, `episode_index`,
  `rolls`, `spells`, and `uncertain`. Do not emit one long record per event.
- `rolls` is a compact object with integer `total`, plus count maps named
  `by_player`, `by_character`, `by_type`, `by_value`, and `by_natural_value`.
  Include only keys observed in this chunk. `total` counts actual die/check
  results, not mentions of the word roll. A sentence such as `I rolled a 20
  and a 17. So 19.` contributes two results, not three; the final check total
  is not another die. `by_natural_value` includes only explicit unmodified die
  values. Also include `relevant_values`, a short list of objects with
  `speaker`, `character`, `type`, `value`, `natural_value`, and `evidence` when
  the question asks for a particular value or roll.
- `spells` is a compact object with integer `total`, count maps named
  `by_player`, `by_character`, and `by_name`, and chronological `ordered` spell
  events. An ordered event has `name`, `speaker`, `character`, `level`, and
  `base_level`; use JSON `null` for unknown levels. Keep only the first/last
  few events when the question asks for first/last spells, but retain counts
  for count questions.
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

Few-shot 1 - identify all real episode blocks and initialize bounded chunks:
```repl
import json
import re

source = context["context_window_text"]
query = context["question"]
start_re = re.compile(r"(?m)^\[START OF EPISODE\]\s*$")
end_re = re.compile(r"(?m)^\[END OF EPISODE\]\s*$")
starts = list(start_re.finditer(source))
mapping = source[:starts[0].start()] if starts else ""
episodes = []
for episode_index, match in enumerate(starts, 1):
    end = end_re.search(source, match.end())
    if end:
        episodes.append((episode_index, source[match.start():end.end()]))

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

requested = [int(x) for x in re.findall(r"episode\s+(\d+)", query, re.I)]
if re.search(r"across all|each episode|all episodes", query, re.I):
    selected = episodes
elif re.search(r"cummulative|cumulative", query, re.I) and requested:
    selected = [item for item in episodes if item[0] <= requested[-1]]
elif requested:
    selected = [item for item in episodes if item[0] == requested[-1]]
else:
    selected = episodes

chunks = [
    {"episode_index": episode_index, "text": chunk}
    for episode_index, episode_text in selected
    for chunk in chunk_text(episode_text)
]
print({"episodes": len(selected), "chunks": len(chunks),
       "max_chunk_chars": max([len(item["text"]) for item in chunks], default=0)})
```

Few-shot 2 - a child returns additive statistics, not a truncated event list:
```repl
import json

toy_report = {
    "chunk_index": 0,
    "episode_index": 1,
    "rolls": {
        "total": 2,
        "by_player": {"Marisha": 2},
        "by_character": {"Keyleth": 2},
        "by_type": {"Perception": 2},
        "by_value": {"14": 1, "7": 1},
        "by_natural_value": {"14": 1},
        "relevant_values": [],
    },
    "spells": {
        "total": 0,
        "by_player": {},
        "by_character": {},
        "by_name": {},
        "ordered": [],
    },
    "uncertain": [],
}
answer["content"] = json.dumps(toy_report, ensure_ascii=False)
answer["ready"] = True
```

Few-shot 3 - fan out independent read-only chunk auditors:
```repl
focus = (
    "This is a roll question. Analyze rolls only and set spells to zero with "
    "empty maps/lists; do not scan spell names. "
    if str(context.get("question_type", "")).endswith("rolls")
    else
    "This is a spell question. Analyze spells only and set rolls to zero with "
    "empty maps/lists; do not scan die results. "
)

requests = [
    {
        "task": (
            focus +
            "[CHILD JSON CONTRACT] Inspect only context['context_window_text'] "
            "for episode context['episode_index']; never delegate or call an "
            "environment tool. The transcript uses natural speaker utterances, "
            "not one numeric result per line, so do semantic extraction rather "
            "than a regex requiring exactly 'Speaker: number'. Count each "
            "explicit die/check result once. Do not count 'roll well', 'what did "
            "you roll?', or a repeated total. Use context['mapping'] to map player "
            "to character. Return exactly one final repl block with a JSON object "
            "containing chunk_index, episode_index, rolls, spells, and uncertain; "
            "set answer['content'] = json.dumps(report) and answer['ready'] = True. "
            "For rolls fill additive totals/maps; for spells fill additive totals, "
            "name maps, and chronological events needed for first/last/list "
            "questions. A sentence like 'I rolled a 20 and a 17. So 19.' means "
            "two rolls, not three. A damage/check total is not a natural value. "
            "Question: " + query
        ),
        "context": {
            "context_window_text": chunk["text"],
            "question": query,
            "mapping": player_to_character,
            "question_type": context.get("question_type"),
            "episode_index": chunk["episode_index"],
            "chunk_index": i,
        },
    }
    for i, chunk in enumerate(chunks)
]
evidence = spawn_subagents(requests)
```

Few-shot 4 - aggregate and submit in the same root execution:
```repl
aggregate = spawn_subagent(
    "[AGGREGATOR JSON CONTRACT] You are read-only: do not call tools or delegate. "
    "Parse only valid JSON child reports from context['reports']; ignore malformed "
    "reports. Reports are disjoint chunks, so sum their additive maps and totals "
    "within each episode before applying the question filter. For cumulative "
    "episode N use episodes 1..N; for across-all use every episode; for a single "
    "episode use that episode. Use by_player/by_character/by_type/by_value and "
    "by_natural_value for roll questions. Use spell by_* maps and ordered events "
    "for spell questions. Do not count prose mentions. Return exactly one repl "
    "block that imports json, sets result = {'candidate': ..., 'used_chunks': [...], "
    "'uncertain': [...]}, then sets answer['content'] = json.dumps(result) and "
    "answer['ready'] = True. The candidate may be an integer or the exact comma "
    "separated string requested by the question. "
    "Question: " + query,
    {
        "question": query,
        "question_type": context.get("question_type"),
        "mapping": player_to_character,
        "reports": evidence,
    },
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
