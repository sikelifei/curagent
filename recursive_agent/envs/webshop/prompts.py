"""Editable WebShop task and environment prompts."""

from __future__ import annotations


DEFAULT_WEBSHOP_AGENT_PROMPT = """WebShop environment guidance:

- The WebShop tools are already registered as REPL globals. Call them directly;
  do not import WebShop, ReCode, or legacy helper modules.
- Call observe() before choosing an action. Use only the current valid_actions.
  Search only when a search action is listed, and copy click targets exactly.
- Extract every hard requirement from the instruction: product type, quantity,
  pack/count, size, color, material, compatibility, and price.
- On a result page, compare visible candidates before clicking. On a product
  page, select every required visible option before clicking Buy Now.
- Open Description or Features only when a required attribute is unclear. Keep
  live environment actions serial; do not repeat a stale action after an error.

Delegation guidance:
When several visible candidates or independent constraints need analysis, the
parent may pass a copied observe() result to spawn_subagents. Children should
analyze that snapshot only, must not call act, and should return:
candidate, matched requirements, missing requirements, evidence, and one
recommended currently valid action. The parent compares the evidence and makes
the next live act() call itself.

Few-shot 1 - parallel candidate analysis:
```repl
state = observe()
checks = spawn_subagents([
    {"task": "Evaluate candidate A against every shopping requirement. "
             "Return matched, missing, evidence, and one valid next action. "
             "Analyze only this snapshot; do not call act.", "context": state},
    {"task": "Evaluate candidate B against every shopping requirement. "
             "Return matched, missing, evidence, and one valid next action. "
             "Analyze only this snapshot; do not call act.", "context": state},
])
print(checks)
```
After comparing the child results, the parent performs exactly one currently
valid act("click[...]"), then observes again.

Few-shot 2 - ordinary navigation:
```text
observe -> search[keywords] -> inspect result candidates -> click[candidate]
-> select required visible options -> click[Buy Now]
```
The exact product names and action labels must always come from the current
observation, not from this example."""

DEFAULT_WEBSHOP_TASK_TEMPLATE = """Complete this WebShop shopping episode.

Shopping instruction:
{instruction}

Use `observe()` to inspect the current page and valid actions. Execute one valid
`act(action)` at a time, print its result, and continue until WebShop reaches a
terminal state after `click[Buy Now]` or the environment step limit. Do not claim
completion before the environment is terminal. The environment guidance also
describes when a child snapshot analysis can help; delegation remains optional."""


def build_webshop_task_prompt(
    instruction: str,
    *,
    template: str = DEFAULT_WEBSHOP_TASK_TEMPLATE,
) -> str:
    instruction = str(instruction).strip()
    if not instruction:
        raise ValueError("WebShop instruction cannot be empty")
    if "{instruction}" not in template:
        raise ValueError("WebShop prompt template must contain {instruction}")
    return template.replace("{instruction}", instruction).strip()
