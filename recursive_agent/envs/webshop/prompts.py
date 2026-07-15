"""Editable WebShop task and environment prompts."""

from __future__ import annotations


DEFAULT_WEBSHOP_AGENT_PROMPT = """WebShop environment guidance:

- The WebShop tools are already registered as REPL globals. Call them directly;
  do not import WebShop, ReCode, or legacy helper modules.
- Call observe() before choosing an action. Use only the current valid_actions.
  `search[keywords]` in valid_actions is a template, not a literal query:
  replace `keywords` with the actual search terms before calling act(). Never
  execute `act("search[keywords]")`; use e.g.
  `act("search[dip powder kit gentle nude]")`. Copy click targets exactly.
- Extract every hard requirement from the instruction: product type, quantity,
  pack/count, size, color, material, compatibility, and price.
- On a result page, compare visible candidates before clicking. On a product
  page, select every required visible option before clicking Buy Now.
- Open Description or Features only when a required attribute is unclear. Keep
  live environment actions serial; after an action error, observe again and do
  not repeat the same stale action. Do not bounce between search, Back, and Next
  without new evidence. At most one alternate search should be tried before
  choosing the best visible candidate.

Delegation guidance:
When several visible candidates or independent constraints need analysis, the
parent may pass a copied observe() result to spawn_subagents. The parent should
say explicitly whether each child is doing snapshot analysis or live environment
operation. For snapshot analysis, the child should not call act and should
return candidate, matched requirements, missing requirements, evidence, and one
recommended currently valid action. For a delegated live operation, a child may
call observe() and act(), but it must check valid_actions immediately before each
action, make one state-changing call at a time, and report the returned state to
the parent. A child should return control after its delegated objective; it
should not spawn further children unless the parent explicitly asks for nested
delegation. Multiple children must not act concurrently on the same session
unless the parent explicitly accepts that risk.

Few-shot 1 - parallel candidate analysis:
```repl
state = observe()
checks = spawn_subagents([
    {"task": "Evaluate candidate A against every shopping requirement. "
             "Return matched, missing, evidence, and one valid next action. "
             "Analyze only this snapshot; this is a read-only delegation.", "context": state},
    {"task": "Evaluate candidate B against every shopping requirement. "
             "Return matched, missing, evidence, and one valid next action. "
             "Analyze only this snapshot; this is a read-only delegation.", "context": state},
])
print(checks)
```
This example is intentionally read-only. In a different delegation, the parent
may authorize a child to operate the live session; that child must follow the
serial observe-check-act-report protocol above.

Few-shot 2 - ordinary navigation:
```text
observe -> act("search[dip powder kit gentle nude]") -> inspect current results
-> act("click[exact visible candidate]") -> select required options
-> act("click[buy now]")
```
The search terms, product names, click labels, and option labels in this example
are illustrative. Always replace them with values from the current observation;
only the literal action pattern is reusable."""

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
