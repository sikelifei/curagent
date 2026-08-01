"""Editable WebShop task and environment prompts."""

from __future__ import annotations


DEFAULT_WEBSHOP_AGENT_PROMPT = """WebShop environment guidance:

The tools are already registered as REPL globals. Use observe() and act().
Do not import WebShop, ReCode, or legacy helpers.

Your first response must be exactly one executable `repl` block, and every
later working response must also contain exactly one executable `repl` block.
Print the returned value, for example:

```repl
state = observe()
print(state)
```

Do not use a `python` fence or write bare `observe()`/`act(...)` calls in prose.
Wait for the actual observation before selecting the next action.

At the start and after every action or action error, print observe() and use only
its current valid_actions. search[keywords] is a template: replace keywords
with real terms. Copy click targets and option labels exactly. Never issue a
search while the current valid actions are clicks; use click[back to search]
first when it is available.

Extract hard requirements before acting: product type, quantity or pack count,
size or capacity, color, material, compatibility, and price. Treat unspecified
attributes as irrelevant and explicit price limits as hard constraints.

Use this serial loop: observe, choose one currently valid action, act, observe.
On search results, compare visible candidates against every hard requirement. If
no candidate is adequate and click[next >] is valid, inspect one next page. If
the query is weak, return to search and try at most one new query. Do not repeat
an equivalent query, stale action, or analysis. On a product page, inspect
Description or Features only when a required attribute is unclear, select every
required option, and use click[buy now] only when the candidate is acceptable.
The episode is successful only after the environment reports a terminal
purchase.

Choose direct work by default. Delegate only a read-only snapshot comparison
when at least two visible candidates need genuinely separate evaluation and the
report will save more work than it costs. Pass the snapshot, requirements, and
one candidate per child. A child must return candidate, matched, missing,
evidence, and one currently valid next_action; it must not call act(), search,
or delegate. The root owns all live actions, checks the report against the
latest observation, and may continue locally after it returns. Never claim
success from a child report alone."""

DEFAULT_WEBSHOP_TASK_TEMPLATE = """Complete this WebShop shopping episode.

Shopping instruction:
{instruction}

Use observe() to inspect the current page and valid actions. Execute one valid
act(action) at a time, print its result, and continue until WebShop reaches a
terminal state after click[Buy Now] or the environment step limit. Do not claim
completion before the environment is terminal. Delegation is optional and must
be justified by a concrete read-only comparison."""

DEFAULT_WEBSHOP_FORCED_FINAL_PROMPT = """No working steps remain. Return a concise plain-text status for this WebShop
shopping episode. State whether the requested item was successfully purchased.
The actual purchase action is act(\"click[Buy Now]\"); buy[...] and [FINISH] are
invalid actions. Do not claim success unless the environment reached the
terminal Buy Now state. Do not use tools, subagents, or the BrowseComp answer
format."""


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
