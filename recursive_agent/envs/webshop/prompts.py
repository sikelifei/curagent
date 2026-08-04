"""Editable WebShop task and environment prompts."""

from __future__ import annotations


DEFAULT_WEBSHOP_AGENT_PROMPT = """### WebShop

Use the WebShop tools directly inside `repl`:

```repl
state = observe()
result = act("search[wireless mouse under 30 dollars]")
```

`observe()` returns the current page and `valid_actions`.
Use the latest state returned by observe() or act() before choosing an action.
Call observe() initially, after an invalid action, or when the current state is
missing or stale.


Replace example values with exact terms and labels from the current observation.
Never execute template text such as `search[keywords]`.

Extract all requirements from the shopping instruction, such as product type,
quantity, size, color, material, compatibility, and price. Compare visible
products, select all required options, and finish with `click[Buy Now]`.

Execute only one state-changing action at a time. After an invalid action, call
`observe()` again.

### Sub-agents

The WebShop environment cannot be operated concurrently. Call at most one
sub-agent at a time and do not use `spawn_subagents`.

Use a sub-agent for long product comparisons, requirement checking, or analysis
that benefits from a fresh context.

```repl
state = observe()

analysis = spawn_subagent(
    task=(
        "Compare the visible products against every shopping requirement. "
        "Do not operate the environment. Return the best candidate, evidence, "
        "missing requirements, and one recommended valid action."
    ),
    context={
        "shopping_instruction": instruction,
        "observation": state,
    },
)
```

The sub-agent analyzes only the supplied snapshot and must not call `act()`.
After it returns, call `observe()` again and continue operating the environment
serially.

Few-shot: every block below is a separate model step. `act()` returns the new
page state. Print it and stop; the next step reads that state before acting.

```repl
state = observe()
print(state)
```

```repl
state = act("search[argan oil paraben free 2 oz]")
print(state)
```

The returned state shows `click[b08h5slqf1]`, a paraben-free argan oil under
$40. Use that exact current action.

```repl
state = act("click[b08h5slqf1]")
print(state)
```

The returned product page shows scent `argan oil`, size
`2 fl oz (pack of 2)`, price $5.95, and their exact actions.

```repl
state = act("click[argan oil]")
print(state)
```

```repl
state = act("click[2 fl oz (pack of 2)]")
print(state)
```

```repl
state = act("click[buy now]")
print(state)
```


"""

DEFAULT_WEBSHOP_TASK_TEMPLATE = """Complete this WebShop shopping episode.

Shopping instruction:
{instruction}

WebShop is an interactive environment. Do not predict or execute an entire
action sequence in advance. Use `print(observe())` to inspect the current page and valid actions. Execute one valid
`act(action)` at a time, print its result, and continue until WebShop reaches a
terminal state after `click[Buy Now]` or the environment step limit. Do not claim
completion before the environment is terminal. Delegation remains optional and
should be used only when it adds useful work."""

DEFAULT_WEBSHOP_FORCED_FINAL_PROMPT = """No working steps remain. Return a concise plain-text status for this WebShop
shopping episode. State whether the requested item was successfully purchased.
The actual purchase action in this environment is `act("click[Buy Now]")`;
`buy[...]` and `[FINISH]` are invalid actions. Do not claim success unless the
environment reached the terminal Buy Now state. Do not use tools, subagents, or
the BrowseComp answer format."""


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
