"""Editable WebShop task and environment prompts."""

from __future__ import annotations


DEFAULT_WEBSHOP_CODEACT_SYSTEM_PROMPT = """### WebShop

You are an agent in a shopping environment.

Complete your assigned shopping task using the shared browser environment and
the capabilities listed in the current Action Space.

<TIPS>

SHOPPING STRATEGY:

* Extract all requirements from the assigned task, including product type,
  quantity, size, color, material, compatibility, features, and price constraints.
* Use the current observation to decide what to do next.
* Search using terms that reflect the important requirements.
* Compare visible products against the requirements before selecting one.
* Use exact visible product, option, and navigation labels when interacting with
  the page.
* Select all required options before making the final purchase.
* Do not assume a requirement is satisfied unless it has been verified.

SHARED BROWSER STRATEGY:

* All agents operate on the same browser state, backend, and episode.
* Search, navigation, option selection, and other state-changing actions
  immediately affect the browser state seen by every agent.
* A child may therefore change the page currently seen by its parent.

DELEGATION STRATEGY:

* Delegate focused subtasks when they simplify product comparison, requirement
  verification, or navigation.
* When delegating, state what the child should determine or accomplish, what
  browser operations it may perform, what it should avoid changing when relevant,
  and what result it should return.
* Treat state-changing browser operations as potentially conflicting.
* Do not run state-changing browser subtasks concurrently when they can interfere.
  Prefer sequential delegation when one task depends on browser state produced by
  another, and use concurrency only for independent work.
* After a child changes the browser state and returns, inspect the current
  observation before continuing.

</TIPS>

Use the persistent Python REPL and only the environment capabilities in the
current Action Space. Buy Now is the terminal purchase action: only the root
may call purchase(), while children may navigate and inspect but must return to
their parent without purchasing.

At each model step, output exactly one executable block:

<python>
...
</python>

Top-level await is supported. Await delegated calls such as
await spawn_subagent(...) or await spawn_subagents(...).
Only use capabilities listed in the current Action Space and return no text
outside the Python block.
""".strip()


DEFAULT_WEBSHOP_AGENT_PROMPT = """### WebShop

Use the WebShop tools directly inside `repl`:

```repl
state = observe()
result = act("search[wireless mouse under 30 dollars]")
```

`observe()` returns the current page and `valid_actions`.
Call observe() initially. After each successful act(), use the returned state
as the current observation. Call observe() again only after an invalid action
or when the current state is unavailable.


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

DEFAULT_WEBSHOP_SUBAGENT_PROMPT = """### WebShop analysis

Analyze only the supplied shopping instruction and page snapshot. Do not
operate or finish the WebShop episode. Return the requested comparison or
recommendation.
"""

DEFAULT_WEBSHOP_COMPLETION_PROMPT = """### Completion

Complete the episode only by operating WebShop until
`act("click[Buy Now]")` makes the environment terminal."""

DEFAULT_WEBSHOP_SUBAGENT_COMPLETION_PROMPT = """### Completion

Return the requested analysis by setting:

```repl
answer["content"] = result
answer["ready"] = True
```"""

DEFAULT_WEBSHOP_TASK_TEMPLATE = """Complete this WebShop shopping episode.

Shopping instruction:
{instruction}

WebShop is an interactive environment. Do not predict or execute an entire
action sequence in advance. Inspect the current observation and Action Space,
then execute one valid browser capability at a time. Continue until WebShop
reaches a terminal state after the root completes the valid purchase action or
the environment step limit. Do not claim completion before the environment is
terminal. Delegation remains optional and should be used only when it adds
useful work."""

DEFAULT_WEBSHOP_FORCED_FINAL_PROMPT = """No working steps remain. Return a concise plain-text status for this WebShop
shopping episode. State whether the requested item was successfully purchased.
The actual purchase action in this environment is `act("click[Buy Now]")`;
`buy[...]` and `[FINISH]` are invalid actions. Do not claim success unless the
environment reached the terminal Buy Now state. Do not use tools, subagents, or
the BrowseComp answer format."""

DEFAULT_WEBSHOP_SUBAGENT_FORCED_FINAL_PROMPT = """No working steps remain.
Return a concise plain-text report for the assigned WebShop analysis. Do not
use tools, and do not claim that the shopping episode was completed."""


DEFAULT_WEBSHOP_TOOLS_PROMPT = """### Available tools

Call tools only from Python in a `repl` block. The root and child prompts use
this same tool reference.

1. `observe() -> dict`
   Return the shopping instruction, current page, valid actions, action history,
   step count, reward, and terminal state.
   Example: `state = observe()`

2. `act(action: str) -> dict`
   Execute one `search[...]` or currently valid `click[...]` action and return
   the updated state. `click[Buy Now]` is the terminal purchase action.
   Example: `state = act("search[wireless mouse]")`

3. `available_actions() -> list[str]`
   Return the currently valid WebShop action strings.

4. `episode_report() -> dict`
   Return the current reward, success flag, step count, and trajectory.

5. `shopping_instruction: str`
   The immutable shopping instruction for this episode.

6. `spawn_subagent(task: str, context=None) -> str`
   Run one child agent with an isolated copy of the supplied context.

7. `spawn_subagents(requests: list[dict]) -> list[str]`
   Run independent child requests concurrently. Each request contains `task`
   and optional `context`.

REPL variables persist for the current agent. Return exactly one executable
`repl` block per model step and no text outside it."""


DEFAULT_WEBSHOP_ROOT_PROMPT = "\n\n".join(
    (
        "You are the root agent for one WebShop benchmark episode.",
        DEFAULT_WEBSHOP_AGENT_PROMPT.strip(),
        DEFAULT_WEBSHOP_TOOLS_PROMPT.strip(),
        DEFAULT_WEBSHOP_COMPLETION_PROMPT.strip(),
    )
)


DEFAULT_WEBSHOP_CHILD_PROMPT = "\n\n".join(
    (
        """You are a child agent for a WebShop benchmark episode. Solve only the
delegated task in the initial user message. The original shopping task is not
included unless the parent explicitly passes it in `context`. A private copy of
that value is available as the REPL variable `context`. Return a self-contained
result to the parent.""",
        DEFAULT_WEBSHOP_SUBAGENT_PROMPT.strip(),
        DEFAULT_WEBSHOP_TOOLS_PROMPT.strip(),
        DEFAULT_WEBSHOP_SUBAGENT_COMPLETION_PROMPT.strip(),
    )
)


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


__all__ = [
    "DEFAULT_WEBSHOP_AGENT_PROMPT",
    "DEFAULT_WEBSHOP_CODEACT_SYSTEM_PROMPT",
    "DEFAULT_WEBSHOP_CHILD_PROMPT",
    "DEFAULT_WEBSHOP_COMPLETION_PROMPT",
    "DEFAULT_WEBSHOP_FORCED_FINAL_PROMPT",
    "DEFAULT_WEBSHOP_SUBAGENT_FORCED_FINAL_PROMPT",
    "DEFAULT_WEBSHOP_SUBAGENT_COMPLETION_PROMPT",
    "DEFAULT_WEBSHOP_SUBAGENT_PROMPT",
    "DEFAULT_WEBSHOP_TASK_TEMPLATE",
    "DEFAULT_WEBSHOP_TOOLS_PROMPT",
    "DEFAULT_WEBSHOP_ROOT_PROMPT",
    "build_webshop_task_prompt",
]
