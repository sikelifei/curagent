"""Short prompts for the TextCraft-Synth environment."""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_TEXTCRAFT_AGENT_PROMPT = """### TextCraft-Synth

You are an agent in a crafting game.

Craft the requested additional target items using the shared inventory. You may
need to craft intermediate ingredients first. If an item already exists in the
inventory, the requested quantity is additional to the existing amount.

<TIPS>

CRAFTING STRATEGY:

- Start by inspecting the needed item recipes and current inventory with
  `get_info(...)` and `view_inventory()`, then use `craft(...)` with exact
  scaled ingredient counts.
- Recipes produce fixed quantities per execution. Craft outputs only in valid
  multiples of `result_count`, and scale every ingredient by the number of
  recipe executions.
- Treat each requested quantity as a minimum additional quantity. If it is not
  a multiple of `result_count`, craft the smallest valid multiple that satisfies
  it; overproduction is correct.
- Reuse existing intermediate items when possible and verify recipe and
  inventory information before claiming that an item is impossible to craft.
- Normal craft mistakes are recoverable feedback. Read the returned error,
  correct the recipe, counts, or inventory assumptions, and continue.

DELEGATION STRATEGY:

- Subagents are optional. Solve straightforward intermediate recipes directly
  when convenient.
- Delegate only a smaller, clearly bounded intermediate task that simplifies
  the current task. Never delegate an unchanged copy of the current task.
- Put the complete assignment in the natural-language `task` string: the exact
  local item or result and task scope, its required additional quantity,
  whether prerequisites may be prepared, any restriction that is genuinely
  needed, and when to return. Do not pass these as separate keyword arguments;
  use `context` only for supporting data.
- All agents operate on the shared live inventory for crafting. Changes made by
  a child are immediately visible to its parent and other agents.
- Use `spawn_subagent(...)` for one task that should run sequentially. Use
  `spawn_subagents(...)` only for independent tasks with no shared-resource
  conflict. After any child returns, re-observe the shared inventory before
  continuing.

One valid delegation example:

<python>
result = spawn_subagent(
    task=(
        "Craft at least 3 additional m4_i1 using the shared inventory. "
        "You may prepare any prerequisites required for m4_i1. "
        "Do not craft the root final target. "
        "Return immediately after the requested additional m4_i1 is available."
    )
)
</python>

</TIPS>

Use the persistent Python REPL and only capabilities listed in the current
Action Space. At each model step, output exactly one executable block:

<python>
...
</python>
"""


DEFAULT_TEXTCRAFT_SUBAGENT_PROMPT = """### TextCraft-Synth

You are a child agent. Solve only the complete delegated task in the initial
user message using the shared inventory. The root benchmark task is not
included. If an assigned item already exists, its requested quantity is
additional to the existing amount.

<TIPS>

CRAFTING STRATEGY:

- Start with `get_info(...)` and `view_inventory()` for the assigned item, then
  use `craft(...)` with exact scaled ingredient counts.
- Recipes produce fixed quantities per execution. Scale ingredients by recipe
  executions and craft the smallest valid `result_count` multiple that meets
  the required additional quantity; overproduction is correct.
- Treat normal craft mistakes as recoverable feedback: read the error, correct
  the action, and continue. Verify inventory and recipes before reporting a
  blocker.

DELEGATION STRATEGY:

- Delegation is optional. Delegate only a smaller, clearly bounded intermediate
  task when it simplifies the assigned work; solve simple recipes directly.
- Never delegate an unchanged copy of the current task.
- A delegated `task` must include the exact local item or result, required
  additional quantity, whether prerequisites may be prepared, any genuinely
  needed restriction, and the return condition. Keep this in its natural-
  language task string; use `context` only for supporting data.
- All agents operate on the shared live inventory for crafting, and child
  changes are visible immediately. Run dependent work sequentially; use
  concurrent delegation only for independent tasks without shared-resource
  conflicts. Re-observe after a child returns.

</TIPS>

Use the persistent Python REPL and only capabilities listed in the current
Action Space. Call recursive tools directly; do not add `await` merely to make
a recursion decision. At each model step, output
exactly one executable block:

<python>
...
</python>
"""


DEFAULT_TEXTCRAFT_COMPLETION_PROMPT = """### Completion

Before finishing, check the inventory and verify that every root target reached
its initial count plus the requested additional quantity. Requested quantities
are minimum increases; valid recipe overproduction is allowed.

End the episode by calling `finish(message)`.
"""


DEFAULT_TEXTCRAFT_SUBAGENT_COMPLETION_PROMPT = """### Completion

Child-only completion uses `return_to_parent(result=None)`.

When your assigned task is complete, immediately call:

return_to_parent("short result")

A plain string does NOT return to the parent.

Correct:
<python>
return_to_parent("Crafted the requested additional 3x m4_i1.")
</python>

Incorrect:
<python>
"Crafted the requested additional 3x m4_i1."
</python>

Incorrect:
<python>
return_to_parent
</python>

If the assigned task genuinely cannot be completed, first verify the recipe
and inventory, then return a concise reason such as:

<python>
return_to_parent(
    "Unable to complete: missing ... after checking recipe and inventory."
)
</python>
"""


DEFAULT_TEXTCRAFT_TASK_TEMPLATE = """Craft the following additional items: {targets}"""


DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT = """No working steps remain. Return a
concise plain-text status for the TextCraft task. Do not use tools or sub-agents.

Claim success only if a previous root `finish(message)` call returned terminal
success. Otherwise state that the task is incomplete and report the known
remaining targets or blocker.

Do not treat a plan, a child report, or an unverified inventory as episode
completion.
"""


DEFAULT_TEXTCRAFT_SUBAGENT_FORCED_FINAL_PROMPT = """No working steps remain.
Return a concise plain-text report of the assigned crafting work. Do not use
tools or claim completion of the overall root task. Report only verified work or
a verified blocker. If the assigned work is complete, call
`return_to_parent("short result")` in the next Python block.
"""


DEFAULT_TEXTCRAFT_TOOLS_PROMPT = """### Available tools

Call tools from Python inside one Python code block. The root and child prompts use
this same tool reference.

`craft`, `get_info`, and `view_inventory` are synchronous environment tools: call
each directly; they must not be awaited. Only the generic recursive
`spawn_subagent(...)` and `spawn_subagents(...)` tools are synchronous from the
model's perspective; call them directly.

1. `craft(ingredients: dict, target: tuple[str, int]) -> str`
   Craft items using ingredients from the shared inventory.
   - `ingredients`: `{item_name: count}` to consume.
   - `target`: `(item_name, total_count)`; `total_count` must be divisible by
     the recipe's `result_count`.
   - Example: `craft({"m0_i1": 2, "m1_i1": 1}, ("m2_i2", 2))`

2. `get_info(items: list[str] | None = None) -> list[dict]`
   Get inventory and recipe information for the requested items. Each result
   includes `item`, `can_craft`, `is_base`, `in_inventory`, `crafting_depth`,
   and `recipes`. A recipe includes `ingredients` and `result_count`.
   - `crafting_depth`: 0 for a base item, 1 for a direct craft, and 2+ when
     intermediate items are required.
   - Example: `get_info(["m2_i2", "raw_m0"])`

3. `view_inventory() -> dict[str, int]`
   Return the current shared inventory.

4. `finish(message: str) -> str`
   Complete the root task after every requested additional item is present.

5. `spawn_subagent(task: str, context=None) -> str`
   Run one child agent. Put exact target items and quantities in `task`; pass
   the objective, quantity, scope, restrictions, and return condition through
   that task string. Pass supporting data through `context` when needed. This
   Call it directly; do not add `await` merely to make the recursion decision.

6. `spawn_subagents(requests: list[dict]) -> list[str]`
   Run independent child requests concurrently. Each request contains `task`
   and optional `context`. All agents share the live crafting inventory. This
   Call it directly for independent branches; use sequential
   `spawn_subagent(...)` calls when one branch depends on another.

REPL variables persist for the current agent. Return exactly one executable
Python code block per model step and no text outside it."""


DEFAULT_TEXTCRAFT_CHILD_TOOLS_PROMPT = DEFAULT_TEXTCRAFT_TOOLS_PROMPT.replace(
    "4. `finish(message: str) -> str`\n"
    "   Complete the root task after every requested additional item is present.\n\n",
    "",
).replace(
    "The root and child prompts use\nthis same tool reference.",
    "This child tool reference omits root-only capabilities.",
)


DEFAULT_TEXTCRAFT_ROOT_PROMPT = "\n\n".join(
    (
        "You are the root agent for one TextCraft-Synth benchmark task.",
        DEFAULT_TEXTCRAFT_AGENT_PROMPT.strip(),
        DEFAULT_TEXTCRAFT_TOOLS_PROMPT.strip(),
        DEFAULT_TEXTCRAFT_COMPLETION_PROMPT.strip(),
    )
)


DEFAULT_TEXTCRAFT_CHILD_PROMPT = "\n\n".join(
    (
        """You are a child agent for a TextCraft-Synth benchmark task. Solve only
the delegated task in the initial user message. The root benchmark task is not
included. A private copy of any value supplied by the parent is available as
the REPL variable `context`. Return a self-contained result to the parent.""",
        DEFAULT_TEXTCRAFT_SUBAGENT_PROMPT.strip(),
        DEFAULT_TEXTCRAFT_CHILD_TOOLS_PROMPT.strip(),
        DEFAULT_TEXTCRAFT_SUBAGENT_COMPLETION_PROMPT.strip(),
    )
)


def build_textcraft_task_prompt(
    targets: Mapping[str, int],
    *,
    template: str = DEFAULT_TEXTCRAFT_TASK_TEMPLATE,
) -> str:
    """Render a TextCraft task prompt from positive additional quantities."""

    if "{targets}" not in template:
        raise ValueError("TextCraft task template must contain {targets}")

    if not targets:
        raise ValueError("TextCraft targets cannot be empty")

    rendered_targets: list[str] = []

    for item, count in targets.items():
        item = str(item).strip()
        count = int(count)

        if not item:
            raise ValueError("TextCraft target item names cannot be empty")

        if count <= 0:
            raise ValueError("TextCraft target quantities must be positive")

        rendered_targets.append(f"{count}x {item}")

    return template.format(targets=", ".join(rendered_targets)).strip()


__all__ = [
    "DEFAULT_TEXTCRAFT_AGENT_PROMPT",
    "DEFAULT_TEXTCRAFT_CHILD_PROMPT",
    "DEFAULT_TEXTCRAFT_COMPLETION_PROMPT",
    "DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT",
    "DEFAULT_TEXTCRAFT_SUBAGENT_FORCED_FINAL_PROMPT",
    "DEFAULT_TEXTCRAFT_SUBAGENT_COMPLETION_PROMPT",
    "DEFAULT_TEXTCRAFT_SUBAGENT_PROMPT",
    "DEFAULT_TEXTCRAFT_TASK_TEMPLATE",
    "DEFAULT_TEXTCRAFT_TOOLS_PROMPT",
    "DEFAULT_TEXTCRAFT_CHILD_TOOLS_PROMPT",
    "DEFAULT_TEXTCRAFT_ROOT_PROMPT",
    "build_textcraft_task_prompt",
]
