"""Short prompts for the TextCraft-Synth environment."""

from __future__ import annotations

# DEFAULT_TEXTCRAFT_AGENT_PROMPT = """TextCraft-Synth guidance:

# You are crafting the requested items from a shared inventory. Use
# `view_inventory()` and `get_info(items)` before planning. A recipe's output and
# ingredient counts are per execution; `craft` takes the exact output quantity,
# which must be divisible by that recipe's result_count. Existing target items
# count toward the goal, and the requested amount is added on top of them.

# Solve directly when the dependency tree is small. For a deeper tree, delegate
# an independent intermediate-item group to `spawn_subagent` or
# `spawn_subagents`; delegated agents use the same live inventory, so parallelize
# only groups that do not compete for the same ingredients. Children may recurse
# when their own dependency tree is still substantial. Check the inventory after
# delegation, assemble remaining targets yourself, and call `finish(message)`
# only after verifying every requested target is present. A child that only
# crafted an intermediate group should return a report to its caller; it should
# not finish the whole episode.

# Use this threshold: for depth 0-3, prefer solving directly. For depth 4 or
# more, when the target has two or more independent direct branches, first
# delegate those branch item/count pairs and reserve the final assembly for
# yourself. Do not expand every branch in the root when those children are
# independent. Otherwise keep the work serial."""

# DEFAULT_TEXTCRAFT_TASK_TEMPLATE = """Craft the following items: {targets}

# Use the registered crafting tools to complete the task. The environment is
# finished only after you call `finish(...)`."""

# DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT = """Return a concise status for the crafting task. Do not use tools or subagents. Do not claim success unless all requested items were crafted."""
# DEFAULT_TEXTCRAFT_DELEGATED_FORCED_FINAL_PROMPT = """Return a concise report of the delegated crafting work. Do not use tools or subagents. State what you completed or could not complete."""

DEFAULT_TEXTCRAFT_AGENT_PROMPT = """TextCraft-Synth guidance:

You are crafting additional requested quantities from one shared inventory.
Begin with `view_inventory()` and `get_info()` for the target items. Record the
initial count of each target; success requires initial count plus requested
count.

Your first response must be exactly one executable `repl` block such as:

```repl
inventory = view_inventory()
info = get_info()
print(inventory)
print(info)
```

Every later working response must use one `repl` block. Print every tool
return value. Never infer an empty result from an unprinted return value, never
use guessed ingredients, and never use a `python` fence or bare tool call.

Plan the dependency graph before crafting. Recipe ingredient counts are per
execution, and `craft` requires exact scaled counts with output divisible by
`result_count`. Round output up only when extra output is allowed. Prefer the
shortest verified dependency path.

Keep planning short: after the initial inventory and target recipe inspection,
inspect only the missing dependency recipes needed for the plan. Then execute
verified independent crafts in one `repl` block with several exact
`print(craft(...))` calls when possible. Do not spend one model turn per craft;
reserve at least two turns for final inventory verification and `finish`.

Choose direct work when the next dependency chain is short, sequential, or
shares scarce ingredients. Delegate a distinct intermediate branch only when
the expected benefit exceeds coordination cost. The request must assign exact
item counts and exclusive ingredients, state whether the child should only plan
or may craft, and reserve final assembly for the root. Use parallel children
only for branches that neither depend on each other nor compete for ingredients.
Pass the branch target, current inventory snapshot, relevant recipes, and
exclusions.

A planning child returns required item counts, exact recipe calls, ingredient
needs, blockers, and confidence without mutating inventory. A child explicitly
assigned exclusive live crafting may craft only that branch, must verify the
result, and must not call `finish`. It may recurse only by assigning a smaller
non-overlapping branch, never the same branch unchanged. The root owns final
assembly and the completion decision.

After a child report, re-read the inventory and verify the plan against current
resources. Execute independent exact crafts in a compact batch, then inspect
the inventory and update remaining targets. If a batch reports an error, trust
the latest inventory, call `get_info()` again, and do not repeat an argument
that was already rejected or a craft that already succeeded.

Reserve steps for final assembly and verification. Call `finish(message)` only
after final inventory confirms every requested target count. If verified
recipes and resources cannot complete a target, return the verified partial
state instead of looping."""

DEFAULT_TEXTCRAFT_TASK_TEMPLATE = """Craft the following additional items: {targets}

Use the registered crafting tools to complete the task. The environment is
finished only after you call `finish(...)`."""

DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT = """Return a concise status for the
crafting task. Do not use tools or subagents. Do not claim success unless all
requested additional items were verified in the final inventory."""

DEFAULT_TEXTCRAFT_DELEGATED_FORCED_FINAL_PROMPT = """Return a concise report of
the delegated crafting work. Do not use tools or subagents. State what you
completed or could not complete, and do not claim completion of the whole
episode."""


def build_textcraft_task_prompt(
    targets: dict[str, int],
    *,
    template: str = DEFAULT_TEXTCRAFT_TASK_TEMPLATE,
) -> str:
    if "{targets}" not in template:
        raise ValueError("TextCraft task template must contain {targets}")
    rendered = ", ".join(
        f"{int(count)}x {item}" for item, count in targets.items()
    )
    return template.format(targets=rendered)


__all__ = [
    "DEFAULT_TEXTCRAFT_AGENT_PROMPT",
    "DEFAULT_TEXTCRAFT_DELEGATED_FORCED_FINAL_PROMPT",
    "DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT",
    "DEFAULT_TEXTCRAFT_TASK_TEMPLATE",
    "build_textcraft_task_prompt",
]
