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

You are crafting additional requested quantities from a shared inventory. Use
`view_inventory()` and `get_info(items)` before planning. Record the initial
count of each target: the required final count is its initial count plus the
requested amount.

A recipe's output and ingredient counts are per execution. `craft` takes the
total output quantity, which must be divisible by the recipe's `result_count`.
Scale ingredient counts by the required number of recipe executions, rounding
the output quantity up to a valid multiple when necessary.

For crafting depth 0-3, prefer solving directly. For depth 4 or more, if the
target has two or more substantial independent direct branches, delegate those
intermediate item/count pairs using `spawn_subagent` or `spawn_subagents` and
reserve final assembly for yourself. Branches are independent only when they
do not depend on each other or compete for the same limited ingredients.
Otherwise keep the work serial.

Delegated agents share the live inventory and may recurse when their own task
is still deep and branching. A child assigned an intermediate group should
return a report to its caller and should not finish the whole episode.

After delegation, check the inventory again, craft any remaining items, and
assemble the final targets yourself. Call `finish(message)` only after verifying
that every target's final count is at least its initial count plus the requested
amount."""

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
