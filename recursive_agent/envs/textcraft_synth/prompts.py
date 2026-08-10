"""Short prompts for the TextCraft-Synth environment."""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_TEXTCRAFT_AGENT_PROMPT = """### TextCraft-Synth guidance:

You are an agent in a crafting game. Craft the requested additional items from
the shared inventory. You may need to craft intermediate ingredients first.

If a target item already exists in the inventory, craft the requested quantity
on top of the existing count.

CRAFTING STRATEGY:

- Recipes produce fixed quantities per execution.
- Craft outputs only in valid multiples of `result_count`.
- Scale every ingredient by the number of recipe executions.
- Always verify what you have before claiming something cannot be crafted.
- Check the inventory and recipe information before crafting.
- Reuse existing intermediate items when possible.
- Only use item names returned by the task, inventory, or `get_info()`.

DELEGATION STRATEGY:

- It is highly recommended to delegate crafting of intermediate ingredients.
- Break complex tasks into smaller, independent subtasks.
- For sufficiently complex tasks, recursively delegate; subagents may further
  delegate smaller subtasks.
- Delegate one group of related items at a time, not the whole task at once.
- Independent intermediate items may be delegated in parallel.
- Use `spawn_subagent` for one subtask and `spawn_subagents` for independent
  subtasks.
- Delegated agents share the live inventory, so crafted items are immediately
  available to the parent and other agents.
- Reserve enough work for the current agent to perform the final assembly after
  delegated subtasks complete.
- After delegated work returns, check the live inventory before continuing.
- A delegated task should be smaller than its parent and specify the exact
  intermediate item and additional quantity to craft.
- Do not delegate an unchanged copy of the parent task.

A good delegated task is:

Craft N additional ITEM in the shared inventory. Inspect the inventory and
recipes, craft required intermediates or recursively delegate them when useful,
verify the requested increase, and return immediately.

Use one repl block per response. First briefly state your strategy in 1-3
sentences, then execute the next useful action.
"""


DEFAULT_TEXTCRAFT_SUBAGENT_PROMPT = """### TextCraft-Synth guidance:

Craft only the supplied additional item/count assignment from the shared
inventory.

You may need to craft intermediate ingredients first. For complex assignments,
delegate intermediate crafting to smaller subagents when useful, and recursively
delegate when appropriate.

Recipes produce fixed quantities per execution. Scale ingredients correctly,
reuse existing inventory when possible, and verify the requested increase before
returning.

Do not continue working on the parent's final target after your assigned item is
ready. Return immediately after completing and verifying your assignment.
"""


DEFAULT_TEXTCRAFT_COMPLETION_PROMPT = """### Completion

Before finishing, check the inventory and verify that every target reached its
initial count plus the requested quantity.

End the episode by calling `finish(message)`.
"""


DEFAULT_TEXTCRAFT_SUBAGENT_COMPLETION_PROMPT = """### Completion

Return the work report by setting:

answer["content"] = result
answer["ready"] = True
"""


DEFAULT_TEXTCRAFT_TASK_TEMPLATE = """Craft the following additional items: {targets}"""


DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT = """No working steps remain. Return a
concise plain-text status for the TextCraft task. Do not use tools or sub-agents.

Claim success only if a previous `finish(message)` call returned terminal
success. Otherwise state that the task is incomplete and report the known
remaining targets or blocker.

Do not treat `answer["ready"]`, a plan, a child report, or an unverified
inventory as episode completion.
"""


DEFAULT_TEXTCRAFT_SUBAGENT_FORCED_FINAL_PROMPT = """No working steps remain.
Return a concise plain-text report of the assigned crafting work. Do not use
tools, do not call `finish`, and do not claim completion of the overall task.
"""


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
    "DEFAULT_TEXTCRAFT_COMPLETION_PROMPT",
    "DEFAULT_TEXTCRAFT_FORCED_FINAL_PROMPT",
    "DEFAULT_TEXTCRAFT_SUBAGENT_FORCED_FINAL_PROMPT",
    "DEFAULT_TEXTCRAFT_SUBAGENT_COMPLETION_PROMPT",
    "DEFAULT_TEXTCRAFT_SUBAGENT_PROMPT",
    "DEFAULT_TEXTCRAFT_TASK_TEMPLATE",
    "build_textcraft_task_prompt",
]