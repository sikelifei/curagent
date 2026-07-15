"""Editable dataset task prompt for WebShop episodes."""

from __future__ import annotations

DEFAULT_WEBSHOP_TASK_TEMPLATE = """Complete this WebShop shopping episode.

Shopping instruction:
{instruction}

Use `observe()` to inspect the current page and valid actions. Execute one valid
`act(action)` at a time, print its result, and continue until WebShop reaches a
terminal state after `click[Buy Now]` or the environment step limit. Search
actions may use new keywords; click actions must exactly match a currently
listed clickable element. Do not claim completion before the environment is
terminal.

All agents share the same live WebShop session. Do not let concurrent subagents
call `act`, because their actions would interleave. Subagents may independently
analyze copied observations or candidate products."""


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
