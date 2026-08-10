"""Prompt text shared by every recursive agent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a recursive agent harness. Solve the task in the user
message using direct reasoning, Python execution, and recursive sub-agents when
useful.

Decide whether to:
- solve the task directly;
- delegate one subtask; or
- delegate multiple independent subtasks concurrently.

### Tools

#### Python

Execute Python inside a `repl` block:

```repl
# Python code
```

REPL variables persist within the current agent.

#### Sub-agents

Sub-agents must be called inside a `repl` block.

```repl
spawn_subagent(task, context=None) -> str
```

Run one fresh sub-agent.

```repl
spawn_subagents(requests) -> list[str]
```

Run multiple independent sub-agents concurrently. Each request contains `task`
and optional `context`.

Put complete, self-contained subtask instructions in task. Clearly state what
the sub-agent must inspect, compute, return, and how its result will be merged.

Do not merely repeat the parent task or use vague instructions. Break the work
into a smaller, concrete subtask with the exact output format and all necessary
constraints.

Each agent has an isolated context and REPL environment. Agents cannot access
another agent's variables or intermediate state. Any required information must
be passed explicitly through `task` or `context`.



### Examples

```repl
result = spawn_subagent(
    task="Solve the supplied equation.",
    context={"equation": equation}
)
```

```repl
results = spawn_subagents([
    {
        "task": "Analyze approach A.",
        "context": {"problem": problem}
    },
    {
        "task": "Analyze approach B.",
        "context": {"problem": problem}
    }
])
```
A sub-agent must solve its assigned task rather than delegate an unchanged or
substantially equivalent copy of that task.

Delegate further only when the child can identify a smaller, distinct subtask
whose result is necessary for completing its own assignment.

"""


DEFAULT_ANSWER_COMPLETION_PROMPT = """### Completion

`answer` is the reserved completion dictionary provided by the harness.

Finish by setting:

```repl
answer["content"] = final_text
answer["ready"] = True
```
"""


FORCED_FINAL_USER = """No working steps remain. Return the best final answer now as plain text.
Do not use the REPL, tools, or subagents."""


def build_system_prompt(
    formatted_tools: str | None,
    *,
    prompt_addendum: str | None = None,
    base_prompt: str | None = None,
    completion_prompt: str | None = None,
) -> str:
    sections = [str(base_prompt).strip() if base_prompt else SYSTEM_PROMPT]
    if prompt_addendum:
        sections.append(str(prompt_addendum).strip())
    if formatted_tools:
        sections.append(f"Custom tools:\n{formatted_tools}")
    sections.append(
        str(completion_prompt).strip()
        if completion_prompt is not None
        else DEFAULT_ANSWER_COMPLETION_PROMPT
    )
    return "\n\n".join(section for section in sections if section)


def build_initial_user(
    task: str,
    *,
    delegated: bool = False,
    delegated_guidance: str | None = None,
) -> str:
    if not delegated:
        return f"Task:\n{task}"
    message = (
        f"Delegated task:\n{task}\n\n"
        "This task was supplied by another agent. A private copy of the context "
        "it supplied is available in the REPL variable `context`; it may be None. "
        "The caller's message history and REPL variables are not available unless "
        "they were explicitly included in this task or context. Use the same tools, "
        "REPL, and delegation abilities as any other agent, and return a "
        "self-contained result for the caller."
    )
    if delegated_guidance:
        message = f"{message}\n\n{delegated_guidance.strip()}"
    return message
