# CurAgent Multi-Environment Recursive CodeAct Harness — Implementation Spec

## 0. Goal

Refactor CurAgent so that **TextCraft, WebShop, Oolong-Synth and future environments share the same recursive CodeAct harness**.

The harness should contain only generic recursive execution logic.

Environment-specific behavior must be provided by each environment through:

```text
environment-specific system prompt
environment-specific observation
environment-specific CodeAct capabilities
environment-specific task/context
```

The target architecture is:

```text
                    Generic CurAgent Harness
                            │
              ┌─────────────┼─────────────┐
              │             │             │
          TextCraft       WebShop      Oolong-Synth
              │             │             │
         prompt/tools   prompt/tools   prompt/tools
              │             │             │
              └─────────────┼─────────────┘
                            │
                      AgentNode Loop
                            │
                     <python>...</python>
```

Do not add benchmark-specific branches such as:

```python
if env_name == "textcraft":
    ...
elif env_name == "webshop":
    ...
```

inside `AgentNode`.

---

# 1. Core Recursive Semantics

A recursive task uses exactly **one Environment instance**.

```text
                         Shared Environment
                                │
               ┌────────────────┼────────────────┐
               │                │                │
             Root            Child A          Child B
          original task    delegated task   delegated task
          private REPL     private REPL     private REPL
               │                │
               │            Grandchild
               │
               └────────────────┼────────────────┘
                                │
                         SharedBudget
```

Required invariant:

```python
root.environment is child.environment
child.environment is grandchild.environment
```

Do not:

```text
clone environment
fork environment
copy environment
create child-local environment
```

Recursion happens in the **agent**, not in the environment.

---

# 2. Shared vs Node-Local State

## Shared across the recursive tree

```text
Environment instance
SharedBudget
model backend
scheduler
global trace recorder
```

## Local to each AgentNode

```text
agent_id
parent_id
depth

task
context

persistent Python namespace
node-local conversation / trajectory
```

Environment mutations are visible to every node.

Python variables are not.

Example:

```python
# root REPL
x = 10
```

The child does not automatically know `x`.

However, if the child modifies shared environment state, root immediately sees that modification.

---

# 3. Different Environments Share the Harness

The generic harness MUST NOT know TextCraft, WebShop or Oolong-specific rules.

Instead, each environment supplies three things:

```text
1. system_prompt
2. observation
3. CodeAct capabilities
```

Conceptually:

```python
class Environment:

    def system_prompt(self) -> str:
        ...

    async def observe(self) -> Any:
        ...

    def codeact_namespace(
        self,
        *,
        is_root: bool,
        depth: int,
    ) -> dict[str, Any]:
        ...
```

Exact method names may differ.

The important requirement is that the harness can dynamically obtain:

```text
task strategy
current observation
available environment operations
```

without benchmark-specific logic.

---

# 4. Common CodeAct Protocol

All environments use the same execution protocol.

At each LLM generation the model outputs:

```text
<python>
...
</python>
```

Do not require:

```text
<thought>
```

One generation may contain multiple Python operations.

Example:

```python
info = get_info(["A"])
print(info)

inventory = view_inventory()
print(inventory)

if need_help:
    result = await spawn_subagent(
        task="Prepare intermediate A and return when complete."
    )
    print(result)
```

The entire code block consumes:

```text
1 LLM step
```

Any model generations performed by the child consume additional shared steps.

---

# 5. Persistent REPL

Every AgentNode owns an independent persistent Python namespace.

The namespace persists across that node's own steps.

Example:

```python
# step 1
records = []
```

Then:

```python
# step 2
records.append(...)
```

must work.

Different nodes have different namespaces.

Normal Python functionality should be available, including imports such as:

```python
import math
import json
import asyncio

from collections import defaultdict
```

Do not unnecessarily restrict the REPL.

Top-level `await` MUST work.

---

# 6. Shared Global Step Budget

The entire recursive tree shares one global step counter.

```python
AgentLimits(
    max_total_steps=N,
    max_depth=D,
)
```

Do not introduce per-agent LLM budgets.

Definition:

> One successful LLM generation consumes exactly one step.

Not:

```text
one Python statement
one environment operation
one tool call
one recursion call
```

Example:

```text
root generation        +1
  child generation     +1
  child generation     +1
root Python resumes    +0
root next generation   +1

total                   4
```

---

# 7. Root Task vs Subagent Task

Root receives the original benchmark task.

Subagent receives the task explicitly assigned by its parent.

The harness must NOT automatically append the original root task to every child.

Example:

```python
await spawn_subagent(
    task="""
Prepare 4 units of intermediate A.
Do not assemble the final target.
Return after A is available.
"""
)
```

The child only receives this task plus explicitly supplied context.

---

# 8. Delegation Is a Policy Decision

The harness must not automatically partition or isolate tasks.

When delegating, the parent agent is responsible for clearly defining:

```text
what the child should accomplish
what scope it should operate on
what it should not modify
important constraints
what result should be returned
when the child should return
```

A delegated task should be self-contained.

Bad:

```python
await spawn_subagent(
    task="Do the first part."
)
```

Good:

```python
await spawn_subagent(
    task="""
Prepare component A and its required intermediates.

Only work on component A.
Do not assemble the final target.
Do not consume materials reserved for component B.

Return when A is ready and report the produced quantity.
"""
)
```

The same rule applies recursively to every subagent.

---

# 9. Shared Environment and Logical Isolation

The harness MUST NOT create automatic task isolation by using:

```text
environment copies
inventory partitions
state namespaces
transactional child worlds
resource partitions
```

Logical isolation comes from delegation instructions.

Therefore:

```text
same environment
+
different clearly-scoped tasks
```

is the intended recursive model.

---

# 10. Concurrent Delegation

Public recursive APIs:

```python
await spawn_subagent(
    task: str,
    context: Any = None,
)
```

and:

```python
await spawn_subagents(
    requests: list[dict],
)
```

`spawn_subagent()` means sequential delegation.

`spawn_subagents()` means the parent intentionally wants independent children to execute concurrently.

All concurrent children still receive:

```python
the exact same Environment instance
the same SharedBudget
```

No child environment copy is allowed.

---

# 11. Conflict Detection Belongs to the Parent

Before using:

```python
spawn_subagents(...)
```

the parent should decide whether subtasks can safely operate concurrently.

The parent should consider:

```text
Do they modify the same state?

Do they consume the same resources?

Could both perform the same operation?

Does one depend on the result of another?

Can one invalidate assumptions made by another?

Is execution order important?
```

If there is meaningful conflict or dependency, the parent should use sequential delegation:

```python
a = await spawn_subagent(...)
b = await spawn_subagent(...)
```

The harness MUST NOT add:

```text
automatic conflict judges
automatic dependency detection
automatic semantic locks
automatic serialization
LLM-based conflict detection
```

Environment implementations may use normal synchronization primitives for atomic state mutations, but semantic independence remains the agent's responsibility.

---

# 12. Recursive Guidance in Every Environment Prompt

Every environment-specific system prompt should communicate the same recursive principle:

```text
When delegating:
- clearly specify the subtask;
- specify relevant scope and restrictions;
- specify what should be returned.

Use concurrent subagents only when their work can safely proceed
against the shared environment without harmful conflicts or ordering dependencies.
Otherwise delegate sequentially.
```

The exact wording may be adapted to the environment.

---

# 13. Root and Subagent Termination

Only root may submit the final benchmark result.

Root gets:

```python
finish(result=None)
```

Subagents MUST NOT get `finish`.

Subagents instead get:

```python
return_to_parent(result=None)
```

Therefore:

```text
Root:
    environment capabilities
    spawn_subagent
    spawn_subagents
    finish

Subagent:
    environment capabilities
    spawn_subagent
    spawn_subagents
    return_to_parent
```

At `max_depth`, recursive spawn functions disappear.

---

# 14. Root-Only finish()

`finish()` means:

> Submit the final result of the original root task and terminate root execution.

Required:

```python
"finish" in root_namespace
"finish" not in child_namespace
```

This is not merely a prompt instruction.

Child must genuinely not have the callable.

---

# 15. Subagent-Only return_to_parent()

`return_to_parent()` means:

> Finish the delegated local task, return a local result to the direct parent, and terminate this subagent.

Required:

```python
"return_to_parent" in child_namespace
"return_to_parent" not in root_namespace
```

A child may modify the shared environment before returning.

Its returned value is for:

```text
summary
local result
statistics
status
evidence
metadata
```

not for synchronizing environment state.

---

# 16. Dynamic Action Space

The Action Space must be generated dynamically from capabilities actually bound into the current node REPL.

Use a single capability collection as the source of truth:

```text
Capabilities
     │
     ├── bind into REPL
     │
     └── render into prompt
```

Do not separately maintain:

```text
prompt tools
runtime tools
```

because they may diverge.

---

# 17. Environment-Specific Capabilities

Different environments naturally expose different functions.

Example:

## TextCraft

```text
get_info(...)
view_inventory(...)
craft(...)
```

## WebShop

```text
observe(...)
search(...)
click(...)
...
```

## Oolong-Synth

May expose very few environment functions because the main work happens through:

```text
context
Python
subagents
```

The harness should not assume all environments have identical APIs.

---

# 18. Same Environment Does Not Require Same Node Permissions

Sharing the same Environment object does NOT mean root and child must see exactly the same callable set.

An environment may expose capabilities based on node role.

Example WebShop:

```text
Root:
    search
    click
    purchase/final-action

Child:
    search
    click
    no final purchase capability
```

This prevents a child from bypassing root-only final submission semantics through an environment-specific terminal action.

Therefore environment capability construction may receive:

```text
is_root
depth
```

or equivalent node metadata.

However, the underlying Environment instance remains exactly the same.

---

# 19. Environment Prompt Interface

Each environment should provide a task-specific system prompt.

The system prompt describes:

```text
what environment/task this is
task-specific hard rules
task-specific solving strategy
task-specific delegation strategy
```

The generic harness then supplies dynamically:

```text
current task
context
observation
remaining global steps
action space
```

Do not hardcode these values into static environment prompts.

---

# 20. Prompt Composition

Conceptual composition:

```text
SYSTEM
--------------------
environment.system_prompt()


USER
--------------------
# Task

{node.task}


# Context

{node.context}


# Observation

{current_environment_observation}


# Remaining Steps

{shared_budget_remaining}


# Action Space

{dynamic_codeact_capabilities}
```

Subsequent turns should additionally include the node's own CodeAct execution history.

---

# 21. TextCraft System Prompt

Use the following baseline.

### TextCraft

You are an agent in a crafting environment.

Craft the requested additional target items using the shared inventory. You may need to prepare intermediate items first.

<TIPS>

CRAFTING STRATEGY:

* Recipes produce fixed quantities per execution.
* Scale ingredients by the number of recipe executions.
* Requested target quantities are additional to the inventory that already exists.
* Check inventory and recipe information before deciding what must be crafted.
* Reuse existing intermediate items when possible.
* Verify quantities carefully before performing crafting actions.

DELEGATION STRATEGY:

* Delegate intermediate crafting when it simplifies a complex task.
* Break independent crafting branches into clear subtasks.
* When delegating, state exactly what should be crafted, relevant restrictions, what resources or scope the child should avoid when necessary, and when it should return.
* All agents operate on the same shared crafting environment and inventory.
* Changes made by a subagent are immediately visible to its parent and other agents.
* Before running multiple subagents concurrently, ensure their work does not have harmful resource conflicts, duplicate operations, or ordering dependencies.
* If subtasks depend on one another or may conflict, run them sequentially.
* After delegated work returns, inspect the shared state before continuing final assembly.

</TIPS>

Use the persistent Python REPL and the available environment capabilities.

At each model step, output exactly one executable block:

<python>
...
</python>

Only use capabilities listed in the current action space.

---

# 22. WebShop System Prompt

Use the following baseline.

### WebShop

You are an agent in a shopping environment.

Complete your assigned shopping task using the shared browser environment.

<TIPS>

SHOPPING STRATEGY:

* Extract all requirements from the assigned task, including product type, quantity, size, color, material, compatibility, features, and price constraints.
* Use the current observation to decide what to do next.
* Search using terms that reflect the important requirements.
* Compare visible products against the requirements before selecting one.
* Use exact visible product, option, and navigation labels when interacting with the page.
* Select all required options before making the final purchase.
* Do not assume a requirement is satisfied unless it has been verified.

SHARED BROWSER STRATEGY:

* All agents operate on the same browser state.
* Search, navigation, option selection, and other state-changing actions immediately affect the browser state seen by every agent.
* A subagent may therefore change the page currently seen by its parent.

DELEGATION STRATEGY:

* Delegate focused subtasks when they simplify product comparison, requirement verification, or navigation.
* When delegating, clearly state what the child should determine or accomplish, what browser operations it may perform, what it should avoid changing when relevant, and what result should be returned.
* Treat state-changing browser operations as potentially conflicting.
* Do not run state-changing browser subtasks concurrently when they can interfere with each other.
* Use concurrent subagents only when their work is independent in the shared browser environment.
* If a task depends on browser state produced by another task, run them sequentially.
* After a child changes the browser state and returns, inspect the current observation before continuing.

</TIPS>

Use the persistent Python REPL and the available browser capabilities.

At each model step, output exactly one executable block:

<python>
...
</python>

Only use capabilities listed in the current action space.

---

# 23. WebShop Root/Child Final-Action Rule

A WebShop environment may contain an action that itself commits the final purchase.

If an environment-specific action is semantically equivalent to final benchmark submission, it should be root-only.

Therefore the environment capability layer should support:

```text
root-visible capabilities
child-visible capabilities
```

while both nodes still share the exact same browser Environment object.

Example:

```text
Root:
    search
    click
    final purchase action

Child:
    search
    click
    no final purchase action
```

This prevents child agents from bypassing root-only `finish()` semantics.

Exact WebShop purchase behavior can be finalized during environment integration.

---

# 24. Oolong-Synth System Prompt

Use the following baseline.

### Oolong-Synth

You are an agent for long-context aggregation tasks.

Solve the assigned question using only the information in your assigned context.

The context may contain dataset instructions, source records, chunk metadata, or other task-specific information passed to this agent.

<TIPS>

CONTEXT STRATEGY:

* Process the assigned records according to the task instructions.

* Classify records using semantic meaning rather than keyword shortcuts or guessed labels.

* Process every complete assigned record needed for the result.

* Use Python for counting, aggregation, bookkeeping, and combining intermediate results.

* If the assigned source text is at most 65,536 characters, solve the assigned task directly.

* If the assigned source text is larger, split it at complete record boundaries into ordered, non-overlapping chunks and delegate the chunks.

* Every source record must belong to exactly one delegated chunk.

DELEGATION STRATEGY:

* Delegate independent chunks when the assigned context is too large to process directly.

* Pass the actual data chunk through the child context instead of duplicating large text inside the task instruction.

* Tell every child exactly:

  * what question or local computation it should solve;
  * which assigned records it is responsible for;
  * what values or statistics it should return.

* Request mergeable intermediate information such as counts, sums, sample sizes, numerator/denominator pairs, group statistics, or local extrema when needed for the final result.

* Do not request only a local winner if the underlying values are required for global comparison.

* Disjoint read-only chunks may normally be processed concurrently.

* If one computation depends on another, run them sequentially.

* Merge child results correctly: add counts and sums, combine numerators and denominators, compare extrema, and merge corresponding groups.

* Never average local averages unless mathematically valid.

</TIPS>

Use the persistent Python REPL to solve the assigned task.

At each model step, output exactly one executable block:

<python>
...
</python>

When delegating, make each child task self-contained and state clearly what it should return.

Only use capabilities listed in the current action space.

---

# 25. Oolong Context Semantics

Oolong data chunks are NOT separate environments.

The recursive structure should be:

```text
one shared Oolong Environment
          │
          ├── root
          │     context = full assigned source
          │
          ├── child A
          │     context = chunk A
          │
          └── child B
                context = chunk B
```

The child context carries its data assignment.

Environment identity remains unchanged.

This is important because the generic harness should not equate:

```text
different child context
```

with:

```text
different environment
```

---

# 26. Oolong Delegation Example

Root may execute:

```python
results = await spawn_subagents([
    {
        "task": """
Process every complete record in the assigned chunk.
Answer the original question locally and return the counts,
sums, denominators, or other statistics needed for global merging.
""",
        "context": {
            "question": question,
            "dataset_intro": dataset_intro,
            "source": chunk_a,
            "chunk_id": 0,
        },
    },
    {
        "task": """
Process every complete record in the assigned chunk.
Answer the original question locally and return the counts,
sums, denominators, or other statistics needed for global merging.
""",
        "context": {
            "question": question,
            "dataset_intro": dataset_intro,
            "source": chunk_b,
            "chunk_id": 1,
        },
    },
])
```

These are safe concurrent tasks because each agent operates on a disjoint read-only chunk.

---

# 27. TextCraft Delegation Example

Parallel delegation is allowed only after parent conflict analysis.

Example:

```python
results = await spawn_subagents([
    {
        "task": """
Craft 4 units of component A and its required intermediates.

Only work on A's branch.
Do not assemble the final target.
Avoid consuming materials reserved for component B.

Return when A is ready and report the produced quantity.
"""
    },
    {
        "task": """
Craft 3 units of component B and its required intermediates.

Only work on B's branch.
Do not assemble the final target.
Avoid consuming materials reserved for component A.

Return when B is ready and report the produced quantity.
"""
    },
])
```

Both children use the same inventory.

If the branches actually compete for required resources, the parent should not use concurrent delegation.

---

# 28. WebShop Delegation Example

Because browser state is shared, sequential delegation should be preferred for page-mutating subtasks.

Example:

```python
comparison = await spawn_subagent(
    task="""
Inspect the currently visible products and determine which candidates
satisfy the required compatibility and price constraints.

You may inspect product details if necessary.
Do not make the final purchase.
Return the best matching candidates and the requirements each satisfies.
"""
)
```

After the child returns, root should inspect the current shared browser state before acting again.

Concurrent WebShop delegation should only be used when the tasks genuinely do not conflict in the shared browser environment.

---

# 29. Generic AgentNode Must Remain Environment-Agnostic

`AgentNode.run()` should conceptually do only:

```python
while True:

    observation = await environment.observe()

    capabilities = environment capabilities
    capabilities += recursive framework capabilities
    capabilities += root_or_child_termination

    prompt = compose_prompt(
        system_prompt=environment.system_prompt(),
        task=node.task,
        context=node.context,
        observation=observation,
        remaining_steps=shared_budget.remaining,
        capabilities=capabilities,
        history=node.history,
    )

    model_output = await model.generate(...)

    code = extract_python(model_output)

    result = await node.repl.execute(code)

    record(result)

    if node terminated:
        return
```

There should be no task-specific policy inside this loop.

---

# 30. Environment Interface Requirements

The exact implementation is flexible, but every environment integration must be able to provide the equivalent of:

```python
class Environment:

    def system_prompt(self) -> str:
        ...

    async def observe(self) -> Any:
        ...

    def codeact_capabilities(
        self,
        *,
        is_root: bool,
        depth: int,
    ) -> Mapping[str, Any]:
        ...

    def codeact_descriptions(
        self,
        *,
        is_root: bool,
        depth: int,
    ) -> str:
        ...
```

Prefer deriving descriptions and runtime bindings from one underlying capability specification rather than maintaining duplicate lists.

---

# 31. Framework Capabilities

The harness adds framework functions after obtaining environment capabilities.

Conceptually:

```python
capabilities = environment.capabilities(node_role)

if depth < max_depth:
    capabilities["spawn_subagent"] = ...
    capabilities["spawn_subagents"] = ...

if is_root:
    capabilities["finish"] = ...
else:
    capabilities["return_to_parent"] = ...
```

The same capability set is used to:

```text
render Action Space
bind REPL namespace
```

---

# 32. Prompt Ownership

Use the following responsibility split.

## Environment owns

```text
task identity
task-specific strategy
task-specific delegation guidance
environment capability descriptions
observation formatting
```

## Harness owns

```text
CodeAct parsing/execution
persistent REPL
shared global step budget
recursive scheduling
node hierarchy
root finish
child return_to_parent
dynamic capability binding
```

## Parent policy owns

```text
subtask decomposition
delegated task wording
scope restrictions
conflict analysis
sequential vs concurrent choice
information placed in child context
```

This separation is a MUST.

---

# 33. Adding a New Benchmark

A new benchmark should ideally require only:

```text
1. implement Environment
2. provide system prompt
3. expose CodeAct capabilities
4. create root task/context
5. implement benchmark evaluator separately
```

It should NOT require modifying:

```text
AgentNode
recursive scheduler semantics
CodeAct parser
shared budget
root/child control flow
```

This is the main extensibility acceptance criterion.

---

# 34. Acceptance Criteria

The architecture is correct when:

```text
[ ] TextCraft, WebShop and Oolong-Synth use the same AgentNode implementation.

[ ] The harness contains no benchmark-name-specific branches.

[ ] Each recursive tree uses one exact Environment instance.

[ ] Different child tasks/contexts do not require different environments.

[ ] Each environment supplies its own system prompt.

[ ] Each environment supplies its own observation.

[ ] Each environment supplies its own CodeAct capabilities.

[ ] Root and subagents may receive different capability sets while still sharing the same Environment object.

[ ] Parent agents are responsible for clear delegation boundaries.

[ ] Parent agents are responsible for deciding whether concurrent work conflicts.

[ ] The harness does not automatically partition environment state.

[ ] Oolong chunks are node-local context, not child environments.

[ ] WebShop browser state is shared across root and children.

[ ] TextCraft inventory is shared across root and children.

[ ] CodeAct protocol is identical across all environments.

[ ] One global LLM-step budget is shared across the recursive tree.

[ ] Root alone owns final submission.

[ ] Subagents only return local results to their parent.

[ ] Adding a new environment does not require modifying AgentNode.
```

---

# 35. Final Architectural Principle

The design should follow this separation:

```text
                    RECURSIVE HARNESS
                           │
                           │
             knows HOW agents recursively run
                           │
                           ▼
                    ENVIRONMENT
                           │
                           │
              knows WHAT task/world this is
                           │
                           ▼
                      PARENT POLICY
                           │
                           │
             decides HOW the task is decomposed
```

In short:

> **Harness defines recursion. Environment defines the task. Parent agents define the decomposition.**

And:

> **Different tasks should reuse the same recursive mechanism without forcing different environments for different subagents. A recursive child changes its assignment, not its world.**
