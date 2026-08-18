Read the current CurAgent repository and the existing TextCraft implementation carefully before making changes.

You are the primary architect, experiment designer, and reviewer.

Use `luna_worker` (GPT-5.6 Luna Max) for implementation work.

Work strictly sequentially:

```text
Sol High
→ inspect / design
→ define ONE implementation task
→ ONE Luna Max implements
→ Sol inspects actual diff
→ Sol reviews tests
→ fix with ONE Luna if needed
→ accept
→ run experiment
→ analyze results
```

Never run multiple Luna workers concurrently.

---

# Goal

Run a controlled TextCraft-Synth sampling experiment to determine:

1. whether the recursive CodeAct harness actually works;
2. whether Qwen3-4B-Instruct-2507 can solve TextCraft medium tasks under a shared global model-step budget;
3. what global step budget is reasonable;
4. whether the model uses CodeAct efficiently;
5. whether the model voluntarily delegates to subagents;
6. whether the current prompt is the main reason recursion is not used.

This is an inference/sampling experiment only.

Do NOT modify:

* evaluator semantics;
* reward;
* RL;
* GRPO;
* advantage calculation;
* Platoon/AReaL training;
* benchmark scoring.

Do not tune the evaluator to increase success.

---

# Model Endpoint

Use the already-running OpenAI-compatible endpoint:

```text
http://192.168.1.134:56782/v1
```

First query:

```bash
curl -s http://192.168.1.134:56782/v1/models
```

Use the exact model ID returned by the endpoint.

Expected underlying model:

```text
/home/zhangwenjian/model/Qwen3-4B-Instruct-2507
```

Do not restart the vLLM server unless explicitly requested.

---

# Critical Dataset Requirement

Before running experiments, inspect the TextCraft-Synth dataset loader and configuration.

The previous validation silently used generated fallback tasks because the configured dataset root was missing.

That is NOT acceptable for this experiment.

MUST use the real intended TextCraft-Synth validation dataset.

Use:

```text
split = val
difficulty = medium
seed = 42
```

If the real TextCraft-Synth dataset cannot be located:

STOP.

Report:

```text
DATASET_NOT_FOUND
```

including:

* paths inspected;
* expected dataset path;
* relevant configuration;
* how the current loader falls back.

Do NOT use generated fallback tasks for the main experiment.

---

# Before Editing

Sol must inspect the current implementation and identify:

* TextCraft-Synth loader;
* TextCraft environment;
* current TextCraft system prompt;
* CodeAct parser/executor;
* AgentNode;
* SharedBudget;
* RecursiveScheduler;
* `spawn_subagent`;
* `spawn_subagents`;
* root `finish`;
* child `return_to_parent`;
* rollout/inference runner;
* existing logging/trajectory format.

Determine whether existing infrastructure can run the experiments without code changes.

Only delegate implementation if a small experiment runner, configuration, or additional diagnostics are actually needed.

Do not redesign the harness during this experiment.

---

# Implementation Workflow

If experiment support needs code changes:

## Sol

Define exactly ONE bounded implementation task.

Examples:

```text
Add a validation runner that:
- selects fixed TextCraft-Synth tasks;
- runs configurable global budgets;
- writes one JSONL row immediately after every rollout;
- preserves raw traces;
- reports recursion and termination statistics.
```

Then spawn exactly ONE `luna_worker`.

## Luna

Luna implements only that task.

It should not change agent semantics.

## Sol Review

After Luna finishes, Sol MUST:

```text
git diff
```

and inspect the changed files directly.

Verify that:

* no evaluator/reward/RL semantics changed;
* no generated fallback dataset is being used;
* global budget semantics remain unchanged;
* the runner persists each completed rollout immediately;
* the experiment is reproducible.

Run focused tests.

If incorrect, assign ONE focused correction task to Luna Max and review again.

Only then begin experiments.

---

# Experiment 0 — Endpoint Sanity

Confirm:

```text
/v1/models
/v1/chat/completions
```

work.

Send a small CodeAct request requiring:

```text
<python>
x = 40
print(x + 2)
</python>
```

Confirm:

* HTTP request succeeds;
* content is non-empty;
* `<python>...</python>` is followed.

Do not continue if endpoint/model communication is broken.

---

# Experiment 1 — Forced Recursion Harness Smoke Test

Purpose:

> Separate "recursion implementation is broken" from "the model chooses not to recurse."

Select ONE real medium TextCraft-Synth task that contains a non-trivial intermediate dependency.

For this experiment only, augment the task instruction with an explicit temporary validation requirement equivalent to:

```text
You must delegate one coherent intermediate crafting subtask to exactly one
subagent before completing the root task.

Give the subagent a clear crafting objective and return condition.
After the subagent returns, inspect the shared environment and continue the
root task.
```

This is a harness integration test, not a benchmark result.

Use:

```text
temperature = 0.0
global max_steps = 64
max_depth = current configured value
1 task
1 rollout
```

Verify from the actual trace:

```text
root
→ spawn_subagent
→ child receives delegated task
→ child uses the exact same Environment object
→ child model generations consume the same SharedBudget
→ child modifies shared environment
→ child calls return_to_parent
→ parent resumes
→ parent observes child changes
```

Also verify:

```text
child does not have finish()
root does not have return_to_parent()
```

Result must be classified as:

```text
RECURSION_PATH_PASS
```

or:

```text
RECURSION_PATH_FAIL
```

If this fails because of harness/runtime behavior, stop all later experiments and diagnose the harness.

---

# Experiment 2 — Global Budget Sweep

Purpose:

> Determine whether the previous 25-step failure is primarily caused by an undersized shared global budget.

Use FIVE fixed real TextCraft-Synth medium tasks.

Select once with:

```text
split = val
difficulty = medium
seed = 42
```

Save the exact task IDs.

Reuse exactly these task IDs for every budget.

Use:

```text
temperature = 0.0
1 rollout per task
```

Test:

```text
max_total_steps = 32
max_total_steps = 64
max_total_steps = 96
max_total_steps = 128
```

Total:

```text
5 tasks × 4 budgets = 20 rollouts
```

Do NOT modify the prompt between budget conditions.

For each rollout record:

```text
task_id
budget
success
termination_reason
global_steps_used
number_of_agents
maximum_depth
spawn_subagent_count
spawn_subagents_count
get_info_count
craft_count
finish_called
parse_errors
runtime_errors
```

Also record the final missing targets for failures when available.

Produce a table:

```text
budget | successes/5 | mean steps | recursion used | budget exhausted
32
64
96
128
```

Interpretation:

* If success rises strongly with budget, budget is a primary bottleneck.
* If success remains near zero even at 128, inspect policy/prompt efficiency.
* Do not automatically choose the largest budget.

Identify the smallest budget where success begins to plateau.

Call this:

```text
candidate_global_budget
```

---

# Experiment 3 — CodeAct Efficiency Inspection

Use trajectories from Experiment 2, especially the candidate budget.

Measure how the model is actually using CodeAct.

For each rollout calculate:

```text
LLM generations
environment function calls
get_info calls
craft calls
average environment calls per generated Python block
```

Inspect whether the model behaves like:

```text
generation 1 → get_info only
generation 2 → get_info only
generation 3 → craft only
...
```

or whether it uses CodeAct as intended:

```text
one generation
→ several related queries/calculations/actions
```

Manually inspect at least:

```text
2 successful trajectories if available
2 failed trajectories
```

Classify CodeAct utilization as:

```text
GOOD
UNDERUTILIZED
PATHOLOGICAL
```

Do not change the prompt yet.

---

# Experiment 4 — Baseline Autonomous Recursion

Using:

```text
candidate_global_budget
```

run the same five tasks with:

```text
temperature = 0.0
1 rollout per task
normal current TextCraft prompt
```

No forced recursion instruction.

Measure:

```text
successes / 5
rollouts using recursion
spawn counts
maximum depth
mean global steps
```

If recursion usage is still:

```text
0/5
```

do not conclude the harness is broken if Experiment 1 passed.

Conclude instead that autonomous delegation is not being selected under the current prompt/model.

---

# Experiment 5 — Minimal Prompt Intervention

Only perform this experiment if:

```text
Experiment 1 recursion path passed
AND
Experiment 4 showed little or no autonomous recursion
```

Create ONE minimal TextCraft prompt variant.

Do not rewrite the whole prompt.

Add only the following strategy concepts:

```text
CODEACT STRATEGY

- A Python block may perform multiple environment operations.
- Batch related information queries when possible.
- Keep useful recipe information and calculations in the persistent REPL.
- Once several crafting operations are known to be valid, they may be
  executed in the same Python block.

DELEGATION STRATEGY

- For deep targets with multiple intermediate dependencies, prefer assigning
  one coherent intermediate branch to a subagent instead of expanding every
  branch yourself.
- Give the child an exact objective, quantity, scope, restrictions, and return
  condition.
- All agents operate on the same shared environment.
- Use concurrent children only when their work is independent and cannot
  conflict in the shared environment.
- Otherwise delegate sequentially.
- After a child returns, inspect the shared state before continuing.
```

Do NOT add hard rules such as:

```text
depth >= 3 must recurse
```

The model should still decide whether recursion is appropriate.

Run the same five fixed tasks:

```text
temperature = 0.0
candidate_global_budget
1 rollout per task
```

Compare:

```text
CURRENT PROMPT
vs
MINIMAL PROMPT VARIANT
```

on:

```text
success
recursion rate
global steps
environment actions per generation
budget exhaustion
```

---

# Experiment 6 — RL-Style Sampling

Only run this after the earlier experiments establish:

```text
endpoint works
real dataset works
recursion path works
reasonable budget is known
prompt choice is fixed
```

Use the same FIVE task IDs.

Run:

```text
8 rollouts per task
temperature = 1.0
top_p = 1.0
candidate_global_budget
```

Total:

```text
40 rollouts
```

Do not run these in a fragile foreground process that loses results at session termination.

The runner MUST append results after every completed rollout.

Use an execution method suitable for long-running work, for example a normal shell/tmux workflow if available.

For each task report:

```text
successes / 8
failures / 8
```

Classify:

```text
0/8     → ALL_FAIL
1-7/8   → MIXED
8/8     → ALL_SUCCESS
```

Report:

```text
ALL_FAIL groups
MIXED groups
ALL_SUCCESS groups
```

MIXED groups are especially important because they provide group-relative RL signal.

---

# Required Output Files

Store everything under a dedicated directory such as:

```text
validation_results/textcraft_sampling_experiment/
```

Include:

```text
config.json
task_ids.json

experiment_1_forced_recursion.jsonl
experiment_2_budget_sweep.jsonl
experiment_4_baseline_recursion.jsonl
experiment_5_prompt_variant.jsonl
experiment_6_rl_sampling.jsonl

raw_trajectories/

report.md
```

Each completed rollout must be written immediately.

Do not wait until an entire experiment finishes before persisting results.

---

# Final Report

`report.md` must include:

## Environment

```text
repository commit
endpoint
served model ID
real TextCraft dataset path
split
difficulty
fixed task IDs
```

## Experiment 1

```text
forced recursion path: PASS / FAIL
shared environment verified: yes/no
shared global budget verified: yes/no
child return verified: yes/no
```

## Experiment 2

```text
budget | success | mean steps | recursion | exhausted
32
64
96
128

candidate_global_budget = ...
```

## Experiment 3

```text
CodeAct utilization:
GOOD / UNDERUTILIZED / PATHOLOGICAL

evidence:
...
```

## Experiment 4

```text
baseline autonomous recursion:
X/5 used recursion
X/5 succeeded
```

## Experiment 5

```text
current prompt:
success X/5
recursion X/5

minimal variant:
success X/5
recursion X/5
```

## Experiment 6

```text
40 requested
40 completed

overall success: X/40

ALL_FAIL: X/5
MIXED: X/5
ALL_SUCCESS: X/5
```

---

# Final Diagnosis

Choose one or more evidence-backed conclusions:

```text
HARNESS_RECURSION_PROBLEM
GLOBAL_BUDGET_TOO_SMALL
CODEACT_UNDERUTILIZED
DELEGATION_PROMPT_TOO_WEAK
MODEL_CAPABILITY_LIMIT
SAMPLING_HEALTHY_FOR_RL
DATASET_CONFIGURATION_PROBLEM
INCONCLUSIVE
```

Do not infer a cause without trace evidence.

---

# Worker Policy

Sol remains responsible for:

```text
architecture understanding
experimental validity
task selection
review
result interpretation
```

Luna Max is used only for bounded implementation tasks such as:

```text
adding experiment runner support
adding diagnostics
adding prompt variant config
fixing experiment-script bugs
```

For every Luna task:

```text
ONE Luna
→ wait
→ Sol reviews actual diff
→ tests
→ accept/fix
→ next task
```

Never delegate the overall experimental interpretation to Luna.

Do not stop after producing an experiment plan.

Carry out the implementation, review, experiments, and final report in order.
