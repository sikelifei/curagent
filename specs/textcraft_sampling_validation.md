# TextCraft Sampling Validation Spec

## Goal

Validate the current CurAgent + TextCraft rollout path using the already-running remote vLLM endpoint.

This is a **sampling validation task only**.

Do not train anything.

Do not modify:

* RL code
* reward logic
* evaluator semantics
* GRPO
* advantage calculation
* Platoon/AReaL integration
* dataset contents

Do not redesign the harness during this task.

The purpose is to determine whether the current TextCraft agent can generate valid recursive trajectories with Qwen3-4B-Instruct-2507 and to characterize why failed rollouts fail.

---

# 1. vLLM Endpoint

Use:

```text
http://192.168.1.134:56782/v1
```

The server was started from:

```text
/home/zhangwenjian/model/Qwen3-4B-Instruct-2507
```

Do not assume the served model name.

First query:

```bash
curl -s http://192.168.1.134:56782/v1/models
```

Read the returned model ID and use that exact ID for subsequent OpenAI-compatible requests.

Do not restart vLLM unless the endpoint is unreachable and the user explicitly asks you to do so.

---

# 2. Endpoint Sanity Check

Before running TextCraft, verify:

```text
/v1/models works
/v1/chat/completions works
model returns non-empty assistant content
```

Send one small CodeAct-format test request.

System instruction:

```text
Output exactly one executable Python block using:

<python>
...
</python>

Do not output a thought block.
```

User task:

```text
Set x to 40 and print x + 2.
```

Expected qualitative result:

```text
<python>
x = 40
print(x + 2)
</python>
```

The exact code does not need to match.

Record:

```text
endpoint reachable
served model ID
HTTP/model errors if any
whether <python> format is followed
```

If this step fails, stop TextCraft sampling and report the endpoint/model problem.

---

# 3. Inspect Existing TextCraft Integration

Before writing new code, inspect the repository.

Find:

```text
TextCraft environment
TextCraft task/dataset loader
TextCraft prompt
CurAgent rollout entry point
existing inference/evaluation scripts
existing sampling configuration
```

Prefer existing runnable infrastructure.

Do not create a new runner if a suitable existing runner already exists.

If this repository is being used together with the Platoon TextCraft plugin, inspect the available TextCraft-Synth inference entry points as well.

The validation should use the current CurAgent harness being tested, not silently substitute another agent implementation.

If both CurAgent and original Platoon implementations are available, clearly identify which one is being executed.

---

# 4. Dataset

Use TextCraft-Synth.

Start with:

```text
split: val
difficulty: medium
```

Use a fixed random seed:

```text
seed = 42
```

The same selected task IDs must be reusable in later validation.

Record the exact task IDs.

Do not silently change tasks between validation phases.

---

# 5. Phase A — Two-Task Smoke Test

Run:

```text
2 medium tasks
1 rollout per task
```

Use deterministic or near-deterministic inference for the first smoke test:

```text
temperature = 0.0
```

Use the current harness step/depth configuration.

Do not tune the limits based on outcomes during this phase.

The purpose is only to verify that the full path works:

```text
dataset
→ prompt
→ model
→ <python>
→ CodeAct execution
→ environment
→ recursion if selected
→ termination
→ evaluator/report
```

For both trajectories, preserve the complete trace.

Inspect them manually.

Check:

```text
Did the model produce valid <python> blocks?

Did Python execute?

Were environment functions callable?

Did persistent REPL state work?

If spawn_subagent was called:
    did the child receive its delegated task?
    did it use the same Environment instance?
    did it consume the shared global step budget?
    did it return through return_to_parent?

Could only root call finish?

Were parse/runtime errors returned as feedback instead of crashing the rollout?
```

If there is a harness/runtime failure, stop before the larger run and report it.

---

# 6. Phase B — Small RL-Style Sampling Test

If Phase A works, select:

```text
5 medium TextCraft-Synth tasks
```

using:

```text
seed = 42
```

Run:

```text
8 rollouts per task
```

Total:

```text
5 × 8 = 40 trajectories
```

This is intended to approximate the sampling pattern later used by grouped RL training.

Do not train on these trajectories.

Use sampling parameters close to the intended RL rollout parameters.

If the repository already defines the intended TextCraft RL sampling temperature/top-p, reuse those values.

If no such configuration exists, use:

```text
temperature = 1.0
top_p = 1.0
```

Do not silently invent additional sampling parameters.

Record the exact values used.

---

# 7. Shared Global Step Budget

For CurAgent, verify the intended semantics:

```text
root and all descendants share one global LLM-generation budget.
```

Count:

```text
one successful LLM generation = one step
```

Do NOT count individual Python/environment function calls as model steps.

For every rollout record:

```text
total global model steps
number of agent nodes
maximum recursion depth reached
```

If possible also record model generations per node for diagnostics, but do not introduce per-agent budgets.

---

# 8. Recursion Statistics

For every trajectory collect:

```text
number of spawn_subagent calls
number of spawn_subagents calls
total created subagents
maximum recursion depth
whether recursion was used at all
```

For concurrent delegation, note whether:

```text
the parent provided separate task scopes
children shared the same Environment object
```

Do not judge recursion quality with an LLM.

Only collect structural statistics and manually inspect representative trajectories.

---

# 9. Termination Statistics

Separate the following outcomes.

Root:

```text
finish called
finish not called
global budget exhausted
environment terminated
runtime failure
model/API failure
```

Subagents:

```text
return_to_parent called
budget exhausted before return
runtime failure
model/API failure
```

Do not collapse every failure into only:

```text
reward = 0
```

The purpose of this validation is to understand why reward is zero.

---

# 10. Error Statistics

Collect counts for at least:

```text
missing <python> block
Python syntax error
Python runtime error
unknown function / NameError
environment action error
spawn error
max-depth rejection
global-budget exhaustion
model API error
timeout
```

Preserve representative traces for each observed failure type.

---

# 11. Success Statistics

Use the existing evaluator unchanged.

For the five-task / eight-rollout experiment report per task:

```text
task_id
successes / 8
failures / 8
recursion usage
mean global steps
min global steps
max global steps
```

Then report overall:

```text
total trajectories = 40
successful trajectories
failed trajectories
success rate

tasks with:
    0/8 success
    1-7/8 mixed success
    8/8 success
```

The mixed groups are particularly important because they indicate useful RL sampling variance.

---

# 12. Sampling Quality Classification

Classify each of the five GRPO-style task groups into:

```text
ALL_FAIL
MIXED
ALL_SUCCESS
```

Example:

```text
task A: 0/8  -> ALL_FAIL
task B: 3/8  -> MIXED
task C: 8/8  -> ALL_SUCCESS
```

Report:

```text
number of ALL_FAIL groups
number of MIXED groups
number of ALL_SUCCESS groups
```

Do not modify rewards or filtering behavior.

This is only an observation of the sampled distribution.

---

# 13. Representative Trace Review

Manually inspect at least:

```text
2 successful trajectories
2 failed trajectories
```

If recursion occurs, prefer examples containing recursion.

For each representative trajectory summarize:

```text
task
root strategy
delegation decision
child task wording
whether child respected its scope
whether environment changes were visible to parent
termination behavior
reason for success/failure
```

Do not use an LLM judge.

Base the analysis on the actual trace.

---

# 14. Output Artifacts

Create a validation directory such as:

```text
validation_results/textcraft_qwen3_4b/
```

Preserve:

```text
selected task IDs
run configuration
raw trajectory/event traces
per-rollout summary
aggregate summary
```

Create:

```text
validation_results/textcraft_qwen3_4b/report.md
```

The report should contain:

```text
1. Endpoint
2. Model
3. Repository commit
4. Harness implementation being tested
5. Dataset / split / difficulty
6. Sampling parameters
7. Step/depth limits
8. Phase A result
9. Phase B aggregate result
10. Per-task 8-rollout results
11. Recursion statistics
12. Failure-reason distribution
13. Representative success traces
14. Representative failure traces
15. Final assessment
```

---

# 15. Final Assessment

Conclude with one of:

```text
READY_FOR_LARGER_SAMPLING

HARNESS_PROBLEM

MODEL_CAPABILITY_PROBLEM

PROMPT_PROBLEM

ENVIRONMENT_INTEGRATION_PROBLEM

INCONCLUSIVE
```

Explain the evidence.

Do not change code just to improve the result.

The purpose of this run is diagnosis.

---

# 16. Optional Phase C

Do NOT run this automatically.

Only recommend it in the final report if Phase B is healthy.

Possible next run:

```text
20 medium tasks
8 rollouts per task
160 trajectories
```

The user will decide whether to run it.

---

# 17. Restrictions

MUST NOT:

```text
train the model
change rewards
change evaluator semantics
change GRPO logic
change task data
lower task difficulty just to improve success
change prompts midway through the same comparison
increase budgets after seeing failures
silently skip failed rollouts
```

If the validation requires a small runner or logging utility that does not currently exist, it MAY be added under a clearly validation-specific location such as:

```text
scripts/validation/
```

or:

```text
validation/
```

Do not refactor the harness as part of this task.

---

# 18. Final Deliverable

At the end print a concise summary containing:

```text
Endpoint: ...
Model: ...
Commit: ...

Smoke:
2 tasks / 2 rollouts
success: X/2

Sampling:
5 tasks × 8 = 40 rollouts
success: X/40

Group distribution:
ALL_FAIL: X/5
MIXED: X/5
ALL_SUCCESS: X/5

Recursion used: X/40
Mean global steps: ...
Parse/runtime failures: ...

Top failure reasons:
1. ...
2. ...
3. ...

Assessment:
...
```

Also give the path to the complete validation report and raw trajectories.
