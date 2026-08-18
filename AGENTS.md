# CurAgent Development Workflow

## Primary/worker roles

For substantial implementation work:

- The primary agent is the architect, orchestrator, and reviewer.
- The primary agent should use GPT-5.6 Sol with High reasoning.
- Actual bounded implementation work should be delegated to the `luna_worker`
  custom agent.
- `luna_worker` uses GPT-5.6 Luna with Max reasoning.

## Strict sequential workflow

Implementation must be strictly sequential.

For each implementation task:

1. The primary agent inspects the current repository and decides the next
   smallest coherent implementation task.

2. The primary agent defines that task precisely, including:
   - goal;
   - relevant architecture;
   - likely files/components;
   - required behavior;
   - constraints;
   - non-goals;
   - tests;
   - acceptance criteria.

3. Spawn exactly ONE `luna_worker`.

4. Wait for that worker to complete.

5. The primary agent MUST then inspect the actual implementation itself:
   - inspect changed files;
   - inspect `git diff`;
   - verify behavior against the specification;
   - inspect or run tests;
   - check architectural consistency.

6. Do not accept the worker merely because its textual report says the task
   succeeded.

7. If review finds a problem:
   - define a focused correction task;
   - spawn exactly ONE `luna_worker`;
   - wait for completion;
   - review again.

8. Only after the current implementation task passes review may the primary
   agent define and delegate the next task.

Never run multiple implementation workers concurrently.

## Architecture ownership

Cross-cutting architecture decisions belong to the primary agent.

Workers implement bounded decisions already made by the primary agent.

If a worker discovers that its task requires a major architectural change
outside the delegated scope, it should report the issue rather than redesigning
the system independently.

## Repository constraints

Preserve existing behavior unless the specification explicitly changes it.

Do not modify evaluator, reward, RL, GRPO, Platoon/AReaL integration, or
benchmark scoring when implementing the recursive CodeAct harness unless a
later explicit specification requests those changes.
