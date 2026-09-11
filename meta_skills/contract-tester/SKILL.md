---
name: contract-tester
description: |
  Executes contract tests and simulation gating for action policies to guarantee safety.
  Use when verifying generated action policies on simulation environments before deployment.
  Do NOT use for observing raw frames or generating python code blocks.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "2.0.0"
  pattern: "meta-workflow"
  inputs:
    - name: skill_name
      type: str
      description: Name of skill to verify
    - name: test_env
      type: GameEnvironment
      description: Game environment simulator instance
  outputs:
    - name: is_solved
      type: bool
      description: Whether the policy cleared the stage within limit
    - name: test_report
      type: dict
      description: Execution details (steps taken, reward, history)
---

# Contract Tester Meta-Skill

## When to use
- Verify newly synthesized game action policies against positive and negative contract tests.
- Run deterministic multi-step game simulation loops to detect collisions, oscillation loops, or timeouts.
- Gatekeeper validation before registering skills into the permanent verified skill library.

## When NOT to use
- Observing initial visual game frames (use `env-observer`).
- Writing or refactoring policy code (use `skill-synthesizer`).
- Deep failure root-cause traceback extraction (use `failure-diagnoser`).

## Workflow
1. Environment Reset: Reset simulation environment and initialize policy globals.
2. Step Loop Execution: Step through policy with closed-loop observations up to `max_steps`:
   ```python
   verification = execute_and_verify_game_policy(code, env, max_steps=50)
   ```
3. Pass/Fail Decision: Check that `verification["success"]` is true with zero hazard/wall collisions.
4. Routing: If passed, hand off to library registration; if failed, forward report to `failure-diagnoser`.

## Examples
- Input: `test_mover` on 1D environment → Output: `{"is_solved": True, "steps_taken": 2, "final_reward": 1.0}`

## Output format
- Structured verification dictionary: `{"is_solved": bool, "success": bool, "steps_taken": int, "final_reward": float, "history": list}`.

## Anti-patterns to avoid
- Never approve policies that pass with negative reward or unresolved collision events.
- Do not bypass negative boundary test cases (wall collision, hazard collision).

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `references/`
- Reference implementations in `src/acr_agi3/agent/llm/arc_tools.py` and `edd_tools.py`.


