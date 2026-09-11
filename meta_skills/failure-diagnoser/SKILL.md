---
name: failure-diagnoser
description: |
  Diagnoses simulation execution failures (wall collisions, trap hazards, oscillation loops).
  Use when analyzing failure traces to generate actionable self-repair feedback for policy synthesis.
  Do NOT use for successful test gating or direct frame observation.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "2.0.0"
  pattern: "meta-workflow"
  inputs:
    - name: test_report
      type: dict
      description: Failure report from contract-tester or environment execution
  outputs:
    - name: diagnostic_summary
      type: str
      description: Root cause summary (collision, oscillation, timeout, exception)
    - name: repair_guidance
      type: str
      description: Targeted feedback prompt for skill-synthesizer self-repair
---

# Failure Diagnoser Meta-Skill

## When to use
- Analyze test failure reports when a policy crashes or fails simulation gating.
- Diagnose wall collision deadlocks, hazard trap collisions, or oscillation loops.
- Formulate prompt-guided self-repair feedback for `skill-synthesizer` iterative retry.

## When NOT to use
- Synthesizing new policies from scratch without failure context (use `skill-synthesizer`).
- Gating passed policies into the verified library (use `contract-tester`).
- Initial visual affordance parsing (use `env-observer`).

## Workflow
1. Failure Log Ingestion: Read execution report `verification` dictionary with `status`, `steps_taken`, and `error`.
2. Root Cause Classification: Categorize into:
   - Obstacle Deadlock: Repeated collisions with same wall cell.
   - Hazard Collision: Negative terminal reward from hazard entry.
   - Oscillation Loop: Alternating movements without spatial displacement.
   - Timeout: Failing to reach goal within step budget.
3. Repair Instruction Generation: Produce concrete algorithmic hints (e.g., add visited history buffer, add orthogonal detour waypoint).

## Examples
- Input: `{"status": "timeout", "steps_taken": 50, "last_pos": (1, 2), "wall": (1, 3)}` → Output: `{"diagnostic_summary": "Wall collision deadlock at (1, 3)", "repair_guidance": "Add orthogonal detour step (UP or DOWN) when wall blocked in heading direction."}`

## Output format
- Dictionary with `diagnostic_summary` and `repair_guidance` string.

## Anti-patterns to avoid
- Do not output vague advice ("try harder"); always specify exact coordinates and corrective maneuver.
- Do not recommend random actions when deterministic detour waypoints are required.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `references/`
- Reference implementations in `src/acr_agi3/agent/evolver.py` and `meta_agent.py`.


