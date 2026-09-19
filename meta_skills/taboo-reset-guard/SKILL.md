---
name: taboo-reset-guard
description: |
  Failure attribution, taboo constraint learning, and active reset guard engine for ARC-AGI-3.
  Monitors pixel stagnation, wall bumps, and 2-step oscillation loops to prune bad actions
  and trigger active level reset when caught in irreversible deadlocks.
  Do NOT use for initial affordance extraction or geometric path generation.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - filter_taboo_actions
    - evaluate_active_reset
  inputs:
    - name: proposed_action
      type: int
      description: Action ID proposed by decision model
    - name: stagnation_count
      type: int
      description: Number of consecutive steps with zero pixel change
    - name: action_history
      type: list[int]
      description: Sequence of recently executed action IDs
    - name: available_actions
      type: list[int]
      description: Set of valid action IDs in current environment
  outputs:
    - name: is_allowed
      type: bool
      description: True if proposed action satisfies all taboo constraints
    - name: sanitized_action
      type: int
      description: Filtered action or safe exploratory alternative
    - name: should_reset
      type: bool
      description: True if active reset is recommended due to unrecoverable deadlock
---

# Taboo Reset Guard

## When to use
- When an agent repeatedly bumps into walls (e.g. 60+ steps of `ACTION1` with zero pixel change).
- When detecting 2-step or 3-step oscillation loops (e.g. `LEFT`, `RIGHT`, `LEFT`, `RIGHT`).
- When evaluating whether a game board is stuck in an irreversible deadlock requiring an active `RESET`.
- To dynamically prune ineffective actions from the choice set.

## When NOT to use
- When state transitions are progressing smoothly with positive pixel changes and level advances.
- For generating original path waypoints (use `backward-planner`).

## Workflow
1. **Stagnation & Bump Detection**:
   - If the previous action resulted in 0 changed pixels, designate that action ID as Taboo for the current step.
2. **Oscillation & Loop Detection**:
   - Check the last 4 to 6 actions in `action_history`. If an alternating cycle (A, B, A, B) is detected, veto action A.
3. **Alternative Action Selection**:
   - If the proposed action is vetoed, select the highest-priority untried action from `available_actions`.
4. **Active Reset Trigger**:
   - If consecutive stagnation exceeds threshold (default: 6 steps) or loop count exceeds 3, issue `RESET` (Action 0).

## Examples
- Example 1 (Wall bump veto):
  - Proposed: `ACTION1`, but last step `ACTION1` produced 0 pixel change.
  - Result: `is_allowed: false`, `sanitized_action: ACTION2` (alternative direction).
- Example 2 (Active reset on deadlock):
  - Stagnation count: 7 consecutive steps without state change.
  - Result: `should_reset: true`, recommending `GameAction.RESET`.

## Anti-patterns to avoid
- Do not keep selecting the same action when it continuously produces 0 pixel change.
- Do not fear calling `RESET`; human players actively reset to prune bad states.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy, standard library (collections, argparse, json, typing)

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/taboo_reset_guard.py`: Taboo filter, oscillation detector, and active reset evaluator CLI.
