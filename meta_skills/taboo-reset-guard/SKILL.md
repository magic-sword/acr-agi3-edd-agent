---
name: taboo-reset-guard
description: |
  Comprehensive failure recovery and safety engine: diagnoses failures, learns taboo constraints, and enforces active resets.
  Identifies deadlocks, cyclic loops, wall collisions, and lethal hazards, maintaining a persistent taboo registry across episodes.
  Use whenever actions become ineffective, state trajectory loops, lethal hazards occur, or deciding whether to reset the episode.
  Do NOT use during smooth progressive gameplay without obstacles or collision signals.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: recent_trajectory
      type: optional[list[dict]]
      description: Sequence of recent state transitions and action outcomes
    - name: consecutive_ineffective_actions
      type: optional[int]
      description: Number of consecutive steps where state remained unchanged
    - name: game_over_occurred
      type: optional[bool]
      description: Whether game-over or lethal trap was triggered
    - name: error_trace
      type: optional[str]
      description: Runtime error or exception trace if an action failed
  outputs:
    - name: is_reset_recommended
      type: bool
      description: Whether the agent should perform an Active Reset
    - name: status
      type: str
      description: Failure classification (OK, DEADLOCK, OSCILLATION_LOOP, COLLISION_STAGNANT, EXCEPTION)
    - name: attribution_reason
      type: str
      description: Structural cause of failure
    - name: new_no_go_constraint
      type: str
      description: Registered taboo state hash to prune from future search
---

# Taboo Reset Guard

## When to use
- When the agent experiences consecutive wall collisions or zero displacement.
- When an oscillation loop (A -> B -> A -> B) is detected in state trajectory.
- When diagnosing exceptions, runtime crashes, or contract test failures.
- Immediately upon encountering a game-over screen or lethal trap penalty.
- To maintain a persistent registry of No-Go taboo states across episode boundaries.

## When NOT to use
- During early unconstrained exploration before collision occurs.
- As a substitute for initial goal decomposition (use `backward-planner`).
- For simple one-off non-terminal action steps.

## Workflow
1. **Trajectory & Collision Ingestion**:
   - Inspect `consecutive_ineffective_actions`, `recent_trajectory`, `game_over_occurred`, or `error_trace`.
2. **Failure Attribution & Diagnosis**:
   - Classify failure into `COLLISION_STAGNANT`, `OSCILLATION_LOOP`, `DEADLOCK`, or `EXCEPTION`.
   ```bash
   python scripts/taboo_reset_guard.py --consecutive-ineffective 5
   ```
3. **Taboo Registry & Reset Recommendation**:
   - Hash and record current state to `persistent_taboo_states`.
   - Recommend `RESET` if repair cost exceeds fresh start cost.

## Examples
- Input: `consecutive_ineffective=5`
  → Output: `{"is_reset_recommended": true, "status": "COLLISION_STAGNANT", "suggested_action": "RESET"}`

## Output format
- Structured JSON with `is_reset_recommended`, `status`, `attribution_reason`, and `new_no_go_constraint`.

## Anti-patterns to avoid
- Do not keep thrashing in a deadlocked state for dozens of steps; trigger active reset early.
- Do not discard learned taboo constraints upon reset; preserve them across episodes.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/taboo_reset_guard.py`: Deterministic CLI tool for failure attribution, taboo learning, and active reset.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Active reset heuristics, oscillation detection, and taboo pruning theory.
