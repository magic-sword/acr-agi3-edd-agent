---
name: constraint-learner
description: |
  Learns environmental constraints, taboo states, and safety invariants from game feedback.
  Use when the user asks to extract gameplay constraints, update safety bounds, or record taboo states.
  Do NOT use for high-level goal scheduling or static grid geometry operations.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: transitions
      type: list[dict]
      description: Trajectory history of actions and results
    - name: failure_diagnostics
      type: optional[dict]
      description: Reports from failure-diagnoser
  outputs:
    - name: constraints
      type: dict
      description: Taboo cells, lethal colors, irreversible transitions
---

# Constraint Learner

## When to use
- Infer environmental hazards, non-walkable coordinates, and death traps from interactive trial history.
- Maintain persistent taboo sets across episodes to avoid repeated mistakes.
- Identify irreversible state changes (e.g., falling into holes, one-way gates, lava pits).

## When NOT to use
- Initial goal decomposition (use `subgoal-decomposer`).
- Executing action movements in the game (use synthesized policy skills).
- Static color frequency analysis for ARC-1/2 puzzles.

## Workflow
1. Trajectory and Failure Ingestion: To inspect failed transitions, penalty signals, and termination causes:
   ```bash
   python scripts/constraint_learner.py --input "data"
   ```
2. Constraint Extraction: To associate game failure with specific cell coordinates, adjacent color tags, or irreversible actions.
3. Constraint Set Update: To merge newly learned constraints into the environment safety specification used by `skill-synthesizer`.

## Examples
- Input: "Transitions show touching color 6 causes immediate game over" → Output: `{"taboo_colors": [6], "hazard_type": "LETHAL_TRAP", "rule": "Never step on color 6"}`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not discard learned constraints upon episode reset; preserve them in memory.
- Do not generalize single-cell obstacles to entire colors without multi-step evidence.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/constraint_learner.py`: Deterministic CLI tool for constraint learning.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, safety invariant models, and taboo sets.
