---
name: subgoal-decomposer
description: |
  Decomposes long-horizon game objectives into ordered intermediate subgoals.
  Use when the user asks to plan gameplay steps, break down levels, or sequence subgoals.
  Do NOT use for single-step primitive action execution or static grid rotations.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: affordances
      type: dict
      description: Locations of player, goal, keys, doors, switches
    - name: game_objective
      type: str
      description: Overall stage clearance criteria
  outputs:
    - name: subgoal_sequence
      type: list[dict]
      description: Ordered milestone checkpoints
---

# Subgoal Decomposer

## When to use
- Decompose complex multi-step ACR-AGI-3 levels into manageable intermediate milestones.
- Formulate sequential dependencies (e.g., Navigate to Key -> Collect Key -> Navigate to Door -> Unlock Door -> Reach Exit).
- Update subgoal sequences dynamically when environment state changes unexpectedly.

## When NOT to use
- Primitive single-step physics simulations (use `env-observer`).
- Direct action policy code execution (use synthesized skills).
- Static ARC puzzle transformation planning.

## Workflow
1. Objective and Topology Inspection: To examine player position, target exit, and locked barriers:
   ```bash
   python scripts/subgoal_decomposer.py --input "data"
   ```
2. Dependency Graph Construction: To resolve topological ordering of prerequisite objects (keys before doors, switches before bridges).
3. Subgoal Plan Emission: To output a structured sequence of intermediate target coordinates with clear termination conditions.

## Examples
- Input: "Player at (0, 0), Key at (2, 2), Door at (4, 4), Goal at (5, 5)" → Output: `[{"subgoal_id": 1, "target": [2, 2], "action": "COLLECT_KEY"}, {"subgoal_id": 2, "target": [4, 4], "action": "UNLOCK_DOOR"}, {"subgoal_id": 3, "target": [5, 5], "action": "REACH_GOAL"}]`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not plan direct paths to the goal when intermediate keys or doors block the way.
- Do not create circular dependency graphs.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/subgoal_decomposer.py`: Deterministic CLI tool for subgoal decomposition.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications and subgoal sequencing patterns.
