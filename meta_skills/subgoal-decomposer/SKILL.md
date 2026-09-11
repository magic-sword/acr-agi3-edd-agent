---
name: subgoal-decomposer
description: |
  Decomposes complex game objectives into verifiable intermediate milestones (Subgoals).
  Use when planning sequential milestones (key retrieval, obstacle bypass, goal arrival).
  Do NOT use for single-step primitive execution or post-mortem failure tracing.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "2.0.0"
  pattern: "meta-workflow"
  inputs:
    - name: observation
      type: numpy.ndarray
      description: Current game observation grid
    - name: affordances
      type: dict
      description: Extracted roles (agent, obstacles, goals, items)
  outputs:
    - name: subgoals
      type: list[dict]
      description: Ordered milestone list with preconditions and objectives
---

# Subgoal Decomposer Meta-Skill

## When to use
- Plan sequential milestones (e.g., collect key, bypass obstacle, reach goal) for a game stage.
- Break down complex multi-objective navigation tasks into independently testable subgoals.
- Establish intermediate waypoint constraints based on Human Visual Concept Guided Thinking (VCGT).

## When NOT to use
- Executing single action primitives (use compiled action policies).
- Low-level frame pixel feature extraction (use `env-observer`).
- Diagnosing why a policy collided with a wall (use `failure-diagnoser`).

## Workflow
1. Affordance Ingestion: Receive current grid state, agent position, item positions, and goal coordinates.
2. Topological Plan Formulation: Identify mandatory sequential bottlenecks (e.g. acquire item 4 before entering door 5).
3. Milestone Generation: Output ordered `DecompositionPlan`:
   ```python
   plan = decomposer.decompose_game(obs)
   ```
4. Hand-off: Deliver subgoal milestones to `skill-synthesizer` for modular skill synthesis.

## Examples
- Input: Map with key at (1, 7), wall at column 4, goal at (8, 8) → Output: 3 subgoals (`Acquire_item_color_4`, `BypassCentralObstacle`, `ReachGoalAndClearStage`).

## Output format
- Structured `DecompositionPlan` with list of `subgoals` containing `index`, `name`, `objective`, `reasoning`.

## Anti-patterns to avoid
- Never attempt to plan an entire multi-room maze as a single monolithic policy without subgoals.
- Do not plan unreachable subgoals without checking impassable boundary connectivity.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `references/`
- Reference implementations in `src/acr_agi3/meta/decomposer.py`.

