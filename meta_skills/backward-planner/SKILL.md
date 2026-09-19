---
name: backward-planner
description: |
  Backward-chaining geometric pathfinding and precondition subgoal decomposition engine for ARC-AGI-3.
  Computes deterministic shortest paths (A* / BFS) around maze walls and orders dependency subgoals
  (e.g., Key to Barrier to Goal) to replace LLM geometric hallucinations with exact waypoint vectors.
  Do NOT use for raw pixel clustering or direct physical button execution.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - plan_geometric_path
    - sequence_preconditions
  inputs:
    - name: current_grid
      type: list[list[int]] | numpy.ndarray
      description: Current 2D visual observation grid
    - name: start_pos
      type: list[int]
      description: Coordinates [col, row] of starting entity
    - name: goal_pos
      type: list[int]
      description: Coordinates [col, row] of target destination
    - name: impassable_colors
      type: optional[list[int]]
      description: Color codes considered non-traversable boundaries (default [1])
  outputs:
    - name: path
      type: list[list[int]]
      description: Sequence of coordinate waypoints from start to goal
    - name: next_direction
      type: optional[str]
      description: Immediate cardinal direction (UP, DOWN, LEFT, RIGHT) to follow path
---

# Backward Planner

## When to use
- When navigating mazes or dynamic grid boards (e.g. `tu93`, `ls20`).
- When planning multi-step routes around obstacles and barriers.
- When decomposing composite tasks with prerequisite items (e.g. collect key before door).
- To generate deterministic directional waypoints instead of letting the LLM guess movements blindly.

## When NOT to use
- In pure click-based single-step toggle puzzles without agent movement (use `spatial-grounder`).
- For primitive hardware-level action execution (use `game-controller`).

## Workflow
1. **Target & Obstacle Isolation**:
   - Identify start coordinates `(c_start, r_start)` and goal coordinates `(c_goal, r_goal)`.
   - Mark impassable cells (static borders, wall color 1, or lethal hazard traps).
2. **Backward / Forward A* Pathfinding**:
   - Compute the optimal Manhattan grid path using A* search.
   - Extract the immediate first waypoint vector to determine `next_direction` (`UP`, `DOWN`, `LEFT`, `RIGHT`).
3. **Causal Precondition Sequencing**:
   - If barriers are present that require prerequisite triggers (keys, switches), order intermediate waypoints: `Start -> Key -> Switch/Door -> Goal`.
4. **Waypoint Delivery**:
   - Supply the deterministic next direction to the action executor (`game-controller`).

## Examples
- Example 1 (Maze navigation):
  - Start: `(1, 1)`, Goal: `(8, 8)`, Wall at `row 3`.
  - Result: Path detouring via column 6, `next_direction: "RIGHT"`.
- Example 2 (Prerequisite key sequence):
  - Preconditions: Key at `(1, 7)`, Door at `(5, 4)`, Exit at `(8, 8)`.
  - Result: Subgoal sequence `[Key(1, 7), Door(5, 4), Exit(8, 8)]`.

## Anti-patterns to avoid
- Do not repeat `ACTION1` or single directions when facing a wall; always follow the computed path waypoints.
- Do not attempt straight-line diagonal moves through impassable obstacle blocks.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy, standard library (heapq, collections, argparse, json, typing)

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/backward_planner.py`: A* pathfinder and precondition sequencer CLI.
