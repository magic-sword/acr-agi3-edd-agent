---
name: constraint-learner
description: |
  Learns irreversible dead-end constraints (corner traps, hazard proximity, blocked pathways).
  Use when analyzing failed trajectories to formulate pruning rules that prevent terminal deadlocks.
  Do NOT use for successful action execution or frame affordance classification.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "1.0.0"
  pattern: "meta-workflow"
  inputs:
    - name: failed_trajectories
      type: list[dict]
      description: Sequences of actions ending in deadlocks or traps
  outputs:
    - name: deadend_constraints
      type: list[dict]
      description: Explicit forbidden states and conditions
    - name: pruning_rules
      type: list[str]
      description: Search pruning rules for planners and synthesis
---

# Constraint Learner Meta-Skill

## When to use
- Analyze failed gameplay trajectories that ended in irreversible deadlocks or traps.
- Extract forbidden state predicates (e.g., pushing movable items into irreversible corner traps).
- Supply search pruning rules to `subgoal-decomposer` and negative contract tests to `skill-synthesizer`.

## When NOT to use
- Real-time action decision-making during normal gameplay.
- Single-step collision diagnostics (use `failure-diagnoser`).
- Direct Python policy writing (use `skill-synthesizer`).

## Workflow
1. Trajectory Ingestion: Parse sequences of states and actions resulting in terminal loss or deadlock.
2. Irreversibility Analysis: Detect points of no return (e.g., moving block adjacent to concave wall corners where no pull action exists).
3. Rule Formulation: Formulate explicit pruning predicates:
   ```json
   {"forbidden_condition": "player at (r, c) when hazard is adjacent in heading direction"}
   ```
4. Rule Propagation: Feed negative constraints into `subgoal-decomposer` and `contract-tester`.

## Examples
- Trajectory: Block pushed into corner (0, 0) unable to be retrieved → Output rule: `Avoid pushing movable objects into concave corner cells`.

## Output format
- Structured list of `pruning_rules` and `deadend_constraints`.

## Anti-patterns to avoid
- Do not formulate overly aggressive pruning rules that eliminate valid narrow passages.
- Do not confuse temporary detours with irreversible dead-ends.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `references/`
- Reference implementations in `src/acr_agi3/meta/human_vcgt.py`.

