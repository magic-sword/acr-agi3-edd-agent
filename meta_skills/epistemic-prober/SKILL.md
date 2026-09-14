---
name: epistemic-prober
description: |
  Selects hypothesis-driven epistemic probing actions to uncover hidden dynamics and mechanics.
  Use when encountering untested interactables, actuator rails, or mirrored linkages before pragmatic exploitation.
  Do NOT use when game rules are completely established and the agent is executing known subgoals.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: step_index
      type: int
      description: Current environment step index
    - name: available_actions
      type: list[int]
      description: Valid primitive action IDs in the environment
    - name: unexplored_affordances
      type: dict
      description: Actuator rails, linkage blocks, or switches needing verification
  outputs:
    - name: is_epistemic
      type: bool
      description: Whether the recommended action is for information-gathering
    - name: probe_type
      type: str
      description: Probing category (PROBE_ACTUATOR_REACH, PROBE_LINKAGE_DYNAMICS, etc.)
    - name: recommended_action_id
      type: int
      description: Chosen minimal intervention action
    - name: hypothesis
      type: str
      description: Environmental rule being tested
---

# Epistemic Prober

## When to use
- During early exploration (Step 1–5) when the physics rules of actuators or objects are untested.
- When new interactable entities (pistons, mirrored units, switches, teleporters) appear.
- When the agent needs to isolate variables using a single-step intervention.

## When NOT to use
- During execution of verified linear paths to the goal.
- For high-level backward planning across multiple subgoals (use `backward-planner`).
- After a terminal failure or deadlock requiring full reset (use `taboo-reset-guard`).

## Workflow
1. **Uncertainty & Mechanics Check**:
   - Inspect `unexplored_affordances` (rails, linkages, switches) and `step_index`.
2. **Epistemic Action Formulation**:
   - Apply the Minimal Intervention Principle (1-step action aimed at maximum information gain).
   ```bash
   python scripts/epistemic_prober.py --step 1 --actions '[0, 1, 2, 3]' --affordances '{"has_actuator_rail": true}'
   ```
3. **Hypothesis Emission**:
   - Output the specific test hypothesis and track the Epistemic Action Ratio (EAR).

## Examples
- Input: `step=1`, `actions=[0, 1, 2]`, `affordances={"has_actuator_rail": true}`
  → Output: `{"is_epistemic": true, "probe_type": "PROBE_ACTUATOR_REACH", "recommended_action_id": 0, "hypothesis": "Testing piston/carriage extension range and holding contact."}`

## Output format
- Structured JSON with `is_epistemic`, `probe_type`, `recommended_action_id`, `hypothesis`, and `target_variable`.

## Anti-patterns to avoid
- Do not spam multiple random actions; keep interventions isolated (1 action, then observe).
- Do not skip epistemic probing when actuators or linkages have unknown physics.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/epistemic_prober.py`: Deterministic CLI tool for epistemic action selection.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Epistemic action ratio (EAR) theory, Kirsh & Maglio principles, and probing types.
