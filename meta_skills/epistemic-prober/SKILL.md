---
name: epistemic-prober
description: |
  Hypothesis-driven active exploration and action dynamics probing engine for ARC-AGI-3.
  Performs minimal-intervention probe actions during early onset to empirically identify
  controller button dynamics (mapping physical actions 1-7 to UP, DOWN, LEFT, RIGHT, CLICK).
  Do NOT use for long-horizon path execution or repetitive backtracking.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - probe_action_dynamics
    - get_dynamics_map
  inputs:
    - name: available_actions
      type: list[int]
      description: List of available action IDs
    - name: step_index
      type: int
      description: Current episode step count
    - name: frame_diff
      type: optional[dict]
      description: Previous vs current frame pixel displacement info
  outputs:
    - name: dynamics_map
      type: dict
      description: Mapping of semantic roles (UP, DOWN, LEFT, RIGHT, CLICK) to action IDs
    - name: probe_recommendation
      type: optional[dict]
      description: Recommended minimal-intervention action to test next hypothesis
---

# Epistemic Prober

## When to use
- During early episode onset (`step_index <= 4`) when button mechanics are unknown.
- When facing a new game level where movement directions (UP, DOWN, LEFT, RIGHT) are scrambled or unmapped.
- When deciding whether an environment requires discrete clicking (`ACTION6`) or directional stepping.
- To execute single-variable isolation probes (1 step move and observe) before executing high-stakes plans.

## When NOT to use
- Mid-to-late episode when action dynamics are already established in `dynamics_map`.
- For standard goal-directed navigation (use `backward-planner`).

## Workflow
1. **Uncertainty Assessment**:
   - Check if `dynamics_map` contains full directional mapping for available actions.
   - If unmapped, select an untested action ID from `available_actions`.
2. **Minimal Intervention Probe Execution**:
   - Issue a single probe action to measure environmental response without causing irreversible deadlock.
3. **Displacement & Causal Grounding**:
   - Compare grid before and after: detect the moving entity's centroid displacement `(dr, dc)`.
   - Classify:
     - `dr < 0, dc == 0` -> `UP`
     - `dr > 0, dc == 0` -> `DOWN`
     - `dr == 0, dc < 0` -> `LEFT`
     - `dr == 0, dc > 0` -> `RIGHT`
     - localized single-point change -> `CLICK` / `TOGGLE`
4. **Dynamics Map Registration**:
   - Update `dynamics_map` with confirmed mapping, freeing the planner to execute directional waypoints deterministically.

## Examples
- Example 1 (Directional probing):
  - Action tested: `ACTION1`. Agent centroid moved from `(10, 5)` to `(9, 5)` (`dr = -1`).
  - Result: Registered `{"UP": 1}` into `dynamics_map`.
- Example 2 (Probing click action):
  - Action tested: `ACTION6` at `(15, 15)`. Pixel at `(15, 15)` toggled from 0 to 2.
  - Result: Registered `{"CLICK": 6}` into `dynamics_map`.

## Anti-patterns to avoid
- Do not continue probing once all available actions have been mapped; switch to pragmatic goal-oriented planning.
- Do not probe with destructive actions when near game-over boundaries.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy, standard library (argparse, json, typing, collections)

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/epistemic_prober.py`: Probing logic, displacement classifier, and dynamics map manager.
