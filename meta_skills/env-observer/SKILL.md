---
name: env-observer
description: |
  Extracts affordances (agent, obstacles, targets, hazards) and transition dynamics from ACR-AGI-3 games.
  Use when observing game frames or analyzing state transitions to infer causal interaction rules.
  Do NOT use for synthesising full action policy scripts or static puzzle grid transforms.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "2.0.0"
  pattern: "meta-workflow"
  inputs:
    - name: observation
      type: numpy.ndarray
      description: Current observation grid array (H, W)
    - name: transition_history
      type: optional[list[dict]]
      description: Past transition logs [obs, action, next_obs, reward, done]
  outputs:
    - name: affordances
      type: dict
      description: Identified roles (agent, obstacles, goals, hazards, interactables)
    - name: dynamics_rules
      type: list[str]
      description: Inferred causality rules
---

# Environment Observer Meta-Skill

## When to use
- Observe raw 2D observation frames of an unknown ACR-AGI-3 game environment.
- Classify visual gestalt components into gameplay affordance roles (Agent, Static Obstacles, Goal, Hazards, Interactables).
- Analyze action transition tuples `(obs, action, next_obs, reward, done)` to extract physical causal dynamics.

## When NOT to use
- Static input-to-output puzzle grid transformations (ARC-1/2 style).
- Synthesizing full executable Python action policies (use `skill-synthesizer`).
- Running contract test simulation loops (use `contract-tester`).

## Workflow
1. Frame Affordance Analysis: Pass raw color observation grid to extract spatial boundaries, background, agent, obstacles, and items:
   ```bash
   python -m acr_agi3.meta.observer --frame "<observation_array>"
   ```
2. Transition Causality Extraction: Feed `(obs_before, action, obs_after)` tuples to identify move displacement, collision elasticity, or key-lock interaction.
3. Structured Hand-off: Pass structured affordance report to `subgoal-decomposer` and `skill-synthesizer`.

## Examples
- Input: 5x5 grid with agent at (1, 1), walls at row 0, goal at (4, 4) → Output: `{"player_pos": (1, 1), "goal_pos": (4, 4), "obstacles": [(0, 0), ...], "background": 0}`

## Output format
- Structured dictionary with keys: `grid_shape`, `background_color`, `player_pos`, `goal_pos`, `obstacles`, `hazards`, `interactables`.

## Anti-patterns to avoid
- Do not assume agent coordinate is always color 2 without checking motion displacement across steps.
- Do not treat dynamic game grids as static matrix math transformations.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `references/`
- Reference implementations in `src/acr_agi3/meta/observer.py`.

