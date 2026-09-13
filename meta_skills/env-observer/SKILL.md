---
name: env-observer
description: |
  Extracts affordances, invariants, and transition dynamics from ACR-AGI-3 games.
  Use when the user asks to observe game frames, extract visual affordances, or infer dynamics.
  Do NOT use for static puzzle grid transforms or final action policy execution.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
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

# Environment Observer

## When to use
- Observe raw 2D observation frames of an unknown ACR-AGI-3 game environment.
- Classify visual gestalt components into gameplay affordance roles (Agent, Static Obstacles, Goal, Hazards, Interactables).
- Analyze action transition tuples `(obs, action, next_obs, reward, done)` to extract physical causal dynamics.

## When NOT to use
- Static input-to-output puzzle grid transformations (ARC-1/2 style).
- Synthesizing full executable Python action policies (use `skill-synthesizer`).
- Running contract test simulation loops (use `contract-tester`).

## Workflow
1. Reconnaissance and Affordance Analysis: To inspect the raw observation grid and identify spatial boundaries, player, and objects:
   ```bash
   python scripts/env_observer.py --input "data"
   ```
2. Transition Causality Extraction: To analyze step transitions `(obs, action, next_obs)` and extract motion vectors and collision rules.
3. Result Verification: To verify the extracted affordance dictionary contains all required fields and pass structured context to downstream meta-skills.

## Examples
- Input: "Observe grid with agent at (1, 1), walls at row 0, goal at (4, 4)" → Output: `{"player_pos": [1, 1], "goal_pos": [4, 4], "obstacles": [[0, 0], [0, 1]], "background": 0}`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not assume agent coordinate is always color 2 without checking motion displacement across steps.
- Do not treat dynamic game grids as static matrix math transformations.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy, acr_agi3
- External packages: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/env_observer.py`: Deterministic CLI tool for environment observation.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications and gameplay affordance guidelines.
