---
name: game-style-intuitor
description: |
  Analyzes visual texture, border permeability, and gestalt patterns to classify game genres.
  Use when the user asks to classify game style, determine exploration strategy, or inspect screen texture.
  Do NOT use for single-step primitive action execution or diagnosing test failures.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: observation
      type: numpy.ndarray
      description: Raw 2D game observation grid (H, W)
  outputs:
    - name: style
      type: str
      description: Classified game genre (OPEN_EXPLORATION, CLOSED_MAZE, ITEM_TRIGGER_PUZZLE, SYMMETRIC_PATTERN)
    - name: recommended_approach
      type: str
      description: Strategic guidance on exploratory vs detour posture
    - name: recommended_domain
      type: str
      description: Target skill domain folder to load
---

# Game Style Intuitor

## When to use
- Classify high-level game genre from raw visual texture and layout before committing to detailed path planning.
- Identify open-boundary exploration games where the goal is off-screen and perimeter traversal is required.
- Direct the agent toward the appropriate skill domain folder (e.g. exploration, navigation, inventory_puzzle).

## When NOT to use
- Executing discrete single-step game actions (UP, DOWN, LEFT, RIGHT).
- Diagnosing Python runtime exceptions or syntax errors (use `failure-diagnoser`).
- Direct static matrix transformations for ARC-1/2 puzzles.

## Workflow
1. Visual Gestalt and Boundary Ingestion: To inspect the outer borders, obstacle density, and color distributions:
   ```bash
   python scripts/game_style_intuitor.py --input "data"
   ```
2. Genre Classification: To categorize the environment into open exploration, closed maze, item trigger, or symmetric pattern.
3. Domain Routing: To emit the recommended strategic posture and load the matching domain skill folder into the active agent context.

## Examples
- Input: 10x10 grid with unblocked edges and sparse obstacles → Output: `{"style": "OPEN_EXPLORATION", "recommended_domain": "exploration"}`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not assume goals are always visible on-screen when perimeter boundaries are wide open.
- Do not load heavy inventory puzzle skills when the environment is an open traversal domain.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/game_style_intuitor.py`: Deterministic CLI tool for visual style intuition.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, visual gestalt taxonomy, and play-style rules.
