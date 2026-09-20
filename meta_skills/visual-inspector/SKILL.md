---
name: visual-inspector
description: |
  Official visual inspection and console observation engine for ARC-AGI-3 dynamic games.
  Provides integrated console screen perception (game board + controller HUD with highlighted buttons)
  and human visual inspection pause protocols for LLM visual recognition and causal grounding.
  Do NOT use for high-level multi-step planning or executing primitive actions.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.3.0"
  adk_additional_tools:
    - inspect_cursor_target
    - inspect_affordances
    - inspect_board_summary
  inputs:
    - name: current_grid
      type: list[list[int]] | numpy.ndarray
      description: Current 2D visual observation grid
    - name: target_pattern
      type: optional[list[int] | list[list[int]]]
      description: Target strip or target goal sequence if available
    - name: step_index
      type: optional[int]
      description: Current environment step index (0 indicates onset)
  outputs:
    - name: diff_type
      type: str
      description: Gestalt diff classification (IDENTITY, REVERSAL_REORDER, INTERLEAVED, TRANSLATION_ONLY)
    - name: style
      type: str
      description: Classified game genre (OPEN_EXPLORATION, CLOSED_MAZE, ITEM_TRIGGER_PUZZLE, SYMMETRIC_PATTERN)
    - name: grid_dimensions
      type: list[int]
      description: Board dimensions [height, width]
    - name: active_colors
      type: list[int]
      description: Palette of active foreground colors present
---

# Visual Inspector

## When to use
- When inspecting the integrated console image (game board + controller HUD).
- When cross-referencing which button on the HUD was highlighted with visual displacement on the board.
- When classifying the spatial gestalt gap between starting layout and target sequence.
- When extracting objective facts (grid dimensions, active colors, diff classification) to feed the planning node.
- When verifying mouse cursor reticle alignment (`inspect_cursor_target`) before firing a click.

## When NOT to use
- For deterministic hardcoded guessing of player/goal positions (the LLM must recognize these visually).
- During mid-episode primitive action execution (use `game-controller`).

## Workflow
1. **Visual Reconnaissance & Inspection**:
   - Observe the full console canvas containing both the game board (top) and the physical controller HUD (bottom).
   - Check the highlighted/glowing button on the HUD to see exactly which action was just executed.
   - Use visual Gestalt perception to observe what moved, appeared, or disappeared on the board.
2. **Gestalt Difference & Genre Classification**:
   - Compare current layout against target layout (if provided) to identify transformation style.
   - Observe whether the game is open movement, maze navigation, click-based toggle, or pattern completion.
3. **Causal Grounding**:
   - Ground the button action to visual pixel changes without hardcoded assumptions.
4. **Mouse Cursor Reticle Verification (On-Demand Visual Inspection)**:
   - When moving mouse cursor with `move_cursor(x, y)` in click-based environments, call `inspect_cursor_target()` to verify aim.
   - Inspect the returned 7x7 local gestalt map, aim status (`CENTERED`, `EDGE`, `OFF_TARGET`), and aimed object.
   - If `status == "CENTERED"`, confidently fire `click_at_cursor()`. If `EDGE`, adjust aim with `move_cursor` to object center.

## Examples
- Example 1 (Initial level onset):
  - Input: `current_grid` at `step_index: 0`
  - Output: `{"grid_dimensions": [15, 15], "style": "OPEN_EXPLORATION", "diff_type": "IDENTITY"}`
- Example 2 (Causal frame inspection):
  - Inspecting console image reveals `ACTION4` (Right) was executed, and a blue square shifted right by 1 cell.

## Anti-patterns to avoid
- Do not use hardcoded programs to guess which object is the player or target; let the multimodal LLM determine roles via visual recognition.
- Do not blindly assume there is only one controllable object; observe if multiple objects move simultaneously or if the game is click-driven.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: standard library (json, collections, argparse), numpy, spatial_grounder

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/visual_inspector.py`: Deterministic helper for grid dimensions, color palette, and gestalt diff classification.
