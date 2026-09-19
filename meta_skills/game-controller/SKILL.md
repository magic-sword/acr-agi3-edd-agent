---
name: game-controller
description: |
  Official 1-step action execution engine for ARC-AGI-3 dynamic games.
  Translates agent intentions (UP, DOWN, LEFT, RIGHT, CLICK, RESET) into exact actions:
  resolves dynamic action maps, auto-snaps clicks to object centers, and guarantees valid IDs.
  Do NOT use for passive static image inspection without taking actions.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.2.0"
  adk_additional_tools:
    - step_action
    - click_at
    - reset_game
  inputs:
    - name: action_call
      type: dict | str
      description: Action specification such as step_action, click_at, or structured JSON
    - name: available_actions
      type: list[int]
      description: List of action IDs valid in current state
    - name: dynamics_map
      type: optional[dict]
      description: "Identified invariant action map (e.g. mapping UP to 3, DOWN to 4)"
    - name: grid_shape
      type: optional[tuple[int, int]]
      description: Current board dimensions (height, width) for bounds checking
  outputs:
    - name: success
      type: bool
      description: Whether the action is valid and executable
    - name: action_name
      type: str
      description: Normalized canonical action name (e.g. UP, RIGHT, ACTION6, RESET)
    - name: action_id
      type: int
      description: Official environment action ID (0-7)
    - name: coordinates
      type: optional[dict]
      description: "Click coordinates if click action ({x: col, y: row})"
    - name: reasoning
      type: str
      description: Agent reasoning explaining why this action was chosen
---

# Game Controller

## When to use
- Every single turn when deciding, validating, and executing your next dynamic game action.
- When executing directional navigation (`UP`, `DOWN`, `LEFT`, `RIGHT`) without needing to remember physical button IDs.
- When targeting an interactive element or button using coordinate clicks (`click_at(x, y)` / `click_at(object_id=...)` / `ACTION6`).
- When navigating via point-and-click to open ground tiles or empty spaces using explicit coordinates (`click_at(x=col, y=row)`).
- When deadlocked or trapped in an irreversible state and requiring an active reset (`reset_game()` / `RESET`).

## When NOT to use
- During pure passive visual inspection when no action is being taken.
- For multi-step macro planning prior to concrete single-step action selection.

## Workflow
1. **Identify Available Actions & Dynamics**:
   - Check `available_actions` provided in the observation text (e.g. `[1, 2, 3, 4]` or `[1, 2, 3, 4, 6]`).
   - The engine automatically resolves `UP`, `DOWN`, `LEFT`, `RIGHT` to the correct physical button via the online dynamics map.
2. **Formulate High-Level 1-Step Decision**:
   - For movement: Choose `UP`, `DOWN`, `LEFT`, `RIGHT`, `ACTION5`, or `ACTION7`.
   - For interactive click: Choose `ACTION6` via `click_at`:
     - **Specific Object Target**: Specify `object_id=N` to automatically and precisely snap to the center of the detected object cluster.
     - **Explicit Tile / Ground Target**: Specify `x=col, y=row` to click exact board coordinates (such as open ground tiles for character movement or placement).
     - **Unspecified Click**: Leave coordinates empty to snap to the highest priority interactive affordance.
   - For reset: Choose `RESET` (0).
3. **Execute via Deterministic Tool**:
   - Execute CLI tool or function tool to output valid environment action:
   ```bash
   python scripts/game_controller.py --action '{"action": "step_action", "direction": "UP"}' --available 1 2 3 4 --dynamics '{"UP": 3}'
   ```

## Examples
- Example 1 (Directional step with dynamic mapping):
  - Input: `{"action": "step_action", "direction": "UP", "reasoning": "Advancing to goal"}` with dynamics `{"UP": 3}`
  - Output: `{"success": true, "action_name": "ACTION3", "action_id": 3, "action_type": "STEP"}`
- Example 2 (Object-targeted click with auto-snap):
  - Input: `{"action": "click_at", "object_id": 0, "reasoning": "Pressing detected HUD button 0"}`
  - Output: `{"success": true, "action_name": "ACTION6", "action_id": 6, "coordinates": {"x": 22, "y": 54}, "action_type": "CLICK"}`
- Example 3 (Explicit ground coordinate click):
  - Input: `{"action": "click_at", "x": 10, "y": 15, "reasoning": "Walking character to open tile at (10, 15)"}`
  - Output: `{"success": true, "action_name": "ACTION6", "action_id": 6, "coordinates": {"x": 10, "y": 15}, "action_type": "CLICK"}`

## Anti-patterns to avoid
- Do not output ambiguous free-form sentences without stating a concrete action identifier (`UP`, `DOWN`, `ACTION1`〜`ACTION7`).
- Do not worry about physical button permutations; trust `game-controller` to map semantic directions (`UP`) to physical actions.
- Do not guess pixel coordinates blindly when targeting objects; use `object_id` to let `game-controller` snap to the exact object center.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: standard library (json, re, argparse), numpy, spatial_grounder, acr_agi3

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/game_controller.py`: Deterministic CLI tool for action validation, coordinate checking, and protocol parsing.
