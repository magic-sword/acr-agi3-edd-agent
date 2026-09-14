---
name: game-controller
description: |
  Official game interaction and action execution engine for ARC-AGI-3 dynamic games.
  Validates and translates agent decisions into environment-compliant actions: movement (UP, DOWN, LEFT, RIGHT, ACTION1-7),
  coordinate clicking (ACTION6 with 0-indexed x, y), and active reset (RESET).
  Use whenever deciding, formatting, or executing an action in any ARC-AGI-3 game environment.
  Do NOT use for passive static image inspection without taking actions.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  inputs:
    - name: action_call
      type: dict | str
      description: Action specification such as step_action, click_at, or structured JSON
    - name: available_actions
      type: list[int]
      description: List of action IDs valid in current state
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
- When formatting an action call for directional navigation (`UP`, `DOWN`, `LEFT`, `RIGHT`, `ACTION1`〜`ACTION7`).
- When targeting an interactive element or button using coordinate clicks (`click_at(x, y)` / `ACTION6`).
- When deadlocked or trapped in an irreversible state and requiring an active reset (`reset_game()` / `RESET`).

## When NOT to use
- During pure passive visual inspection when no action is being taken.
- For high-level subgoal decomposition prior to concrete action formulation.

## Workflow
1. **Identify Available Actions**:
   - Inspect `available_actions` provided in the observation text (e.g. `[1, 2, 3, 4]` or `[1, 2, 3, 4, 6]`).
2. **Formulate Action Decision**:
   - For movement: Select one of `UP` (1), `DOWN` (2), `LEFT` (3), `RIGHT` (4), `ACTION5` (5), `ACTION7` (7).
   - For interactive click: Select `ACTION6` with integer coordinate `(x=column, y=row)`.
   - For reset: Select `RESET` (0).
3. **Execute via Script or Function Call**:
   - Execute CLI tool to validate parameters:
   ```bash
   python scripts/game_controller.py --action '{"action": "step_action", "action_name": "RIGHT"}' --available 1 2 3 4
   ```

## Examples
- Example 1 (Directional step):
  - Input: `{"action": "step_action", "action_name": "RIGHT", "reasoning": "Advancing to goal"}`
  - Output: `{"success": true, "action_name": "ACTION4", "action_id": 4, "action_type": "STEP"}`
- Example 2 (Coordinate click):
  - Input: `{"action": "click_at", "x": 5, "y": 3, "reasoning": "Toggling switch"}`
  - Output: `{"success": true, "action_name": "ACTION6", "action_id": 6, "coordinates": {"x": 5, "y": 3}, "action_type": "CLICK"}`

## Anti-patterns to avoid
- Do not output ambiguous free-form sentences without stating a concrete action identifier (`UP`, `DOWN`, `ACTION1`〜`ACTION7`).
- Do not pass out-of-bounds coordinates (e.g. negative numbers or values exceeding board width/height).
- Do not select action IDs that are absent from `available_actions`.
- Do not repeat unparseable action formats; follow the canonical `step_action` or `click_at` syntax.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: standard library only (json, re, argparse)

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/game_controller.py`: Deterministic CLI tool for action validation, coordinate checking, and protocol parsing.
