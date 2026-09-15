---
name: visual-inspector
description: |
  Comprehensive visual perception engine: extracts affordances, visual objects, gestalt differences, and game style.
  Implements the human visual inspection pause protocol at level onset and classifies environment genres.
  Use whenever inspecting game frames, extracting affordances (agent, obstacles, goals), comparing targets, or classifying game style.
  Do NOT use for high-level multi-step planning or executing primitive actions.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
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
    - name: pause_required
      type: bool
      description: Whether the agent should pause actions for visual inspection at onset
    - name: diff_type
      type: str
      description: Gestalt diff classification (IDENTITY, REVERSAL_REORDER, INTERLEAVED, TRANSLATION_ONLY)
    - name: style
      type: str
      description: Classified game genre (OPEN_EXPLORATION, CLOSED_MAZE, ITEM_TRIGGER_PUZZLE, SYMMETRIC_PATTERN)
    - name: affordances
      type: dict
      description: Extracted agent position, goals, rails, obstacles, and open travel space
    - name: invariants_hypothesized
      type: list[str]
      description: High-level physical rule hypotheses
---

# Visual Inspector

## When to use
- At Step 0 of any new ACR-AGI-3 level or newly initialized board.
- When inspecting observation frames to extract visual affordances (agent position, target candidates, obstacles).
- When analyzing the spatial gestalt gap between starting layout and target sequence.
- When classifying the game style/genre (Open Exploration, Closed Maze, Item Trigger, Symmetric Pattern).
- When formulating initial invariant hypotheses before taking premature actions.

## When NOT to use
- During mid-episode primitive action execution (use compiled macro-skills).
- For resolving deep causal deadlocks or lethal trap collisions (use `taboo-reset-guard`).
- For multi-step backward subgoal decomposition (use `backward-planner`).

## Workflow
1. **Visual Reconnaissance & Inspection**:
   - Inspect the main game grid to identify agent position, target candidates, fixed rails, and obstacle boundaries.
   - Inspect the lower game console HUD (D-Pad, Action Buttons, and RESET). Cross-reference the actively highlighted button (last executed action) with visual displacement on the board to verify action-outcome contingency.
2. **Gestalt Difference & Genre Classification**:
   - Classify layout difference against target (Reversal, Interleaved, Identity) and detect overall game style.
   ```bash
   python scripts/visual_inspector.py --grid '[[0,0],[1,2]]' --target '[2,1]' --step 0
   ```
3. **Affordance & Invariant Report Emission**:
   - Emit structured report to guide downstream `epistemic-prober` and `backward-planner`.

## Examples
- Input: `current_grid=[[0, 0], [1, 2]]`, `target=[2, 1]`, `step=0`
  → Output: `{"pause_required": true, "diff_type": "REVERSAL_REORDER", "style": "CLOSED_MAZE", "affordances": {...}}`

## Output format
- Structured JSON containing `pause_required`, `diff_type`, `style`, `affordances`, and `invariants_hypothesized`.

## Anti-patterns to avoid
- Do not execute actions at step 0 before evaluating target difference and affordance layout.
- Do not assume agent coordinate is always color 2 without checking motion displacement.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/visual_inspector.py`: Deterministic CLI tool for unified visual inspection & affordance extraction.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, human inspection pause patterns, visual gestalt, and game genre taxonomy.
