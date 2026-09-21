---
name: game-controller
description: >-
  Execute one justified action in an unknown dynamic game. Use for a causal
  experiment, a checked plan step, or a diagnosed strategic reset.
  Do NOT use for passive inspection, free-text actions, or ungrounded moves.
license: MIT
allowed-tools: load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  adk_additional_tools:
    - step_action
    - click_at
    - reset_game
---

# Game Controller

## When to use
Use after causal-deliberation has entered EXPERIMENT or EXECUTE and the current
screen has been observed. A strategic reset is available in PLAN or CAUSAL.

## When NOT to use
Do not use for looking at the screen, speculative planning, or before reviewing
the actual result of a pending action.

## Workflow
1. Read available_actions. Physical IDs are game-specific; learn their effects.
2. For an experiment, explain the intervention and predicted observation.
   For a plan, check the next step's precondition using visual-inspector.
3. Call step_action(action_id=1, reasoning="...") for a physical button, or
   click_at(x=10, y=15, reasoning="...") for ACTION6. Coordinates are original
   game coordinates: x is column, y is row. Cropped or enlarged images do not
   change these coordinates; add the crop origin when locating a target.
4. Stop after one scheduled action. The gateway performs it and supplies the
   next frame. In REVIEW, inspect both frames and assess the predicted effect.
5. A strategic reset requires reset_game(diagnosis="...", revised_approach="...").
   Diagnose the problem and explain a changed strategy. Repeated resets of the
   same board are rejected. Game lifecycle initialization is handled by the host.

## Examples
- An unknown button may move a piece: prepare an experiment, then call
  step_action(action_id=3, reasoning="Test whether this button moves the piece left").
- A verified plan requires pressing a switch at (10, 15): observe it, then call
  click_at(x=10, y=15, reasoning="The switch is visible and currently unpressed").

## Anti-patterns to avoid
- Never assume ACTION1 means UP or use directional aliases instead of physical IDs.
- Do not call load_skill with an action name, or emit action JSON as a substitute
  for calling an execution tool.
- Do not execute a second action before observing the first result.
- Do not invent cursor, object-snapping, or CLI execution capabilities. Use the
  currently exposed tool schemas. Bundled scripts do not operate the live gateway.

## Requirements & Prerequisites
- Bundled offline helpers additionally use numpy and spatial_grounder.
- Python >= 3.12, acr_agi3, official Google ADK SkillToolset.
- A live DeliberativeGamePlayer session and on-demand visual observation.

## Resources
- scripts/game_controller.py: standalone protocol validation helper for offline tests;
  it does not execute live game actions.
- tests/: controller validation contracts.
