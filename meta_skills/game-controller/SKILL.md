---
name: game-controller
description: >-
  Execute one justified action in an unknown dynamic game. Use for a causal
  experiment, a checked plan step, or a diagnosed strategic reset.
  Do NOT use for passive inspection, free-text actions, or ungrounded moves.
license: MIT
allowed-tools: step_action move_cursor click_at_cursor reset_game load_skill_resource
metadata:
  pattern: workflow
  version: "3.0.0"
  adk_additional_tools:
    - step_action
    - move_cursor
    - click_at_cursor
    - reset_game
---

# Game Controller

## When to use
Use cursor aiming in any reasoning mode. Execute a button or confirmed click only
in EXPERIMENT or EXECUTE after observing. Reset is available in PLAN or CAUSAL.

## When NOT to use
Do not use for looking at the screen, speculative planning, or before reviewing
the actual result of a pending action.

## Workflow
1. Read available_actions. Physical IDs are game-specific; learn their effects.
2. For an experiment, explain the intervention and predicted observation.
   For a plan, check the next step's precondition using visual-inspector.
3. For a physical button, call step_action with an ID from available_actions.
   For ACTION6, call move_cursor with the target's original game coordinates:
   x is column, y is row. This changes an internal reticle, not the environment.
   Load visual-inspector and observe_screen after moving. Inspect the reticle
   over the intended target. Adjust and observe again if it is misplaced.
   Only then call click_at_cursor with visual_evidence and reasoning; it accepts
   no coordinates. A new frame or another cursor movement invalidates the look.
   Observation crops must contain the reticle. Add crop origin when locating it.
4. Stop after one scheduled action. The gateway performs it and supplies the
   next frame. In REVIEW, inspect both frames and assess the predicted effect.
5. A strategic reset requires reset_game(diagnosis="...", revised_approach="...").
   Diagnose the problem and explain a changed strategy. Repeated resets of the
   same board are rejected. Game lifecycle initialization is handled by the host.

## Examples
- Test an unknown button only after stating different predicted outcomes.
- Aim at a suspected switch, inspect the displayed reticle and surrounding board,
  then confirm the click if aligned. If misaligned, move and inspect again.
- Aim while planning to inspect a candidate without spending a game action;
  confirmation still requires EXPERIMENT or EXECUTE.

## Anti-patterns to avoid
- Never assume ACTION1 means UP or use directional aliases instead of physical IDs.
- Do not call load_skill with an action name, or emit action JSON as a substitute
  for calling an execution tool.
- Do not execute a second action before observing the first result.
- Do not bypass cursor verification with direct coordinate clicks, implicit snapping,
  or scripts. Bundled scripts do not operate the live gateway.

## Requirements & Prerequisites
- Bundled offline helpers additionally use numpy and spatial_grounder.
- Python >= 3.12, acr_agi3, official Google ADK SkillToolset.
- A live DeliberativeGamePlayer session and on-demand visual observation.

## Resources
- scripts/game_controller.py: standalone protocol validation helper for offline tests;
  it does not execute live game actions.
- tests/: controller validation contracts.
