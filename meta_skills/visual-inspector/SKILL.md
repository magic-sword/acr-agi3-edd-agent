---
name: visual-inspector
description: >-
  Observe the latest game screen on demand during any reasoning mode. Use to
  locate objects, check planning preconditions, design experiments, or compare
  an action's before and after frames. Do NOT use to execute game actions or
  claim causal knowledge from appearance alone.
license: MIT
allowed-tools: observe_screen load_skill_resource
metadata:
  pattern: workflow
  version: "5.0.0"
  adk_additional_tools:
    - observe_screen
---

# Visual Inspector

## When to use
Look whenever a needed fact is not known, in PLAN, CAUSAL, EXPERIMENT, EXECUTE,
or REVIEW. Observation is a skill, not a mandatory initial pipeline stage.

## When NOT to use
Do not use to advance the game or replace an experiment with an unsupported
causal conclusion. Repeated viewing does not produce a new environment frame.

## Workflow
1. Call observe_screen(view="current") to see the latest gateway frame.
   The result delivers an actual rendered PNG to the local VLM at its next
   inference. No initial screen image is automatically injected.
2. Use view="previous" or view="both" to inspect the last action's consequences.
   Compare object positions, boundaries, goals and counters. Use frame_id to
   cite the actual before and after observations in assess_result.
3. For detail, provide x, y, width, height to crop in original game coordinates.
   The image is enlarged for visibility. Coordinates for a click remain game
   coordinates; translate crop-local positions using the returned origin.
4. After move_cursor, observe the current frame with the reticle visible. Check
   its alignment against the intended target, not just the reported coordinates.
   A crop excluding the reticle or a previous-frame view cannot authorize a click.
   Moving again, receiving a new frame or resetting requires a fresh inspection.
   The reticle is an internal annotation, not a game object or game progress.
   Before/after views use the same reticle overlay for a fair comparison.
   Game frames use the official ARC-AGI-3 16-color palette.
5. Separate visible facts from hypothesized roles and causal relations. A moving
   animation is not necessarily player movement or progress. A stationary object
   can mean an obstacle, a wrong control hypothesis, or insufficient evidence.
6. Ground experiment and plan predictions in an observed target region:
   x, y, width, height and a description in original frame coordinates. Include
   an expected movement destination; avoid unrelated HUD regions. A crop outside
   the target does not count as inspecting it. Inspect the region in both frames
   for review; an invisible or ambiguous effect is inconclusive.
7. Resume the interrupted thought. A visible factual question can be answered
   without a game action; causal uncertainty may require a controlled experiment.

## Examples
- During backward planning, inspect the goal and the obstacle blocking its access.
- During inference, crop a suspected switch before designing an intervention.
- During review, observe_screen(view="both") compares the two actual frames;
  assess_result records whether the previously predicted effect occurred.

## Anti-patterns to avoid
- Do not assume an image has been supplied before calling observe_screen.
- Do not invent a controller HUD, cursor reticle or object IDs absent from the image.
- Do not mistake crop pixels or enlarged display pixels for original coordinates.
- Do not treat pixel changes alone as proof of a causal rule.
- Do not assume every scene contains a single player or a movement-based goal.

## Requirements & Prerequisites
- Bundled offline helpers additionally use numpy and spatial_grounder.
- Python >= 3.12, acr_agi3, numpy, Pillow, official Google ADK SkillToolset.
- Offline local multimodal inference; no external image service is needed.

## Resources
- scripts/visual_inspector.py: offline grid-summary helper; live viewing uses
  observe_screen through the bound ScreenTools instance.
- tests/: visual analysis contracts;
  tests/test_verified_observation_loop.py in the repository covers reticle and palette delivery; tests/test_deliberative_player.py in the
  repository also verifies live image delivery and invalid observation rejection.
