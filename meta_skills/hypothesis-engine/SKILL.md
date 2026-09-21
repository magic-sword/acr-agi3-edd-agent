---
name: hypothesis-engine
description: |
  Scientific hypothesis lifecycle engine for ARC-AGI-3 gameplay.
  Formulates falsifiable causal claims, tracks minimal probe interventions, evaluates pixel outcomes,
  archives refuted hypotheses into lightweight memory bookmarks (TOC), and proposes taboo-free alternative hypotheses.
  Do NOT use for passive affordance detection or unconstrained random trial-and-error.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - formulate_hypothesis
    - evaluate_hypothesis
    - archive_refutation
    - get_refuted_bookmarks
    - propose_alternative_hypothesis
  inputs:
    - name: claim
      type: str
      description: The falsifiable causal hypothesis statement
    - name: action_name
      type: str
      description: Name of the action being tested (e.g. ACTION1, ACTION6)
    - name: action_id
      type: int
      description: Integer ID of the action being tested
    - name: coords
      type: optional[dict]
      description: "Target coordinates {x: int, y: int} if applicable"
    - name: step_index
      type: int
      description: Current step number
  outputs:
    - name: status
      type: str
      description: Result status (ok, rejected, refuted, verified)
    - name: active_hypothesis
      type: optional[dict]
      description: Formulated active hypothesis data
    - name: refuted_record
      type: optional[dict]
      description: Refuted hypothesis record if falsified
---

# Hypothesis Engine

## When to use
- During cognitive planning (`CAUSAL_PROGRAMMER` / `PROBING_SCIENTIST`) to formulate testable cause-and-effect hypotheses.
- When evaluating the immediate outcome of an action:
  - If `pixels_changed == 0`: immediately falsify the active hypothesis and archive it to memory bookmarks (`hypothesis.refuted.*`).
  - If `pixels_changed > 0`: confirm the hypothesis and update causal models.
- When in `TABOO_RECOVERY` to inspect lightweight bookmarks of previously refuted hypotheses and propose an unrefuted alternative.
- To prevent the agent from repeatedly trying identical failing actions or coordinates.

## When NOT to use
- For raw visual parsing or OpenCV color clustering (use `visual-inspector`).
- For geometric shortest path finding across empty terrain (use `backward-planner`).
- For unconditional random exploration without a falsifiable prediction.

## Workflow
1. **Single-Variable Falsifiable Hypothesis Formulation**:
   - Before taking an action, define a specific prediction: e.g., "Clicking at (col=3, row=4) will toggle the door switch".
   - Register the claim using `formulate_hypothesis`.
   - Verify that the target action/coordinate pair is NOT already in `get_refuted_bookmarks()`.
2. **Minimal Probe Intervention**:
   - Execute the action via `game-controller`.
3. **Outcome Evaluation & Immediate Falsification**:
   - If the action resulted in `0 pixel changes` (or collided with an invisible barrier):
     - Automatically trigger `evaluate_hypothesis` -> falsify the claim.
     - The falsified claim is archived to `hypothesis.refuted.s{step}_{action}_{coords}` with tags `refuted,falsified,constraint`.
     - The active hypothesis is immediately cleared.
     - A 1-line bookmark is visible in the memory notebook TOC.
   - If `pixels_changed > 0`:
     - Mark the hypothesis as verified/supported.
4. **Taboo-Free Alternative Selection**:
   - In subsequent steps, query `get_refuted_bookmarks()` or `propose_alternative_hypothesis()` to pick the next untried hypothesis.

## Examples
- **Example 1 (Probing Phase)**:
  - Formulate: `formulate_hypothesis(claim="ACTION1 moves avatar UP", action_name="ACTION1", action_id=1)`
  - Outcome: ΔPixels = 4, displacement = (-1, 0)
  - Result: Confirmed directional dynamics for ACTION1.
- **Example 2 (Causal Interaction Falsification)**:
  - Formulate: `formulate_hypothesis(claim="Click at (5, 5) opens door", action_name="ACTION6", action_id=6, coords={"x": 5, "y": 5})`
  - Outcome: ΔPixels = 0
  - Result: Hypothesis refuted, archived to `hypothesis.refuted.s3_ACTION6_5_5`, (5, 5) marked as taboo.

## Progressive Disclosure Guidelines
- **Level 1 (Metadata & Bookmarks)**: Keep refuted hypotheses strictly in the notebook TOC bookmarks (`[hypothesis.refuted.*]`). Do NOT dump entire refuted texts into the LLM system prompt.
- **Level 2 (Instructions)**: On-demand retrieval of this scientific protocol via `load_skill("hypothesis-engine")`.
- **Level 3 (Execution Tools)**: Python deterministic methods (`formulate_hypothesis`, `evaluate_hypothesis`, `archive_refutation`, `get_refuted_bookmarks`, `propose_alternative_hypothesis`).

## Anti-patterns to avoid
- Never repeat an action at coordinates listed in `get_refuted_bookmarks()`.
- Never formulate multiple simultaneous vague hypotheses in a single step.
- Do not dump hundreds of past refuted hypotheses into the LLM context; rely on TOC bookmarks and targeted `memory_read`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: Google ADK 2.0, MemoryNotebook

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/hypothesis_engine.py`: Deterministic state machine and hypothesis lifecycle tracker.
