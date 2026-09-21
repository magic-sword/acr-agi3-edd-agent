---
name: rule-inducer
description: |
  Empirical causal rule induction and invariant win-condition discovery engine for ARC-AGI-3.
  Analyzes environmental state transitions (before vs after), pixel changes, and level advance events
  to induce deterministic game rules (mechanics, barriers, hazards, and victory conditions).
  Do NOT use for ungrounded hypothetical daydreaming or single-pixel noise fitting.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - induce_rule_from_transition
    - get_known_rules
    - get_win_condition_hypotheses
  inputs:
    - name: action_name
      type: str
      description: Name of the executed action (e.g. ACTION1, ACTION6)
    - name: action_id
      type: int
      description: Numerical action ID
    - name: pixels_changed
      type: int
      description: Number of pixels changed by the transition
    - name: coords
      type: optional[dict]
      description: "Coordinates {x: int, y: int} if applicable"
    - name: level_before
      type: int
      description: Level index before the action
    - name: level_after
      type: int
      description: Level index after the action
  outputs:
    - name: status
      type: str
      description: Processing status (ok, error)
    - name: is_new
      type: bool
      description: True if a new rule was discovered
    - name: rule
      type: dict
      description: The induced rule record
---

# Rule Inducer

## When to use
- Whenever an environmental state transition occurs (`CognitiveMode.CAUSAL_PROGRAMMER` and `CognitiveState.PROBING` / `PLANNING`).
- When a level advances (`level_after > level_before`), to deduce and record the exact win condition invariant.
- When an interaction causes state changes, to record causal interaction rules (e.g. clicking key tile toggles door).
- To accumulate permanent game laws in the Memory Notebook (`rules.*`).

## When NOT to use
- For raw visual parsing or entity boundary segmentation (use `visual-inspector` or `spatial-grounder`).
- For geometric shortest path planning (use `backward-planner`).
- For guessing rules without empirical transition evidence.

## Workflow
1. **Transition Observation**:
   - Collect transition data: `(action_name, action_id, coords, pixels_changed, level_before, level_after)`.
2. **Causal Rule Induction**:
   - Call `induce_rule_from_transition(...)`.
   - If `level_after > level_before`: induce a `win_condition` rule (e.g. reaching this state wins the level).
   - If `pixels_changed > 0`: induce an environmental affordance/interaction rule.
   - If `pixels_changed == 0`: induce an invariant obstacle/barrier rule.
3. **Knowledge Transfer & Persistence**:
   - Query `get_known_rules()` and `get_win_condition_hypotheses()` to inform future subgoals and avoid repeating barrier collisions.

## Examples
- **Example 1 (Win Condition Discovery)**:
  - Input: `action_name="ACTION1", level_before=0, level_after=1`
  - Output: `rule_type="win_condition", statement="Triggering ACTION1 advances level from 0 to 1."`
- **Example 2 (Causal Interaction Discovery)**:
  - Input: `action_name="ACTION6", coords={"x": 3, "y": 2}, pixels_changed=5`
  - Output: `rule_type="causal_interaction", effect="Causes environmental state change (5 pixels changed)."`

## Progressive Disclosure Guidelines
- **Level 1 (Metadata & Catalog)**: Minimal frontmatter catalog in LLM context.
- **Level 2 (Instructions)**: On-demand retrieval of this inductive reasoning protocol via `load_skill("rule-inducer")`.
- **Level 3 (Execution Tools)**: Python deterministic methods (`induce_rule_from_transition`, `get_known_rules`, `get_win_condition_hypotheses`).

## Anti-patterns to avoid
- Never assert a win condition without an observed level progression event.
- Do not hallucinate causal links when `pixels_changed == 0`.
- Do not clear invariant win conditions on episode retry.

## Requirements & Prerequisites
- Python: >= 3.10

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/rule_inducer.py`: Deterministic transition rule induction engine.
