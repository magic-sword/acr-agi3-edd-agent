---
name: causal-deliberation
description: >-
  Goal-directed reasoning for unknown dynamic games. Use when planning is blocked
  by missing causal knowledge, evidence requires an experiment, or an action result
  must be reviewed before resuming a suspended question. Do NOT use for static grid transformations or general factual questions.
license: MIT
allowed-tools: set_goal need_causal_knowledge need_experiment assess_result resolve_question plan_actions continue_plan answer_visible_question use_known_rules load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - set_goal
    - need_causal_knowledge
    - need_experiment
    - assess_result
    - resolve_question
    - plan_actions
    - continue_plan
    - answer_visible_question
    - use_known_rules
---

# Causal Deliberation

## When to use
Use in every goal-directed game reasoning session, beginning in PLAN. The returned
mode is authoritative. A mode is a reasoning purpose, not an elapsed-step label.

## When NOT to use
Do not use for static grid transformations, factual questions outside a game, or
blind action emission without a goal and observation.

## Workflow
1. **PLAN**: Reason backwards from the winning condition. If the goal or a
   precondition is not visible, load visual-inspector and call observe_screen.
   Record the grounded goal with set_goal. Identify the conditions that must hold
   immediately before success; allow staging and temporary detours.
2. **Missing knowledge**: Call need_causal_knowledge with the specific unanswered
   question blocking the plan. This suspends the current goal and enters CAUSAL.
   Another missing prerequisite can suspend causal inference recursively.
   If the missing fact is visible, answer_visible_question resolves it by observation.
   If existing causal rules suffice, use_known_rules resumes the parent without a new experiment.
3. **CAUSAL**: Consider conditional hypotheses: precondition → intervention → effect.
   Inspect the screen whenever needed. If existing evidence cannot distinguish
   explanations, call need_experiment with a hypothesis, a predicted visible
   result and a competing explanation. Random button enumeration is not a goal.
4. **EXPERIMENT**: Load game-controller. Perform one minimal intervention via
   step_action(action_id, reasoning) or click_at(x, y, reasoning). Use physical
   IDs; never assume button direction from its number. Stop for the gateway result.
5. **REVIEW**: In the next observation, call observe_screen(view="both"). Use
   assess_result with the actual before/after frame IDs and visual evidence.
   supported/refuted concern the predicted effect; inconclusive is legitimate.
   Screen animation alone does not verify a causal hypothesis. A wall collision
   does not refute the meaning of a movement button.
6. **Resume**: After experiment review, return to CAUSAL. Use resolve_question
   with cited result IDs, a precondition and an effect. Rules remain provisional.
   This pops the suspended question and returns to the mode that needed its answer.
7. **EXECUTE**: In PLAN, call plan_actions with a subgoal, cited rule IDs and an
   ordered list of steps. Each step needs action_id, precondition, expected_result;
   clicks need x and y. Execute only its first step. After review, inspect current
   preconditions and continue_plan, revise the plan, or ask a new causal question.

## Observation and memory
Observation is a read-only skill available in every mode; it never advances the
game. Images are not automatically supplied. Keep the goal, suspended questions,
conditional rules and their evidence in the returned thought state. Existing
rules are transferable hypotheses: recheck their preconditions on a new level.

## Recovery
Review failure before choosing a reset. Explain both a causal diagnosis and a
changed approach to reset_game. Do not retry resets on the same board. When the
reasoning budget expires without a justified action, stop without inventing a move.

## Anti-patterns to avoid
- Never explore merely because the step count is small. Identify the missing fact.
- Do not equate any pixel change with verification, or erase a rule after one wall collision.
- Do not infer an action from free text, or resolve a question with inconclusive evidence.

## Examples
- A door blocks the goal: ask what opens it, hypothesize a switch, inspect both,
  press the switch once, compare frames, record the conditional rule, resume planning.
- An object did not move: distinguish a blocked destination from an incorrect
  control hypothesis before discarding a previously supported rule.

## Requirements & Prerequisites
- Python >= 3.12; project package acr_agi3.
- Official Google ADK SkillToolset and offline local multimodal inference.

## Resources
- scripts/causal_deliberation.py: offline thought-transition demonstration, no game actions.
- tests/test_contract.py: three positive and three negative transition contracts.
