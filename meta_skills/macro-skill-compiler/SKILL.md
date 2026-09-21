---
name: macro-skill-compiler
description: |
  Reusable procedural macro-skill compiler and fast-path action sequence execution engine for ARC-AGI-3.
  Compiles multi-step action patterns (such as Aim-Inspect-Click, Cardinal Probing, Isolate-and-Stage)
  to enable rapid, deterministic execution without invoking LLM inference every turn.
  Do NOT use for unvalidated speculative action bursts into dangerous or unknown hazards.

license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - compile_macro_sequence
    - execute_macro_step
    - list_available_macros
  inputs:
    - name: macro_name
      type: str
      description: Identifier name of the macro skill
    - name: parameters
      type: optional[dict]
      description: Execution parameters such as target coordinates
  outputs:
    - name: status
      type: str
      description: Status of macro operation (ok, aborted, completed)
    - name: next_step
      type: optional[dict]
      description: Next executable step action details
---

# Macro Skill Compiler

## When to use
- When executing known multi-step procedural routines (e.g. Aiming cursor at coordinates, inspecting, and clicking).
- During fast-path execution (`CognitiveState.EXECUTING`) to bypass slow LLM generation for routine deterministic actions.
- During early exploration (`CognitiveMode.PROBING_SCIENTIST`) to execute standard cardinal probing (`CARDINAL_PROBE`).
- To drastically reduce GPU/CPU inference time and stay within strict submission time limits.

## When NOT to use
- When navigating dangerous dynamic environments with unpredictable moving enemies.
- When an unexpected 0-pixel change or collision occurs (the macro MUST be aborted).
- For high-level strategic reasoning or goal decomposition (use `subgoal-decomposer`).

## Workflow
1. **Macro Selection or Registration**:
   - Query `list_available_macros()` to see registered patterns (`AIM_AND_CLICK`, `CARDINAL_PROBE`).
   - Or register a custom macro using `compile_macro_sequence(name, steps)`.
2. **Instantiation & Parameter Binding**:
   - Instantiate the macro with runtime arguments: e.g. `instantiate_macro("AIM_AND_CLICK", {"coords": {"x": 3, "y": 2}})`.
3. **Fast-Path Step-by-Step Execution**:
   - Call `execute_macro_step()` to pop each atomic action in sequence.
   - Execute the action via `game-controller`.
   - If an unexpected collision or 0-pixel change occurs, abort the macro and transition to `TABOO_RECOVERY`.

## Examples
- **Example 1 (Aim and Click Routine)**:
  - Macro: `AIM_AND_CLICK` with `coords={"x": 5, "y": 5}`
  - Execution Sequence:
    1. `move_cursor(x=5, y=5)`
    2. `inspect_cursor_target()`
    3. `click_at_cursor()`
- **Example 2 (Cardinal Probing)**:
  - Macro: `CARDINAL_PROBE`
  - Execution Sequence:
    1. `ACTION1` (UP)
    2. `ACTION2` (DOWN)
    3. `ACTION3` (LEFT)
    4. `ACTION4` (RIGHT)

## Progressive Disclosure Guidelines
- **Level 1 (Metadata & Catalog)**: Minimal catalog representation in planner context.
- **Level 2 (Instructions)**: On-demand retrieval of compilation guidelines via `load_skill("macro-skill-compiler")`.
- **Level 3 (Execution Tools)**: Python deterministic methods (`compile_macro_sequence`, `execute_macro_step`, `list_available_macros`).

## Anti-patterns to avoid
- Never continue executing a macro after a collision or 0-pixel change.
- Do not hardcode rigid coordinate paths across games; use parameterized macros.
- Do not instantiate a macro when in `TABOO_RECOVERY` without first verifying constraints.

## Requirements & Prerequisites
- Python: >= 3.10

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/macro_skill_compiler.py`: Deterministic macro compilation and queue runner.
