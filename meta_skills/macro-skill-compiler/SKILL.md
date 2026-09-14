---
name: macro-skill-compiler
description: |
  Comprehensive policy synthesis and action compilation engine.
  Synthesizes executable policies (navigation, interaction, exploration) and compiles human archetypes (Isolate-and-Stage, Pair-and-Carry, Lane-Splitting) into verified action sequences.
  Use whenever compiling subgoals into concrete primitive actions or synthesizing dynamic gameplay policies.
  Do NOT use for high-level backward planning or initial affordance inspection.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: subgoal
      type: optional[dict]
      description: Target milestone description, phase, and target colors
    - name: affordance_report
      type: optional[dict]
      description: Dynamic affordance report for policy synthesis
    - name: available_action_mapping
      type: optional[dict]
      description: Mapping of semantic actions (UP, DOWN, EXTEND) to environment action IDs
  outputs:
    - name: archetype_applied
      type: str
      description: Applied human archetype (ISOLATE_AND_STAGE, PAIR_AND_CARRY, etc.)
    - name: compiled_action_ids
      type: list[int]
      description: Sequence of primitive action IDs ready for execution
    - name: contract_specification
      type: str
      description: Safety and behavioral contract guarantees
---

# Macro Skill Compiler

## When to use
- Converting decomposed subgoals (from `backward-planner`) into concrete executable action sequences.
- Synthesizing dynamic action policies (obstacle-avoiding navigation, interactive clicking, frontier exploration).
- Applying recognized human archetypes (`Isolate-and-Stage`, `Pair-and-Carry`, `Lane-Splitting`, `Anchor-Cross`).
- Creating reusable macro-skill snippets that can be cached and transferred across levels.

## When NOT to use
- For initial step 0 visual inspection (use `visual-inspector`).
- For determining overall goal order or solving dependencies (use `backward-planner`).
- For emergency deadlock reset (use `taboo-reset-guard`).

## Workflow
1. **Subgoal & Affordance Ingestion**:
   - Inspect input subgoal, phase, and affordance layout.
2. **Archetype Mapping & Policy Synthesis**:
   - Match the phase to the corresponding human archetype macro or instantiate navigation/exploration policies.
   ```bash
   python scripts/macro_skill_compiler.py --subgoal '{"phase": "ISOLATE_AND_STAGE", "target_colors": [4]}'
   ```
3. **Contract Attachment & Output Emission**:
   - Verify safety contracts (e.g. preserving already formed bonds) and emit compiled action IDs.

## Examples
- Input: `{"phase": "ISOLATE_AND_STAGE", "target_colors": [4]}`
  → Output: `{"archetype_applied": "ISOLATE_AND_STAGE", "compiled_action_ids": [4, 1, 5], "reusable_macro_id": "macro_isolate_and_stage"}`

## Output format
- Structured JSON with `archetype_applied`, `compiled_action_ids`, `sequence_length`, and `contract_specification`.

## Anti-patterns to avoid
- Do not compile actions that violate already learned taboo states.
- Do not break established multi-block pairs during translation.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/macro_skill_compiler.py`: Deterministic CLI tool for unified policy synthesis & archetype compilation.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Catalog of human macro archetypes, policy classes, and action mapping protocols.
