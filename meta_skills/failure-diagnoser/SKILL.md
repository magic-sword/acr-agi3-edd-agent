---
name: failure-diagnoser
description: |
  Analyzes failed trajectories and diagnoses root causes for ACR-AGI-3 skills.
  Use when the user asks to diagnose test failures, extract error patterns, or propose patches.
  Do NOT use for synthesizing initial skills or running baseline simulations.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: failed_report
      type: dict
      description: Detailed test failure logs and tracebacks
    - name: skill_source
      type: str
      description: Source code of failed skill
  outputs:
    - name: failure_mode
      type: str
      description: Categorized failure reason
    - name: repair_plan
      type: dict
      description: Specific repair instructions
---

# Failure Diagnoser

## When to use
- Diagnose failed contract tests or episode terminations in ACR-AGI-3 gameplay.
- Categorize failure modes (collision with wall, fatal trap contact, oscillatory loop, out-of-bounds).
- Produce targeted bug fixes and patch plans for `skill-synthesizer`.

## When NOT to use
- Running initial passing simulations (use `contract-tester`).
- Extracting raw environmental affordances (use `env-observer`).
- Static ARC puzzle transformation debugging.

## Workflow
1. Trace Ingestion and Error Parsing: To extract the failing frame, target coordinates, and action trace:
   ```bash
   python scripts/failure_diagnoser.py --input "data"
   ```
2. Failure Root-Cause Classification: To classify whether failure is caused by path planning, incorrect affordance labeling, or unmet game state constraints.
3. Repair Directive Formulation: To generate targeted modification instructions and pass them to `skill-synthesizer` for iterative self-repair.

## Examples
- Input: "Action RIGHT resulted in lethal hazard contact at (2, 3)" → Output: `{"failure_mode": "HAZARD_COLLISION", "culprit_action": "RIGHT", "required_fix": "Add cell (2, 3) to taboo set"}`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not make vague suggestions; provide precise code lines and constraint fixes.
- Do not repeat the same broken action sequence without updating environmental constraints.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/failure_diagnoser.py`: Deterministic CLI tool for failure diagnosis.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, failure taxonomy, and repair patterns.
