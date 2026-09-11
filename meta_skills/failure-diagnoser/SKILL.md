---
name: failure-diagnoser
description: |
  Analyzes failed trajectories and diagnoses abstract failure modes for ACR-AGI-3 skills.
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
      description: Categorized failure reason (ContractViolation, ArrayAmbiguity, Stagnation, InvariantBreach)
    - name: repair_plan
      type: dict
      description: Specific concise repair directive
---

# Failure Diagnoser

## When to use
- Diagnose failed contract tests or episode terminations in ACR-AGI-3 gameplay.
- Categorize failure modes into abstract taxonomies (contract violation, runtime exception, behavioral stagnation, safety breach).
- Produce concise, actionable repair directives for `skill-synthesizer`.

## When NOT to use
- Running initial passing simulations (use `contract-tester`).
- Extracting raw environmental affordances (use `env-observer`).
- Static ARC puzzle transformation debugging.

## Workflow
1. Trace Ingestion and Error Parsing: To extract the failing frame, stack trace, and execution metrics:
   ```bash
   python scripts/failure_diagnoser.py --input "data"
   ```
2. Failure Root-Cause Classification: To classify failure into contract violations, language exceptions, behavioral stagnation, or safety breaches.
3. Repair Directive Formulation: To generate concise, targeted modification instructions and pass them to `skill-synthesizer` for iterative self-repair.

## Examples
- Input: "The truth value of an array with more than one element is ambiguous" → Output: `{"failure_category": "ArrayComparisonAmbiguity", "directive": "Use np.argwhere(obs == color) instead of direct boolean comparison"}`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not pass verbose raw stack traces to the LLM; distill into concise directives.
- Do not make suggestions dependent on specific board coordinates; maintain abstract contract guidance.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/failure_diagnoser.py`: Deterministic CLI tool for abstract failure diagnosis.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, abstract failure taxonomy, and repair patterns.
