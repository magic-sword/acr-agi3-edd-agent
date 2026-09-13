---
name: contract-tester
description: |
  Executes EDD contract test gates and sandboxed simulations for ACR-AGI-3 skills.
  Use when the user asks to validate generated skills, run contract tests, or check firewall gates.
  Do NOT use for synthesizing new skills or diagnosing failure causes.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: skill_path
      type: str
      description: Directory path of candidate skill
    - name: eval_cases
      type: list[dict]
      description: 3 positive and 3 negative test cases
  outputs:
    - name: passed
      type: bool
      description: True if 100% of test cases pass
    - name: report
      type: dict
      description: Detailed test run report
---

# Contract Tester

## When to use
- Execute Evaluation-Driven Development (EDD) contract test suites against candidate skills.
- Enforce the 100% pass firewall gate (3 positive + 3 negative cases) before skill adoption.
- Run deterministic sandbox simulations to detect boundary violations, infinite loops, and exceptions.

## When NOT to use
- Generating new skills or writing test definitions (use `skill-synthesizer`).
- Diagnosing why a contract test failed (use `failure-diagnoser`).
- Direct static transformation tests for ARC-1/2 puzzles.

## Workflow
1. Reconnaissance and Test Ingestion: To inspect the candidate skill directory, test config, and evaluation cases:
   ```bash
   python scripts/contract_tester.py --input "data"
   ```
2. Sandboxed Execution: To execute all 3 positive and 3 negative contract tests in an isolated Python environment with timeout safeguards.
3. Firewall Gate Verdict: To verify all test assertions pass; emit promotion signal if passed, or route error logs to `failure-diagnoser`.

## Examples
- Input: "Run contract tests on generated_skills/maze-solver" → Output: `Passed 6/6 contract tests (100%). Gate: APPROVED`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Never promote a skill if even 1 negative test fails (e.g., trap avoidance).
- Do not run un-sandboxed code without timeout limits.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: pytest, numpy, acr_agi3

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/contract_tester.py`: Deterministic CLI tool for executing contract tests.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, failure criteria, and evaluation rules.
