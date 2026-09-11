---
name: skill-synthesizer
description: |
  Synthesizes executable action policies and contract tests for ACR-AGI-3 games.
  Use when the user asks to generate Python skills, create contract tests, or build action policies.
  Do NOT use for static grid color transformations or diagnosing test failures.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: template_generator
  version: "2.0.0"
  inputs:
    - name: affordances
      type: dict
      description: Identified roles (agent, obstacles, goals, hazards)
    - name: dynamics_rules
      type: list[str]
      description: Inferred game mechanics
  outputs:
    - name: skill_code
      type: str
      description: Complete Python policy code
    - name: contract_tests
      type: list[dict]
      description: 3 positive and 3 negative test cases
---

# Skill Synthesizer

## When to use
- Synthesize actionable Python policy scripts for unknown ACR-AGI-3 game environments.
- Generate mandatory Evaluation-Driven Development (EDD) contract test suites (3 positive + 3 negative cases).
- Package domain algorithms (A* pathfinding, BFS maze routing, key-door solvers) into modular skill units.

## When NOT to use
- Static matrix math transformations for ARC-1/2 puzzles.
- Failure diagnosis or repairing broken skills (use `failure-diagnoser`).
- Evaluating contract tests against simulation environments (use `contract-tester`).

## Workflow
1. Reconnaissance and Specification Review: To inspect affordances, subgoals, and environmental constraints:
   ```bash
   python scripts/skill_synthesizer.py --help
   ```
2. Core Synthesis: To generate deterministic policy code with `choose_action(obs) -> Action` signature:
   ```bash
   python scripts/skill_synthesizer.py --input "data"
   ```
3. Contract Test Construction: To produce 3 positive reachable trajectories and 3 negative boundary scenarios (collision, traps, out-of-bounds).

## Examples
- Input: "Synthesize grid navigation skill with 3 positive and 3 negative tests" → Output: `Generated skill 'grid-navigator' with 6 contract test cases`

## Output format
- Return direct operational summary and structured result files.

## Anti-patterns to avoid
- Do not commit generated concrete skills into `meta_skills/`; output to `generated_skills/`.
- Never produce policy code without accompanying 3 positive and 3 negative contract tests.
- Do not use conversational phrasing in generated code comments.

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/skill_synthesizer.py`: Core CLI tool for Skill Synthesizer.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Specifications, templates, and contract test design rules.
