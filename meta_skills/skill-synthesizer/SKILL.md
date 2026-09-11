---
name: skill-synthesizer
description: |
  Synthesizes deterministic action policies (`choose_action`) with contract test specifications.
  Use when creating a new game action skill for a subgoal or composing verified skills.
  Do NOT use for environment frame observation or direct failure diagnosis.
license: MIT
allowed-tools: run_skill_script
metadata:
  version: "2.0.0"
  pattern: "meta-workflow"
  inputs:
    - name: affordances
      type: dict
      description: Identified roles (agent, obstacles, goals)
    - name: subgoal
      type: dict
      description: Target milestone coordinates and conditions
  outputs:
    - name: policy_code
      type: str
      description: Implementation of choose_action(obs, info) -> Action
    - name: skill_md
      type: str
      description: Generated SKILL.md documentation
---

# Skill Synthesizer Meta-Skill

## When to use
- Synthesize a modular, testable Python action policy `choose_action(obs, info=None) -> Action`.
- Generate contract test cases (3 positive + 3 negative cases) for an action subgoal.
- Compose existing verified skills from the skill library into a higher-order strategy.

## When NOT to use
- Initial spatial affordance observation from raw grids (use `env-observer`).
- Diagnosing traceback errors or timeout failures (use `failure-diagnoser`).
- Running execution loops in the simulation environment (use `contract-tester`).

## Workflow
1. Milestone Requirement Ingestion: Ingest subgoal objective, player position, and obstacle layout.
2. Code Synthesis: Generate deterministic policy adhering to standard signature:
   ```python
   from acr_agi3.game.env import Action
   def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:
       ...
   ```
3. Contract Test Generation: Scaffold 3 positive scenarios (direct path, detour, arrival) and 3 negative scenarios (wall collision, hazard entry, out-of-bounds).

## Examples
- Subgoal: "Reach Key (4) at (1, 3)" → Generates `choose_action` prioritizing horizontal movement along row 1 with wall avoidance.

## Output format
- Python policy code block and corresponding `SKILL.md` specifications.

## Anti-patterns to avoid
- Never output hardcoded step lists without closed-loop observation checks.
- Do not skip negative contract tests (avoiding obstacles and hazards).

## Requirements & Prerequisites
- Python: >= 3.10
- External packages: numpy

## Bundled Resources
### `references/`
- Reference implementations in `src/acr_agi3/agent/llm_agent.py` and `meta_agent.py`.


