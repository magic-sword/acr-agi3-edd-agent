# Skill Synthesizer Reference Guide (ACR-AGI-3)

## Overview
Skill Synthesizer compiles environmental affordances, subgoals, and constraints into executable Python policy skills.

## Structure of Generated Skills
1. `SKILL.md`: Frontmatter adhering to Google ADK 2.0 with Progressive Disclosure sections.
2. `scripts/<skill_name>.py`: Zero-dependency deterministic execution script implementing `choose_action(obs) -> Action`.
3. `tests/test_config.json`: Evaluation criteria configuration.
4. `tests/<skill_name>.test.json`: Full 6-case EvalSet (3 positive reachable trajectories + 3 negative boundary scenarios).

## Contract Test Standards (EDD Firewall Gate)
- **Positive Test 1**: Direct path from start to goal without obstacles.
- **Positive Test 2**: Path requiring obstacle avoidance.
- **Positive Test 3**: Path involving key-door or switch interaction.
- **Negative Test 1**: Target blocked entirely by walls (assert exception or NOOP).
- **Negative Test 2**: Action into lethal trap/hazard (assert prohibition).
- **Negative Test 3**: Out-of-bounds action request (assert boundary clamp).
