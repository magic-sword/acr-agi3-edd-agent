# Failure Diagnoser Reference Guide (ACR-AGI-3)

## Overview
Failure Diagnoser analyzes why an action or synthesized skill failed and produces precise repair directives.

## Failure Taxonomy
- **WALL_COLLISION**: Agent attempted to move into an impassable obstacle cell. Fix: update walkable mask.
- **HAZARD_COLLISION**: Agent touched a lethal trap/enemy resulting in game over. Fix: register cell/color in taboo set.
- **OSCILLATORY_LOOP**: Agent repeated actions A -> B -> A without making progress. Fix: add visited state penalty or cycle detection.
- **TIMEOUT_STAGNATION**: Episode budget exceeded before reaching goal. Fix: optimize heuristic or break down into closer subgoals.
- **PRECONDITION_UNMET**: Attempted to open door without holding required key. Fix: route through key subgoal first.
