# Constraint Learner Reference Guide (ACR-AGI-3)

## Overview
Constraint Learner records safety boundaries and taboo conditions from gameplay experience.

## Learned Constraint Types
1. **Lethal Elements (Taboo)**: Cells or colors that immediately trigger episode failure when entered.
2. **One-Way Passages**: Edges in the state graph that cannot be traversed in reverse (ledges, one-way conveyor belts).
3. **Resource Exhaustion**: Minimum step budgets or move limits required to prevent stagnation penalties.
4. **State-Dependent Hazards**: Enemies that move in patrol patterns requiring timing-dependent constraints.
