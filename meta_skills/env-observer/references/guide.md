# Environment Observer Reference Guide (ACR-AGI-3)

## Overview
ACR-AGI-3 operates on interactive, dynamic game environments where an agent takes discrete actions (UP, DOWN, LEFT, RIGHT, etc.) to achieve a goal.
The environment observer extracts visual gestalts, affordances, and physical invariants.

## Affordance Roles
- **Player/Agent**: Controllable entity that changes position upon valid actions.
- **Static Obstacles (Walls)**: Non-walkable cells that block movement without terminating the game.
- **Goal (Target)**: Cell or item that triggers a positive reward and successful episode termination.
- **Hazards (Traps/Enemies)**: Lethal elements causing immediate failure or penalty upon contact.
- **Interactables (Keys, Switches, Doors)**: Elements that toggle state upon agent contact or proximity.

## Dynamics Inference
Given transitions `(s, a, s')`:
- Displacement vector: `delta = pos(s') - pos(s)`.
- Action alignment: If `a = UP` and `delta = (-1, 0)`, action maps to standard grid movement.
- Collision elasticity: If `pos(s') == pos(s)` despite directional action, cell is impassable.
