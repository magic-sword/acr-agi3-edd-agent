# Game Style Intuitor Reference Guide (ACR-AGI-3)

## Overview
Game Style Intuitor perceives overall screen texture, border permeability, obstacle density, and spatial symmetry to infer high-level game genre and initial strategic posture.

## Visual Gestalt Taxonomy
1. **OPEN_EXPLORATION**:
   - Visual Sign: Perimeter borders (top, bottom, left, right edges) are open without continuous enclosing walls.
   - Gameplay Reality: Goal is often off-screen, or episode progress requires camera-scrolling across outer boundaries.
   - Initial Posture: Prioritize perimeter edge exploration rather than hunting for visible goals.

2. **CLOSED_MAZE**:
   - Visual Sign: Outer edges are fully enclosed by solid boundary walls, and internal obstacle density is high (>20%).
   - Gameplay Reality: Traditional indoor dungeon or labyrinth puzzle.
   - Initial Posture: Execute deterministic topological detour search (BFS/A*).

3. **ITEM_TRIGGER_PUZZLE**:
   - Visual Sign: Multiple isolated 1-3 cell colored objects distributed across the board (keys, switches, gates).
   - Gameplay Reality: Sequential causal dependencies. Direct path to goal is blocked until prerequisites are triggered.
   - Initial Posture: Formulate precondition milestones targeting isolated interactables first.

4. **SYMMETRIC_PATTERN**:
   - Visual Sign: Strong horizontal or vertical reflectional symmetry (>80%).
   - Gameplay Reality: Puzzle requires maintaining balance, mirror reflection, or complementary shape completion.
   - Initial Posture: Check symmetrical action parity across axes.
