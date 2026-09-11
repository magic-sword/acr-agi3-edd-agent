# Subgoal Decomposer Reference Guide (ACR-AGI-3)

## Overview
Subgoal Decomposer transforms long-horizon planning problems into short-horizon reachable tasks.

## Subgoal Hierarchy
1. **Precondition Resolution**: Acquiring keys, flipping switches, clearing movable boxes.
2. **Path Segmentation**: Reaching topological choke points (corridors, portals, portals).
3. **Terminal Execution**: Navigating from the final milestone to the stage clearance exit.

## Dynamic Replanning Trigger
- If an agent encounters an unexpected obstacle or locked door, the current subgoal is suspended and an intermediate resolution subgoal is inserted.
