# Backward Planner Reference Guide

## 1. Backward Chaining in Combinatorial Gameplay
Standard reinforcement learning easily gets trapped in local minima (deadlocks) when solving block reordering tasks because moving a block to a temporary siding increases distance metrics. Human solvers operate via Backward Chaining:
1. Fix the prerequisite order by starting from the final unmovable/target boundary.
2. Accept temporary displacement of secondary pieces.
3. Use staging buffers as temporary stack registers.
