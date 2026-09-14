---
name: backward-planner
description: |
  Comprehensive goal planning engine: decomposes objectives via backward chaining, topological ordering, and staging buffers.
  Resolves causal prerequisite dependencies (keys before doors, switches before bridges) and multi-block rearrangement milestones.
  Use whenever breaking down game objectives into subgoals, scheduling milestone checkpoints, or planning detour routes.
  Do NOT use for single primitive action steps or diagnosing test crashes.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  inputs:
    - name: current_sequence
      type: optional[list[int]]
      description: Current arrangement of movable pieces
    - name: target_sequence
      type: optional[list[int]]
      description: Goal sequence order or clearance objective
    - name: open_grid_spaces
      type: optional[list[list[int]]]
      description: Coordinates available as temporary staging buffers
  outputs:
    - name: backward_chaining_applied
      type: bool
      description: Whether reverse-engineering decomposition was applied
    - name: subgoals
      type: list[dict]
      description: Ordered milestone checkpoints with staging phases
    - name: staging_buffers
      type: list[list[int]]
      description: Allocated buffer coordinates
---

# Backward Planner

## When to use
- Decomposing multi-step ACR-AGI-3 levels into manageable intermediate milestones.
- Formulating prerequisite dependencies (e.g. Collect Key -> Unlock Door -> Reach Goal).
- Planning multi-stage object rearrangements (such as `sk48` block reordering) from terminal targets backward.
- When direct pathing creates a deadlock, requiring intermediate holding areas (buffers).

## When NOT to use
- Executing single-step primitive actuator actions (use compiled macro-skills).
- When observation rules and affordances are completely unknown (use `visual-inspector` and `epistemic-prober`).
- For emergency recovery when state is irreversibly deadlocked (use `taboo-reset-guard`).

## Workflow
1. **Target-Initial Gap & Prerequisite Inspection**:
   - Inspect required final arrangement and topological prerequisites (interactables, barriers, goal).
2. **Buffer Space Allocation & Topological Ordering**:
   - Identify open corridor or sideline spaces to serve as temporary holding zones.
3. **Backward Subgoal Decomposition**:
   - Apply the Isolate-and-Stage -> Secure Keystone -> Incremental Chain -> Final Alignment pipeline.
   ```bash
   python scripts/backward_planner.py --current '[4, 3, 2, 1]' --target '[1, 2, 3, 4]'
   ```

## Examples
- Input: `current=[4, 3, 2, 1]`, `target=[1, 2, 3, 4]`, `buffers=[[0, 0]]`
  → Output: `{"backward_chaining_applied": true, "total_subgoals": 5, "buffer_allocated": true}`

## Output format
- Structured JSON containing `backward_chaining_applied`, `total_subgoals`, `staging_buffers`, and `subgoals`.

## Anti-patterns to avoid
- Do not plan direct paths to the goal when intermediate keys or doors block the way.
- Do not plan forward greedily without allocating a buffer when blocks are in reverse order.
- Do not read large scripts into LLM context window without running `--help`.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy

## Bundled Resources
### `scripts/` (Executable Tools - Zero-dependency)
- `scripts/backward_planner.py`: Deterministic CLI tool for unified milestone planning and backward decomposition.

### `references/` (On-Demand Knowledge)
- `references/guide.md`: Reverse architect principles, topological prerequisite ordering, and buffer staging heuristics.
