---
name: subgoal-decomposer
description: |
  Hierarchical backward subgoal decomposition and temporary buffer staging engine for ARC-AGI-3.
  Decomposes macro goals into topologically ordered subgoals (keys before doors, buffer staging before final alignment)
  to prevent deadlocks, irreversible errors, and blind forward hill-climbing.
  Do NOT use for single-step reflex actions or raw OpenCV color thresholding.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
    - decompose_hierarchical_subgoals
    - get_active_subgoal
    - advance_subgoal
  inputs:
    - name: entities
      type: list[dict]
      description: List of detected objects with type, coordinates, and bounding box
    - name: target_pattern
      type: optional[list[dict]]
      description: Desired target configuration or sequence of entities
  outputs:
    - name: subgoals
      type: list[dict]
      description: Topologically ordered list of subgoals
    - name: active_subgoal
      type: optional[dict]
      description: Current incomplete subgoal to be executed next
---

# Subgoal Decomposer

## When to use
- When solving complex multi-step puzzle levels requiring prerequisite ordering (e.g. collecting a key before reaching a locked door).
- When multiple blocks or pieces must be rearranged, requiring temporary buffer staging (accepting temporary displacement of secondary pieces).
- During cognitive planning (`CognitiveMode.BACKWARD_ARCHITECT` and `CognitiveState.PLANNING`) to break a large goal into manageable waypoints.
- To prevent forward-greedy agents from walking into dead ends without the required keys or prerequisites.

## When NOT to use
- For simple 1-step reflex actions or single-click toggle environments without dependencies.
- For low-level pixel diff computation (use `visual-inspector`).
- For raw geometric shortest-path calculations along an open grid (use `backward-planner`).

## Workflow
1. **Entity & Dependency Identification**:
   - Inspect detected entities from `spatial-grounder` or `visual-inspector`.
   - Identify prerequisite items (keys, switches, toggles), barriers (doors, gates), movable blocks, and terminal goals.
2. **Backward Chaining & Topological Ordering**:
   - Call `decompose_hierarchical_subgoals(entities, target_pattern)`.
   - The engine automatically prioritizes:
     - 0: Buffer staging for disordered/blocking pieces (Detour/Staging).
     - 1: Keys, switches, and prerequisite interactions.
     - 2: Barrier unlocking and clearance.
     - 3: Keystone piece placement.
     - 4: Terminal goal entry.
3. **Incremental Subgoal Execution & Tracking**:
   - Query `get_active_subgoal()` to get the current immediate objective.
   - Once the objective is achieved (e.g. key collected, door opened), call `advance_subgoal()` to progress to the next subgoal.

## Examples
- **Example 1 (Key and Locked Door)**:
  - Input Entities: Key at (2, 5), Door at (6, 5), Exit at (9, 5)
  - Output Subgoals:
    1. `precondition_interaction`: Acquire key at (2, 5)
    2. `barrier_clearance`: Unlock door at (6, 5)
    3. `reach_goal`: Reach terminal exit at (9, 5)
- **Example 2 (Buffer Staging in Block Reordering)**:
  - Input: Red block blocking Blue block from reaching target slot.
  - Output Subgoals:
    1. `stage_buffer`: Move Red block into adjacent temporary buffer cell.
    2. `item_positioning`: Slide Blue block directly into target slot.
    3. `item_positioning`: Retrieve Red block from buffer and place in target slot.

## Progressive Disclosure Guidelines
- **Level 1 (Metadata & Catalog)**: Lightweight frontmatter visible to planner node.
- **Level 2 (Instructions)**: On-demand retrieval via `load_skill("subgoal-decomposer")`.
- **Level 3 (Execution Tools)**: Python deterministic methods (`decompose_hierarchical_subgoals`, `get_active_subgoal`, `advance_subgoal`).

## Anti-patterns to avoid
- Do not attempt to reach the final goal before prerequisite keys or switches are unlocked.
- Do not treat temporary displacement into buffer areas as a failure (hill-climbing detour is necessary).
- Do not generate circular dependency subgoals.

## Requirements & Prerequisites
- Python: >= 3.10

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/subgoal_decomposer.py`: Deterministic topological dependency sorter and hierarchical planner.
