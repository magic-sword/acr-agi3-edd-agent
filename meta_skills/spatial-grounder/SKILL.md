---
name: spatial-grounder
description: |
  Advanced hierarchical object detection and geometric spatial grounding engine for ARC-AGI-3.
  Segments multi-color composite objects (buttons, sprites, dynamic entities), tracks causal pixel differences,
  and provides discrete coordinate targets for ACTION6 (click_at) to prevent hallucinated clicks and misclicks.
  Do NOT use for high-level multi-step planning or general directional movement.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "2.0.0"
  adk_additional_tools:
    - inspect_detected_objects
    - get_object_coordinates
    - inspect_clickable_anchors
    - snap_to_anchor
  inputs:
    - name: current_grid
      type: list[list[int]] | numpy.ndarray
      description: Current 2D visual observation grid
    - name: exclude_background
      type: optional[bool]
      description: Whether to auto-detect and exclude the dominant background color
  outputs:
    - name: objects
      type: list[dict]
      description: List of detected composite objects with id, type, center, bbox, colors, area, and is_dynamic
    - name: anchors
      type: list[dict]
      description: List of detected clickable anchors with id, centroid (x, y), bounding box, color, and size
    - name: anchor_summary
      type: str
      description: Human/LLM readable bullet list of available click targets
---

# Spatial Grounder

## When to use
- When the environment requires clicking (`ACTION6` / `click_at`).
- When detecting cohesive multi-color objects (buttons, sprites, icons) on the screen without splitting them by color.
- When querying the system for a structured list of clickable entities via `inspect_detected_objects()`.
- When getting exact pixel coordinates for a specific object via `get_object_coordinates(object_id)`.
- When determining the exact `(col, row)` coordinates of interactive buttons, colored tiles, or key items.
- When snapping an imprecise or approximate click coordinate to the nearest valid object center.

## When NOT to use
- In games where only directional steps (`UP`, `DOWN`, `LEFT`, `RIGHT`) are allowed and clicks are disabled.
- For high-level path planning or causal condition graph traversal (use `backward-planner`).

## Workflow
1. **Hierarchical Object Detection**:
   - Call `inspect_detected_objects(filter_mode='interactive')` to get discrete, cohesive objects on the screen.
   - The engine automatically filters out giant background regions (>12% of screen) and noise.
   - Multi-color sprites (e.g. black outline with red center) are grouped as single objects.
2. **Dynamic Causal Tracking**:
   - Objects that changed appearance in response to recent actions are tagged with `is_dynamic: true` and `DYNAMIC_ENTITY`.
3. **Discrete Object Selection**:
   - Choose a target object by its ID (e.g. Object #1, Object #2).
   - Use `get_object_coordinates(object_id)` to retrieve exact `(x=col, y=row)` coordinates.
4. **Coordinate Snapping & Taboo Guard**:
   - When executing `click_at(x=col, y=row)`, `snap_to_anchor` automatically avoids previously failed taboo coordinates.

## Examples
- Example 1 (Querying detected objects):
  - Call: `inspect_detected_objects(filter_mode='interactive')`
  - Output: `[{"id": 1, "type": "BUTTON_CANDIDATE", "center": {"x": 14, "y": 12}, "colors": ["Red", "Black"], "area": 16, "is_dynamic": true}]`
- Example 2 (Getting pinpoint coordinates):
  - Call: `get_object_coordinates(object_id=1)`
  - Output: `{"x": 14, "y": 12, "type": "BUTTON_CANDIDATE", "success": true}`

## Anti-patterns to avoid
- Do not output arbitrary or default coordinates like `(0, 0)` or `(5, 5)` when clicking; always select a valid object or anchor.
- Do not perform click sweeps on giant background areas; use `inspect_detected_objects` to target valid foreground entities.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy, scipy, cv2 (opencv-python), standard library (argparse, json, typing)

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/spatial_grounder.py`: CLI and helper class for composite object detection, anchor extraction, and coordinate snapping.
