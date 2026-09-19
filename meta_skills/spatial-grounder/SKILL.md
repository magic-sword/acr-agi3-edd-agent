---
name: spatial-grounder
description: |
  Geometric spatial grounding and clickable anchor perception engine for ARC-AGI-3.
  Extracts connected components, object centroids, and interactive element anchors
  to provide discrete coordinate targets for ACTION6 (click_at) and prevent misclicks.
  Do NOT use for high-level multi-step planning or general directional movement.
license: MIT
allowed-tools: run_skill_script load_skill_resource
metadata:
  pattern: workflow
  version: "1.0.0"
  adk_additional_tools:
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
- When determining the exact `(col, row)` coordinates of interactive buttons, colored tiles, or key items.
- When mapping user intent ("click the red block") to an exact pixel anchor rather than hallucinating coordinates like `(0, 0)`.
- When snapping an imprecise or approximate click coordinate to the nearest valid object center.

## When NOT to use
- In games where only directional steps (`UP`, `DOWN`, `LEFT`, `RIGHT`) are allowed and clicks are disabled.
- For high-level path planning or causal condition graph traversal (use `backward-planner`).

## Workflow
1. **Geometric Component Segmentation**:
   - Detect background color as the most frequent color in the border / overall grid.
   - Segment all contiguous foreground components of distinct colors using connected component analysis.
2. **Anchor Centroid Calculation**:
   - Compute the center of mass `(centroid_x, centroid_y)` for each component.
   - Tag each anchor with a unique index, bounding box `[min_x, min_y, max_x, max_y]`, pixel area, and color name.
3. **Discrete Target Presentation**:
   - Present candidates to the action decision node as numbered discrete choices:
     - `Target 0: Red block at (col=14, row=22), size=9px`
     - `Target 1: Blue button at (col=30, row=5), size=16px`
4. **Coordinate Snapping Guard**:
   - If an agent issues an approximate or unspecified click, snap to the nearest target anchor within allowable radius.

## Examples
- Example 1 (Interactive keypad grounding):
  - Input: 64x64 grid with yellow background and 3 distinct colored buttons at row 3.
  - Output: `[{"id": 0, "color_name": "Red", "centroid": [23, 3], "area": 20}, {"id": 1, "color_name": "Teal", "centroid": [31, 3], "area": 20}]`
- Example 2 (Snapping imprecise click):
  - Input: Loose click at `(24, 4)`, nearest anchor is `(23, 3)`.
  - Output: Snapped coordinate `(23, 3)` with target ID 0.

## Anti-patterns to avoid
- Do not output arbitrary or default coordinates like `(0, 0)` or `(5, 5)` when clicking; always select a valid anchor ID.
- Do not perform click sweeps on the background color; only interact with foreground anchors.

## Requirements & Prerequisites
- Python: >= 3.10
- Dependencies: numpy, scipy, standard library (argparse, json, typing)

## Bundled Resources
### `scripts/` (Executable Tools)
- `scripts/spatial_grounder.py`: CLI and helper class for anchor extraction and coordinate snapping.
