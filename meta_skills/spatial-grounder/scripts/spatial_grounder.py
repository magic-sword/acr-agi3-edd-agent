"""Spatial Grounder: Geometric spatial grounding and clickable anchor perception engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


COLOR_NAMES = {
    0: "Black",
    1: "Blue",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Gray",
    6: "Magenta",
    7: "Orange",
    8: "Teal",
    9: "Maroon",
}


def _normalize_2d(grid: Any) -> np.ndarray:
    if isinstance(grid, np.ndarray):
        arr = grid
    else:
        arr = np.array(grid, dtype=int)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[-1]
    return arr


class SpatialGrounder:
    """Extracts clickable anchors and grounds spatial coordinates on game boards."""

    @staticmethod
    def extract_clickable_anchors(
        grid: Any,
        exclude_colors: Optional[List[int]] = None,
        min_size: int = 1,
        max_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Finds connected components of distinct foreground colors and returns their centroids."""
        arr = _normalize_2d(grid)
        if arr.size == 0:
            return []

        h, w = arr.shape
        # Auto-detect background color: most frequent color in border
        border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        counts = np.bincount(border_pixels, minlength=10)
        bg_color = int(np.argmax(counts))

        ignored = set(exclude_colors or [])
        ignored.add(bg_color)

        anchors: List[Dict[str, Any]] = []
        anchor_id = 0

        # Connected component labeling per color
        try:
            from scipy.ndimage import label
            has_scipy = True
        except ImportError:
            has_scipy = False

        unique_colors = [int(c) for c in np.unique(arr) if c not in ignored]

        for c in unique_colors:
            mask = (arr == c)
            if has_scipy:
                labeled, num_features = label(mask)
                for feat_idx in range(1, num_features + 1):
                    feat_mask = (labeled == feat_idx)
                    size = int(np.sum(feat_mask))
                    if size < min_size:
                        continue
                    if max_size is not None and size > max_size:
                        continue
                    rows, cols = np.where(feat_mask)
                    min_r, max_r = int(np.min(rows)), int(np.max(rows))
                    min_c, max_c = int(np.min(cols)), int(np.max(cols))
                    cy, cx = int(np.round(np.mean(rows))), int(np.round(np.mean(cols)))

                    anchors.append({
                        "id": anchor_id,
                        "color": c,
                        "color_name": COLOR_NAMES.get(c, f"Color {c}"),
                        "centroid": (cx, cy),
                        "x": cx,
                        "y": cy,
                        "bbox": [min_c, min_r, max_c, max_r],
                        "area": size,
                    })
                    anchor_id += 1
            else:
                # Pure python BFS fallback
                visited = np.zeros_like(mask, dtype=bool)
                for r in range(h):
                    for col in range(w):
                        if mask[r, col] and not visited[r, col]:
                            # BFS
                            comp_r, comp_c = [], []
                            queue = [(r, col)]
                            visited[r, col] = True
                            while queue:
                                curr_r, curr_c = queue.pop(0)
                                comp_r.append(curr_r)
                                comp_c.append(curr_c)
                                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nr, nc = curr_r + dr, curr_c + dc
                                    if 0 <= nr < h and 0 <= nc < w:
                                        if mask[nr, nc] and not visited[nr, nc]:
                                            visited[nr, nc] = True
                                            queue.append((nr, nc))

                            size = len(comp_r)
                            if size < min_size:
                                continue
                            if max_size is not None and size > max_size:
                                continue

                            min_r, max_r = min(comp_r), max(comp_r)
                            min_c, max_c = min(comp_c), max(comp_c)
                            cy = int(round(sum(comp_r) / size))
                            cx = int(round(sum(comp_c) / size))

                            anchors.append({
                                "id": anchor_id,
                                "color": c,
                                "color_name": COLOR_NAMES.get(c, f"Color {c}"),
                                "centroid": (cx, cy),
                                "x": cx,
                                "y": cy,
                                "bbox": [min_c, min_r, max_c, max_r],
                                "area": size,
                            })
                            anchor_id += 1

        return anchors

    @staticmethod
    def snap_to_anchor(
        x: int,
        y: int,
        anchors: List[Dict[str, Any]],
        max_dist: float = 12.0,
        taboo_coords: Optional[List[Tuple[int, int]]] = None,
        taboo_dist: float = 3.0,
    ) -> Tuple[int, int, Optional[int]]:
        """Snaps an arbitrary coordinate (x, y) to the closest anchor centroid if within max_dist.

        Avoids anchors that are within taboo_dist of any coordinate in taboo_coords.
        """
        if not anchors:
            return x, y, None

        best_dist = float("inf")
        best_anchor = None
        taboos = taboo_coords or []

        for a in anchors:
            ax, ay = a["x"], a["y"]
            # 禁忌座標に近いアンカーは除外
            if any(np.hypot(ax - tx, ay - ty) <= taboo_dist for tx, ty in taboos):
                continue
            dist = np.hypot(ax - x, ay - y)
            if dist < best_dist:
                best_dist = dist
                best_anchor = a

        if best_anchor and best_dist <= max_dist:
            return best_anchor["x"], best_anchor["y"], best_anchor["id"]
        return x, y, None

    @classmethod
    def format_anchors_prompt(cls, anchors: List[Dict[str, Any]], max_items: int = 15) -> str:
        """Formats the list of clickable anchors into a clear prompt snippet."""
        if not anchors:
            return "No distinct clickable anchors detected on board (default to center or active region)."

        lines = [f"Clickable Target Anchors (Total {len(anchors)}):"]
        for a in anchors[:max_items]:
            lines.append(
                f"- [Anchor {a['id']}] {a['color_name']} at (col={a['x']}, row={a['y']}), "
                f"bbox=[{a['bbox'][0]},{a['bbox'][1]}]..[{a['bbox'][2]},{a['bbox'][3]}], area={a['area']}px"
            )
        if len(anchors) > max_items:
            lines.append(f"... and {len(anchors) - max_items} more smaller anchors.")
        return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Spatial Grounder CLI for clickable anchor extraction.")
    parser.add_argument("--grid", type=str, help="JSON string representing 2D grid array")
    parser.add_argument("--file", type=str, help="Path to JSON file containing grid array")
    parser.add_argument("--snap-x", type=int, default=None, help="Optional X coordinate to snap")
    parser.add_argument("--snap-y", type=int, default=None, help="Optional Y coordinate to snap")
    args = parser.parse_args()

    grid = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            grid = data.get("grid", data)
    elif args.grid:
        grid = json.loads(args.grid)
    else:
        grid = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]

    anchors = SpatialGrounder.extract_clickable_anchors(grid)
    result = {
        "anchors": anchors,
        "summary": SpatialGrounder.format_anchors_prompt(anchors),
    }

    if args.snap_x is not None and args.snap_y is not None:
        sx, sy, aid = SpatialGrounder.snap_to_anchor(args.snap_x, args.snap_y, anchors)
        result["snap_result"] = {"x": sx, "y": sy, "anchor_id": aid}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
