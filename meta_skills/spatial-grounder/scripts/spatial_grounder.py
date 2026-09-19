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
    def detect_composite_objects(
        grid: Any,
        last_grid: Optional[Any] = None,
        min_size: int = 2,
        max_area_ratio: float = 0.12,
    ) -> List[Dict[str, Any]]:
        """Extracts multi-color composite objects (sprites, buttons, entities) using contour analysis.

        Groups multi-color pixels enclosed in the same boundary into a single cohesive entity.
        Filters out background-scale giant areas (walls, letterboxing) and tags dynamically changing elements.
        """
        arr = _normalize_2d(grid)
        if arr.size == 0:
            return []

        h, w = arr.shape
        total_pixels = h * w
        max_size = int(total_pixels * max_area_ratio)

        # 1. 自動背景色検出: 外周境界の最頻色
        border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        counts = np.bincount(border_pixels, minlength=10)
        bg_color = int(np.argmax(counts))

        # 2. 前景バイナリマスクの作成
        fg_mask = (arr != bg_color).astype(np.uint8)

        # 3. 前後フレーム差分マスクの作成 (動的オブジェクト同定用)
        diff_mask = None
        if last_grid is not None:
            last_arr = _normalize_2d(last_grid)
            if last_arr.shape == arr.shape:
                diff_mask = (arr != last_arr)

        objects: List[Dict[str, Any]] = []

        try:
            import cv2
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, cnt in enumerate(contours):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # 輪郭内マスク
                roi_mask = np.zeros((bh, bw), dtype=np.uint8)
                roi_cnt = cnt - np.array([bx, by])
                cv2.drawContours(roi_mask, [roi_cnt], -1, 1, thickness=-1)
                size = int(np.sum(roi_mask))

                if size < min_size or size > max_size:
                    continue

                # 重心計算
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(round(M["m10"] / M["m00"]))
                    cy = int(round(M["m01"] / M["m00"]))
                else:
                    cx = bx + bw // 2
                    cy = by + bh // 2

                cx = max(0, min(w - 1, cx))
                cy = max(0, min(h - 1, cy))

                # 輪郭内に含まれるユニーク色
                roi_grid = arr[by:by+bh, bx:bx+bw]
                contained_colors = [int(c) for c in np.unique(roi_grid[roi_mask == 1]) if c != bg_color]
                color_names = [COLOR_NAMES.get(c, f"Color {c}") for c in contained_colors]

                # 動的変化フラグの判定
                is_dynamic = False
                if diff_mask is not None:
                    roi_diff = diff_mask[by:by+bh, bx:bx+bw]
                    is_dynamic = bool(np.any(roi_diff[roi_mask == 1]))

                # スタイル・種別の分類
                aspect = bw / max(1, bh)
                if is_dynamic:
                    obj_type = "DYNAMIC_ENTITY"
                    confidence = "HIGH"
                elif 4 <= size <= 64 and 0.4 <= aspect <= 2.5:
                    obj_type = "BUTTON_CANDIDATE"
                    confidence = "HIGH"
                else:
                    obj_type = "SPRITE_CANDIDATE"
                    confidence = "MEDIUM"

                objects.append({
                    "id": idx,
                    "type": obj_type,
                    "center": {"x": cx, "y": cy},
                    "x": cx,
                    "y": cy,
                    "bbox": [bx, by, bx + bw, by + bh],
                    "width": bw,
                    "height": bh,
                    "area": size,
                    "colors": color_names,
                    "color_ids": contained_colors,
                    "is_dynamic": is_dynamic,
                    "confidence": confidence,
                })
        except ImportError:
            # OpenCV がない場合のフォールバック（scipy.ndimage.label）
            from scipy.ndimage import label
            labeled, num_features = label(fg_mask)
            for idx in range(1, num_features + 1):
                mask = (labeled == idx)
                size = int(np.sum(mask))
                if size < min_size or size > max_size:
                    continue
                rows, cols = np.where(mask)
                min_r, max_r = int(np.min(rows)), int(np.max(rows))
                min_c, max_c = int(np.min(cols)), int(np.max(cols))
                cy, cx = int(round(np.mean(rows))), int(round(np.mean(cols)))
                contained_colors = [int(c) for c in np.unique(arr[mask]) if c != bg_color]

                is_dynamic = bool(np.any(diff_mask[mask])) if diff_mask is not None else False
                bw, bh = max_c - min_c + 1, max_r - min_r + 1
                aspect = bw / max(1, bh)
                obj_type = "DYNAMIC_ENTITY" if is_dynamic else ("BUTTON_CANDIDATE" if 4 <= size <= 64 and 0.4 <= aspect <= 2.5 else "SPRITE_CANDIDATE")

                objects.append({
                    "id": idx - 1,
                    "type": obj_type,
                    "center": {"x": cx, "y": cy},
                    "x": cx,
                    "y": cy,
                    "bbox": [min_c, min_r, max_c, max_r],
                    "width": bw,
                    "height": bh,
                    "area": size,
                    "colors": [COLOR_NAMES.get(c, f"Color {c}") for c in contained_colors],
                    "color_ids": contained_colors,
                    "is_dynamic": is_dynamic,
                    "confidence": "HIGH" if (is_dynamic or obj_type == "BUTTON_CANDIDATE") else "MEDIUM",
                })

        # 優先度順にソート (動的変化 > ボタン候補 > 面積適正度)
        def _score(o: Dict[str, Any]) -> Tuple[int, int, int]:
            dyn_score = 100 if o["is_dynamic"] else 0
            type_score = 50 if o["type"] == "BUTTON_CANDIDATE" else 10
            # 面積が極端すぎず適度なものを好む (16〜40px が最高)
            size_score = -abs(o["area"] - 25)
            return (dyn_score + type_score, size_score, -o["area"])

        objects.sort(key=_score, reverse=True)
        # ID の再採番 (1-indexed for LLM user friendliness)
        for i, obj in enumerate(objects, 1):
            obj["id"] = i

        return objects

    @staticmethod
    def extract_clickable_anchors(
        grid: Any,
        last_grid: Optional[Any] = None,
        exclude_colors: Optional[List[int]] = None,
        min_size: int = 2,
        max_area_ratio: float = 0.12,
    ) -> List[Dict[str, Any]]:
        """Finds clickable anchors, prioritizing composite objects over raw color blobs."""
        # まず複合オブジェクト検出を実行
        comp_objs = SpatialGrounder.detect_composite_objects(
            grid=grid,
            last_grid=last_grid,
            min_size=min_size,
            max_area_ratio=max_area_ratio,
        )

        anchors: List[Dict[str, Any]] = []
        for obj in comp_objs:
            primary_color = obj["color_ids"][0] if obj.get("color_ids") else 0
            if exclude_colors and primary_color in exclude_colors:
                continue
            anchors.append({
                "id": obj["id"],
                "color": primary_color,
                "color_name": obj["colors"][0] if obj.get("colors") else "Mixed",
                "all_colors": obj.get("colors", []),
                "type": obj.get("type", "SPRITE_CANDIDATE"),
                "centroid": (obj["x"], obj["y"]),
                "x": obj["x"],
                "y": obj["y"],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "is_dynamic": obj.get("is_dynamic", False),
            })

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
            type_str = f" [{a.get('type')}]" if a.get('type') else ""
            dyn_str = " [DYNAMIC]" if a.get('is_dynamic') else ""
            lines.append(
                f"- [Anchor {a['id']}] {a['color_name']} at (col={a['x']}, row={a['y']}), "
                f"bbox=[{a['bbox'][0]},{a['bbox'][1]}]..[{a['bbox'][2]},{a['bbox'][3]}], area={a['area']}px{type_str}{dyn_str}"
            )
        if len(anchors) > max_items:
            lines.append(f"... and {len(anchors) - max_items} more smaller anchors.")
        return "\n".join(lines)

    @classmethod
    def format_detected_objects_prompt(cls, objects: List[Dict[str, Any]], max_items: int = 10) -> str:
        """Formats the list of detected composite objects into an LLM-friendly summary."""
        if not objects:
            return "No distinct interactive objects detected on screen."

        lines = [f"Detected Objects & Sprites (Total {len(objects)}):"]
        for o in objects[:max_items]:
            dyn_str = " [DYNAMIC / RECENTLY CHANGED]" if o.get("is_dynamic") else ""
            color_str = "/".join(o.get("colors", [])) or "Mixed"
            lines.append(
                f"- [Object #{o['id']}] Type: {o['type']}, Center: (col={o['x']}, row={o['y']}), "
                f"Colors: [{color_str}], Size: {o['area']}px, BBox: {o['bbox']}{dyn_str}"
            )
        if len(objects) > max_items:
            lines.append(f"... and {len(objects) - max_items} more smaller background objects.")
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
