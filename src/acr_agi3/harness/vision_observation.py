"""ARC-AGI-3 視覚認識・マルチモーダル観測ハーネス.

ゲーム環境のグリッド観測（FrameData）を公式 10 色カラー画像 (PIL Image / PNG Blob) に変換し、
Google ADK 2.0 / GenAI の Part オブジェクトとして LLM/VLM に提示します。
"""

from __future__ import annotations

import io
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.genai.types import Blob, Part

from acr_agi3.dsl.renderer import ARC_COLORS, render_console_observation, render_grid_to_image


def normalize_grid(grid_data: Any) -> np.ndarray:
    """3次元テンソル (N, H, W) またはリストを最新フレームの 2D numpy 配列 (H, W) へ正規化."""
    arr = np.array(grid_data, dtype=int)
    if arr.ndim == 3:
        arr = arr[-1]
    elif arr.ndim == 1:
        arr = np.array([arr])
    return arr


def detect_interactable_objects(grid: np.ndarray, max_objects: int = 15) -> List[Dict[str, Any]]:
    """グリッド内の前景オブジェクト（複合スプライト・ボタン）を検出し、重心や BBox を抽出."""
    if grid.ndim != 2:
        return []
    h, w = grid.shape
    if h == 0 or w == 0:
        return []

    try:
        from pathlib import Path
        import sys
        _sg_dir = Path(__file__).resolve().parents[3] / "meta_skills" / "spatial-grounder" / "scripts"
        if str(_sg_dir) not in sys.path and _sg_dir.exists():
            sys.path.insert(0, str(_sg_dir))
        from spatial_grounder import SpatialGrounder

        comp_objs = SpatialGrounder.detect_composite_objects(grid, min_size=1, max_area_ratio=0.12)
        if comp_objs:
            results = []
            for obj in comp_objs[:max_objects]:
                min_x = obj["bbox"][0]
                min_y = obj["bbox"][1]
                max_x = max(min_x, obj["bbox"][2] - 1)
                max_y = max(min_y, obj["bbox"][3] - 1)
                results.append({
                    "id": obj["id"],
                    "color": obj["color_ids"][0] if obj.get("color_ids") else 0,
                    "color_name": "/".join(obj.get("colors", [])) or "Mixed",
                    "size": obj["area"],
                    "center": obj["center"],
                    "bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
                    "type": obj.get("type", "SPRITE_CANDIDATE"),
                    "is_dynamic": obj.get("is_dynamic", False),
                })
            return results
    except Exception:
        pass

    # フォールバック (連結成分探索)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg_color = int(np.argmax(counts))

    visited = np.zeros((h, w), dtype=bool)
    objects: List[Dict[str, Any]] = []
    max_size = int(h * w * 0.12)

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg_color or visited[r, c]:
                continue

            q = [(r, c)]
            visited[r, c] = True
            pixels = [(r, c)]
            head = 0
            while head < len(q):
                curr_r, curr_c = q[head]
                head += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if grid[nr, nc] == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            pixels.append((nr, nc))

            if len(pixels) < 2 or len(pixels) > max_size:
                continue

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            mean_r = sum(rows) / len(rows)
            mean_c = sum(cols) / len(cols)

            best_p = min(pixels, key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2)
            objects.append({
                "id": len(objects) + 1,
                "color": color,
                "size": len(pixels),
                "center": {"x": best_p[1], "y": best_p[0]},
                "bbox": {"min_x": min_c, "min_y": min_r, "max_x": max_c, "max_y": max_r},
                "pixels": pixels,
            })

    # 適正サイズ（ボタン状）を好むスコアでソート
    objects.sort(key=lambda o: -abs(o["size"] - 25))
    return objects[:max_objects]


class VisionObservationHarness:
    """グリッド観測をカラー画像および構造化テキスト Part に変換するハーネス."""

    COLOR_NAMES = {
        0: "Black (0)",
        1: "Blue (1)",
        2: "Red (2)",
        3: "Green (3)",
        4: "Yellow (4)",
        5: "Gray (5)",
        6: "Magenta (6)",
        7: "Orange (7)",
        8: "Teal (8)",
        9: "Maroon (9)",
    }

    def __init__(self, cell_size: int = 12, use_console_ui: bool = True) -> None:
        self.cell_size = cell_size
        self.use_console_ui = use_console_ui

    def create_observation_parts(
        self,
        grid_data: Any,
        step_index: int = 0,
        available_actions: Optional[List[str]] = None,
        last_action_info: Optional[Dict[str, Any]] = None,
        additional_hint: Optional[str] = None,
    ) -> List[Part]:
        """観測グリッドを画像 Part および状態テキスト Part に変換して返却."""
        arr = normalize_grid(grid_data)
        h, w = arr.shape

        parts: List[Part] = []

        # 1. カラー画像 (盤面単体または統合コンソール画面) の生成と Part 化
        if h > 0 and w > 0:
            if self.use_console_ui:
                last_act = None
                if last_action_info:
                    last_act = last_action_info.get("action") or last_action_info.get("action_id")
                pil_img = render_console_observation(
                    grid=arr,
                    available_actions=available_actions,
                    last_action=last_act,
                    step_index=step_index,
                    cell_size=self.cell_size,
                )
            else:
                pil_img = render_grid_to_image(arr, cell_size=self.cell_size)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            parts.append(
                Part(
                    inline_data=Blob(
                        mime_type="image/png",
                        data=png_bytes,
                    )
                )
            )

        # 2. 空間・コンテキスト情報の構築
        unique_colors = np.unique(arr).tolist() if (h > 0 and w > 0) else []
        color_desc = [self.COLOR_NAMES.get(c, f"Color {c}") for c in unique_colors]

        text_lines = [
            f"=== [GAME STEP {step_index}] CURRENT OBSERVATION ===",
            f"- Grid Size: {h} rows x {w} cols",
            f"- Active Colors: {', '.join(color_desc)}",
        ]

        if self.use_console_ui:
            text_lines.append(
                "- Controller Visual HUD: The lower section of the observation image displays the physical controller. "
                "The D-Pad (1-4: UP/DOWN/LEFT/RIGHT), Action Buttons (5-7, 6: CLICK/ACTION), and RESET (0) are shown. "
                "The actively highlighted/glowing button indicates the last executed action, and brightly lit buttons indicate valid available actions."
            )

        if last_action_info:
            action_name = last_action_info.get("action", "NONE")
            pixels_changed = last_action_info.get("pixels_changed", 0)
            is_eff = last_action_info.get("is_effective", False)
            status_str = f"EFFECTIVE (changed {pixels_changed} pixels)" if is_eff else "INEFFECTIVE (no change / hit barrier)"
            text_lines.append(f"- Last Action: `{action_name}` -> {status_str}")

        if available_actions:
            text_lines.append(f"- Available Actions: {', '.join(available_actions)}")

        # 3. クリック操作が可能な場合の客観的座標リファレンス
        if available_actions and any("ACTION6" in a or "CLICK" in a for a in available_actions):
            detected_objects = detect_interactable_objects(arr)
            if detected_objects:
                text_lines.append("\n=== [VISIBLE OBJECT CLUSTERS (Reference Coordinates for ACTION6 / click_at)] ===")
                for idx, obj in enumerate(detected_objects[:8], 1):
                    c_name = self.COLOR_NAMES.get(obj["color"], f"Color {obj['color']}")
                    cx, cy = obj["center"]["x"], obj["center"]["y"]
                    sz = obj["size"]
                    text_lines.append(f"  * Cluster #{idx}: {c_name} at Center (x={cx}, y={cy}), size={sz}px")

        if additional_hint:
            text_lines.append(f"- Strategic Note: {additional_hint}")

        text_lines.append(
            "\nAnalyze the game console image carefully. Observe the game board and the controller HUD "
            "(the highlighted button indicates your last executed action). "
            "Use your visual recognition to determine what changed on screen, what objects exist, "
            "and decide your next 1-step action (`step_action` or `click_at`)."
        )

        parts.append(Part.from_text(text="\n".join(text_lines)))
        return parts
