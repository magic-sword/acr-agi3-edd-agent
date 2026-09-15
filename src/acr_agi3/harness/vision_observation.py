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
    """グリッド内の前景オブジェクト（連結成分・スプライト）を検出し、重心や BBox を抽出."""
    if grid.ndim != 2:
        return []
    h, w = grid.shape
    if h == 0 or w == 0:
        return []

    # 最頻色を背景色とする
    counts = np.bincount(grid.ravel(), minlength=10)
    bg_color = int(np.argmax(counts))

    visited = np.zeros((h, w), dtype=bool)
    objects: List[Dict[str, Any]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg_color or visited[r, c]:
                continue

            # BFS による同一色の連結成分探索
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

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            mean_r = sum(rows) / len(rows)
            mean_c = sum(cols) / len(cols)

            # 重心に最も近いオブジェクト内ピクセルを代表座標 (center) とする
            best_p = min(pixels, key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2)
            center_x = best_p[1]
            center_y = best_p[0]

            objects.append({
                "id": len(objects) + 1,
                "color": color,
                "size": len(pixels),
                "center": {"x": center_x, "y": center_y},
                "bbox": {"min_x": min_c, "min_y": min_r, "max_x": max_c, "max_y": max_r},
                "pixels": pixels,
            })

    # 画面外周の横長・縦長バー（ステータスバーやUI枠線）を除外
    valid_targets = []
    for obj in objects:
        bbox = obj["bbox"]
        b_width = bbox["max_x"] - bbox["min_x"] + 1
        b_height = bbox["max_y"] - bbox["min_y"] + 1

        is_border_bar = (
            (b_width >= w - 2 and (bbox["min_y"] <= 1 or bbox["max_y"] >= h - 2))
            or (b_height >= h - 2 and (bbox["min_x"] <= 1 or max_x >= w - 2))
        )
        if is_border_bar:
            continue
        valid_targets.append(obj)

    # 1. ゲームのメインボード盤面（コンテナ）を検出 (画面の 8%〜65% を占める大きな正方形・長方形領域)
    board_boxes: List[Tuple[int, int, int, int]] = []
    for obj in objects:
        sz = obj["size"]
        bbox = obj["bbox"]
        bw = bbox["max_x"] - bbox["min_x"] + 1
        bh = bbox["max_y"] - bbox["min_y"] + 1
        if 0.08 * (h * w) <= sz <= 0.65 * (h * w) and bw >= w * 0.25 and bh >= h * 0.25:
            board_boxes.append((bbox["min_x"], bbox["min_y"], bbox["max_x"], bbox["max_y"]))

    # 各オブジェクトがメインボードコンテナ内にあるか判定
    for obj in valid_targets:
        cx, cy = obj["center"]["x"], obj["center"]["y"]
        obj["in_board"] = any(bx1 <= cx <= bx2 and by1 <= cy <= by2 for bx1, by1, bx2, by2 in board_boxes)

    # 2. 形状の出現頻度（同一サイズのタイルが反復配置されているグリッド構造を検出）
    shape_counts: Counter[Tuple[int, int]] = Counter()
    for obj in valid_targets:
        bb = obj["bbox"]
        shape_counts[(bb["max_x"] - bb["min_x"] + 1, bb["max_y"] - bb["min_y"] + 1)] += 1

    # 典型的なスプライト・操作タイル (メインボード内、反復タイル群、アスペクト比 <= 1.4、中央盤面付近) を最優先
    def _rank_key(obj: Dict[str, Any]) -> Tuple[int, int, int, float]:
        sz = obj["size"]
        bbox = obj["bbox"]
        wb = bbox["max_x"] - bbox["min_x"] + 1
        hb = bbox["max_y"] - bbox["min_y"] + 1
        aspect = max(wb / max(hb, 1), hb / max(wb, 1))
        is_square = aspect <= 1.4

        # 0. メインボード内部ボーナス (見本エリアや外周装飾よりメインボード内のピースを最優先)
        in_board_val = 1 if obj.get("in_board", False) else 0

        # 1. タイル反復度 (同じ (w, h) のスプライトが複数個存在する = 盤面タイル群)
        rep = shape_counts.get((wb, hb), 1)
        if rep >= 4 and is_square and sz >= 12:
            tier = 3  # 最優先: 盤面の反復グリッドタイル
        elif (12 <= sz <= 150) and is_square:
            tier = 2  # 次点: まとまった正方形スプライト
        elif (4 <= sz <= 300) and aspect <= 2.5:
            tier = 1  # 一般スプライト
        else:
            tier = 0  # 細長い枠線や極小ノイズ

        # 2. サイズ優先度 (8px以下の極小破片・装飾アイコンは後回し)
        size_priority = sz if sz >= 9 else -abs(sz - 9) * 10

        # 3. 画面中央度 (外周ギリギリより内部を優先)
        cx, cy = obj["center"]["x"], obj["center"]["y"]
        dist_to_center = -((cx - w / 2) ** 2 + (cy - h / 2) ** 2)

        return (in_board_val, tier, size_priority, dist_to_center)

    valid_targets.sort(key=_rank_key, reverse=True)
    return valid_targets[:max_objects]


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

        # 3. 前景オブジェクト（クリック可能ターゲット候補）の座標アフォーダンス提示
        detected_objects = detect_interactable_objects(arr)
        if detected_objects:
            text_lines.append("\n=== [DETECTED INTERACTABLE OBJECTS (Click Targets)] ===")
            for idx, obj in enumerate(detected_objects[:8], 1):
                c_name = self.COLOR_NAMES.get(obj["color"], f"Color {obj['color']}")
                cx, cy = obj["center"]["x"], obj["center"]["y"]
                sz = obj["size"]
                bbox = obj["bbox"]
                text_lines.append(
                    f"  * Target #{idx}: {c_name} at Center (x={cx}, y={cy}) "
                    f"[size={sz}px, bbox=({bbox['min_x']}..{bbox['max_x']}, {bbox['min_y']}..{bbox['max_y']})]"
                )
            text_lines.append("  [Directive]: When executing ACTION6 / click_at, specify the exact center (x, y) of your target object!")

        if additional_hint:
            text_lines.append(f"- Strategic Note: {additional_hint}")

        text_lines.append(
            "\nAnalyze the board image carefully. Identify yourself (agent), targets, obstacles, and interactable objects. "
            "Refer to active meta-skills via `load_skill` if needed, then execute an action using `step_action` or `click_at`."
        )

        parts.append(Part.from_text(text="\n".join(text_lines)))
        return parts
