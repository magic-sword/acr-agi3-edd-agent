"""ARC-AGI-3 視覚認識・マルチモーダル観測ハーネス.

ゲーム環境のグリッド観測（FrameData）を公式 10 色カラー画像 (PIL Image / PNG Blob) に変換し、
Google ADK 2.0 / GenAI の Part オブジェクトとして LLM/VLM に提示します。
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.genai.types import Blob, Part

from acr_agi3.dsl.renderer import ARC_COLORS, render_grid_to_image


def normalize_grid(grid_data: Any) -> np.ndarray:
    """3次元テンソル (N, H, W) またはリストを最新フレームの 2D numpy 配列 (H, W) へ正規化."""
    arr = np.array(grid_data, dtype=int)
    if arr.ndim == 3:
        arr = arr[-1]
    elif arr.ndim == 1:
        arr = np.array([arr])
    return arr


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

    def __init__(self, cell_size: int = 24) -> None:
        self.cell_size = cell_size

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

        # 1. カラー画像の生成と Part 化
        if h > 0 and w > 0:
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

        if last_action_info:
            action_name = last_action_info.get("action", "NONE")
            pixels_changed = last_action_info.get("pixels_changed", 0)
            is_eff = last_action_info.get("is_effective", False)
            status_str = f"EFFECTIVE (changed {pixels_changed} pixels)" if is_eff else "INEFFECTIVE (no change / hit barrier)"
            text_lines.append(f"- Last Action: `{action_name}` -> {status_str}")

        if available_actions:
            text_lines.append(f"- Available Actions: {', '.join(available_actions)}")

        if additional_hint:
            text_lines.append(f"- Strategic Note: {additional_hint}")

        text_lines.append(
            "\nAnalyze the board image carefully. Identify yourself (agent), targets, obstacles, and interactable objects. "
            "Refer to active meta-skills via `load_skill` if needed, then execute an action using `step_action` or `click_at`."
        )

        parts.append(Part.from_text(text="\n".join(text_lines)))
        return parts
