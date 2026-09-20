"""Google ADK 2.0 準拠 Visual Inspector 実行ツール (Level 3 Tools).

Perceive Agent が盤面のアフォーダンスや客観的幾何情報を能動的に取得するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# meta_skills/visual-inspector から VisualInspector をインポート
_SKILL_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "visual-inspector" / "scripts"
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

try:
    from visual_inspector import VisualInspector
except ImportError:
    VisualInspector = None

logger = logging.getLogger(__name__)


class VisionTools:
    """Perceive Agent 向け Level 3 実行ツールセット."""

    def __init__(self, controller: Optional[Any] = None) -> None:
        self.inspector = VisualInspector() if VisualInspector is not None else None
        self.current_grid: Optional[np.ndarray] = None
        self.step_index: int = 0
        self.cursor_pos: Optional[Tuple[int, int]] = (32, 32)
        self.controller = controller

    def set_context(
        self,
        grid: Optional[np.ndarray],
        step_index: int = 0,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """現在のフレーム、ステップ番号、およびカーソル位置を更新."""
        self.current_grid = grid
        self.step_index = step_index
        if cursor_pos is not None:
            self.cursor_pos = cursor_pos

    def set_cursor(self, col: int, row: int) -> None:
        """最新のカーソル位置を同期."""
        self.cursor_pos = (col, row)

    def set_controller(self, controller: Any) -> None:
        """ゲームコントローラーインスタンスをバインドしてカーソル位置を自動同期."""
        self.controller = controller

    def inspect_cursor_target(self, radius: int = 3, cursor_pos: Optional[Tuple[int, int]] = None) -> str:
        """現在のマウスカーソル照準位置を検証し、対象オブジェクト、中央/端判定、および周辺 7x7 ゲシュタルトマップを取得します。

        Args:
            radius: 局所マップの半径 (デフォルト 3 -> 7x7 グリッド)。
            cursor_pos: 任意指定のカーソル座標 (col, row)。省略時は現在コントローラーが指す照準位置。

        Returns:
            照準検証結果（所属オブジェクト、中央判定、周囲カラーマップ、クリック可否ガイダンス）。
        """
        if self.current_grid is None or self.inspector is None:
            return "Error: No observation grid currently set."

        target_cursor = cursor_pos
        if target_cursor is None:
            if self.controller is not None and hasattr(self.controller, "cursor"):
                target_cursor = self.controller.cursor
            else:
                target_cursor = self.cursor_pos

        try:
            res = self.inspector.inspect_cursor_target(
                self.current_grid, cursor_pos=target_cursor, radius=radius
            )
            if not res.get("success", False):
                return f"Error inspecting cursor target: {res.get('error', 'Unknown error')}"

            status = res["status"]
            cursor = res["cursor"]
            pixel = res["target_pixel"]
            guidance = res["guidance"]
            ascii_map = res["local_ascii_map"]
            obj_info = res.get("object_info")

            obj_summary = "None (Background/Empty space)"
            if obj_info:
                obj_summary = (
                    f"{obj_info['type']} #{obj_info['id']} (Center: ({obj_info['center']['x']}, {obj_info['center']['y']}), "
                    f"Dist: {obj_info['distance_to_center']})"
                )

            return (
                f"=== [CURSOR TARGET INSPECTION] ===\n"
                f"Cursor Position: (col={cursor['col']}, row={cursor['row']})\n"
                f"Aim Status: [{status}]\n"
                f"Aimed Pixel Color: {pixel['color']} ({pixel['color_name']})\n"
                f"Overlapping Object: {obj_summary}\n"
                f"Guidance: {guidance}\n\n"
                f"Local {2*radius+1}x{2*radius+1} Visual Gestalt Map:\n"
                f"{ascii_map}\n"
                f"=================================="
            )
        except Exception as e:
            return f"Error inspecting cursor target: {e}"

    def inspect_affordances(self, mode: str = "deep") -> str:
        """現在の盤面グリッドからアフォーダンス（自機候補、ゴール候補、障害物、スタイル）を深層抽出します。

        Args:
            mode: 解析モード ('deep', 'fast')。
        """
        if self.current_grid is None or self.inspector is None:
            return json.dumps({"error": "No observation grid currently set"}, ensure_ascii=False)

        try:
            report = self.inspector.analyze_frame(self.current_grid, step_index=self.step_index)
            res = {
                "grid_shape": list(report.grid_shape),
                "background_color": report.background_color,
                "style": report.style,
                "agent_pos": list(report.agent_pos) if report.agent_pos else None,
                "goal_pos": list(report.goal_pos) if report.goal_pos else None,
                "obstacles_count": len(report.obstacles),
                "target_candidates_count": len(report.target_candidates),
                "interactables": {k: list(v) for k, v in report.interactables.items()},
            }
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Failed to inspect affordances: {e}"}, ensure_ascii=False)

    def inspect_board_summary(self) -> str:
        """盤面の幾何寸法、出現色パレット、および基本構造サマリーを取得します。"""
        if self.current_grid is None or self.inspector is None:
            return json.dumps({"error": "No observation grid currently set"}, ensure_ascii=False)

        try:
            data = self.inspector.inspect_board(self.current_grid, step_index=self.step_index)
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Failed to inspect board summary: {e}"}, ensure_ascii=False)

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すための関数ツール一覧を返却."""
        return [
            self.inspect_cursor_target,
            self.inspect_affordances,
            self.inspect_board_summary,
        ]
