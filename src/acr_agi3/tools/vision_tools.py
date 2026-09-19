"""Google ADK 2.0 準拠 Visual Inspector 実行ツール (Level 3 Tools).

Perceive Agent が盤面のアフォーダンスや客観的幾何情報を能動的に取得するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
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

    def __init__(self) -> None:
        self.inspector = VisualInspector() if VisualInspector is not None else None
        self.current_grid: Optional[np.ndarray] = None
        self.step_index: int = 0

    def set_context(self, grid: Optional[np.ndarray], step_index: int = 0) -> None:
        """現在のフレームとステップ番号を更新."""
        self.current_grid = grid
        self.step_index = step_index

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
        return [self.inspect_affordances, self.inspect_board_summary]
