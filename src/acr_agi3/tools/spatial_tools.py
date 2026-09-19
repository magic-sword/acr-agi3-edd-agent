"""Google ADK 2.0 準拠 Spatial Grounder 実行ツール (Level 3 Tools).

Perceive Agent および Act Agent が盤面のクリック可能アンカーを特定し、
座標を接地（Grounding）するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# meta_skills/spatial-grounder から SpatialGrounder をインポート
_SKILL_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "spatial-grounder" / "scripts"
if str(_SKILL_DIR) not in sys.path and _SKILL_DIR.exists():
    sys.path.insert(0, str(_SKILL_DIR))

try:
    from spatial_grounder import SpatialGrounder
except ImportError:
    SpatialGrounder = None

logger = logging.getLogger(__name__)


class SpatialTools:
    """Perceive & Act Agent 向け空間接地・クリックアンカーツールセット."""

    def __init__(self) -> None:
        self.grounder = SpatialGrounder() if SpatialGrounder is not None else None
        self.current_grid: Optional[np.ndarray] = None
        self.step_index: int = 0
        self.cached_anchors: List[Dict[str, Any]] = []

    def set_context(self, grid: Optional[np.ndarray], step_index: int = 0) -> None:
        """現在のフレームとステップ番号を更新し、アンカーを抽出."""
        self.current_grid = grid
        self.step_index = step_index
        if grid is not None and self.grounder is not None:
            self.cached_anchors = self.grounder.extract_clickable_anchors(grid)
        else:
            self.cached_anchors = []

    def inspect_clickable_anchors(self) -> str:
        """盤面上のクリック可能な有色要素・ボタンの重心座標アンカー一覧を取得します。"""
        if self.current_grid is None or self.grounder is None:
            return json.dumps({"error": "No observation grid currently set"}, ensure_ascii=False)

        res = {
            "total_anchors": len(self.cached_anchors),
            "anchors": self.cached_anchors,
            "prompt_summary": self.grounder.format_anchors_prompt(self.cached_anchors),
        }
        return json.dumps(res, ensure_ascii=False)

    def snap_to_anchor(self, x: int, y: int) -> str:
        """指定された座標 (x, y) を直近の有効なオブジェクト中心アンカーへ吸着（Snap）します。"""
        if not self.cached_anchors or self.grounder is None:
            return json.dumps({"x": x, "y": y, "snapped": False, "anchor_id": None})

        sx, sy, aid = self.grounder.snap_to_anchor(x, y, self.cached_anchors)
        return json.dumps({
            "x": sx,
            "y": sy,
            "snapped": (aid is not None),
            "anchor_id": aid,
        })

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すための関数ツール一覧を返却."""
        return [self.inspect_clickable_anchors, self.snap_to_anchor]
