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
        self.last_grid: Optional[np.ndarray] = None
        self.step_index: int = 0
        self.cached_anchors: List[Dict[str, Any]] = []
        self.cached_objects: List[Dict[str, Any]] = []
        self.taboo_coords: List[Tuple[int, int]] = []

    def set_taboo_coords(self, coords: List[Tuple[int, int]]) -> None:
        """空振り等で禁忌となったクリック座標リストを更新."""
        self.taboo_coords = list(coords)

    def set_context(
        self,
        grid: Optional[np.ndarray],
        step_index: int = 0,
        last_grid: Optional[np.ndarray] = None,
    ) -> None:
        """現在のフレームと直前フレームを更新し、複合オブジェクトとアンカーを抽出."""
        self.last_grid = last_grid
        self.current_grid = grid
        self.step_index = step_index
        if grid is not None and self.grounder is not None:
            self.cached_objects = self.grounder.detect_composite_objects(grid, last_grid=last_grid)
            self.cached_anchors = self.grounder.extract_clickable_anchors(grid, last_grid=last_grid)
        else:
            self.cached_objects = []
            self.cached_anchors = []

    def inspect_detected_objects(self, filter_mode: str = "interactive") -> str:
        """階層的CVと動的差分によって検出された複合オブジェクト（ボタン、スプライト、キャラ）一覧を取得します。

        Args:
            filter_mode: "all"（全検出オブジェクト）, "interactive"（ボタン・動的候補を優先）, "dynamic"（変化した動的オブジェクトのみ）
        """
        if self.current_grid is None or self.grounder is None:
            return json.dumps({"error": "No observation grid currently set"}, ensure_ascii=False)

        objs = self.cached_objects
        if filter_mode == "dynamic":
            objs = [o for o in objs if o.get("is_dynamic")]
        elif filter_mode == "interactive":
            objs = [o for o in objs if o.get("type") in ["BUTTON_CANDIDATE", "DYNAMIC_ENTITY", "SPRITE_CANDIDATE"]]

        res = {
            "total_objects": len(objs),
            "filter_mode": filter_mode,
            "objects": objs,
            "summary": self.grounder.format_detected_objects_prompt(objs),
        }
        return json.dumps(res, ensure_ascii=False)

    def get_object_coordinates(self, object_id: int) -> str:
        """指定されたオブジェクト ID のクリック推奨座標 (x=col, y=row) を取得します。

        Args:
            object_id: inspect_detected_objects で取得したオブジェクトの ID 番号。
        """
        for obj in self.cached_objects:
            if obj["id"] == object_id:
                return json.dumps({
                    "object_id": object_id,
                    "type": obj["type"],
                    "x": obj["x"],
                    "y": obj["y"],
                    "colors": obj["colors"],
                    "area": obj["area"],
                    "is_dynamic": obj.get("is_dynamic", False),
                    "success": True,
                }, ensure_ascii=False)

        return json.dumps({"error": f"Object #{object_id} not found", "success": False}, ensure_ascii=False)

    def inspect_clickable_anchors(self) -> str:
        """盤面上のクリック可能な有色要素・ボタンの重心座標アンカー一覧を取得します。"""
        if self.current_grid is None or self.grounder is None:
            return json.dumps({"error": "No observation grid currently set"}, ensure_ascii=False)

        # 禁忌座標に近いアンカーにはフラグを付与
        visible_anchors = []
        for a in self.cached_anchors:
            is_taboo = any(np.hypot(a["x"] - tx, a["y"] - ty) <= 3.0 for tx, ty in self.taboo_coords)
            a_copy = dict(a)
            a_copy["is_taboo"] = is_taboo
            visible_anchors.append(a_copy)

        res = {
            "total_anchors": len(visible_anchors),
            "anchors": visible_anchors,
            "prompt_summary": self.grounder.format_anchors_prompt([a for a in visible_anchors if not a.get("is_taboo")]),
        }
        return json.dumps(res, ensure_ascii=False)

    def snap_to_anchor(self, x: int, y: int) -> str:
        """指定された座標 (x, y) を直近の有効なオブジェクト中心アンカーへ吸着（Snap）します。"""
        if not self.cached_anchors or self.grounder is None:
            return json.dumps({"x": x, "y": y, "snapped": False, "anchor_id": None})

        sx, sy, aid = self.grounder.snap_to_anchor(
            x, y, self.cached_anchors, taboo_coords=self.taboo_coords
        )
        return json.dumps({
            "x": sx,
            "y": sy,
            "snapped": (aid is not None),
            "anchor_id": aid,
        })

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すための関数ツール一覧を返却."""
        return [
            self.inspect_detected_objects,
            self.get_object_coordinates,
            self.inspect_clickable_anchors,
            self.snap_to_anchor,
        ]
