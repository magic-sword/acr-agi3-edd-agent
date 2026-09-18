"""Google ADK 2.0 準拠・自律観測ツールセット (ObservationTools).

LLM Agent が反復思考 (ReAct ループ) 中に、オンデマンドでゲーム盤面の状態、
動的アフォーダンス (操作対象、目標、障害物)、直前アクションの因果変化、
および局所領域 (ROI) を詳細に観測・検査するための ADK 2.0 FunctionTool 群を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# visual-inspector スクリプトのインポートパス解決
_VISUAL_INSPECTOR_DIR = (
    Path(__file__).resolve().parents[3]
    / "meta_skills"
    / "visual-inspector"
    / "scripts"
)
if str(_VISUAL_INSPECTOR_DIR) not in sys.path and _VISUAL_INSPECTOR_DIR.exists():
    sys.path.insert(0, str(_VISUAL_INSPECTOR_DIR))

try:
    from visual_inspector import VisualInspector, DynamicAffordanceReport
except ImportError:
    VisualInspector = None
    DynamicAffordanceReport = None


@dataclass
class ObservationContext:
    """現在のアクティブな盤面および環境状態を保持するコンテキスト."""

    grid: Optional[np.ndarray] = None
    prev_grid: Optional[np.ndarray] = None
    step_index: int = 0
    last_action_info: Optional[Dict[str, Any]] = None
    dynamics_map: Dict[str, int] = field(default_factory=dict)
    available_actions: List[str] = field(default_factory=lambda: ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    target_pattern: Optional[List[int]] = None


class ObservationTools:
    """エージェントの ReAct 思考ループに提供される Google ADK 2.0 観測ツール群."""

    def __init__(self, visual_inspector: Optional[Any] = None) -> None:
        self.context = ObservationContext()
        self.inspector = visual_inspector or (VisualInspector() if VisualInspector else None)

    def update_context(
        self,
        grid: Any,
        prev_grid: Optional[Any] = None,
        step_index: int = 0,
        last_action_info: Optional[Dict[str, Any]] = None,
        dynamics_map: Optional[Dict[str, int]] = None,
        available_actions: Optional[List[str]] = None,
        target_pattern: Optional[List[int]] = None,
    ) -> None:
        """各ステップの最新観測でコンテキストを更新."""
        if grid is not None:
            self.context.grid = np.asarray(grid, dtype=int)
        else:
            self.context.grid = None

        if prev_grid is not None:
            self.context.prev_grid = np.asarray(prev_grid, dtype=int)
        else:
            self.context.prev_grid = None

        self.context.step_index = step_index
        self.context.last_action_info = last_action_info
        if dynamics_map is not None:
            self.context.dynamics_map = dynamics_map
        if available_actions is not None:
            self.context.available_actions = available_actions
        if target_pattern is not None:
            self.context.target_pattern = target_pattern

    def inspect_board(self, target_pattern: Optional[List[int]] = None) -> Dict[str, Any]:
        """Inspect general board geometry, color palette, gestalt difference, and estimated game style.

        Call this tool to perceive the global game board layout, active colors, and puzzle invariants.

        Args:
            target_pattern: Optional target sequence or target color pattern to compare against.

        Returns:
            Dict containing grid_dimensions, background_color, foreground_colors, style, and invariants.
        """
        if self.context.grid is None:
            return {"error": "No board observation currently available in context."}

        target = target_pattern if target_pattern is not None else self.context.target_pattern

        if self.inspector is not None:
            try:
                res = self.inspector.inspect_board(
                    self.context.grid,
                    target_grid_or_sequence=target,
                    step_index=self.context.step_index,
                )
                return res
            except Exception as e:
                logger.warning("VisualInspector.inspect_board failed: %s", e)

        # フォールバック
        grid = self.context.grid
        h, w = grid.shape
        unique_colors = [int(c) for c in np.unique(grid)]
        bg = unique_colors[0] if unique_colors else 0
        fg = [c for c in unique_colors if c != bg]
        return {
            "success": True,
            "step_index": self.context.step_index,
            "grid_dimensions": [h, w],
            "background_color": bg,
            "foreground_colors": fg,
            "style": "GENERAL_EXPLORATION",
            "available_actions": self.context.available_actions,
        }

    def inspect_affordances(self, detail: bool = True) -> Dict[str, Any]:
        """Inspect dynamic affordances including candidate player objects, target goals, obstacles, and interactables.

        Call this tool when formulating a plan to identify where the controllable agent is,
        what goals or target objects exist, and what paths or obstacles surround them.

        Args:
            detail: Whether to return full object bounding boxes and pixel coordinates.

        Returns:
            Dict containing player_pos, goal_pos, target_candidates, obstacles_count, and game style.
        """
        if self.context.grid is None:
            return {"error": "No board observation currently available in context."}

        if self.inspector is not None:
            try:
                action_taken = None
                if self.context.last_action_info:
                    action_taken = self.context.last_action_info.get("action_id")

                report = self.inspector.analyze_frame(
                    grid=self.context.grid,
                    prev_grid=self.context.prev_grid,
                    action_taken=action_taken,
                )

                out: Dict[str, Any] = {
                    "success": True,
                    "grid_shape": list(report.grid_shape),
                    "background_color": report.background_color,
                    "player_pos": list(report.player_pos) if report.player_pos else None,
                    "goal_pos": list(report.goal_pos) if report.goal_pos else None,
                    "style": report.style,
                    "target_count": len(report.target_candidates),
                    "obstacles_count": len(report.obstacles),
                }

                if detail:
                    out["target_candidates"] = [
                        {
                            "obj_id": t.obj_id,
                            "color": t.color,
                            "size": t.size,
                            "center": [float(t.center_r), float(t.center_c)],
                            "bbox": list(t.bounding_box),
                        }
                        for t in report.target_candidates[:5]
                    ]
                    out["interactables"] = {
                        k: list(v) for k, v in report.interactables.items()
                    }

                return out
            except Exception as e:
                logger.warning("VisualInspector.analyze_frame failed: %s", e)

        # フォールバック
        return {
            "success": True,
            "grid_shape": list(self.context.grid.shape),
            "player_pos": None,
            "goal_pos": None,
            "style": "UNKNOWN",
            "message": "Affordance extraction completed with default heuristic.",
        }

    def inspect_action_effect(self) -> Dict[str, Any]:
        """Inspect the causal feedback and visual delta caused by the most recently executed action.

        Call this tool to verify whether the previous action had an effect, how many pixels moved,
        and whether dynamic keybinding mappings (e.g. ACTION1 -> UP) were verified.

        Returns:
            Dict containing last_action, pixels_changed, is_effective, and current dynamics_map.
        """
        if self.context.last_action_info is None:
            return {
                "success": True,
                "message": "No previous action executed yet (step 0 or episode onset).",
                "pixels_changed": 0,
                "is_effective": False,
                "dynamics_map": self.context.dynamics_map,
            }

        last = self.context.last_action_info
        pixels_changed = last.get("pixels_changed", 0)
        is_effective = last.get("is_effective", pixels_changed > 0)

        return {
            "success": True,
            "last_action": last.get("action"),
            "last_action_id": last.get("action_id"),
            "pixels_changed": pixels_changed,
            "is_effective": is_effective,
            "dynamics_map": self.context.dynamics_map,
            "reasoning_used": last.get("reasoning"),
        }

    def inspect_roi(self, top: int, left: int, height: int, width: int) -> Dict[str, Any]:
        """Inspect a specific sub-region (Region of Interest) of the board in close detail.

        Call this tool when you need to closely examine a small area, such as a crowded maze junction,
        a suspected trap, or an intricate pattern cluster.

        Args:
            top: Starting row index (0-indexed).
            left: Starting column index (0-indexed).
            height: Height of the region.
            width: Width of the region.

        Returns:
            Dict containing subgrid values, unique colors in ROI, and center coordinates.
        """
        if self.context.grid is None:
            return {"error": "No board observation currently available in context."}

        grid = self.context.grid
        max_h, max_w = grid.shape
        t = max(0, min(top, max_h - 1))
        l = max(0, min(left, max_w - 1))
        b = min(max_h, t + max(1, height))
        r = min(max_w, l + max(1, width))

        subgrid = grid[t:b, l:r]
        colors = [int(c) for c in np.unique(subgrid)]

        return {
            "success": True,
            "roi_bounds": {"top": t, "left": l, "bottom": b, "right": r},
            "subgrid_shape": list(subgrid.shape),
            "colors_in_roi": colors,
            "subgrid": subgrid.tolist(),
        }

    def get_tools(self) -> List[FunctionTool]:
        """Google ADK 2.0 公式 FunctionTool のリストを生成して返却."""
        return [
            FunctionTool(self.inspect_board),
            FunctionTool(self.inspect_affordances),
            FunctionTool(self.inspect_action_effect),
            FunctionTool(self.inspect_roi),
        ]
