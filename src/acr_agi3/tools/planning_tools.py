"""Google ADK 2.0 準拠 Planning & Action Guard 実行ツール (Level 3 Tools).

Plan Agent および Act Agent が幾何パスファインディング、操作力学同定、
および禁忌ループ防止を実行するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# meta_skills scripts からインポート
_PLANNER_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "backward-planner" / "scripts"
_PROBER_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "epistemic-prober" / "scripts"
_GUARD_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "taboo-reset-guard" / "scripts"

for p in [_PLANNER_DIR, _PROBER_DIR, _GUARD_DIR]:
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

try:
    from backward_planner import BackwardPlanner
except ImportError:
    BackwardPlanner = None

try:
    from epistemic_prober import EpistemicProber
except ImportError:
    EpistemicProber = None

try:
    from taboo_reset_guard import TabooResetGuard
except ImportError:
    TabooResetGuard = None

logger = logging.getLogger(__name__)


class PlanningTools:
    """Plan & Act Agent 向け計画・力学・禁忌ツールセット."""

    def __init__(self) -> None:
        self.planner = BackwardPlanner() if BackwardPlanner is not None else None
        self.prober = EpistemicProber() if EpistemicProber is not None else None
        self.guard = TabooResetGuard() if TabooResetGuard is not None else None
        self.current_grid: Optional[np.ndarray] = None
        self.step_index: int = 0
        self.available_actions: List[int] = [1, 2, 3, 4]

    def set_context(
        self,
        grid: Optional[np.ndarray],
        step_index: int = 0,
        available_actions: Optional[List[int]] = None,
    ) -> None:
        self.current_grid = grid
        self.step_index = step_index
        if available_actions is not None:
            self.available_actions = available_actions

    def plan_geometric_path(
        self,
        start_col: int,
        start_row: int,
        goal_col: int,
        goal_row: int,
        impassable_colors: Optional[List[int]] = None,
    ) -> str:
        """迷路や障害物盤面において、指定された始点から終点への A* 最短幾何経路と直近の推奨進行方向を計算します。"""
        if self.current_grid is None or self.planner is None:
            return json.dumps({"error": "No observation grid or planner available"}, ensure_ascii=False)

        path, next_dir = self.planner.plan_path(
            grid=self.current_grid,
            start_pos=(start_col, start_row),
            goal_pos=(goal_col, goal_row),
            impassable_colors=impassable_colors,
        )
        return json.dumps({
            "path_length": len(path),
            "next_direction": next_dir,
            "waypoints": [list(p) for p in path[:10]],
            "reachable": len(path) > 0,
        }, ensure_ascii=False)

    def plan_action_sequence(
        self,
        start_col: int,
        start_row: int,
        goal_col: int,
        goal_row: int,
        impassable_colors: Optional[List[int]] = None,
    ) -> List[str]:
        """A* 最短経路から、実行可能な方向アクションのリスト (例: ['RIGHT', 'RIGHT', 'UP']) を生成."""
        if self.current_grid is None or self.planner is None:
            return []
        path, _ = self.planner.plan_path(
            grid=self.current_grid,
            start_pos=(start_col, start_row),
            goal_pos=(goal_col, goal_row),
            impassable_colors=impassable_colors,
        )
        if len(path) < 2:
            return []
        actions: List[str] = []
        for i in range(len(path) - 1):
            c0, r0 = path[i]
            c1, r1 = path[i + 1]
            dc = c1 - c0
            dr = r1 - r0
            if dr < 0:
                actions.append("UP")
            elif dr > 0:
                actions.append("DOWN")
            elif dc < 0:
                actions.append("LEFT")
            elif dc > 0:
                actions.append("RIGHT")
        return actions

    def probe_action_dynamics(self) -> str:
        """未確定のコントローラー操作力学（UP/DOWN/LEFT/RIGHT/CLICK）の同定状況と次の推奨プローブ手を取得します。"""
        if self.prober is None:
            return json.dumps({"dynamics_map": {}, "recommendation": None})

        is_needed = self.prober.is_probing_needed(self.step_index, self.available_actions)
        rec = self.prober.recommend_probe_action(self.available_actions) if is_needed else None
        return json.dumps({
            "probing_needed": is_needed,
            "recommended_probe_action": rec,
            "dynamics_map": self.prober.get_dynamics_map(),
        }, ensure_ascii=False)

    def filter_taboo_actions(
        self,
        proposed_action: int,
        last_action_effective: bool,
        last_action_id: Optional[int],
        stagnation_count: int,
    ) -> str:
        """提案された行動を壁衝突・振動ループ制約と照合し、禁忌（Taboo）を刈り込みます。"""
        if self.guard is None:
            return json.dumps({"is_allowed": True, "sanitized_action": proposed_action, "should_reset": False})

        is_allowed, sanitized, reason = self.guard.filter_taboo_actions(
            proposed_action=proposed_action,
            last_action_effective=last_action_effective,
            last_action_id=last_action_id,
            available_actions=self.available_actions,
            stagnation_count=stagnation_count,
        )
        should_reset = self.guard.should_active_reset(stagnation_count)
        return json.dumps({
            "is_allowed": is_allowed,
            "sanitized_action": sanitized,
            "should_reset": should_reset,
            "reason": reason,
        }, ensure_ascii=False)

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すための関数ツール一覧を返却."""
        return [
            self.plan_geometric_path,
            self.probe_action_dynamics,
            self.filter_taboo_actions,
        ]
