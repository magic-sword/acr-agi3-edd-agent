"""ARC-AGI-3 ゲーム操作ツールセット (Google ADK 2.0 Function Tools).

LLM/VLM がゲーム環境を操作するための 1 手実行ツール群を提供します。
決定論的なハードコードではなく、メタスキル `game-controller` の検証エンジンと
同定済み操作力学マップ（Invariant Action Map）に基づき、
方向移動（step_action）、座標指定・幾何吸着クリック（click_at）、能動的リセット（reset_game）を実行します。
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# meta_skills/game-controller から GameController をインポート
_SKILL_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "game-controller" / "scripts"
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

try:
    from game_controller import GameController
except ImportError:
    # パスが異なる場合のフォールバックインポート
    GameController = None


@dataclasses.dataclass
class ActionDecision:
    """LLM によって決定された 1 手の操作."""

    action_type: str  # "STEP", "CLICK", "RESET"
    action_name: str  # "UP", "DOWN", "LEFT", "RIGHT", "ACTION1"〜"ACTION7", "RESET"
    action_id: int  # 0〜7
    coordinates: Optional[Dict[str, int]] = None  # {"x": c, "y": r}
    reasoning: str = ""
    loaded_skill: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class GameActionTools:
    """ゲーム環境操作を ADK 2.0 ツールとして公開し、決定をバッファリングするハーネス."""

    def __init__(
        self,
        available_actions: Optional[List[int]] = None,
        dynamics_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.available_action_ids: List[int] = available_actions or [1, 2, 3, 4]
        self.dynamics_map: Dict[str, int] = dict(dynamics_map or {})
        if GameController is not None:
            self.controller = GameController(
                available_actions=self.available_action_ids,
                dynamics_map=self.dynamics_map,
            )
        else:
            self.controller = None
        self.pending_decision: Optional[ActionDecision] = None
        self.history: List[ActionDecision] = []
        self.grid: Optional[Any] = None

    def set_grid(self, grid: Any) -> None:
        """現在の観測グリッド配列を保存."""
        self.grid = grid

    def set_available_actions(self, available_actions: List[int]) -> None:
        """現在のターンで利用可能なアクション ID を更新."""
        self.available_action_ids = available_actions
        self.pending_decision = None
        if self.controller is not None:
            self.controller.set_available_actions(available_actions)

    def set_dynamics_map(self, dynamics_map: Dict[str, int]) -> None:
        """同定された操作力学マップを注入・更新."""
        self.dynamics_map = dict(dynamics_map or {})
        if self.controller is not None:
            self.controller.set_dynamics_map(self.dynamics_map)

    def step_action(self, action: str, reasoning: str = "") -> str:
        """ゲーム環境で指定された方向キーまたはボタンのアクションを実行します。

        Args:
            action: 移動方向 ("UP", "DOWN", "LEFT", "RIGHT") またはアクション名 ("ACTION1"〜"ACTION7")。
            reasoning: この行動を選択した戦略的理由。
        """
        if self.controller is not None:
            res = self.controller.step_action(direction=action, reasoning=reasoning)
            if not res.get("success", False):
                return f"Error from game-controller: {res.get('error', 'Invalid action')}"
            self.pending_decision = ActionDecision(
                action_type=res["action_type"],
                action_name=res["action_name"],
                action_id=res["action_id"],
                coordinates=None,
                reasoning=res["reasoning"],
                loaded_skill="game-controller",
            )
        else:
            # フォールバック (controller なし)
            act_id = self.available_action_ids[0] if self.available_action_ids else 1
            if action.upper() not in ["UP", "DOWN", "LEFT", "RIGHT"] and not any(action.upper() == f"ACTION{i}" for i in self.available_action_ids):
                return f"Error from game-controller: Action '{action}' is disabled. Available actions: {self.available_action_ids}."
            self.pending_decision = ActionDecision(
                action_type="STEP",
                action_name=action.upper(),
                action_id=act_id,
                coordinates=None,
                reasoning=reasoning,
                loaded_skill="game-controller",
            )

        self.history.append(self.pending_decision)
        return (
            f"Action `{action}` (ID: {self.pending_decision.action_id}) "
            f"scheduled. Reason: {self.pending_decision.reasoning}"
        )

    def click_at(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        object_id: Optional[int] = None,
        grid: Optional[Any] = None,
        reasoning: str = "",
    ) -> str:
        """指定された盤面座標または検出された特定オブジェクトをクリックします (ACTION6).

        Args:
            x: クリック対象の列インデックス (0-indexed)。地面や特定位置をクリックする場合に指定。
            y: クリック対象の行インデックス (0-indexed)。地面や特定位置をクリックする場合に指定。
            object_id: 検出されたオブジェクトのID (0, 1, ...)。特定物体・ボタンを確実にクリックする場合に指定（重心へ自動スナップ）。
            grid: 盤面グリッド配列。
            reasoning: このクリックを選択した戦略的理由。
        """
        target_grid = grid if grid is not None else self.grid
        if self.controller is not None:
            res = self.controller.click_at(
                x=x, y=y, object_id=object_id, grid=target_grid, reasoning=reasoning
            )
            if not res.get("success", False):
                return f"Error from game-controller: {res.get('error', 'Invalid click action')}"
            self.pending_decision = ActionDecision(
                action_type=res["action_type"],
                action_name=res["action_name"],
                action_id=res["action_id"],
                coordinates=res["coordinates"],
                reasoning=res["reasoning"],
                loaded_skill="game-controller",
            )
        else:
            if 6 not in self.available_action_ids:
                return f"Error from game-controller: Click action (ACTION6 / click_at) is disabled in this environment. Available actions: {self.available_action_ids}."
            safe_x = x if x is not None else 0
            safe_y = y if y is not None else 0
            self.pending_decision = ActionDecision(
                action_type="CLICK",
                action_name="ACTION6",
                action_id=6,
                coordinates={"x": safe_x, "y": safe_y},
                reasoning=reasoning,
                loaded_skill="game-controller",
            )

        self.history.append(self.pending_decision)
        coords = self.pending_decision.coordinates
        return f"Click scheduled at coordinate {coords}. Reason: {self.pending_decision.reasoning}"

    def reset_game(self, reasoning: str = "") -> str:
        """現在のレベルをリセットして初期状態に戻します（手詰まり時の能動的リセット）.

        Args:
            reasoning: リセットを決断した理由。
        """
        if self.controller is not None:
            res = self.controller.reset_game(reasoning=reasoning)
            self.pending_decision = ActionDecision(
                action_type="RESET",
                action_name="RESET",
                action_id=0,
                coordinates=None,
                reasoning=res["reasoning"],
                loaded_skill="game-controller",
            )
        else:
            self.pending_decision = ActionDecision(
                action_type="RESET",
                action_name="RESET",
                action_id=0,
                coordinates=None,
                reasoning=reasoning,
                loaded_skill="game-controller",
            )

        self.history.append(self.pending_decision)
        return f"Environment reset scheduled. Reason: {reasoning}"

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すためのツール関数一覧を返却."""
        return [self.step_action, self.click_at, self.reset_game]
