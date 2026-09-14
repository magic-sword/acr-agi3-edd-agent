"""ARC-AGI-3 ゲーム操作ツールセット (Google ADK 2.0 Function Tools).

LLM/VLM がゲーム環境を操作するためのツール群を提供します。
決定論的なハードコードではなく、LLM 自身の Function Calling により
移動アクション（step_action）や座標指定クリック（click_at）、能動的リセット（reset_game）を選択・実行します。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ActionDecision:
    """LLM によって決定された 1 手の操作."""

    action_type: str  # "STEP", "CLICK", "RESET"
    action_name: str  # "UP", "DOWN", "LEFT", "RIGHT", "ACTION1"〜"ACTION7", "RESET"
    action_id: int  # 0〜7
    coordinates: Optional[Dict[str, int]] = None  # {"x": c, "y": r}
    reasoning: str = ""


class GameActionTools:
    """ゲーム環境操作を ADK 2.0 ツールとして公開し、決定をバッファリングするハーネス."""

    ACTION_MAP = {
        "RESET": 0,
        "ACTION1": 1,
        "ACTION2": 2,
        "ACTION3": 3,
        "ACTION4": 4,
        "ACTION5": 5,
        "ACTION6": 6,
        "ACTION7": 7,
        "UP": 1,
        "DOWN": 2,
        "LEFT": 3,
        "RIGHT": 4,
        "CLICK": 6,
    }

    def __init__(self, available_actions: Optional[List[int]] = None) -> None:
        self.available_action_ids: List[int] = available_actions or [1, 2, 3, 4]
        self.pending_decision: Optional[ActionDecision] = None
        self.history: List[ActionDecision] = []

    def set_available_actions(self, available_actions: List[int]) -> None:
        """現在のターンで利用可能なアクション ID を更新."""
        self.available_action_ids = available_actions
        self.pending_decision = None

    def step_action(self, action: str, reasoning: str = "") -> str:
        """ゲーム環境で指定されたボタンまたは方向キーのアクションを実行します。

        Args:
            action: 実行するアクション名 ("UP", "DOWN", "LEFT", "RIGHT", "ACTION1"〜"ACTION7")。
            reasoning: この行動を選択した戦略的理由。
        """
        act_clean = action.strip().upper()
        act_id = self.ACTION_MAP.get(act_clean)

        if act_id is None:
            # 数値文字列対応
            try:
                act_id = int(act_clean)
            except ValueError:
                act_id = self.available_action_ids[0] if self.available_action_ids else 1

        if act_id not in self.available_action_ids and act_id != 0:
            act_id = self.available_action_ids[0] if self.available_action_ids else 1

        self.pending_decision = ActionDecision(
            action_type="STEP",
            action_name=act_clean,
            action_id=act_id,
            reasoning=reasoning,
        )
        self.history.append(self.pending_decision)
        return f"Action `{act_clean}` (ID: {act_id}) scheduled for execution. Reason: {reasoning}"

    def click_at(self, x: int, y: int, reasoning: str = "") -> str:
        """指定された盤面座標 (列 x, 行 y) をクリックします。

        Args:
            x: クリック対象の列インデックス (Column, 0-indexed horizontal coordinate)。
            y: クリック対象の行インデックス (Row, 0-indexed vertical coordinate)。
            reasoning: この座標をクリックする戦略的理由。
        """
        click_act_id = 6  # ACTION6 (標準クリックアクション)
        self.pending_decision = ActionDecision(
            action_type="CLICK",
            action_name="ACTION6",
            action_id=click_act_id,
            coordinates={"x": int(x), "y": int(y)},
            reasoning=reasoning,
        )
        self.history.append(self.pending_decision)
        return f"Click scheduled at coordinate (x={x}, y={y}). Reason: {reasoning}"

    def reset_game(self, reasoning: str = "") -> str:
        """現在のレベルをリセットして初期状態に戻します（デッドロックや手詰まり時の能動的リセット）。

        Args:
            reasoning: リセットを決断した理由（ループ検知、行き止まりなど）。
        """
        self.pending_decision = ActionDecision(
            action_type="RESET",
            action_name="RESET",
            action_id=0,
            reasoning=reasoning,
        )
        self.history.append(self.pending_decision)
        return f"Environment reset scheduled. Reason: {reasoning}"

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すためのツール関数一覧を返却."""
        return [self.step_action, self.click_at, self.reset_game]
