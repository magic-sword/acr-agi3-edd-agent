"""ACR-AGI-3 公式提出用エージェント (MyAgent - Google ADK 2.0 ネイティブ).

ARC Gateway / Kaggle 提出仕様に準拠し、
決定論的プログラムではなく Google ADK 2.0 ネイティブの ADKGamePlayer を通じて
画面認識（カラー画像）、Progressive Disclosure (SKILL.md)、
および Function Calling による自律的行動決定を実行するエージェント。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.harness.game_action_tools import ActionDecision

logger = logging.getLogger(__name__)

try:
    from arcengine import FrameData, GameAction, GameState
except ImportError:
    from enum import Enum

    FrameData = Any

    class GameState(str, Enum):
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class GameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        @classmethod
        def from_id(cls, i: int) -> GameAction:
            for a in cls:
                if a.value == i:
                    return a
            return cls.ACTION1

try:
    from agents.agent import Agent
except ImportError:
    Agent = object


class ActionDataWrapper:
    """クリック座標等の追加データを保持するラッパー."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> Dict[str, Any]:
        return self._data

    def to_dict(self) -> Dict[str, Any]:
        return self._data


class MyAgent(Agent):
    """Google ADK 2.0 準拠・自律推論ゲームプレイエージェント."""

    def __init__(
        self,
        card_id: str = "local",
        game_id: str = "default",
        agent_name: str = "MyAgent",
        ROOT_URL: str = "http://local",
        record: bool = False,
        arc_env: Any = None,
        model: Optional[Any] = None,
        autonomous_probing: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, *args, **kwargs)
        except Exception:
            pass
        self.game_id = game_id or getattr(self, "game_id", "default")
        self.step_count = 0

        # ADK 2.0 ネイティブゲームプレイヤー
        import re
        clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", self.game_id)
        if autonomous_probing is not None:
            auto_probe = autonomous_probing
        else:
            # model が明示的に渡されたカスタム推論（テストモック等）の場合は False、
            # デフォルト（本番自律推論）の場合は True
            auto_probe = (model is None)

        self.player = ADKGamePlayer(
            model=model,
            name=f"adk_player_{clean_id}",
            app_name=f"app_{clean_id}",
            autonomous_probing=auto_probe,
        )

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """クリア判定."""
        state = getattr(latest_frame, "state", None)
        return state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> Any:
        """現在の観測フレームから、ADK 2.0 エージェントの思考を経て行動を選択."""
        self.step_count += 1
        state = getattr(latest_frame, "state", None)

        # ゲーム開始時または終了時はリセットを発行
        if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.step_count = 0
            self.player.reset()
            return GameAction.RESET

        # 利用可能アクションの抽出
        avail = getattr(latest_frame, "available_actions", None)
        reset_val = getattr(GameAction.RESET, "value", 0)
        avail_ids = [a for a in (avail or [1, 2, 3, 4]) if a != reset_val]
        if not avail_ids:
            avail_ids = [1, 2, 3, 4]

        grid = getattr(latest_frame, "frame", [])

        # ADK 2.0 エージェントによる自律的行動決定 (マルチモーダル画面認識 + SKILL.md + Function Calling)
        decision: ActionDecision = self.player.decide_next_action(
            grid=grid,
            available_actions=avail_ids,
            state_str=str(state),
        )

        # 決定されたアクション ID から GameAction を生成
        act_id = decision.action_id
        try:
            action = GameAction.from_id(act_id)
        except Exception:
            action = GameAction.ACTION1

        # クリック座標や理由データの付与
        if decision.coordinates:
            action.action_data = ActionDataWrapper(decision.coordinates)
        else:
            action.action_data = ActionDataWrapper({})

        action_reasoning = {
            "strategy": decision.reasoning,
            "step": self.step_count,
            "loaded_skill": decision.loaded_skill,
        }
        if hasattr(decision, "metadata") and isinstance(decision.metadata, dict):
            action_reasoning.update(decision.metadata)
        action.reasoning = action_reasoning
        return action
