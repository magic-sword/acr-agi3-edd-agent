"""ACR-AGI-3 公式提出用エージェント (MyAgent).

ARC Gateway / Kaggle 提出仕様に準拠し、MetaSkillHarnessPlanner を通じて
動的アフォーダンス同定と行動スキル実行を行うエージェント。
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from acr_agi3.agent.planner import MetaSkillHarnessPlanner

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


class MyAgent(Agent):
    """ACR-AGI-3 自律適応型メタスキルエージェント (Meta-Skill Harness Agent)."""

    def __init__(
        self,
        card_id: str = "local",
        game_id: str = "default",
        agent_name: str = "MyAgent",
        ROOT_URL: str = "http://local",
        record: bool = False,
        arc_env: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, *args, **kwargs)
        except Exception:
            pass
        self.game_id = game_id or getattr(self, "game_id", "default")
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.step_count = 0
        self.action_history: List[int] = []
        self.last_frame_hash: Optional[int] = None
        self.planner = MetaSkillHarnessPlanner(game_id=self.game_id)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        state = getattr(latest_frame, "state", None)
        return state is GameState.WIN

    def _get_cands(self, latest_frame: FrameData) -> List[Any]:
        avail = getattr(latest_frame, "available_actions", None)
        reset_val = getattr(GameAction.RESET, "value", 0)
        cands = []
        if avail:
            for act_id in avail:
                if act_id != reset_val:
                    try:
                        cands.append(GameAction.from_id(act_id))
                    except Exception:
                        pass
        if not cands:
            all_actions = list(GameAction) if hasattr(GameAction, "__iter__") else [
                getattr(GameAction, f"ACTION{i}", None) for i in range(1, 8)
            ]
            cands = [a for a in all_actions if a is not None and getattr(a, "value", -1) != reset_val]
        return cands

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> Any:
        self.step_count += 1
        state = getattr(latest_frame, "state", None)

        if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.step_count = 0
            self.action_history.clear()
            self.planner = MetaSkillHarnessPlanner(game_id=self.game_id)
            return GameAction.RESET

        try:
            cands = self._get_cands(latest_frame)
            if not cands:
                return GameAction.RESET

            grid = getattr(latest_frame, "frame", [])
            cand_ids = [getattr(a, "value", 1) for a in cands]

            def _hash_grid(g: Any) -> int:
                try:
                    if not g:
                        return 0
                    if isinstance(g, (list, tuple)) and len(g) > 0:
                        if isinstance(g[0], (list, tuple)) and len(g[0]) > 0 and isinstance(g[0][0], (list, tuple)):
                            g = g[-1]
                        elif len(g) == 1 and isinstance(g[0], (list, tuple)):
                            g = g[0]
                    return hash(tuple(tuple(int(c[0]) if isinstance(c, (list, tuple)) else int(c) for c in row) for row in g))
                except Exception:
                    return 0

            current_hash = _hash_grid(grid)
            is_eff = self.last_frame_hash is not None and current_hash != self.last_frame_hash
            self.planner.on_feedback(is_effective=is_eff, pixels_changed=1 if is_eff else 0)
            self.last_frame_hash = current_hash

            act_id, act_data, reasoning = self.planner.decide_action(grid, cand_ids)
            chosen_action = GameAction.from_id(act_id)

            if hasattr(chosen_action, "is_complex") and chosen_action.is_complex():
                chosen_action.set_data(act_data)
                chosen_action.reasoning = {
                    "desired_action": f"{chosen_action.value}",
                    "my_reason": reasoning,
                }
            else:
                chosen_action.reasoning = reasoning

            self.action_history.append(act_id)
            return chosen_action

        except Exception as e:
            avail = getattr(latest_frame, "available_actions", None)
            if avail:
                act_id = [x for x in avail if x != 0][0] if any(x != 0 for x in avail) else 0
                return GameAction.from_id(act_id)
            return GameAction.from_id(1)
