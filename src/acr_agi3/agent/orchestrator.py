"""ACR-AGI-3 推論統括エージェント (Game Orchestrator)."""

from typing import Any, Dict, Optional

from acr_agi3.agent.meta_agent import MetaSkillDrivenAgent
from acr_agi3.game.env import GameEnvironment


class ARCOrchestrator:
    """メタスキルと契約テストを統括してゲームステージを自律クリアするオーケストレーター."""

    def __init__(
        self,
        agent: Optional[Any] = None,
    ) -> None:
        self.agent = agent or MetaSkillDrivenAgent()

    def solve_game(
        self,
        env: GameEnvironment,
        max_steps: int = 50,
        task_id: str = "game_task",
    ) -> Dict[str, Any]:
        """与えられたゲーム環境に対してメタスキル分析と行動ポリシー実行を行いクリアを目指す."""

        if hasattr(self.agent, "solve_game"):
            return self.agent.solve_game(env, max_steps=max_steps, task_id=task_id)
        elif hasattr(self.agent, "solve"):
            return self.agent.solve(env, max_steps=max_steps)
        else:
            raise ValueError(f"Agent {self.agent} does not support game solving.")

    def solve(
        self,
        env: GameEnvironment,
        max_steps: int = 50,
        task_id: str = "game_task",
    ) -> Dict[str, Any]:
        """solve_game へのエイリアス."""
        return self.solve_game(env, max_steps=max_steps, task_id=task_id)
