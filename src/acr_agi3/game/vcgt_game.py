"""VCGT 人間プレイデータに基づく決定論的ゲームシミュレータ環境."""

from __future__ import annotations

from typing import Any

import numpy as np

from acr_agi3.game.env import Action, GameEnvironment, StepResult


class GridWorldGameEnv(GameEnvironment):
    """アフォーダンス（プレイヤー、壁、ゴール、トラップ）を持つインタラクティブゲーム環境."""

    def __init__(
        self,
        grid_shape: tuple[int, int] = (10, 10),
        initial_player_pos: tuple[int, int] = (1, 1),
        goal_pos: tuple[int, int] = (8, 8),
        walls: set[tuple[int, int]] | None = None,
        hazards: set[tuple[int, int]] | None = None,
        background_color: int = 0,
        wall_color: int = 1,
        player_color: int = 2,
        goal_color: int = 3,
        hazard_color: int = 4,
        max_steps: int = 100,
        player_pos: tuple[int, int] | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.initial_player_pos = player_pos if player_pos is not None else initial_player_pos
        self.goal_pos = goal_pos
        self.walls = walls or set()
        self.hazards = hazards or set()


        self.background_color = background_color
        self.wall_color = wall_color
        self.player_color = player_color
        self.goal_color = goal_color
        self.hazard_color = hazard_color
        self.max_steps = max_steps

        self.player_pos = initial_player_pos
        self.steps_taken = 0
        self.done = False

    def reset(self) -> np.ndarray:
        """環境リセット."""
        self.player_pos = self.initial_player_pos
        self.steps_taken = 0
        self.done = False
        return self.render()

    def step(self, action: Action | int) -> StepResult:
        """アクションの実行."""
        if self.done:
            return StepResult(
                observation=self.render(),
                reward=0.0,
                done=True,
                info={"message": "Game already completed"},
            )

        self.steps_taken += 1
        act = Action(action) if isinstance(action, int) else action
        r, c = self.player_pos
        dr, dc = 0, 0

        if act == Action.UP:
            dr, dc = -1, 0
        elif act == Action.DOWN:
            dr, dc = 1, 0
        elif act == Action.LEFT:
            dr, dc = 0, -1
        elif act == Action.RIGHT:
            dr, dc = 0, 1
        elif act in (Action.INTERACT, Action.WAIT):
            dr, dc = 0, 0

        nr, nc = r + dr, c + dc
        # 境界判定
        if 0 <= nr < self.grid_shape[0] and 0 <= nc < self.grid_shape[1]:
            # 壁（障害物）判定
            if (nr, nc) not in self.walls:
                self.player_pos = (nr, nc)

        # 危険物判定
        if self.player_pos in self.hazards:
            self.done = True
            return StepResult(
                observation=self.render(),
                reward=-1.0,
                done=True,
                info={"status": "hazard_collision", "steps": self.steps_taken},
            )

        # ゴール判定
        if self.player_pos == self.goal_pos:
            self.done = True
            return StepResult(
                observation=self.render(),
                reward=1.0,
                done=True,
                info={"status": "goal_reached", "steps": self.steps_taken},
            )

        # 最大ステップ超過
        if self.steps_taken >= self.max_steps:
            self.done = True
            return StepResult(
                observation=self.render(),
                reward=0.0,
                done=True,
                info={"status": "timeout", "steps": self.steps_taken},
            )

        return StepResult(
            observation=self.render(),
            reward=0.0,
            done=False,
            info={"status": "in_progress", "steps": self.steps_taken},
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "player_pos": self.player_pos,
            "goal_pos": self.goal_pos,
            "walls_count": len(self.walls),
            "hazards_count": len(self.hazards),
            "steps_taken": self.steps_taken,
            "done": self.done,
        }

    def render(self) -> np.ndarray:
        """盤面を 2 次元カラーグリッド配列としてレンダリング."""
        grid = np.full(self.grid_shape, self.background_color, dtype=int)
        for wr, wc in self.walls:
            if 0 <= wr < self.grid_shape[0] and 0 <= wc < self.grid_shape[1]:
                grid[wr, wc] = self.wall_color
        for hr, hc in self.hazards:
            if 0 <= hr < self.grid_shape[0] and 0 <= hc < self.grid_shape[1]:
                grid[hr, hc] = self.hazard_color

        gr, gc = self.goal_pos
        if 0 <= gr < self.grid_shape[0] and 0 <= gc < self.grid_shape[1]:
            grid[gr, gc] = self.goal_color

        pr, pc = self.player_pos
        if 0 <= pr < self.grid_shape[0] and 0 <= pc < self.grid_shape[1]:
            grid[pr, pc] = self.player_color

        return grid
