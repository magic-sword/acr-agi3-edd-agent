"""ACR-AGI-3 インタラクティブゲームプレイ環境 (Interactive Game Environment)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class Action(IntEnum):
    """ACR-AGI-3 ゲーム操作アクション."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    WAIT = 5

    @classmethod
    def from_str(cls, val: str) -> Action:
        v = val.strip().upper()
        mapping = {
            "UP": cls.UP,
            "DOWN": cls.DOWN,
            "LEFT": cls.LEFT,
            "RIGHT": cls.RIGHT,
            "INTERACT": cls.INTERACT,
            "WAIT": cls.WAIT,
        }
        if v not in mapping:
            raise ValueError(f"Unknown action string: {val}")
        return mapping[v]


@dataclass
class StepResult:
    """環境ステップ実行結果."""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class GameEnvironment(ABC):
    """ACR-AGI-3 未知ゲーム環境の抽象基底クラス."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """環境を初期状態にリセットし、初期盤面観測 (Observation) を返す."""
        pass

    @abstractmethod
    def step(self, action: Action | int) -> StepResult:
        """アクションを実行し、次状態・報酬・終了フラグ・メタ情報を返す."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """現在の内部ゲーム状態（プレイヤー位置、ゴール位置、スコア等）を返す."""
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """現在の盤面を 2 次元カラーグリッド配列としてレンダリング."""
        pass
