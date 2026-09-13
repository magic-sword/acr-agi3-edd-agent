"""EDD テレメトリ収集モジュール (Telemetry Adapter).

エージェントの各ステップにおける行動、グリッド差分、状態遷移をリアルタイム記録する。
上流の edd_agent_tools が利用可能な場合はそちらを優先利用し、
Kaggle 本番等のオフライン環境ではローカル定義にフォールバックする。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

try:
    from edd_agent_tools.models.telemetry import SessionTelemetry, StepTelemetry  # type: ignore
except ImportError:
    @dataclasses.dataclass
    class StepTelemetry:
        """単一ステップの計測データ."""

        step_index: int
        action_id: int
        action_name: str
        action_data: Dict[str, Any]
        reasoning: Optional[str | Dict[str, Any]]
        state_before: str
        state_after: str
        levels_completed: int
        win_levels: int
        pixels_changed: int
        changed_colors: List[int]
        is_effective: bool  # 画面に変化があったか
        time_taken_ms: float = 0.0

    @dataclasses.dataclass
    class SessionTelemetry:
        """1ゲーム全体のセッション計測データ."""

        game_id: str
        title: str
        baseline_actions: List[int]
        steps: List[StepTelemetry] = dataclasses.field(default_factory=list)
        initial_shape: Tuple[int, int] = (0, 0)
        final_state: str = "NOT_FINISHED"
        total_levels_completed: int = 0
        total_win_levels: int = 0
        total_time_sec: float = 0.0

        def add_step(self, step: StepTelemetry) -> None:
            self.steps.append(step)

        @property
        def total_steps(self) -> int:
            return len(self.steps)

        @property
        def effective_steps(self) -> int:
            return sum(1 for s in self.steps if s.is_effective)

        @property
        def effective_ratio(self) -> float:
            if not self.steps:
                return 0.0
            return self.effective_steps / len(self.steps)
