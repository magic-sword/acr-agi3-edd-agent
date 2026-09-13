"""EDD 診断アナライザーモジュール (Diagnostic Analyzer Adapter).

収集されたテレメトリから、膠着・空振り・即死・クリックミス等の根本原因を数理的に分析し、
診断レポート (DiagnosticReport) を生成する。
上流の edd_agent_tools が利用可能な場合はそちらへ委譲し、
Kaggle 本番等のオフライン環境ではローカルロジックにフォールバックする。
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Dict, List, Optional

from acr_agi3.edd.telemetry import SessionTelemetry

try:
    from edd_agent_tools.evaluation import DiagnosticAnalyzer as UpstreamAnalyzer  # type: ignore
    from edd_agent_tools.models.telemetry import DiagnosticReport  # type: ignore
except ImportError:
    UpstreamAnalyzer = None

    @dataclasses.dataclass
    class DiagnosticReport:
        """単一ゲームの診断レポート."""

        game_id: str
        title: str
        total_steps: int
        levels_completed: int
        win_levels: int
        effective_ratio: float
        max_consecutive_stagnation: int
        dominant_failure_category: str
        primary_recommendation: str
        action_stats: Dict[int, Dict[str, Any]]
        click_stats: Dict[str, Any]

        def to_dict(self) -> Dict[str, Any]:
            return dataclasses.asdict(self)


class DiagnosticAnalyzer:
    """セッションテレメトリを数理的に診断するアナライザー (Adapter)."""

    @staticmethod
    def analyze(session: SessionTelemetry) -> DiagnosticReport:
        if UpstreamAnalyzer is not None:
            return UpstreamAnalyzer.analyze(session)
        total_steps = session.total_steps
        if total_steps == 0:
            return DiagnosticReport(
                game_id=session.game_id,
                title=session.title,
                total_steps=0,
                levels_completed=0,
                win_levels=0,
                effective_ratio=0.0,
                max_consecutive_stagnation=0,
                dominant_failure_category="NoActionsExecuted",
                primary_recommendation="Agent exited without taking any step. Check initialization.",
                action_stats={},
                click_stats={},
            )

        # 1. アクション別統計
        act_counts: Dict[int, int] = collections.defaultdict(int)
        act_effective: Dict[int, int] = collections.defaultdict(int)
        act_names: Dict[int, str] = {}

        for s in session.steps:
            act_counts[s.action_id] += 1
            if s.is_effective:
                act_effective[s.action_id] += 1
            act_names[s.action_id] = s.action_name

        action_stats = {}
        for aid, count in act_counts.items():
            eff = act_effective[aid]
            action_stats[aid] = {
                "name": act_names.get(aid, f"ACTION{aid}"),
                "count": count,
                "effective_count": eff,
                "effective_rate": round(eff / count, 3) if count > 0 else 0.0,
            }

        # 2. クリック (ACTION6) の詳細統計
        clicks = [s for s in session.steps if s.action_id == 6]
        click_stats = {
            "total_clicks": len(clicks),
            "effective_clicks": sum(1 for c in clicks if c.is_effective),
            "effective_rate": (
                round(sum(1 for c in clicks if c.is_effective) / len(clicks), 3)
                if clicks
                else 0.0
            ),
            "sampled_coords": [
                (c.action_data.get("x", 0), c.action_data.get("y", 0)) for c in clicks[:5]
            ],
        }

        # 3. 最大連続停滞 (Max Consecutive Stagnation)
        curr_stag = 0
        max_stag = 0
        for s in session.steps:
            if not s.is_effective:
                curr_stag += 1
                if curr_stag > max_stag:
                    max_stag = curr_stag
            else:
                curr_stag = 0

        # 4. 根本原因 (Failure Category) の分類
        effective_ratio = session.effective_ratio
        if session.total_levels_completed >= session.total_win_levels and session.total_win_levels > 0:
            category = "COMPLETED"
            recommendation = "Task fully solved. Preserve strategy."
        elif len(clicks) > 0 and click_stats["effective_rate"] < 0.1:
            category = "ClickMissAndSpatialDisorientation"
            recommendation = (
                f"ACTION6 click hit-rate is extremely low ({click_stats['effective_rate']*100:.1f}%). "
                "Focus clicks specifically on active/changed pixel bounding boxes instead of uniform random."
            )
        elif effective_ratio < 0.2:
            category = "WallStagnationAndActionRejection"
            recommendation = (
                f"Only {effective_ratio*100:.1f}% of moves changed the screen (heavy wall collision). "
                "Apply momentum on effective moves and immediately branch away from ineffective actions."
            )
        elif session.final_state in ["GAME_OVER", "GameState.GAME_OVER"]:
            category = "LethalTrapCollision"
            recommendation = (
                "Agent triggered GAME_OVER. Track pre-reset positions as lethal hazards "
                "and enforce strict negative constraint pruning."
            )
        else:
            category = "ExplorationTimeout"
            recommendation = (
                "Actions are changing the environment, but goal condition was not reached within max steps. "
                "Increase step horizon or guide agent with heuristic potential towards rare color clusters."
            )

        return DiagnosticReport(
            game_id=session.game_id,
            title=session.title,
            total_steps=total_steps,
            levels_completed=session.total_levels_completed,
            win_levels=session.total_win_levels,
            effective_ratio=round(effective_ratio, 3),
            max_consecutive_stagnation=max_stag,
            dominant_failure_category=category,
            primary_recommendation=recommendation,
            action_stats=action_stats,
            click_stats=click_stats,
        )
