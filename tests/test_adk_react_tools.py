"""Google ADK 2.0 スキル構造＆3フェーズワークフローテスト (test_adk_react_tools.py).

検証対象:
1. meta_skills/visual-inspector の VisualInspector コアエンジン
2. SkillHarness の get_scoped_toolset による最小権限スキル分離
3. ADKGamePlayer による 3フェーズ (Perceive -> Plan -> Act) パイプライン
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

# visual-inspector スクリプトのインポートパス解決
_VISUAL_INSPECTOR_DIR = (
    Path(__file__).resolve().parents[1]
    / "meta_skills"
    / "visual-inspector"
    / "scripts"
)
if str(_VISUAL_INSPECTOR_DIR) not in sys.path and _VISUAL_INSPECTOR_DIR.exists():
    sys.path.insert(0, str(_VISUAL_INSPECTOR_DIR))

from visual_inspector import VisualInspector
from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.meta.skill_harness import SkillHarness


class TestVisualInspectorUnit:
    """meta_skills/visual-inspector の各分析機能の単体契約テスト."""

    @pytest.fixture
    def sample_grid(self) -> np.ndarray:
        # 5x5 グリッド: 0=背景, 1=プレイヤー, 2=ゴール, 3=障害物
        grid = np.zeros((5, 5), dtype=int)
        grid[1, 1] = 1
        grid[3, 3] = 2
        grid[2, 2] = 3
        return grid

    def test_inspect_board(self, sample_grid: np.ndarray) -> None:
        vi = VisualInspector()
        res = vi.inspect_board(sample_grid, step_index=1)

        assert res.get("grid_dimensions") == [5, 5]
        assert 1 in res.get("foreground_colors", [])
        assert 2 in res.get("foreground_colors", [])

    def test_inspect_affordances(self, sample_grid: np.ndarray) -> None:
        vi = VisualInspector()
        report = vi.analyze_frame(sample_grid)

        assert report.grid_shape == (5, 5)
        assert report.style in ["OPEN_EXPLORATION", "CLOSED_MAZE", "ITEM_TRIGGER_PUZZLE", "SYMMETRIC_PATTERN"]

    def test_scoped_toolsets_per_node(self) -> None:
        """各ノードへ渡す SkillToolset が指定スキルのみに限定開示されているかを検証."""
        harness = SkillHarness()

        # Node 1: visual-inspector のみ
        ts1 = harness.get_scoped_toolset(["visual-inspector"])
        s1_names = list(ts1._skills.keys())
        assert "visual-inspector" in s1_names
        assert "game-controller" not in s1_names
        assert "memory-notebook" not in s1_names

        # Node 2: memory-notebook のみ
        ts2 = harness.get_scoped_toolset(["memory-notebook"])
        s2_names = list(ts2._skills.keys())
        assert "memory-notebook" in s2_names
        assert "visual-inspector" not in s2_names
        assert "game-controller" not in s2_names

        # Node 3: game-controller のみ
        ts3 = harness.get_scoped_toolset(["game-controller"])
        s3_names = list(ts3._skills.keys())
        assert "game-controller" in s3_names
        assert "visual-inspector" not in s3_names
        assert "memory-notebook" not in s3_names


class TestADKThreePhaseWorkflow:
    """エージェントによる 3フェーズ (Perceive -> Plan -> Act) パイプラインテスト."""

    def test_agent_three_phase_workflow_execution(self) -> None:
        """3つのフェーズが順番に実行され、最終アクションが正しく決定されることを検証."""
        calls = []

        def simulation_fn(prompt: str, images: Any = None) -> Any:
            calls.append(prompt)
            # フェーズごとに適切な応答をシミュレート
            if len(calls) == 1:
                # Phase 1 (Perceive) への応答
                return "Observation Summary: Player at (1, 1), Goal at (4, 4), layout is OPEN_EXPLORATION."
            elif len(calls) == 2:
                # Phase 2 (Plan) への応答
                return "Plan Strategy: Immediate subgoal is to move towards Goal at (4, 4) avoiding deadlocks."
            else:
                # Phase 3 (Act) への応答
                return (
                    '```json\n'
                    '{\n'
                    '  "action": "ACTION1",\n'
                    '  "reasoning": "Moving up to advance towards goal"\n'
                    '}\n'
                    '```'
                )

        mock_vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=simulation_fn)
        player = ADKGamePlayer(model=mock_vlm, name="test_3phase_player")

        grid = np.zeros((6, 6), dtype=int)
        grid[1, 1] = 1  # player
        grid[4, 4] = 2  # goal

        decision = player.decide_next_action(
            grid=grid,
            available_actions=[1, 2, 3, 4],
            state_str="NOT_FINISHED",
        )

        # 3 つのフェーズ（Perceive -> Plan -> Act）が実行されたことを確認
        assert len(calls) >= 3
        assert decision.action_id == 1
        assert decision.action_name == "ACTION1"
        assert decision.metadata.get("success") is True
