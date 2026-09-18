"""Google ADK 2.0 ネイティブ ReAct 観測ツール＆反復思考テスト (test_adk_react_tools.py).

検証対象:
1. ObservationTools の各ツール関数 (inspect_board, inspect_affordances, inspect_action_effect, inspect_roi)
2. Google ADK 2.0 FunctionTool 宣言および SkillToolset との共存
3. ADKGamePlayer + LocalQwenVL による自律的マルチターン ReAct 思考ループ (思考 -> ツール呼出 -> 観測反映 -> 行動決定)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
import numpy as np
import pytest

from google.adk.tools import FunctionTool
from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.tools.observation_tools import ObservationTools


class TestObservationToolsUnit:
    """ObservationTools の各ツールの単体契約テスト."""

    @pytest.fixture
    def sample_grid(self) -> np.ndarray:
        # 5x5 グリッド: 0=背景, 1=プレイヤー, 2=ゴール, 3=障害物
        grid = np.zeros((5, 5), dtype=int)
        grid[1, 1] = 1
        grid[3, 3] = 2
        grid[2, 2] = 3
        return grid

    def test_inspect_board(self, sample_grid: np.ndarray) -> None:
        ot = ObservationTools()
        ot.update_context(grid=sample_grid, step_index=1, available_actions=["ACTION1", "ACTION2"])
        res = ot.inspect_board()

        assert res.get("success") is True
        assert res.get("grid_dimensions") == [5, 5]
        assert 1 in res.get("foreground_colors", [])
        assert 2 in res.get("foreground_colors", [])

    def test_inspect_affordances(self, sample_grid: np.ndarray) -> None:
        ot = ObservationTools()
        ot.update_context(grid=sample_grid, step_index=1)
        res = ot.inspect_affordances(detail=True)

        assert res.get("success") is True
        assert res.get("grid_shape") == [5, 5]
        assert "style" in res

    def test_inspect_action_effect(self, sample_grid: np.ndarray) -> None:
        ot = ObservationTools()
        # 直前アクションなし
        res0 = ot.inspect_action_effect()
        assert res0["is_effective"] is False

        # 直前アクションあり
        last_info = {
            "action": "ACTION1",
            "action_id": 1,
            "pixels_changed": 4,
            "is_effective": True,
            "reasoning": "Moving up",
        }
        ot.update_context(
            grid=sample_grid,
            step_index=2,
            last_action_info=last_info,
            dynamics_map={"UP": 1},
        )
        res1 = ot.inspect_action_effect()
        assert res1["is_effective"] is True
        assert res1["pixels_changed"] == 4
        assert res1["last_action"] == "ACTION1"
        assert res1["dynamics_map"] == {"UP": 1}

    def test_inspect_roi(self, sample_grid: np.ndarray) -> None:
        ot = ObservationTools()
        ot.update_context(grid=sample_grid)
        res = ot.inspect_roi(top=0, left=0, height=3, width=3)

        assert res.get("success") is True
        assert res.get("subgrid_shape") == [3, 3]
        # (1, 1) にある色 1 が含まれる
        assert 1 in res.get("colors_in_roi", [])

    def test_get_tools_returns_adk_function_tools(self) -> None:
        ot = ObservationTools()
        tools = ot.get_tools()
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "inspect_board" in tool_names
        assert "inspect_affordances" in tool_names
        assert "inspect_action_effect" in tool_names
        assert "inspect_roi" in tool_names
        for t in tools:
            assert isinstance(t, FunctionTool)


class TestADKReActLoop:
    """エージェントによるマルチターン自律 ReAct 思考ループテスト."""

    def test_agent_multi_turn_react_loop(self) -> None:
        """エージェントが複数ツールを自律的に呼び出し、観測結果を反映して決定することを検証."""
        turn_counter = 0

        def react_simulation_fn(prompt: str, images: Any = None) -> Any:
            nonlocal turn_counter
            turn_counter += 1

            if turn_counter == 1:
                # ターン 1: 盤面の大域情報を観測したい
                return {"tool_call": "inspect_board", "tool_args": {}}
            elif turn_counter == 2:
                # ターン 2: アフォーダンス (プレイヤー位置等) を詳細観測したい
                assert "inspect_board" in prompt or "Tool response" in prompt
                return {"tool_call": "inspect_affordances", "tool_args": {"detail": True}}
            else:
                # ターン 3: 十分な観測が集まったので、最終アクションを決定
                assert "inspect_affordances" in prompt or "Tool response" in prompt
                return (
                    '```json\n'
                    '{\n'
                    '  "hypothesis": "Visual inspection verified clear path to target",\n'
                    '  "goal": "Reach the green objective",\n'
                    '  "action": "ACTION1",\n'
                    '  "reasoning": "Moving forward based on multi-turn tool observation"\n'
                    '}\n'
                    '```'
                )

        mock_vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=react_simulation_fn)
        player = ADKGamePlayer(model=mock_vlm, name="test_react_player")

        grid = np.zeros((6, 6), dtype=int)
        grid[1, 1] = 1  # player
        grid[4, 4] = 2  # goal

        decision = player.decide_next_action(
            grid=grid,
            available_actions=[1, 2, 3, 4],
            state_str="NOT_FINISHED",
        )

        # 3ターン (2回のツール呼び出し + 1回の最終決定) が実行されたことを確認
        assert turn_counter == 3
        assert decision.action_id == 1
        assert decision.action_name == "ACTION1"
        assert "multi-turn tool observation" in decision.reasoning
