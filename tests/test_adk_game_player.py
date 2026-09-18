"""Google ADK 2.0 準拠 ADKGamePlayer および MyAgent の統合テスト."""

import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.agent.my_agent import MyAgent
from acr_agi3.harness.game_action_tools import ActionDecision, GameActionTools
from acr_agi3.harness.vision_observation import VisionObservationHarness
from arcengine import FrameData, GameAction, GameState


def test_vision_observation_harness_generates_image_and_text_parts():
    """グリッド観測から画像 Blob とテキストを含む Part リストが正しく構築されるかを検証."""
    harness = VisionObservationHarness(cell_size=20)

    # 5x5 テストグリッド (0: 背景, 1: 青, 2: 赤)
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 2
    grid[3, 3] = 1

    parts = harness.create_observation_parts(
        grid_data=grid,
        step_index=1,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        last_action_info={"action": "ACTION1", "pixels_changed": 1, "is_effective": True},
    )

    assert len(parts) >= 2
    # 最初の Part は画像 Blob
    img_part = parts[0]
    assert hasattr(img_part, "inline_data")
    assert img_part.inline_data.mime_type == "image/png"
    assert len(img_part.inline_data.data) > 0

    # 2つ目の Part はテキスト説明
    text_part = parts[1]
    assert hasattr(text_part, "text")
    assert "Grid Size: 5 rows x 5 cols" in text_part.text
    assert "Red (2)" in text_part.text
    assert "EFFECTIVE" in text_part.text


def test_game_action_tools_function_calling():
    """GameActionTools のツール呼び出しによる ActionDecision 生成を検証."""
    tools = GameActionTools(available_actions=[1, 2, 3, 4, 6])

    # 1. 移動アクション
    res_step = tools.step_action("RIGHT", reasoning="Move towards key")
    assert "Action `RIGHT`" in res_step
    assert tools.pending_decision is not None
    assert tools.pending_decision.action_name in ("RIGHT", "ACTION4")
    assert tools.pending_decision.action_id == 4
    assert "Move towards key" in tools.pending_decision.reasoning

    # 2. クリックアクション
    res_click = tools.click_at(x=10, y=15, reasoning="Click isolated blue switch")
    assert "Click scheduled at coordinate" in res_click
    assert tools.pending_decision.action_type == "CLICK"
    assert tools.pending_decision.action_id == 6
    assert tools.pending_decision.coordinates == {"x": 10, "y": 15}

    # 3. 能動的リセット
    res_reset = tools.reset_game(reasoning="Deadlocked in oscillation")
    assert "Environment reset scheduled" in res_reset
    assert tools.pending_decision.action_type == "RESET"
    assert tools.pending_decision.action_id == 0


def test_adk_game_player_end_to_end_decision():
    """モック推論モデルを用いた ADKGamePlayer のマルチモーダル思考・行動決定テスト."""
    def mock_generation_fn(prompt: str) -> str:
        # LLM が思考を行い、テキスト中で行動を表明するシナリオ
        return (
            "I have inspected the visual board. The red agent is at (1, 1) and goal is to the right. "
            "I decide to take ACTION4 (RIGHT) to progress toward the objective."
        )

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_generation_fn)
    player = ADKGamePlayer(model=mock_llm)

    grid = np.zeros((6, 6), dtype=int)
    grid[1, 1] = 2  # Agent
    grid[1, 4] = 3  # Goal

    decision = player.decide_next_action(
        grid=grid,
        available_actions=[1, 2, 3, 4],
        state_str="NOT_FINISHED",
    )

    assert isinstance(decision, ActionDecision)
    assert decision.action_id == 4
    assert decision.action_name in ["ACTION4", "RIGHT"]


def test_my_agent_integration_with_adk_player():
    """MyAgent が ADKGamePlayer を通じて arcengine の choose_action を正常に実行できるかを検証."""
    def mock_click_generation_fn(prompt: str) -> str:
        return "I see an interactive tile at (5, 8). Let's click (5, 8) to activate the switch."

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_click_generation_fn)
    agent = MyAgent(model=mock_llm)

    grid = np.zeros((10, 10), dtype=int)
    grid[5, 8] = 4  # Switch

    frame = FrameData(
        levels_completed=0,
        state=GameState.NOT_FINISHED,
        frame=[grid.tolist()],
        available_actions=[1, 2, 3, 4, 6],
    )

    action = agent.choose_action([frame], frame)
    assert action == GameAction.ACTION6
    assert hasattr(action, "action_data")
    data = action.action_data.model_dump()
    # grid[5, 8] は row=5, col=8 のため、正しい画面クリック座標は x=8, y=5 (自動反転・スナップ補正)
    assert data.get("x") == 8
    assert data.get("y") == 5


def test_adk_game_player_rethink_on_rejected_action():
    """無効アクション (click_at) 提示時に game-controller から拒絶され、自律的に Re-think で有効アクションへ修復できるかを検証."""
    call_count = 0

    def mock_rethink_generation_fn(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Phase 1: Perceive
            return "Perceive: Found player at (0, 0)."
        elif call_count == 2:
            # Phase 2: Plan
            return "Plan: Move player towards goal."
        elif call_count == 3:
            # Phase 3 (Act 初回): 移動専用ゲームなのに誤って click_at を提案
            return '```json\n{\n  "action": "click_at",\n  "coordinates": {"x": 1, "y": 1},\n  "reasoning": "Try clicking tile"\n}\n```'
        else:
            # Phase 3 (Act 再検討): エラー通知を受けて ACTION2 (下移動) に自己修正
            assert "ACTION REJECTED BY GAME CONTROLLER" in prompt
            return '```json\n{\n  "action": "ACTION2",\n  "reasoning": "Click was disabled, moving down instead"\n}\n```'

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_rethink_generation_fn)
    player = ADKGamePlayer(model=mock_llm)

    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 2

    decision = player.decide_next_action(
        grid=grid,
        available_actions=[1, 2, 3, 4],  # 6 (click) は利用不可
        state_str="NOT_FINISHED",
    )

    assert call_count == 4
    assert decision.action_id == 2
    assert decision.action_name == "ACTION2"
    assert decision.metadata.get("rethink_attempts") == 1
    assert decision.metadata.get("success") is True

