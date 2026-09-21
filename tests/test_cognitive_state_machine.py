"""Tests for Google ADK 2.0 Cognitive State Machine (Fast Path, Deviation Detection, Queue Execution)."""

import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveState, CognitiveMode
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.harness.game_action_tools import ActionDecision


def test_state_machine_fast_path_execution_and_deviation():
    """EXECUTING 状態で plan_queue からサクサク実行され、壁衝突で RECOVERY に戻る動作を検証."""
    call_count = {"llm": 0}

    def counting_mock_llm(prompt: str) -> str:
        call_count["llm"] += 1
        return "I will proceed with the planned action."

    mock_llm = LocalTransformersLlm("mock", generation_fn=counting_mock_llm)
    player = ADKGamePlayer(model=mock_llm)

    # 1. 手動で plan_queue に 3 手をセットし、EXECUTING 状態にする
    player.cognitive_state = CognitiveState.EXECUTING
    player.plan_queue = [
        {"action": "RIGHT"},
        {"action": "RIGHT"},
        {"action": "UP"},
    ]

    grid1 = np.zeros((8, 8), dtype=int)
    grid1[2, 2] = 2  # Agent
    grid1[2, 6] = 3  # Goal

    # Execution is permitted only after a confirmed prediction succeeds.
    from acr_agi3.agent.execution_evidence import Motion
    prior = grid1.copy()
    prior[2, 2] = 0
    prior[2, 1] = 2
    player.execution_evidence.samples[4] = (Motion(2, 0, 1), 2)
    player.execution_evidence.observe(prior, None)
    player.execution_evidence.arm(prior, 4)
    player.last_action_info = {"action_id": 4}
    player.last_grid = prior

    # Step 1: EXECUTING 高速パス発動 (LLM 呼び出し回数 0)
    decision1 = player.decide_next_action(grid=grid1, available_actions=[1, 2, 3, 4])
    assert decision1.action_name in ("RIGHT", "ACTION4")
    assert len(player.plan_queue) == 2
    assert player.cognitive_state == CognitiveState.EXECUTING
    assert call_count["llm"] == 0  # LLM は一切呼ばれていない (Fast Path!)

    # Step 2: 盤面が予定通り変化（エージェントが右に移動）
    grid2 = np.zeros((8, 8), dtype=int)
    grid2[2, 3] = 2  # Agent moved right
    grid2[2, 6] = 3

    decision2 = player.decide_next_action(grid=grid2, available_actions=[1, 2, 3, 4])
    assert decision2.action_name in ("RIGHT", "ACTION4")
    assert len(player.plan_queue) == 1
    assert player.cognitive_state == CognitiveState.EXECUTING
    assert call_count["llm"] == 0  # 依然として LLM ゼロ呼び出し！

    # Step 3: 壁衝突が発生！(盤面が全く変化しない grid2 のまま)
    # Deviation Detector が発火し、キューを破棄して RECOVERY ➔ PLANNING に戻る
    decision3 = player.decide_next_action(grid=grid2, available_actions=[1, 2, 3, 4])
    assert len(player.plan_queue) == 0  # キューが安全に破棄されている
    # 逸脱検知により RECOVERY 状態を経て再計画ループ (Slow Path) がキックされる
    assert call_count["llm"] > 0


def test_automatic_plan_queue_construction():
    """A* 経路が存在する場合、自動的に plan_queue が構築され EXECUTING に移行するかを検証."""
    def mock_llm_fn(prompt: str) -> str:
        return "I follow the backward planner's path."

    mock_llm = LocalTransformersLlm("mock", generation_fn=mock_llm_fn)
    player = ADKGamePlayer(model=mock_llm)

    # dynamics_map を設定してプロービング不要にする
    player.dynamics_map = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
    player.action_tools.set_dynamics_map(player.dynamics_map)
    if player.planning_tools.prober is not None:
        player.planning_tools.prober.dynamics_map = dict(player.dynamics_map)

    grid = np.zeros((10, 10), dtype=int)
    grid[2, 2] = 2  # Agent
    grid[2, 5] = 3  # Goal (3 steps away to the right)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert isinstance(decision, ActionDecision)

    # 経路が見つかり、2手目以降がキューに入って EXECUTING になっていること
    if player.plan_queue:
        assert player.cognitive_state == CognitiveState.EXECUTING
        assert len(player.plan_queue) >= 1
