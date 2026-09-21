"""Tests for Cognitive Mode State Machine and Workflow in ADKGamePlayer."""

import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveMode
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.harness.game_action_tools import ActionDecision


def test_determine_cognitive_mode_transitions():
    """determine_cognitive_mode の全認知的状態遷移および優先順位を検証."""
    player = ADKGamePlayer(model=LocalTransformersLlm("mock", generation_fn=lambda p: "ACTION1"))

    # 1. 停滞・壁衝突時は最優先で TABOO_RECOVERY
    mode = player.determine_cognitive_mode(
        step_index=1,
        stagnation_count=1,
        available_action_ids=[1, 2, 3, 4],
        has_probe_rec=True,
        has_nav_path=True,
        has_anchors=True,
        has_preconditions=True,
    )
    assert mode == CognitiveMode.TABOO_RECOVERY

    # 2. 初期ステップ (<= 4) かつプロービング推奨時は PROBING_SCIENTIST
    mode = player.determine_cognitive_mode(
        step_index=2,
        stagnation_count=0,
        available_action_ids=[1, 2, 3, 4],
        has_probe_rec=True,
        has_nav_path=False,
        has_anchors=False,
    )
    assert mode == CognitiveMode.PROBING_SCIENTIST

    # 3. 因果前提条件 (鍵/扉) が存在する場合は CAUSAL_PROGRAMMER
    mode = player.determine_cognitive_mode(
        step_index=5,
        stagnation_count=0,
        available_action_ids=[1, 2, 3, 4],
        has_probe_rec=False,
        has_nav_path=True,
        has_anchors=False,
        has_preconditions=True,
    )
    assert mode == CognitiveMode.CAUSAL_PROGRAMMER

    # 4. ゴールまでの A* 幾何経路が存在する場合は BACKWARD_ARCHITECT
    mode = player.determine_cognitive_mode(
        step_index=5,
        stagnation_count=0,
        available_action_ids=[1, 2, 3, 4],
        has_probe_rec=False,
        has_nav_path=True,
        has_anchors=False,
        has_preconditions=False,
    )
    assert mode == CognitiveMode.BACKWARD_ARCHITECT

    # 5. クリック可能アンカーが存在し ACTION6 がある場合は CAUSAL_PROGRAMMER
    mode = player.determine_cognitive_mode(
        step_index=6,
        stagnation_count=0,
        available_action_ids=[1, 2, 3, 4, 6],
        has_probe_rec=False,
        has_nav_path=False,
        has_anchors=True,
        has_preconditions=False,
    )
    assert mode == CognitiveMode.CAUSAL_PROGRAMMER

    # 6. それ以外は通常の探索・障害物回避 RISK_NAVIGATOR
    mode = player.determine_cognitive_mode(
        step_index=6,
        stagnation_count=0,
        available_action_ids=[1, 2, 3, 4],
        has_probe_rec=False,
        has_nav_path=False,
        has_anchors=False,
        has_preconditions=False,
    )
    assert mode == CognitiveMode.RISK_NAVIGATOR


def test_cognitive_mode_in_decide_next_action():
    """decide_next_action 実行時に current_cognitive_mode が更新され、ログ・フォールバックに反映されるかを検証."""
    def mock_generation_fn(prompt: str) -> str:
        if "PROBING_SCIENTIST" in prompt:
            return "I am in probing mode. I will test ACTION2 to probe physics."
        return "I will take ACTION4 to move toward the goal."

    mock_llm = LocalTransformersLlm("mock", generation_fn=mock_generation_fn)
    player = ADKGamePlayer(model=mock_llm)

    grid = np.zeros((8, 8), dtype=int)
    grid[2, 2] = 2  # Agent
    grid[2, 6] = 3  # Goal

    # Step 1: 初動プロービング (step_index=1, 未検証アクションあり)
    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert player.current_cognitive_mode == CognitiveMode.PROBING_SCIENTIST
    assert isinstance(decision, ActionDecision)

    # Step 2: 停滞なし (エージェントが移動して盤面が変化)、力学同定済みの場合の遷移テスト
    # 強制的に dynamics_map をすべて埋めてプロービング不要にする
    player.dynamics_map = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
    player.action_tools.set_dynamics_map(player.dynamics_map)
    if player.planning_tools.prober is not None:
        player.planning_tools.prober.dynamics_map = dict(player.dynamics_map)

    grid2 = np.zeros((8, 8), dtype=int)
    grid2[2, 3] = 2  # Agent moved right
    grid2[2, 6] = 3  # Goal

    decision2 = player.decide_next_action(grid=grid2, available_actions=[1, 2, 3, 4])
    # A manually supplied mapping and one displacement are not confirmation.
    assert player.current_cognitive_mode == CognitiveMode.PROBING_SCIENTIST
    assert not player.plan_queue

    # Step 3: 壁衝突等で 0 変化が起きた場合の TABOO_RECOVERY
    # 盤面が変化しなかった場合 (grid2 のまま)
    decision3 = player.decide_next_action(grid=grid2, available_actions=[1, 2, 3, 4])
    assert player.current_cognitive_mode == CognitiveMode.TABOO_RECOVERY
