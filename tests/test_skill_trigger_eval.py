"""EDD Contract Tests for Autonomous Meta-Skill Triggering in Google ADK 2.0.

エージェントが状況（難所、仮説検証、逆算計画、視覚検査など）に応じて、
メタスキルを自律的かつ適切にトリガーできるかを契約テスト（正例3件＋負例3件）で評価・検証する。
"""

import json
from typing import Any, Dict, List
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.harness.game_action_tools import ActionDecision


# -----------------------------------------------------------------------------
# 正例 3件 (Positive Scenarios): メタスキルが適切に要求・展開されるシナリオ
# -----------------------------------------------------------------------------

def test_positive_backward_planner_trigger_on_complex_maze():
    """正例 1: 遠距離・迷路状の盤面で backward-planner が自律トリガーされ展開されること."""
    call_idx = 0
    responses = [
        # 1. Planner: 迷路状でゴールが遠いため backward-planner を要求
        json.dumps({
            "hypothesis": "Complex labyrinth, target is isolated in top-right corner",
            "goal": "Decompose route backwards from destination",
            "action": "ACTION1",
            "load_skill": "backward-planner",
            "reasoning": "Need backward-planner to formulate subgoal checkpoints",
        }),
        # 2. Planner (Progressive Disclosure 展開後の確定計画):
        json.dumps({
            "hypothesis": "Subgoal 1: Reach doorway at (3, 5)",
            "goal": "Move right toward door",
            "action": "RIGHT",
            "reasoning": "First milestone determined by backward plan",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_bp_player", app_name="test_app_bp")
    grid = np.zeros((15, 15), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert decision.loaded_skill == "backward-planner"
    assert decision.action_id == 4
    assert decision.action_name == "ACTION4"


def test_positive_visual_inspector_trigger_on_novel_board():
    """正例 2: 未知のカラー配置・多色構造の盤面で visual-inspector が自律トリガーされること."""
    call_idx = 0
    responses = [
        # 1. Planner: 盤面精査のため visual-inspector を要求
        json.dumps({
            "hypothesis": "Novel multi-colored board with symmetrical patterns",
            "goal": "Inspect color distributions and active clusters",
            "action": "ACTION1",
            "load_skill": "visual-inspector",
            "reasoning": "Consult visual-inspector for gestalt analysis",
        }),
        # 2. Planner (展開後):
        json.dumps({
            "hypothesis": "Cyan cluster acts as boundary, player is magenta dot at (2, 2)",
            "goal": "Step down into open passage",
            "action": "DOWN",
            "reasoning": "Passage identified below player",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_vi_player", app_name="test_app_vi")
    grid = np.zeros((12, 12), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert decision.loaded_skill == "visual-inspector"
    assert decision.action_id == 2
    assert decision.action_name == "ACTION2"


def test_positive_epistemic_prober_trigger_on_ambiguous_affordance():
    """正例 3: 未知のオブジェクトアフォーダンスに対し epistemic-prober が自律トリガーされること."""
    call_idx = 0
    responses = [
        # 1. Planner: 未知オブジェクトの相互作用仮説検証のため epistemic-prober を要求
        json.dumps({
            "hypothesis": "Unknown yellow object at (4, 4), unclear if pushable or obstacle",
            "goal": "Probe affordance with minimal risk",
            "action": "ACTION3",
            "load_skill": "epistemic-prober",
            "reasoning": "Deploy probe action to test collision physics",
        }),
        # 2. Planner (展開後):
        json.dumps({
            "hypothesis": "Probe hypothesis: yellow object is a pushable block",
            "goal": "Push yellow object from left",
            "action": "RIGHT",
            "reasoning": "Contact test",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_ep_player", app_name="test_app_ep")
    grid = np.zeros((8, 8), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert decision.loaded_skill == "epistemic-prober"
    assert decision.action_id == 4


# -----------------------------------------------------------------------------
# 負例 3件 (Negative Scenarios): 不要・異常なメタスキル呼び出しを正しく制御
# -----------------------------------------------------------------------------

def test_negative_no_skill_trigger_on_trivial_straight_move():
    """負例 1: ゴールが隣接する自明な局面でメタスキルを無駄呼びせず、即座に直接行動できること."""
    call_idx = 0
    responses = [
        # 1. Planner: 直前1マスにゴールがあるためメタスキル要求なし (load_skill: None)
        json.dumps({
            "hypothesis": "Player is at (3, 3), green goal is at (3, 4)",
            "goal": "Step right to finish level",
            "action": "RIGHT",
            "load_skill": None,
            "reasoning": "Goal is adjacent, immediate 1-step clear",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_triv_player", app_name="test_app_triv")
    grid = np.zeros((6, 6), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    # メタスキルは呼び出されない (None)
    assert decision.loaded_skill is None
    assert decision.action_id == 4
    assert decision.action_name == "ACTION4"


def test_negative_invalid_skill_fallback():
    """負例 2: 存在しない架空のスキルが指定された場合、クラッシュせず安全にフォールバックすること."""
    call_idx = 0
    responses = [
        # 1. Planner: 存在しないスキル名を指定
        json.dumps({
            "hypothesis": "Random thought",
            "goal": "Try magic skill",
            "action": "UP",
            "load_skill": "super-imaginary-cheat-skill-999",
            "reasoning": "Trying unknown skill",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_fallback_player", app_name="test_app_fb")
    grid = np.zeros((5, 5), dtype=np.uint8)

    # クラッシュせずに正常終了すること
    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert decision.action_id == 1
    assert decision.action_name == "ACTION1"


def test_negative_invalid_action_fallback():
    """負例 3: モデルが無効・存在しないアクション名を指定した場合、利用可能アクションへ安全にフォールバックすること."""
    call_idx = 0
    responses = [
        # 1. Planner: 存在しないアクション (TELEPORT) を提案
        json.dumps({
            "hypothesis": "Confused model output",
            "goal": "Instant teleport",
            "action": "TELEPORT",
            "reasoning": "Hallucinated action",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_invalid_act_player", app_name="test_app_inv")
    grid = np.zeros((8, 8), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    # 利用可能なアクション（available_actions）の中から有効なIDが選ばれること
    assert decision.action_id in [1, 2, 3, 4]
