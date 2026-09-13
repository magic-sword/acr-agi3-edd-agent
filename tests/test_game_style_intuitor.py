"""GameStyleIntuitor メタスキルの単体テスト."""

import numpy as np
import pytest

from acr_agi3.meta.skill_harness import SkillHarness

GameStyleIntuitor = SkillHarness().get_skill_module("game-style-intuitor").GameStyleIntuitor


@pytest.fixture
def intuitor():
    return GameStyleIntuitor()


def test_open_exploration_style(intuitor):
    """外周が開いておりゴールが画面外にあるような探索型ゲームの直感テスト."""
    # 8x8 グリッドで外周が完全に開いており、障害物が中央にまばらにある
    obs = np.zeros((8, 8), dtype=int)
    obs[3, 3] = 1
    obs[4, 4] = 1

    res = intuitor.analyze_style(obs)
    assert res["style"] == "OPEN_EXPLORATION"
    assert res["recommended_domain"] == "exploration"
    assert "EXPLORATION FIRST" in res["recommended_approach"]
    assert res["features"]["has_edge_exit"] is True


def test_closed_maze_style(intuitor):
    """外周が壁で囲まれ、プレイヤーとゴールのみが存在する閉鎖型迷路の直感テスト."""
    obs = np.zeros((8, 8), dtype=int)
    # 外周を壁 (color 1) で囲む
    obs[0, :] = 1
    obs[7, :] = 1
    obs[:, 0] = 1
    obs[:, 7] = 1
    # 内部に壁
    obs[2:6, 3] = 1
    # プレイヤー (color 2) と ゴール (color 3)
    obs[1, 1] = 2
    obs[6, 6] = 3

    res = intuitor.analyze_style(obs)
    assert res["style"] == "CLOSED_MAZE"
    assert res["recommended_domain"] == "navigation"
    assert "PATHFINDING FIRST" in res["recommended_approach"]
    assert res["features"]["has_edge_exit"] is False


def test_hazard_avoidance_style(intuitor):
    """致死トラップ帯（4マス）が配置された危険地帯の直感テスト."""
    obs = np.zeros((8, 8), dtype=int)
    # 外周壁
    obs[0, :] = 1
    obs[7, :] = 1
    obs[:, 0] = 1
    obs[:, 7] = 1
    # プレイヤーとゴール
    obs[1, 1] = 2
    obs[6, 6] = 3
    # 致死溶岩帯 (color 4, 4マス帯)
    obs[2:6, 4] = 4

    res = intuitor.analyze_style(obs)
    assert res["style"] == "HAZARD_AVOIDANCE"
    assert res["recommended_domain"] == "hazard_avoidance"
    assert "SAFETY FIRST" in res["recommended_approach"]


def test_item_trigger_puzzle_style(intuitor):
    """プレイヤー、ゴールに加え、鍵などの追加アイテムが散在するパズルの直感テスト."""
    obs = np.zeros((8, 8), dtype=int)
    # 外周壁
    obs[0, :] = 1
    obs[7, :] = 1
    obs[:, 0] = 1
    obs[:, 7] = 1
    # プレイヤー (2), ゴール (3), 鍵 (4) -> 孤立アイテムが3個
    obs[1, 1] = 2
    obs[6, 6] = 3
    obs[2, 5] = 4

    res = intuitor.analyze_style(obs)
    assert res["style"] == "ITEM_TRIGGER_PUZZLE"
    assert res["recommended_domain"] == "inventory_puzzle"
    assert "INTERACTION FIRST" in res["recommended_approach"]


def test_symmetric_pattern_style(intuitor):
    """空間対称性が高いパズルの直感テスト."""
    # 左右対称なブロック配置
    obs = np.zeros((8, 8), dtype=int)
    obs[2, 2] = 1
    obs[2, 5] = 1
    obs[3, 1:7] = 2
    obs[4, 1:7] = 2
    obs[5, 2] = 1
    obs[5, 5] = 1

    res = intuitor.analyze_style(obs)
    assert res["style"] == "SYMMETRIC_PATTERN"
    assert res["recommended_domain"] == "symmetry_pattern"
    assert res["features"]["symmetry_score"] >= 0.80
