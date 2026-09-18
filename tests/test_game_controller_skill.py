"""Contract Tests for 1-Step Action Execution Meta-Skill (game-controller).

Google ADK 2.0 準拠の 1 手決定論的実行スキル (game-controller / GameActionTools) が
動的操作力学マップの動的解決、幾何アフォーダンスへの自動吸着クリック、
および安全なフォールバックを正しく実行できるかを契約テスト（正例4件＋負例2件）で検証する。
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# meta_skills/game-controller を sys.path に追加
_SKILL_DIR = Path(__file__).resolve().parents[1] / "meta_skills" / "game-controller" / "scripts"
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

from acr_agi3.harness.game_action_tools import GameActionTools
from game_controller import GameController


def test_positive_dynamic_action_semantics_mapping():
    """正例 1: 操作力学マップ (UP -> ACTION3) が注入された場合、確実に ACTION3 が発行されること."""
    tools = GameActionTools(
        available_actions=[1, 2, 3, 4],
        dynamics_map={"UP": 3, "DOWN": 4, "LEFT": 1, "RIGHT": 2},
    )

    tools.step_action("UP", reasoning="Moving up via identified dynamics")
    decision = tools.pending_decision
    assert decision is not None
    assert decision.action_id == 3
    assert decision.action_name == "ACTION3"
    assert decision.loaded_skill == "game-controller"


def test_positive_click_auto_snaps_unspecified_coordinates():
    """正例 2: クリック座標が省略された場合、幾何アフォーダンス検出により対象オブジェクト中心へ自動吸着すること."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[4, 7] = 5  # 有色オブジェクト (色5: グレー/スイッチ)

    tools = GameActionTools(available_actions=[1, 2, 3, 4, 6])
    tools.click_at(x=None, y=None, grid=grid, reasoning="Toggling discovered switch")

    decision = tools.pending_decision
    assert decision is not None
    assert decision.action_type == "CLICK"
    assert decision.action_id == 6
    assert decision.coordinates == {"x": 7, "y": 4}


def test_positive_click_snaps_empty_cell_to_nearest_object():
    """正例 3: 空セルをクリックしようとした場合、近傍の前景ピクセルへ自動吸着すること."""
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[3, 3] = 2  # 前景オブジェクト

    tools = GameActionTools(available_actions=[1, 2, 3, 4, 6])
    tools.click_at(x=4, y=3, grid=grid, reasoning="Clicking near object")

    decision = tools.pending_decision
    assert decision is not None
    assert decision.action_type == "CLICK"
    assert decision.coordinates == {"x": 3, "y": 3}


def test_positive_active_reset():
    """正例 4: reset_game を呼んだ場合、安全に ACTION0 (RESET) が発行されること."""
    tools = GameActionTools(available_actions=[1, 2, 3, 4])
    tools.reset_game(reasoning="Deadlocked in corner")

    decision = tools.pending_decision
    assert decision is not None
    assert decision.action_type == "RESET"
    assert decision.action_id == 0


def test_negative_invalid_action_rejection():
    """負例 1: 未知・無効なアクション名が渡された場合、暗黙のフォールバックを行わずエラーを返却すること."""
    tools = GameActionTools(available_actions=[2, 4])
    res = tools.step_action("TELEPORT", reasoning="Hallucinated action")

    assert "Error" in res
    assert "disabled" in res or "not in available actions" in res
    assert tools.pending_decision is None


def test_negative_click_unavailable_rejection():
    """負例 2: ACTION6 が利用不可の状態でクリックが呼ばれた場合、勝手に移動へ書き換えずにエラーを返却すること."""
    tools = GameActionTools(available_actions=[1, 2, 3, 4])  # 6 は含まれない
    res = tools.click_at(x=5, y=5, reasoning="Click when click disabled")

    assert "Error" in res
    assert "disabled" in res or "not in available actions" in res
    assert tools.pending_decision is None


def test_negative_controller_validation_flags_unavailable_action():
    """負例 3: GameController.parse_and_validate が無効アクションに対して success=False と error を正しく返すこと."""
    controller = GameController(available_actions=[1, 2, 3, 4])
    validation = controller.parse_and_validate({"action": "click_at", "x": 5, "y": 5})

    assert validation["success"] is False
    assert validation["error"] is not None
    assert "disabled" in validation["error"] or "not in available actions" in validation["error"]

