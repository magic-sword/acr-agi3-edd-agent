"""ARC-AGI-3 アフォーダンス誘導型クリック精度向上 (Click Precision) の契約テスト.

EDD (Evaluation-Driven Development) 防壁ゲート:
- 正例 3 件:
  1. object_id 指定による検出オブジェクト重心への正確な自動スナップ
  2. 背景色（地面）上の明示的座標 (x, y) 指定時、近隣物体に歪められず生座標がそのまま出力されること
  3. 座標未指定時の最優先検出オブジェクトへの自動スナップ
- 負例 3 件:
  1. 盤面外座標の明確なエラー却下
  2. 存在しない object_id 指定時の安全なフォールバック
  3. ACTION6 不許可環境でのクリック要求抑止
"""

import numpy as np
import pytest

from pathlib import Path
import sys

GAME_CONTROLLER_DIR = Path(__file__).resolve().parent.parent / "meta_skills" / "game-controller" / "scripts"
if str(GAME_CONTROLLER_DIR) not in sys.path and GAME_CONTROLLER_DIR.exists():
    sys.path.insert(0, str(GAME_CONTROLLER_DIR))

from acr_agi3.harness.vision_observation import detect_interactable_objects
from game_controller import GameController


# ==============================================================================
# 正例 3 件: 正常系アフォーダンス検出・object_id 自動スナップ・明示的地面クリック
# ==============================================================================

def test_positive_click_snapping_to_specified_object_id():
    """正例 1: object_id を指定した場合、検出された該当オブジェクトの重心へ正確に自動スナップすること."""
    grid = np.zeros((30, 30), dtype=int)
    # Magenta (6) の 3x3 ボタンを (x=10..12, y=15..17) に配置 -> center is (11, 16)
    grid[15:18, 10:13] = 6
    # Teal (8) の 2x2 ボタンを (x=20..21, y=5..6) に配置 -> center is (20, 5)
    grid[5:7, 20:22] = 8

    controller = GameController(available_actions=[1, 2, 3, 4, 6])

    # object_id = 0 を指定してクリック
    payload = {"action": "ACTION6", "object_id": 0, "reasoning": "Click first button"}
    res = controller.parse_and_validate(payload, grid_shape=(30, 30), grid=grid)

    assert res["success"] is True
    assert res["action_name"] == "ACTION6"
    assert "auto-snapped to object #0" in res["reasoning"]
    # いずれかのオブジェクトの重心にスナップされていること
    cx, cy = res["coordinates"]["x"], res["coordinates"]["y"]
    assert (cx, cy) in [(11, 16), (20, 5)]


def test_positive_click_explicit_ground_coordinates_respected():
    """正例 2: エージェントが明示的に座標 (x, y) を指定した場合、背景色（地面）上であっても物体に吸着されず生座標が尊重されること."""
    grid = np.zeros((30, 30), dtype=int)
    # Target: Green (3) at (x=15, y=15)
    grid[15, 15] = 3

    controller = GameController(available_actions=[1, 2, 3, 4, 6])
    # 近傍の背景色セル (x=17, y=16) を地面としてクリック指定（距離 2px だが物体ではなく地面をクリックしたい意図）
    payload = {"action": "ACTION6", "x": 17, "y": 16, "reasoning": "Move character to ground tile at (17, 16)"}

    res = controller.parse_and_validate(payload, grid_shape=(30, 30), grid=grid)
    assert res["success"] is True
    assert res["action_name"] == "ACTION6"
    # 近隣の (15, 15) に強制吸着されず、指定通りの (17, 16) が維持されること
    assert res["coordinates"]["x"] == 17
    assert res["coordinates"]["y"] == 16
    assert "snapped" not in res["reasoning"]


def test_positive_click_unspecified_defaults_to_detected_object():
    """正例 3: x, y が未指定 (None) の場合、最優先の検出オブジェクトへ自動スナップすること."""
    grid = np.zeros((30, 30), dtype=int)
    # Red (2) オブジェクトを col=5, row=25 に配置
    grid[25, 5] = 2

    controller = GameController(available_actions=[1, 2, 6])
    # 座標未指定のクリック要求
    payload = {"action": "ACTION6", "reasoning": "Click whatever is interactive"}

    res = controller.parse_and_validate(payload, grid_shape=(30, 30), grid=grid)
    assert res["success"] is True
    assert res["coordinates"]["x"] == 5
    assert res["coordinates"]["y"] == 25
    assert "auto-snapped unspecified click" in res["reasoning"]


# ==============================================================================
# 負例 3 件: 異常系・制約違反の適切なハンドリング
# ==============================================================================

def test_negative_click_out_of_bounds_rejection():
    """負例 1: 盤面境界外の無謀な座標は明確にエラー却下されること."""
    grid = np.zeros((20, 20), dtype=int)
    grid[5, 5] = 4

    controller = GameController(available_actions=[1, 2, 6])
    payload = {"action": "ACTION6", "x": 999, "y": 999}

    res = controller.parse_and_validate(payload, grid_shape=(20, 20), grid=grid)
    assert res["success"] is False
    assert "out of grid bounds" in res["error"]


def test_negative_click_invalid_object_id_fallback():
    """負例 2: 存在しない object_id が指定された場合でも、安全にフォールバックしてクラッシュしないこと."""
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 5

    controller = GameController(available_actions=[1, 2, 6])
    # 存在しない object_id=999 を指定
    payload = {"action": "ACTION6", "object_id": 999, "reasoning": "Click phantom object"}

    res = controller.parse_and_validate(payload, grid_shape=(20, 20), grid=grid)
    assert res["success"] is True
    assert "object #999 not found" in res["reasoning"]
    # 有効な盤面内座標であること
    assert 0 <= res["coordinates"]["x"] < 20
    assert 0 <= res["coordinates"]["y"] < 20


def test_negative_click_action_not_available_suppressed():
    """負例 3: ACTION6 が利用可能アクションに含まれていない場合、クリック要求が抑止されること."""
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 5

    # 利用可能アクションに 6 (CLICK) がない
    controller = GameController(available_actions=[1, 2, 3, 4])
    payload = {"action": "ACTION6", "x": 10, "y": 10}

    res = controller.parse_and_validate(payload, grid_shape=(20, 20), grid=grid)
    assert res["success"] is False
    assert "not in available actions" in res["error"]


def test_positive_taboo_click_coordinates_tracked_on_failure():
    """正例: クリックが失敗（ピクセル変化0）した際、last_action_info の coordinates から正確な座標が taboo_click_coords に追加されること."""
    from unittest.mock import MagicMock
    from acr_agi3.agent.adk_game_player import ADKGamePlayer

    player = ADKGamePlayer(model="mock_model")
    # 前回のアクション情報にクリック座標 (34, 34) が記録されていたとする
    player.last_action_info = {
        "action": "ACTION6",
        "action_id": 6,
        "coordinates": {"x": 34, "y": 34},
        "reasoning": "Test click",
        "state_before": "NOT_FINISHED",
        "pixels_changed": 0,
        "is_effective": False,
    }
    player.last_grid = np.zeros((64, 64), dtype=int)
    player.taboo_click_coords = []

    # 次のステップをモック呼び出し（ピクセル変化0でステップが進む）
    grid = np.zeros((64, 64), dtype=int)
    from unittest.mock import AsyncMock
    from acr_agi3.harness.game_action_tools import ActionDecision
    dummy_decision = ActionDecision(
        action_type="CLICK",
        action_name="ACTION6",
        action_id=6,
        coordinates={"x": 38, "y": 38},
        reasoning="Next click",
    )
    player._run_plan_act_workflow = AsyncMock(return_value=dummy_decision)

    decision = player.decide_next_action(grid=grid, available_actions=[6], state_str="NOT_FINISHED")

    # (34, 34) が確実に taboo_click_coords に追加されていること
    assert (34, 34) in player.taboo_click_coords
    assert (0, 0) not in player.taboo_click_coords
    # そして今回の決定の coordinates が last_action_info に記録されていること
    assert player.last_action_info["coordinates"] == {"x": 38, "y": 38}

