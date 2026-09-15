"""ARC-AGI-3 アフォーダンス誘導型クリック精度向上 (Click Precision) の契約テスト.

EDD (Evaluation-Driven Development) 防壁ゲート:
- 正例 3 件: 重心完全一致、微小ズレの自動スナップ吸着、row/col 反転自動補正
- 負例 3 件: 盤面外座標、前景皆無時のフォールバック、ACTION6 不許可時の抑止
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
# 正例 3 件: 正常系アフォーダンス検出・吸着・反転補正
# ==============================================================================

def test_positive_detect_interactable_objects():
    """正例 1: 盤面内の有色オブジェクト（連結成分）の重心と BBox が正確に検出されること."""
    grid = np.zeros((30, 30), dtype=int)
    # Magenta (6) の 3x3 タイルを (x=10..12, y=15..17) に配置
    grid[15:18, 10:13] = 6
    # Blue (1) の 2x2 タイルを (x=20..21, y=5..6) に配置
    grid[5:7, 20:22] = 1

    objects = detect_interactable_objects(grid)
    assert len(objects) >= 2

    # Magenta オブジェクトの検証
    magenta_obj = next((o for o in objects if o["color"] == 6), None)
    assert magenta_obj is not None
    assert magenta_obj["size"] == 9
    assert magenta_obj["center"]["x"] == 11
    assert magenta_obj["center"]["y"] == 16
    assert magenta_obj["bbox"]["min_x"] == 10
    assert magenta_obj["bbox"]["max_x"] == 12
    assert magenta_obj["bbox"]["min_y"] == 15
    assert magenta_obj["bbox"]["max_y"] == 17


def test_positive_click_snapping_to_nearest_object():
    """正例 2: わずかにずれた座標 (空セル) を指定した場合、最近傍の有色オブジェクト重心へ自動吸着すること."""
    grid = np.zeros((30, 30), dtype=int)
    # Target: Green (3) at (x=15, y=15)
    grid[15, 15] = 3

    controller = GameController(available_actions=[1, 2, 3, 4, 6])
    # LLM が 2 ピクセルずれた (x=17, y=16) を指定
    payload = {"action": "ACTION6", "x": 17, "y": 16, "reasoning": "Click green object"}

    res = controller.parse_and_validate(payload, grid_shape=(30, 30), grid=grid)
    assert res["success"] is True
    assert res["action_name"] == "ACTION6"
    assert res["coordinates"]["x"] == 15
    assert res["coordinates"]["y"] == 15
    assert "snapped click" in res["reasoning"]


def test_positive_click_transposed_coordinates_correction():
    """正例 3: row と col を取り違えて (y, x) と指定した場合、自動的に反転補正されること."""
    grid = np.zeros((40, 20), dtype=int)  # H=40, W=20
    # Red (2) オブジェクトを col=5, row=25 に配置
    grid[25, 5] = 2

    controller = GameController(available_actions=[1, 2, 6])
    # LLM が row=25, col=5 を取り違えて x=25, y=5 と指定 (x=25 は W=20 の範囲外または空セル)
    payload = {"action": "ACTION6", "x": 25, "y": 5}

    res = controller.parse_and_validate(payload, grid_shape=(40, 20), grid=grid)
    assert res["success"] is True
    assert res["coordinates"]["x"] == 5
    assert res["coordinates"]["y"] == 25
    assert "auto-transposed" in res["reasoning"]


# ==============================================================================
# 負例 3 件: 異常系・制約違反の適切なハンドリング
# ==============================================================================

def test_negative_click_out_of_bounds_rejection():
    """負例 1: 盤面境界外かつスナップ範囲外の無謀な座標は明確にエラー却下されること."""
    grid = np.zeros((20, 20), dtype=int)
    grid[5, 5] = 4

    controller = GameController(available_actions=[1, 2, 6])
    payload = {"action": "ACTION6", "x": 999, "y": 999}

    res = controller.parse_and_validate(payload, grid_shape=(20, 20), grid=grid)
    assert res["success"] is False
    assert "out of grid bounds" in res["error"]


def test_negative_click_all_background_grid_fallback():
    """負例 2: 盤面に前景オブジェクトが一切存在しない場合、スナップせず指定座標の境界検証を行うこと."""
    grid = np.zeros((20, 20), dtype=int)  # All background

    controller = GameController(available_actions=[1, 2, 6])
    payload = {"action": "ACTION6", "x": 10, "y": 10}

    res = controller.parse_and_validate(payload, grid_shape=(20, 20), grid=grid)
    assert res["success"] is True
    # 前景なしのためスナップされず指定値のまま
    assert res["coordinates"] == {"x": 10, "y": 10}
    assert "snapped" not in res["reasoning"]


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
