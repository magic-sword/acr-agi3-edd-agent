"""契約テスト: マウスカーソル照準検証 (inspect_cursor_target) の検証.

EDD防壁ゲート規約（正例3件＋負例3件）に準拠し、
中央判定 (CENTERED)、端判定 (EDGE)、背景判定 (OFF_TARGET)、
7x7局所ゲシュタルトマップ生成、境界外クランプ、およびグリッド未設定エラーハンドリングを検証。
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# visual-inspector を sys.path に追加
VI_DIR = Path(__file__).resolve().parent.parent / "meta_skills" / "visual-inspector" / "scripts"
if str(VI_DIR) not in sys.path and VI_DIR.exists():
    sys.path.insert(0, str(VI_DIR))

from visual_inspector import VisualInspector
from acr_agi3.tools.vision_tools import VisionTools
from acr_agi3.harness.game_action_tools import GameActionTools
from game_controller import GameController


# =============================================================================
# 正例テスト (Positive Tests: 3件)
# =============================================================================

def test_positive_cursor_centered_on_button():
    """正例 1: オブジェクト中心にカーソルがある場合、status='CENTERED' となること."""
    grid = np.zeros((30, 30), dtype=int)
    # (15..17, 10..12) に 3x3 の Green (3) ボタンを配置 -> center=(11, 16)
    grid[15:18, 10:13] = 3

    inspector = VisualInspector()
    res = inspector.inspect_cursor_target(grid, cursor_pos=(11, 16))

    assert res["success"] is True
    assert res["status"] == "CENTERED"
    assert res["cursor"] == {"col": 11, "row": 16}
    assert res["target_pixel"]["color"] == 3
    assert res["target_pixel"]["color_name"] == "Green"
    assert res["object_info"] is not None
    assert res["object_info"]["distance_to_center"] == 0.0
    assert "well-centered" in res["guidance"]


def test_positive_cursor_on_edge_recommends_center():
    """正例 2: オブジェクトの端にカーソルがある場合、status='EDGE' となり中心が案内されること."""
    grid = np.zeros((30, 30), dtype=int)
    # (15..19, 10..14) に 5x5 の Teal (8) ボタンを配置 -> center=(12, 17)
    grid[15:20, 10:15] = 8

    inspector = VisualInspector()
    # ボタンの左上隅 (10, 15) にカーソルを置く
    res = inspector.inspect_cursor_target(grid, cursor_pos=(10, 15))

    assert res["success"] is True
    assert res["status"] == "EDGE"
    assert res["cursor"] == {"col": 10, "row": 15}
    assert res["object_info"] is not None
    assert res["object_info"]["center"] == {"x": 12, "y": 17}
    assert "Recommended center: (col=12, row=17)" in res["guidance"]


def test_positive_local_ascii_gestalt_map_with_reticle():
    """正例 3: 7x7 局所ゲシュタルトマップにカーソル位置 [val] が正しく重畳されること."""
    grid = np.zeros((30, 30), dtype=int)
    grid[15:18, 10:13] = 4  # Yellow button

    inspector = VisualInspector()
    res = inspector.inspect_cursor_target(grid, cursor_pos=(11, 16), radius=3)

    ascii_map = res["local_ascii_map"]
    # カーソル位置にレティクル角カッコ [4] が描画されていること
    assert "[4]" in ascii_map
    # 行インジケータ <-- が含まれること
    assert "<--" in ascii_map


# =============================================================================
# 負例テスト (Negative Tests: 3件)
# =============================================================================

def test_negative_cursor_out_of_bounds_clamped():
    """負例 1: 境界外座標 (-10, 999) を指定した場合、安全に盤面内にクランプされること."""
    grid = np.zeros((20, 20), dtype=int)
    inspector = VisualInspector()

    # 負の座標
    res_neg = inspector.inspect_cursor_target(grid, cursor_pos=(-5, -10))
    assert res_neg["success"] is True
    assert res_neg["cursor"] == {"col": 0, "row": 0}

    # 盤面サイズ超過
    res_over = inspector.inspect_cursor_target(grid, cursor_pos=(100, 200))
    assert res_over["success"] is True
    assert res_over["cursor"] == {"col": 19, "row": 19}


def test_negative_cursor_on_background_is_off_target():
    """負例 2: 背景色（地面）にカーソルがある場合、status='OFF_TARGET' となること."""
    grid = np.zeros((30, 30), dtype=int)
    grid[15:18, 10:13] = 2  # Red button far away

    inspector = VisualInspector()
    # ボタンから離れた背景 (5, 5) にカーソルを置く
    res = inspector.inspect_cursor_target(grid, cursor_pos=(5, 5))

    assert res["success"] is True
    assert res["status"] == "OFF_TARGET"
    assert res["object_info"] is None
    assert "Reticle is on background/open space" in res["guidance"]


def test_negative_empty_or_none_grid_handled():
    """負例 3: グリッドが None または空配列の場合、クラッシュせず安全にエラーを返すこと."""
    inspector = VisualInspector()

    res_none = inspector.inspect_cursor_target(None, cursor_pos=(10, 10))
    assert res_none["success"] is False
    assert "error" in res_none

    res_empty = inspector.inspect_cursor_target(np.zeros((0, 0)), cursor_pos=(10, 10))
    assert res_empty["success"] is False
    assert "error" in res_empty


# =============================================================================
# 統合テスト: VisionTools と GameController の動的連携
# =============================================================================

def test_integration_vision_tools_dynamic_sync():
    """統合: move_cursor でコントローラーのカーソルが移動した際、VisionTools が引数なしでそれを自動反映すること."""
    grid = np.zeros((30, 30), dtype=int)
    grid[10:13, 10:13] = 6  # Magenta button at center=(11, 11)

    controller = GameController(available_actions=[6], cursor=(0, 0))
    action_tools = GameActionTools(controller=controller, available_actions=[6])
    vision_tools = VisionTools(controller=controller)
    vision_tools.set_context(grid=grid, step_index=1)

    # 1. 最初は (0, 0)
    report_init = vision_tools.inspect_cursor_target()
    assert "Cursor Position: (col=0, row=0)" in report_init
    assert "Aim Status: [OFF_TARGET]" in report_init

    # 2. カーソルをボタン中心 (11, 11) へ移動
    action_tools.move_cursor(x=11, y=11)

    # 3. VisionTools は引数なしで自動的に (11, 11) を検証
    report_aimed = vision_tools.inspect_cursor_target()
    assert "Cursor Position: (col=11, row=11)" in report_aimed
    assert "Aim Status: [CENTERED]" in report_aimed
    assert "Aimed Pixel Color: 6 (Magenta)" in report_aimed
