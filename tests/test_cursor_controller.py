"""契約テスト: マウスカーソル（Mouse Cursor）移動・照準・クリック分離エンジンの検証.

EDD防壁ゲート規約（正例3件＋負例3件）に準拠し、
カーソル照準、二段階クリック、画像レティクル描画、境界値クランプ、アクション可用性制御を検証。
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image

GAME_CONTROLLER_DIR = Path(__file__).resolve().parent.parent / "meta_skills" / "game-controller" / "scripts"
if str(GAME_CONTROLLER_DIR) not in sys.path and GAME_CONTROLLER_DIR.exists():
    sys.path.insert(0, str(GAME_CONTROLLER_DIR))

from game_controller import GameController
from acr_agi3.dsl.renderer import render_grid_to_image, render_console_observation
from acr_agi3.harness.game_action_tools import GameActionTools


# =============================================================================
# 正例テスト (Positive Tests: 3件)
# =============================================================================

def test_positive_move_cursor_updates_cursor_position():
    """正例 1: move_cursor を呼ぶとカーソル位置のみが更新され、環境アクションは発行されないこと."""
    controller = GameController(available_actions=[6], cursor=(10, 10))
    tools = GameActionTools(controller=controller, available_actions=[6])

    # 初期位置の確認
    assert controller.cursor == (10, 10)
    assert tools.pending_decision is None

    # カーソルを (38, 38) へ移動（照準・Aim）
    res_str = tools.move_cursor(x=38, y=38, reasoning="Aiming at bottom-right button")

    # カーソル位置が (38, 38) に更新されていること
    assert controller.cursor == (38, 38)
    assert "Mouse cursor moved to (col=38, row=38)" in res_str
    # 環境アクションは発行されておらず None のままであること（誤クリック防止）
    assert tools.pending_decision is None


def test_positive_click_at_cursor_fires_action6():
    """正例 2: click_at_cursor を呼ぶと、現在のカーソル座標に対して正確に ACTION6 が発火すること."""
    controller = GameController(available_actions=[6], cursor=(46, 38))
    tools = GameActionTools(controller=controller, available_actions=[6])

    # 現在のカーソル位置 (46, 38) でクリック発火
    res_str = tools.click_at_cursor(reasoning="Pressing aimed button")

    # ACTION6 が座標 (46, 38) で正しくスケジュールされていること
    assert tools.pending_decision is not None
    assert tools.pending_decision.action_id == 6
    assert tools.pending_decision.action_name == "ACTION6"
    assert tools.pending_decision.coordinates == {"x": 46, "y": 38}
    assert "Click fired at mouse cursor coordinate {'x': 46, 'y': 38}" in res_str


def test_positive_renderer_draws_cursor_reticle():
    """正例 3: renderer.py で cursor_pos を指定した際、画像上にレティクルが正しく重畳描画されること."""
    grid = np.zeros((20, 20), dtype=int)
    # カーソルなし画像
    img_no_cursor = render_grid_to_image(grid, cell_size=16, cursor_pos=None)
    # カーソルあり画像 (セル (10, 10))
    img_with_cursor = render_grid_to_image(grid, cell_size=16, cursor_pos=(10, 10))

    # 画像サイズが等しいこと
    assert img_no_cursor.size == img_with_cursor.size

    # カーソル位置のセル周辺ピクセルが変化していること（白・黄色のレティクルが描画されたこと）
    arr_no = np.array(img_no_cursor)
    arr_with = np.array(img_with_cursor)
    diff = np.abs(arr_with.astype(int) - arr_no.astype(int))
    assert np.sum(diff) > 0, "Cursor reticle must draw visible pixels on the canvas"

    # コンソール画像でも CURSOR テキストとレティクルが描画されること
    console_img = render_console_observation(grid, available_actions=[6], cursor_pos=(10, 10))
    assert isinstance(console_img, Image.Image)


# =============================================================================
# 負例テスト (Negative Tests: 3件)
# =============================================================================

def test_negative_move_cursor_out_of_bounds_clamped():
    """負例 1: 画面外 (-10, 999) へのカーソル移動がクラッシュせず、安全に盤面内にクランプされること."""
    controller = GameController(available_actions=[6], cursor=(10, 10))

    # 負の座標への移動
    res_neg = controller.move_cursor(x=-10, y=-5, grid_shape=(64, 64))
    assert res_neg["success"] is True
    assert controller.cursor == (0, 0)
    assert res_neg["cursor"] == {"x": 0, "y": 0}

    # 盤面サイズ超過への移動
    res_over = controller.move_cursor(x=100, y=80, grid_shape=(64, 64))
    assert res_over["success"] is True
    assert controller.cursor == (63, 63)
    assert res_over["cursor"] == {"x": 63, "y": 63}


def test_negative_click_at_cursor_when_click_not_available():
    """負例 2: ACTION6 が利用可能アクションに含まれていない場合、click_at_cursor が安全に抑止されること."""
    # 利用可能アクションが十字キー [1, 2, 3, 4] のみ
    controller = GameController(available_actions=[1, 2, 3, 4], cursor=(15, 15))
    tools = GameActionTools(controller=controller, available_actions=[1, 2, 3, 4])

    res_str = tools.click_at_cursor(reasoning="Attempting click when click is disabled")

    # エラーメッセージが返り、pending_decision は None のままであること
    assert "Error from game-controller" in res_str
    assert "is disabled" in res_str
    assert tools.pending_decision is None


def test_negative_direct_click_at_out_of_bounds_rejection():
    """負例 3: click_at で盤面外座標 (999, 999) を指定した場合、境界外エラーで拒絶されること."""
    controller = GameController(available_actions=[6], cursor=(10, 10))
    res = controller.click_at(x=999, y=999, grid_shape=(30, 30))

    assert res["success"] is False
    assert "out of grid bounds" in res["error"]
    # 失敗したためカーソル位置は直前のまま保たれること
    assert controller.cursor == (10, 10)
