"""Tests for console renderer and gamepad observation harness."""

import io
import numpy as np
import pytest
from PIL import Image

from acr_agi3.dsl.renderer import (
    _normalize_action_id,
    render_console_observation,
    render_gamepad_panel,
    render_grid_to_image,
)
from acr_agi3.harness.vision_observation import VisionObservationHarness


def test_normalize_action_id() -> None:
    """アクション名や数値、オブジェクトが正しく 0〜7 の ID に正規化されるか検証."""
    assert _normalize_action_id(0) == 0
    assert _normalize_action_id(1) == 1
    assert _normalize_action_id("RESET") == 0
    assert _normalize_action_id("UP") == 1
    assert _normalize_action_id("DOWN") == 2
    assert _normalize_action_id("LEFT") == 3
    assert _normalize_action_id("RIGHT") == 4
    assert _normalize_action_id("CLICK") == 6
    assert _normalize_action_id("ACTION6") == 6
    assert _normalize_action_id("action7") == 7
    assert _normalize_action_id("UNKNOWN_INVALID") is None
    assert _normalize_action_id(None) is None

    # Enum ライクなモック
    class DummyAction:
        value = 6

    assert _normalize_action_id(DummyAction()) == 6


def test_render_gamepad_panel_dimensions_and_mode() -> None:
    """ゲームパッドパネルが指定サイズ・RGBモードで描画されるか検証."""
    panel = render_gamepad_panel(
        width=400,
        height=140,
        available_actions=[1, 2, 3, 4, 6],
        last_action=4,
        step_index=5,
        game_state="PLAYING",
    )
    assert isinstance(panel, Image.Image)
    assert panel.mode == "RGB"
    assert panel.size == (400, 140)


def test_render_console_observation_integration() -> None:
    """ゲームグリッドとコントローラーが統合されたコンソール画像が正しく生成されるか検証."""
    # 10x10 のテスト用グリッド
    grid = np.zeros((10, 10), dtype=int)
    grid[2:5, 2:5] = 1  # 青
    grid[5, 5] = 2  # 赤

    console_img = render_console_observation(
        grid=grid,
        available_actions=["UP", "DOWN", "LEFT", "RIGHT", "CLICK"],
        last_action="UP",
        step_index=3,
        game_state="PLAYING",
        cell_size=16,
        min_console_width=380,
    )

    assert isinstance(console_img, Image.Image)
    assert console_img.mode == "RGB"
    assert console_img.width >= 380
    assert console_img.height > 10 * 16  # ゲーム盤面より十分に高い（HUD+パネル込み）

    # PNG としてエンコード可能か検証
    buf = io.BytesIO()
    console_img.save(buf, format="PNG")
    assert len(buf.getvalue()) > 0


def test_vision_observation_harness_console_ui() -> None:
    """VisionObservationHarness が use_console_ui=True で動作した際、正常に Part を生成するか検証."""
    harness = VisionObservationHarness(cell_size=12, use_console_ui=True)
    assert harness.use_console_ui is True

    grid = np.ones((8, 8), dtype=int) * 3  # 緑
    parts = harness.create_observation_parts(
        grid_data=grid,
        step_index=1,
        available_actions=["ACTION1", "ACTION2", "ACTION6"],
        last_action_info={"action": "ACTION1", "action_id": 1, "pixels_changed": 4, "is_effective": True},
    )

    assert len(parts) >= 2
    # 最初の Part は画像 Blob
    img_part = parts[0]
    assert img_part.inline_data is not None
    assert img_part.inline_data.mime_type == "image/png"
    assert len(img_part.inline_data.data) > 0

    # 画像として復元可能か
    loaded_img = Image.open(io.BytesIO(img_part.inline_data.data))
    assert loaded_img.mode == "RGB"
    assert loaded_img.width >= 380  # min_console_width

    # テキスト Part の検証
    text_part = parts[-1]
    assert text_part.text is not None
    assert "Controller Visual HUD" in text_part.text
    assert "D-Pad" in text_part.text


def test_vision_observation_harness_legacy_toggle() -> None:
    """use_console_ui=False の場合、従来の単一グリッド画像が出力されるか検証."""
    harness = VisionObservationHarness(cell_size=10, use_console_ui=False)
    grid = np.zeros((5, 5), dtype=int)
    parts = harness.create_observation_parts(grid_data=grid, step_index=0)

    img_part = parts[0]
    loaded_img = Image.open(io.BytesIO(img_part.inline_data.data))
    # 従来のグリッド画像幅は 5 * 10 = 50px
    assert loaded_img.width == 50
    assert loaded_img.height == 50
