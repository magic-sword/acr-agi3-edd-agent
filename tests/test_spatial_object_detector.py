"""Tests for SpatialGrounder composite object detection and SpatialTools."""

import numpy as np
import pytest

from spatial_grounder import SpatialGrounder
from acr_agi3.tools.spatial_tools import SpatialTools


def test_detect_composite_objects_filters_large_background():
    grounder = SpatialGrounder()
    # 64x64 グリッド
    grid = np.zeros((64, 64), dtype=np.int32)
    
    # 巨大背景エリア: 500ピクセル (25x20 = 500ピクセル, 500 / 4096 ≈ 12.2% > 12%)
    grid[10:35, 10:30] = 4  # 黄色エリア
    
    # 小さなボタン/スプライト: 3x3 = 9ピクセル
    grid[40:43, 40:43] = 2  # 赤色スプライト
    
    # 別の小さなスプライト: 4x4 = 16ピクセル
    grid[50:54, 50:54] = 3  # 緑色スプライト
    
    objects = grounder.detect_composite_objects(grid, max_area_ratio=0.12)
    
    # 巨大エリア (color 4) は除外され、赤と緑のスプライトのみ抽出されること
    assert len(objects) == 2
    colors_detected = [obj["color_ids"] for obj in objects]
    assert [2] in colors_detected
    assert [3] in colors_detected
    assert [4] not in colors_detected


def test_detect_composite_objects_multi_color_merge():
    grounder = SpatialGrounder()
    grid = np.zeros((64, 64), dtype=np.int32)
    
    # 複数色の複合スプライト: 上半分が色2、下半分が色3で隣接している
    grid[20:23, 20:25] = 2
    grid[23:26, 20:25] = 3
    
    objects = grounder.detect_composite_objects(grid)
    
    # 単一の複合オブジェクトとしてまとまり、color_ids に [2, 3] を持つこと
    assert len(objects) == 1
    assert sorted(objects[0]["color_ids"]) == [2, 3]
    assert objects[0]["area"] == 30
    assert objects[0]["bbox"] == [20, 20, 25, 26]  # [bx, by, bx+bw, by+bh]


def test_dynamic_difference_tagging():
    grounder = SpatialGrounder()
    last_grid = np.zeros((64, 64), dtype=np.int32)
    grid = np.zeros((64, 64), dtype=np.int32)
    
    # 静的オブジェクト
    last_grid[10:14, 10:14] = 2
    grid[10:14, 10:14] = 2
    
    # 動的オブジェクト (新規出現または位置変化)
    grid[30:34, 30:34] = 3
    
    objects = grounder.detect_composite_objects(grid, last_grid=last_grid)
    
    assert len(objects) == 2
    obj_static = next(o for o in objects if 2 in o["color_ids"])
    obj_dynamic = next(o for o in objects if 3 in o["color_ids"])
    
    assert obj_static["is_dynamic"] is False
    assert obj_dynamic["is_dynamic"] is True


def test_spatial_tools_inspect_and_get_coords():
    import json
    tools = SpatialTools()
    grid = np.zeros((64, 64), dtype=np.int32)
    grid[15:18, 15:18] = 2  # 3x3
    grid[30:33, 30:33] = 4  # 3x3
    
    tools.set_context(grid, step_index=1)
    
    # inspect_detected_objects の検証
    res_str = tools.inspect_detected_objects(filter_mode="all")
    res = json.loads(res_str)
    assert res["total_objects"] == 2
    
    # get_object_coordinates の検証
    obj_id = res["objects"][0]["id"]
    coords_str = tools.get_object_coordinates(obj_id)
    coords = json.loads(coords_str)
    assert coords["success"] is True
    assert "x" in coords
    assert "y" in coords
    
    # 存在しないID
    bad_res_str = tools.get_object_coordinates(99999)
    bad_res = json.loads(bad_res_str)
    assert bad_res["success"] is False


def test_detect_composite_objects_rescues_buttons_inside_large_container():
    """巨大な台座・サブパネル枠（size > max_size）の内部に配置された子ボタンが正しく救出・検出されること."""
    grounder = SpatialGrounder()
    grid = np.zeros((64, 64), dtype=np.int32)

    # 巨大な操作パネル台座: 30x30 = 900px (max_size = 491px を大幅超過)
    grid[30:60, 30:60] = 4  # 黄色台座

    # 台座の内部に配置された 3x3 のボタン 2 個
    grid[35:38, 35:38] = 9  # ボタン 1 (Maroon)
    grid[45:48, 45:48] = 9  # ボタン 2 (Maroon)

    objects = grounder.detect_composite_objects(grid, max_area_ratio=0.12)

    # 900pxの巨大台座自身は除外され、内部の2つのボタンが正しく検出されること
    assert len(objects) >= 2
    button_centers = [(o["center"]["x"], o["center"]["y"]) for o in objects if 9 in o["color_ids"]]
    assert len(button_centers) == 2
    assert (36, 36) in button_centers
    assert (46, 46) in button_centers
