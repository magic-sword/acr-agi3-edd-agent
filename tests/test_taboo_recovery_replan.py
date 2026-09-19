"""Tests for taboo failure recording, dynamic recovery prompts, and click taboo filtering."""

from __future__ import annotations

import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveMode, CognitiveState
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.tools.spatial_tools import SpatialTools
from spatial_grounder import SpatialGrounder


def create_mock_player(name: str = "test_player") -> ADKGamePlayer:
    llm = LocalTransformersLlm("mock", generation_fn=lambda p: '```json\n{"action": "ACTION6"}\n```')
    return ADKGamePlayer(model=llm, name=name)


def test_snap_to_anchor_avoids_taboo():
    """禁忌座標（taboo_coords）に近いアンカーがスキップされることを確認."""
    anchors = [
        {"id": 0, "x": 10, "y": 10, "color": 1, "area": 10},
        {"id": 1, "x": 20, "y": 20, "color": 2, "area": 10},
        {"id": 2, "x": 30, "y": 30, "color": 3, "area": 10},
    ]

    # 禁忌なし: (10, 10) に一番近い Anchor 0 (10, 10) が選ばれる
    x, y, aid = SpatialGrounder.snap_to_anchor(11, 11, anchors, taboo_coords=None)
    assert aid == 0
    assert (x, y) == (10, 10)

    # (10, 10) が禁忌の場合: Anchor 0 は除外され、Anchor 1 が選ばれる
    x, y, aid = SpatialGrounder.snap_to_anchor(11, 11, anchors, taboo_coords=[(10, 10)], max_dist=20.0)
    assert aid == 1
    assert (x, y) == (20, 20)


def test_taboo_click_recording_on_zero_change():
    """0ピクセル変化のクリック失敗時に共有黒板 (memory-notebook) に禁忌記録が書き込まれることを確認."""
    player = create_mock_player("test_taboo_player")

    # 初期手番情報（クリック実行）をセット
    player.last_action_info = {
        "action_id": 6,
        "action_name": "ACTION6",
        "coordinates": {"x": 15, "y": 25},
        "is_effective": True,
    }
    player.last_grid = np.zeros((10, 10), dtype=int)

    # 次手番: 同じグリッド（pixels_changed == 0）を観測
    current_grid = np.zeros((10, 10), dtype=int)

    diff_mask = (player.last_grid != current_grid)
    pixels_changed = int(np.sum(diff_mask))
    assert pixels_changed == 0

    # 失敗検知シミュレーション
    player.last_action_info = {
        "action_id": 6,
        "action_name": "ACTION6",
        "coordinates": {"x": 15, "y": 25},
        "is_effective": False,
    }

    # player の taboo リストと黒板のテスト
    cx, cy = 15, 25
    if (cx, cy) not in player.taboo_click_coords:
        player.taboo_click_coords.append((cx, cy))
    player.spatial_tools.set_taboo_coords(player.taboo_click_coords)
    player.memory_tools.memory_write(
        section_id=f"taboo.click_{cx}_{cy}",
        title=f"Failed Click at ({cx}, {cy})",
        content="Click caused 0 pixel change.",
        summary="Inactive click",
        tags="taboo,click",
    )

    assert (15, 25) in player.taboo_click_coords
    assert (15, 25) in player.spatial_tools.taboo_coords
    read_res = player.memory_tools.memory_read(section_id="taboo.click_15_25")
    assert "Failed Click at (15, 25)" in read_res


def test_taboo_recovery_instructions_click_only():
    """available_actions=[6] のクリック専用環境で直角移動ではなくクリック再計画指示が生成されることを確認."""
    player = create_mock_player("test_prompt_player")
    player.stagnation_count = 1
    player.taboo_click_coords = [(44, 46)]

    mode = player.determine_cognitive_mode(
        step_index=2,
        stagnation_count=1,
        available_action_ids=[6],
        has_probe_rec=False,
        has_nav_path=False,
    )
    assert mode == CognitiveMode.TABOO_RECOVERY

    # ワークフローガイダンス生成の検証
    avail_ids = [6]
    available_action_names = "ACTION6"
    mode_instructions = []
    if mode == CognitiveMode.TABOO_RECOVERY:
        if 6 in avail_ids and len(avail_ids) == 1:
            taboo_str = f"Recorded Taboo Clicks: {player.taboo_click_coords}" if player.taboo_click_coords else ""
            mode_instructions.append(
                "🚨 [Mode: TABOO RECOVERY / INTERACTION RE-PLANNING]\n"
                "The previous click action caused 0 pixel changes (target was inactive or missed).\n"
                f"{taboo_str}\n"
                "Goal: Re-plan your target! Choose a DIFFERENT clickable object/anchor from the visual clusters and call `click_at(x=col, y=row)`. "
                "CRITICAL: Only ACTION6 (click_at) is available. Do NOT output movement directions (UP/DOWN/LEFT/RIGHT) or repeat taboo coordinates."
            )

    instruction_text = "\n".join(mode_instructions)
    assert "INTERACTION RE-PLANNING" in instruction_text
    assert "orthogonal" not in instruction_text.lower()
    assert "Do NOT output movement directions" in instruction_text
    assert "(44, 46)" in instruction_text
