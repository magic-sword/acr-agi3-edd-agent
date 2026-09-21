"""Tests for Game-Scoped Memory Notebook Persistence & Selective Episode Reset.

Verifies:
1. Same-game retry (reset()) retains invariant knowledge (causality.*, rules.*, taboo.*, dynamics_map)
   while safely erasing transient plans (plan.active).
2. Environment switching (game_A -> game_B) provides completely isolated notebook spaces.
3. Instant Recall: Returning to a previously played game (game_B -> game_A) completely restores
   the prior game's learned dynamics and notebook entries without contamination.
4. Full wipe (reset(full_wipe=True)) completely cleans all notebooks and mappings.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveState
from acr_agi3.agent.llm.local_model import LocalTransformersLlm


def create_test_player() -> ADKGamePlayer:
    llm = LocalTransformersLlm("mock", generation_fn=lambda p: '```json\n{"action": "ACTION1"}\n```')
    return ADKGamePlayer(model=llm)


def test_same_game_reset_preserves_dynamics_and_rules():
    """同一ゲーム内でのリトライ (reset()) では永続知識 (causality, rules, taboo) が保持され、plan.active のみ消去される."""
    player = create_test_player()
    grid = np.zeros((5, 5), dtype=np.uint8)

    # game_1 でアクション実行
    player.decide_next_action(grid, available_actions=[1, 2, 3, 4], game_id="game_1")

    # 永続知識と一時的計画を書き込み
    player.memory_tools.memory_write("causality.dynamics", '{"RIGHT": 3}', title="Controls", tags="causality")
    player.memory_tools.memory_write("rules.hazards", "Red tiles are fatal", title="Hazards", tags="rule")
    player.memory_tools.memory_write("taboo.step_2", "Bumped into wall at (1, 2)", title="Taboo", tags="taboo")
    player.memory_tools.memory_write("plan.active", '{"steps": ["RIGHT", "UP"]}', title="Active Plan", tags="plan")
    player.dynamics_map["RIGHT"] = 3

    # 同一ゲームのリセット (リトライ)
    player.reset(full_wipe=False)

    # 一時的計画 plan.active は消しゴムで消えている
    plan_res = json.loads(player.memory_tools.memory_read("plan.active"))
    assert plan_res["status"] == "error"

    # 永続知識はすべて保持されている
    causality_res = json.loads(player.memory_tools.memory_read("causality.dynamics"))
    assert causality_res["status"] == "ok"
    assert "RIGHT" in causality_res["content"]

    rule_res = json.loads(player.memory_tools.memory_read("rules.hazards"))
    assert rule_res["status"] == "ok"
    assert "Red tiles" in rule_res["content"]

    taboo_res = json.loads(player.memory_tools.memory_read("taboo.step_2"))
    assert taboo_res["status"] == "ok"

    # 操作力学マップも保持されている
    assert player.dynamics_map.get("RIGHT") == 3


def test_game_switching_and_instant_recall():
    """異なるゲーム環境間でノートブックが完全分離され、過去の環境に再訪した際に即座に知識が復元される."""
    player = create_test_player()
    grid = np.zeros((5, 5), dtype=np.uint8)

    # 1. game_Alpha で学習
    player.decide_next_action(grid, available_actions=[1, 2, 3, 4], game_id="game_Alpha")
    player.memory_tools.memory_write("rules.switches", "Yellow button opens gate", title="Switch Rule", tags="rule")
    player.dynamics_map["UP"] = 1
    from acr_agi3.agent.execution_evidence import Motion
    player.execution_evidence.samples[1] = (Motion(2, -1, 0), 2)

    # 2. 未知の環境 game_Beta に切り替え
    player.decide_next_action(grid, available_actions=[1, 2, 3, 4], game_id="game_Beta")
    assert player.current_game_id == "game_Beta"

    # game_Beta では game_Alpha の知識は読めない (白紙)
    res_in_beta = json.loads(player.memory_tools.memory_read("rules.switches"))
    assert res_in_beta["status"] == "error"
    assert "UP" not in player.dynamics_map

    # game_Beta で独自の知識を蓄積
    player.memory_tools.memory_write("rules.teleport", "Blue circle is a teleporter", title="Teleport Rule", tags="rule")
    player.dynamics_map["UP"] = 4  # Beta では UP が 4

    # 3. 再び game_Alpha に再訪 (Instant Recall)
    player.decide_next_action(grid, available_actions=[1, 2, 3, 4], game_id="game_Alpha")
    assert player.current_game_id == "game_Alpha"

    # game_Alpha の知識が完全に復元されている
    res_recalled = json.loads(player.memory_tools.memory_read("rules.switches"))
    assert res_recalled["status"] == "ok"
    assert "Yellow button" in res_recalled["content"]
    assert player.dynamics_map.get("UP") == 1

    # game_Beta の知識は混ざっていない
    res_beta_in_alpha = json.loads(player.memory_tools.memory_read("rules.teleport"))
    assert res_beta_in_alpha["status"] == "error"


def test_full_wipe_reset():
    """full_wipe=True の場合、すべての記憶と力学マップが完全に初期化される."""
    player = create_test_player()
    grid = np.zeros((5, 5), dtype=np.uint8)

    player.decide_next_action(grid, available_actions=[1, 2, 3, 4], game_id="game_test")
    player.memory_tools.memory_write("rules.gravity", "Downwards gravity active", title="Gravity", tags="rule")
    player.dynamics_map["DOWN"] = 2

    # 完全初期化
    player.reset(full_wipe=True)

    rule_res = json.loads(player.memory_tools.memory_read("rules.gravity"))
    assert rule_res["status"] == "error"
    assert len(player.dynamics_map) == 0
