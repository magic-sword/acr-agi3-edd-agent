"""反証仮説（Refuted Hypotheses）の黒板保持とステートマシンの単体テスト."""

import json
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveMode, CognitiveState
from acr_agi3.agent.llm.local_vlm import LocalQwenVL


@pytest.fixture
def mock_player():
    """モック VLM を持つ ADKGamePlayer インスタンス."""
    mock_vlm = LocalQwenVL(
        model_name_or_path="mock",
        generate_fn=lambda prompt, images=None: "Test a different observed target after the refuted click.",
    )
    player = ADKGamePlayer(model=mock_vlm, name="test_hypo_player")
    return player


def test_hypothesis_active_and_refuted_archiving(mock_player):
    """行動が失敗（0ピクセル変化）した際、hypothesis.active が hypothesis.refuted.* へ退避保存されること."""
    player = mock_player

    # 1. 仮説を立てる
    player.memory_tools.memory_write(
        section_id="hypothesis.active",
        title="Hypothesis Step 1",
        content="Testing click at coordinate (0, 1) to rotate sprite.",
        summary="Click (0, 1) rotation",
        tags="hypothesis,active,plan",
    )
    assert player.memory_tools.memory_read("hypothesis.active") != ""

    # 2. 直前のアクション情報を設定 (click at (0, 1) failed)
    player.last_action_info = {
        "action": "ACTION6",
        "action_name": "ACTION6",
        "action_id": 6,
        "coordinates": {"x": 0, "y": 1},
        "reasoning": "Click (0, 1) rotation",
        "is_effective": False,
        "pixels_changed": 0,
    }
    player.last_grid = np.zeros((10, 10), dtype=int)

    # 3. 次の手の決定を呼び出す（盤面変化なし: 0ピクセル変化）
    dummy_grid = np.zeros((10, 10), dtype=int)
    player.decide_next_action(dummy_grid, available_actions=[6])

    # 4. 検証: 前回の仮説が hypothesis.refuted.* に退避されていること
    toc = player.memory_tools.memory_toc(as_markdown=True)
    assert "hypothesis.refuted." in toc
    assert "0_1" in toc

    # 退避された反証仮説の内容を確認
    refuted_sections = [
        s for s in player.memory_tools.notebook._sections.keys()
        if s.startswith("hypothesis.refuted.")
    ]
    assert len(refuted_sections) == 1
    refuted_content = player.memory_tools.memory_read(refuted_sections[0])
    assert "Testing click at coordinate (0, 1)" in refuted_content
    assert "0 pixel changes" in refuted_content

    # 現在の hypothesis.active は新しい計画（Step 1）で上書き更新されていること
    active_content = player.memory_tools.memory_read("hypothesis.active")
    assert json.loads(active_content)["status"] == "ok"


def test_refuted_hypotheses_preserved_on_reset(mock_player):
    """リトライ (reset_episode) 時に hypothesis.refuted.* は消去されず永続保持されること."""
    player = mock_player

    # 反証仮説とアクティブ計画を登録
    player.memory_tools.memory_write(
        section_id="hypothesis.refuted.s1_ACTION6_0_1",
        title="Refuted Click (0, 1)",
        content="Click (0, 1) produced 0 changes",
        summary="Refuted (0, 1)",
        tags="hypothesis,refuted,falsified",
    )
    player.memory_tools.memory_write(
        section_id="hypothesis.active",
        title="Active Hypothesis",
        content="Old active hypothesis",
        summary="Old hypothesis",
        tags="hypothesis,active,plan",
    )
    player.memory_tools.memory_write(
        section_id="plan.active",
        title="Active Plan",
        content="Old active plan",
        summary="Old plan",
        tags="plan,active",
    )

    # エピソードリセットを実行
    player.memory_tools.reset_episode()

    # plan.active と hypothesis.active は消去されるが、hypothesis.refuted は保持されること
    read_plan = json.loads(player.memory_tools.memory_read("plan.active"))
    assert read_plan["status"] == "error"
    read_active_hypo = json.loads(player.memory_tools.memory_read("hypothesis.active"))
    assert read_active_hypo["status"] == "error"
    read_refuted = json.loads(player.memory_tools.memory_read("hypothesis.refuted.s1_ACTION6_0_1"))
    assert read_refuted["status"] == "ok"
    assert "Click (0, 1) produced 0 changes" in read_refuted["content"]


def test_click_only_game_blocks_backward_architect(mock_player):
    """利用可能アクションが ACTION6 のみのゲームでは、幾何移動 A* (BACKWARD_ARCHITECT) に遷移しないこと."""
    player = mock_player

    mode = player.determine_cognitive_mode(
        step_index=5,
        stagnation_count=0,
        available_action_ids=[6],
        has_probe_rec=False,
        has_nav_path=True,  # 幾何パスファインダーが誤って True を返した場合の防御
        has_anchors=True,
    )
    # BACKWARD_ARCHITECT ではなく CAUSAL_PROGRAMMER になること
    assert mode != CognitiveMode.BACKWARD_ARCHITECT
    assert mode == CognitiveMode.CAUSAL_PROGRAMMER


def test_hypothesis_engine_redirects_refuted_click(mock_player):
    """反証済みのクリック座標が選択された場合、HypothesisEngine が別のアフォーダンスへ安全にリダイレクトすること."""
    player = mock_player

    # (0, 1) を反証済みとして登録
    player.hypothesis_tools.formulate_hypothesis(
        claim="Click (0, 1)",
        action_name="ACTION6",
        action_id=6,
        coords={"x": 0, "y": 1},
    )
    player.hypothesis_tools.evaluate_hypothesis(pixels_changed=0, is_effective=False)
    assert player.hypothesis_tools.is_action_refuted(6, {"x": 0, "y": 1}) is True

    # 有効なアンカー候補として (10, 10) を設定
    player.spatial_tools.cached_anchors = [{"id": "obj1", "x": 10, "y": 10, "color": 2}]
    player.hypothesis_tools.set_context(step_index=2, available_actions=[6])

    # エージェントが誤って反証済み (0, 1) をクリックしようとした場合の自動回避判定
    cand_anchors = [(a["x"], a["y"]) for a in player.spatial_tools.cached_anchors]
    alt_str = player.hypothesis_tools.propose_alternative_hypothesis(candidate_coords=cand_anchors)
    alt_data = json.loads(alt_str)
    assert alt_data["status"] == "ok"
    assert alt_data["proposal"]["coords"] == {"x": 10, "y": 10}


