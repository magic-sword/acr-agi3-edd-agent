"""Google ADK 2.0 認知グラフワークフロー (CognitiveGameWorkflow) の単体テスト.

人間の5段階認知プロセス（初手視覚点検、能動探索、逆算、禁忌リセット、直接実行）の
ルーティング条件と各ノードの実行を検証する。
"""

import pytest
from unittest.mock import MagicMock
from google.adk import Context

from acr_agi3.agent.cognitive_workflow import (
    CognitiveState,
    assess_cognitive_state,
    build_cognitive_workflow,
    visual_inspection_node,
    epistemic_probe_node,
    backward_plan_node,
    taboo_reset_node,
    direct_plan_node,
    reviewer_audit_node,
    act_finalizer_node,
)
from acr_agi3.agent.workflow_schemas import PlanProposal


def test_assess_initial_step_routes_to_visual_inspection():
    """正例1: 初手 (step=1) は必ず visual_inspection ルートへ遷移する."""
    ctx = MagicMock(spec=Context)
    state = CognitiveState(step=1, stagnation_count=0, working_memory="")
    res = assess_cognitive_state(ctx, state)

    assert ctx.route == "route_visual_inspection"
    assert res.selected_skill == "visual-inspector"
    assert res.cognitive_mode == "visual_inspection"


def test_assess_stagnation_routes_to_taboo_reset():
    """正例2: 停滞 (stagnation >= 2) または壁衝突・トラップは taboo_reset ルートへ遷移する."""
    ctx = MagicMock(spec=Context)
    state = CognitiveState(step=5, stagnation_count=2, working_memory="barrier hit detected")
    res = assess_cognitive_state(ctx, state)

    assert ctx.route == "route_taboo_reset"
    assert res.selected_skill == "taboo-reset-guard"
    assert res.cognitive_mode == "taboo_reset"


def test_assess_unknown_elements_routes_to_epistemic_probe():
    """正例3: 未知要素 (unknown / probe) は epistemic_probe ルートへ遷移する."""
    ctx = MagicMock(spec=Context)
    state = CognitiveState(step=3, stagnation_count=0, working_memory="unknown purple key found")
    res = assess_cognitive_state(ctx, state)

    assert ctx.route == "route_epistemic_probe"
    assert res.selected_skill == "epistemic-prober"
    assert res.cognitive_mode == "epistemic_probe"


def test_assess_distant_target_routes_to_backward_plan():
    """正例4: 遠隔ターゲット・迂回必要時は backward_plan ルートへ遷移する."""
    ctx = MagicMock(spec=Context)
    state = CognitiveState(step=4, stagnation_count=0, working_memory="target far across maze")
    res = assess_cognitive_state(ctx, state)

    assert ctx.route == "route_backward_plan"
    assert res.selected_skill == "backward-planner"
    assert res.cognitive_mode == "backward_plan"


def test_assess_normal_step_routes_to_direct_plan():
    """正例5: 通常の前進フェーズは direct_plan ルートへ遷移する."""
    ctx = MagicMock(spec=Context)
    state = CognitiveState(step=6, stagnation_count=0, working_memory="moving forward safely")
    res = assess_cognitive_state(ctx, state)

    assert ctx.route == "route_direct_plan"
    assert res.selected_skill == "macro-skill-compiler"
    assert res.cognitive_mode == "direct_plan"


def test_workflow_structure_instantiation():
    """ADK 2.0 ワークフローのグラフ構築（エッジ・ノード数）の検証."""
    wf = build_cognitive_workflow("test_cog_wf")
    assert wf.name == "test_cog_wf"
    # エッジ定義が存在することを確認
    assert len(wf.edges) >= 7


def test_act_finalizer_decision():
    """最終アクション確定ノードの動作検証."""
    state = CognitiveState(
        step=2,
        selected_skill="visual-inspector",
        plan_proposal=PlanProposal(
            hypothesis="Test hyp",
            goal="Test goal",
            action="RIGHT",
            reasoning="Testing finalizer",
            load_skill="visual-inspector",
        ),
    )
    res = act_finalizer_node(state)
    assert res.final_decision is not None
    assert res.final_decision.action_name == "RIGHT"
    assert res.final_decision.loaded_skill == "visual-inspector"
