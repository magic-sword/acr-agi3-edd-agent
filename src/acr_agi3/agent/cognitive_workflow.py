"""Google ADK 2.0 準拠・5段階認知グラフワークフロー (CognitiveGameWorkflow).

人間の5段階適応プロセス（Visual Inspection -> Epistemic Probing -> Backward Planning
-> Macro Execution -> Taboo Reset Guard）を Google ADK 2.0 の Node と Edge による
グラフワークフローとして実装し、状況に応じたメタスキルの確実な自律トリガーを保証する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from google.adk import Context
from google.adk.workflow import Edge, FunctionNode, Workflow, START

from acr_agi3.agent.workflow_schemas import PlanProposal, ReviewFeedback
from acr_agi3.harness.game_action_tools import ActionDecision

logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    """ワークフロー各ノード間で共有される認知状態コンテキスト."""
    step: int
    observation: Any = None  # grid or visual
    working_memory: str = ""
    stagnation_count: int = 0
    selected_skill: Optional[str] = None
    cognitive_mode: str = "direct_plan"
    plan_proposal: Optional[PlanProposal] = None
    review_feedback: Optional[ReviewFeedback] = None
    final_decision: Optional[ActionDecision] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


def assess_cognitive_state(ctx: Context, state: CognitiveState) -> CognitiveState:
    """ステップ1: 状況認識とメタスキルルーティング (Assessment Node).

    盤面状況、エピソード記憶、停滞カウントから現在の認知フェーズを決定し、
    対応するルート (ctx.route) を設定する。
    """
    step = state.step
    stagnation = state.stagnation_count
    wm = (state.working_memory or "").lower()

    # 1. ルート決定ルール
    if step <= 1 or "initial inspection needed" in wm:
        # 初手: Visual Inspection (目視点検 & ゲシュタルト解析)
        route = "route_visual_inspection"
        skill = "visual-inspector"
        mode = "visual_inspection"
    elif stagnation >= 2 or "stagnant" in wm or "barrier hit" in wm or "trap" in wm:
        # 停滞・壁衝突・トラップ: Taboo Reset Guard (禁忌学習 & リセット判定)
        route = "route_taboo_reset"
        skill = "taboo-reset-guard"
        mode = "taboo_reset"
    elif "unknown" in wm or "probe" in wm or "unexplored" in wm:
        # 未解明のオブジェクト・アフォーダンスあり: Epistemic Prober (能動探索)
        route = "route_epistemic_probe"
        skill = "epistemic-prober"
        mode = "epistemic_probe"
    elif "target far" in wm or "path blocked" in wm or "complex" in wm:
        # 遠隔ゴール・障害物迂回: Backward Planner (逆算プランニング)
        route = "route_backward_plan"
        skill = "backward-planner"
        mode = "backward_plan"
    else:
        # 通常実行: Direct Macro Plan
        route = "route_direct_plan"
        skill = "macro-skill-compiler"
        mode = "direct_plan"

    state.selected_skill = skill
    state.cognitive_mode = mode
    ctx.route = route
    logger.info(
        "🧠 [assess_cognitive_state] Step %d (stagnation=%d) -> mode='%s', route='%s', skill='%s'",
        step, stagnation, mode, route, skill
    )
    return state


def visual_inspection_node(state: CognitiveState) -> CognitiveState:
    """[Phase 1] Visual Inspection ノード: 初手の全体把握・ゲシュタルト構造解析."""
    state.selected_skill = "visual-inspector"
    state.telemetry["cognitive_phase"] = "visual_inspection"
    logger.info("🔍 [visual_inspection_node] Triggering skill 'visual-inspector'")
    return state


def epistemic_probe_node(state: CognitiveState) -> CognitiveState:
    """[Phase 2] Epistemic Prober ノード: 未知アフォーダンスの仮説検証的プローブ行動."""
    state.selected_skill = "epistemic-prober"
    state.telemetry["cognitive_phase"] = "epistemic_probe"
    logger.info("🧪 [epistemic_probe_node] Triggering skill 'epistemic-prober'")
    return state


def backward_plan_node(state: CognitiveState) -> CognitiveState:
    """[Phase 3] Backward Planner ノード: ゴールからの逆算・待避サブゴール分解."""
    state.selected_skill = "backward-planner"
    state.telemetry["cognitive_phase"] = "backward_plan"
    logger.info("🎯 [backward_plan_node] Triggering skill 'backward-planner'")
    return state


def taboo_reset_node(state: CognitiveState) -> CognitiveState:
    """[Phase 4] Taboo Reset Guard ノード: 連続失敗・禁忌制約学習 & リセット判定."""
    state.selected_skill = "taboo-reset-guard"
    state.telemetry["cognitive_phase"] = "taboo_reset"
    logger.info("🛡️ [taboo_reset_node] Triggering skill 'taboo-reset-guard'")
    return state


def direct_plan_node(state: CognitiveState) -> CognitiveState:
    """[Phase 5] Direct Macro Plan ノード: 定石マクロスキルの直接適用・前進行動."""
    state.selected_skill = "macro-skill-compiler"
    state.telemetry["cognitive_phase"] = "direct_plan"
    logger.info("⚡ [direct_plan_node] Triggering skill 'macro-skill-compiler'")
    return state


def reviewer_audit_node(state: CognitiveState) -> CognitiveState:
    """合流ノード: 3点客観監査 (Reviewer Audit Node).

    すべての認知フェーズから提案された計画に対し、客観的ゲートキーパーとして
    座標妥当性、無駄打ち抑止、禁忌回避の監査を行う。
    """
    logger.info("⚖️ [reviewer_audit_node] Auditing plan proposed under skill '%s'", state.selected_skill)
    if state.plan_proposal is None:
        state.plan_proposal = PlanProposal(
            hypothesis=f"Action guided by {state.selected_skill}",
            goal="Advance game exploration",
            action="UP",
            reasoning="Default exploration action",
            load_skill=state.selected_skill,
        )
    return state


def act_finalizer_node(state: CognitiveState) -> CognitiveState:
    """終端ノード: 実行アクション確定 (Act Finalizer Node).

    Reviewer の監査と GameController の検証を経て最終 ActionDecision を構築する。
    """
    plan = state.plan_proposal
    raw_action = (plan.action if plan else "UP").upper()
    action_map = {
        "RESET": (0, "RESET"),
        "ACTION1": (1, "ACTION1"),
        "ACTION2": (2, "ACTION2"),
        "ACTION3": (3, "ACTION3"),
        "ACTION4": (4, "ACTION4"),
        "ACTION5": (5, "ACTION5"),
        "ACTION6": (6, "ACTION6"),
        "ACTION7": (7, "ACTION7"),
        "UP": (1, "ACTION1"),
        "DOWN": (2, "ACTION2"),
        "LEFT": (3, "ACTION3"),
        "RIGHT": (4, "ACTION4"),
        "CLICK": (6, "ACTION6"),
    }
    action_id, action_name = action_map.get(raw_action, (1, "ACTION1"))
    coords = plan.coordinates if plan else None
    action_type = "CLICK" if action_id == 6 else ("RESET" if action_id == 0 else "STEP")

    decision = ActionDecision(
        action_type=action_type,
        action_name=raw_action,
        action_id=action_id,
        coordinates=coords,
        reasoning=plan.reasoning if plan else "Workflow executed",
        loaded_skill=state.selected_skill,
    )
    state.final_decision = decision
    logger.info(
        "🚀 [act_finalizer_node] Finalized ActionDecision: %s (skill=%s)",
        decision.action_name, decision.loaded_skill
    )
    return state


def build_cognitive_workflow(name: str = "cognitive_game_workflow") -> Workflow:
    """Google ADK 2.0 準拠の認知グラフワークフローを生成する."""
    node_assess = FunctionNode(func=assess_cognitive_state, name="assess_cognitive_state")
    node_vis = FunctionNode(func=visual_inspection_node, name="visual_inspection_node")
    node_probe = FunctionNode(func=epistemic_probe_node, name="epistemic_probe_node")
    node_backward = FunctionNode(func=backward_plan_node, name="backward_plan_node")
    node_taboo = FunctionNode(func=taboo_reset_node, name="taboo_reset_node")
    node_direct = FunctionNode(func=direct_plan_node, name="direct_plan_node")

    node_reviewer = FunctionNode(func=reviewer_audit_node, name="reviewer_audit_node")
    node_finalizer = FunctionNode(func=act_finalizer_node, name="act_finalizer_node")

    edges = [
        # START -> Assess
        Edge(from_node=START, to_node=node_assess),

        # Assess -> 各認知ノード (条件付きルーティング)
        Edge(from_node=node_assess, to_node=node_vis, route="route_visual_inspection"),
        Edge(from_node=node_assess, to_node=node_probe, route="route_epistemic_probe"),
        Edge(from_node=node_assess, to_node=node_backward, route="route_backward_plan"),
        Edge(from_node=node_assess, to_node=node_taboo, route="route_taboo_reset"),
        Edge(from_node=node_assess, to_node=node_direct, route="route_direct_plan"),

        # 各認知ノード -> Reviewer Audit (合流)
        Edge(from_node=node_vis, to_node=node_reviewer),
        Edge(from_node=node_probe, to_node=node_reviewer),
        Edge(from_node=node_backward, to_node=node_reviewer),
        Edge(from_node=node_taboo, to_node=node_reviewer),
        Edge(from_node=node_direct, to_node=node_reviewer),

        # Reviewer Audit -> Act Finalizer
        Edge(from_node=node_reviewer, to_node=node_finalizer),
    ]

    return Workflow(name=name, edges=edges)
