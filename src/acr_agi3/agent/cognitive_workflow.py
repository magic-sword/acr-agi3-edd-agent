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

from acr_agi3.agent.workflow_schemas import PlanProposal
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
    current_subgoal: str = ""
    cognitive_intent: str = ""
    method_known: bool = True
    plan_proposal: Optional[PlanProposal] = None
    final_decision: Optional[ActionDecision] = None
    routing_reason: str = ""
    is_probing: bool = False
    active_macro_skill: Optional[str] = None
    is_macro_mode: bool = False
    telemetry: Dict[str, Any] = field(default_factory=dict)


def assess_cognitive_state(ctx: Context, state: CognitiveState) -> CognitiveState:
    """ステップ1: 状況認識とメタスキルルーティング (Assessment Node).

    盤面状況、エピソード記憶、停滞カウント、サブゴール進捗、認知意図から
    現在の認知フェーズを決定し、対応するルート (ctx.route) を設定する。
    """
    step = state.step
    stagnation = state.stagnation_count
    wm = (state.working_memory or "").lower()

    # 1. ルート決定ルール (人間の認知優先度順)
    if stagnation >= 2 or "stagnant" in wm or "barrier hit" in wm or "trap" in wm:
        # 停滞・壁衝突・手詰まり: Taboo Reset Guard (禁忌学習 & マクロ解除 & リセット判定)
        route = "route_taboo_reset"
        skill = "taboo-reset-guard"
        mode = "taboo_reset"
        reason = f"Stagnation detected (stagnation_count={stagnation} >= 2) or barrier in memory, engaging taboo guard"
    elif state.is_macro_mode and state.active_macro_skill and not state.is_probing:
        # 承認済みマクロスキル実行: Fast Macro Execution (LLM推論バイパス)
        route = "route_macro_execution"
        skill = state.active_macro_skill
        mode = "macro_execution"
        reason = f"Executing verified macro skill '{state.active_macro_skill}'"
    elif (
        state.is_probing
        or state.cognitive_intent == "PROBE"
        or not state.method_known
        or "unknown" in wm
        or "probe" in wm
    ):
        # サブゴール達成手法が未知 / 操作法則未解明: Epistemic Prober (能動探索・実験)
        route = "route_epistemic_probe"
        skill = "epistemic-prober"
        mode = "epistemic_probe"
        reason = f"Subgoal '{state.current_subgoal or 'unknown'}' method is unknown or probing requested, requiring epistemic probing"
    elif step <= 1 or "initial inspection needed" in wm or "inspect" in wm:
        # 初手: Visual Inspection (目視点検 & ゲシュタルト構造解析)
        route = "route_visual_inspection"
        skill = "visual-inspector"
        mode = "visual_inspection"
        reason = f"Initial step (step={step}): inspecting overall gestalt and entities"
    elif "target far" in wm or "path blocked" in wm or "complex" in wm or state.cognitive_intent == "PLAN":
        # 遠隔ゴール・障害物迂回・大目標逆算: Backward Planner (逆算プランニング & サブゴール分解)
        route = "route_backward_plan"
        skill = "backward-planner"
        mode = "backward_plan"
        reason = "Decomposing goal into milestones and sequencing subgoals via backward chaining"
    else:
        # 通常前進行動: Direct Macro Plan
        route = "route_direct_plan"
        skill = "macro-skill-compiler"
        mode = "direct_plan"
        reason = "Standard progression without stagnation, direct action plan"

    state.selected_skill = skill
    state.cognitive_mode = mode
    state.routing_reason = reason
    state.telemetry["routing_reason"] = reason
    ctx.route = route
    logger.info(
        "🧠 [assess_cognitive_state] Step %d (stagnation=%d) -> mode='%s', route='%s', skill='%s' (Reason: %s)",
        step, stagnation, mode, route, skill, reason
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


def macro_execution_node(state: CognitiveState) -> CognitiveState:
    """[Phase 5-Macro] Macro Execution ノード: 検証済み生成スキルの自律マクロ実行."""
    skill_name = state.active_macro_skill or "macro-skill-compiler"
    state.selected_skill = skill_name
    state.telemetry["cognitive_phase"] = "macro_execution"
    logger.info("🚀 [macro_execution_node] Executing verified macro skill '%s'", skill_name)
    return state


def act_finalizer_node(state: CognitiveState) -> CognitiveState:
    """終端ノード: 実行アクション確定 (Act Finalizer Node).

    提案された計画と GameController の検証を経て最終 ActionDecision を構築する。
    """
    plan = state.plan_proposal
    raw_action = (plan.action if plan else "UP").upper()
    # meta_skills/game-controller の GameController による安全・決定論的検証
    coords = plan.coordinates if plan else None
    reasoning = plan.reasoning if plan else "Workflow executed"
    action_type = "STEP"
    action_id = 1
    action_name = "ACTION1"

    try:
        from game_controller import GameController
        controller = GameController()
        payload = {"action": raw_action, "reasoning": reasoning}
        if coords:
            payload["coordinates"] = coords
        val = controller.parse_and_validate(payload, grid=state.observation)
        action_type = val["action_type"]
        action_id = val["action_id"]
        action_name = val["action_name"]
        coords = val.get("coordinates")
        reasoning = val.get("reasoning", reasoning)
    except Exception:
        # 最低限のフォールバック
        action_type = "CLICK" if "CLICK" in raw_action or "6" in raw_action else "STEP"
        action_id = 6 if action_type == "CLICK" else 1
        action_name = "ACTION6" if action_type == "CLICK" else "ACTION1"

    decision = ActionDecision(
        action_type=action_type,
        action_name=action_name,
        action_id=action_id,
        coordinates=coords,
        reasoning=reasoning,
        loaded_skill=state.selected_skill or "game-controller",
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
    node_macro = FunctionNode(func=macro_execution_node, name="macro_execution_node")
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
        Edge(from_node=node_assess, to_node=node_macro, route="route_macro_execution"),

        # 各認知ノード -> Act Finalizer (直接合流)
        Edge(from_node=node_vis, to_node=node_finalizer),
        Edge(from_node=node_probe, to_node=node_finalizer),
        Edge(from_node=node_backward, to_node=node_finalizer),
        Edge(from_node=node_taboo, to_node=node_finalizer),
        Edge(from_node=node_direct, to_node=node_finalizer),
        Edge(from_node=node_macro, to_node=node_finalizer),
    ]

    return Workflow(name=name, edges=edges)
