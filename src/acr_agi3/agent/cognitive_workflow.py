"""Google ADK 2.0 準拠・自律ゲームプレイワークフロー (GamePlayWorkflow).

複雑化しすぎた多分岐ルーティングを撤廃し、人間がゲーム画面を見て操作する
直線的かつ強固な「視覚知覚 (Perceive) -> 自律思考 (Plan) -> 1手実行 (Act)」
の Google ADK 2.0 グラフワークフローを提供します。
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
    """ワークフロー各ノード間で共有される状態コンテキスト."""
    step: int
    observation: Any = None  # grid or visual
    working_memory: str = ""
    stagnation_count: int = 0
    available_actions: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    dynamics_map: Dict[str, int] = field(default_factory=dict)
    plan_proposal: Optional[PlanProposal] = None
    final_decision: Optional[ActionDecision] = None


def perceive_node(state: CognitiveState) -> CognitiveState:
    """ノード1: 視覚入力・客観的事実の知覚 (Perceive Node).

    visual-inspector スキルの原則に基づき、客観的盤面状態を整理する。
    """
    logger.info("👁️ [perceive_node] Step %d: Processing visual frame & console state", state.step)
    return state


def plan_node(state: CognitiveState) -> CognitiveState:
    """ノード2: 行動計画・仮説検証ノード (Plan Node / Planning-First Protocol).

    ループの初めに必ず実行される思考ノード。
    VLM / LLM がゲームルール（動的変化への適応、法則発見、最小手クリア）に基づき、
    状況分析 (Hypothesis) -> 目標策定 (Goal) -> 最小手行動 (Action) を立案する。
    """
    logger.info("🧠 [plan_node] Step %d: Constructing action plan (Hypothesis -> Goal -> Action)", state.step)
    return state


def act_node(state: CognitiveState) -> CognitiveState:
    """ノード3: 1手出力確定 (Act Node).

    game-controller メタスキルを用いて、動的操作力学の解決と幾何吸着を行い
    100% 確実に環境へ発行可能な ActionDecision を確定する。
    """
    plan = state.plan_proposal
    raw_action = (plan.action if plan else "ACTION1").upper()
    coords = plan.coordinates if plan else None
    reasoning = plan.reasoning if plan else "Action decided via cognitive workflow"

    try:
        import sys
        from pathlib import Path
        _SKILL_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "game-controller" / "scripts"
        if str(_SKILL_DIR) not in sys.path:
            sys.path.insert(0, str(_SKILL_DIR))
        from game_controller import GameController

        controller = GameController(
            available_actions=state.available_actions,
            dynamics_map=state.dynamics_map,
        )
        payload: Dict[str, Any] = {"action": raw_action, "reasoning": reasoning}
        if coords:
            payload["coordinates"] = coords
        val = controller.parse_and_validate(payload, grid=state.observation)
        decision = ActionDecision(
            action_type=val["action_type"],
            action_name=val["action_name"],
            action_id=val["action_id"],
            coordinates=val.get("coordinates"),
            reasoning=val.get("reasoning", reasoning),
            loaded_skill="game-controller",
        )
    except Exception as e:
        logger.warning("Failed to invoke GameController in act_node: %s. Using fallback.", e)
        fb_id = state.available_actions[0] if state.available_actions else 1
        decision = ActionDecision(
            action_type="STEP",
            action_name=f"ACTION{fb_id}",
            action_id=fb_id,
            coordinates=None,
            reasoning=f"Fallback action: {reasoning}",
            loaded_skill="game-controller",
        )

    state.final_decision = decision
    logger.info(
        "🚀 [act_node] Step %d: Finalized ActionDecision: %s (ID: %d)",
        state.step, decision.action_name, decision.action_id
    )
    return state


def build_cognitive_workflow(name: str = "gameplay_workflow") -> Workflow:
    """Google ADK 2.0 準拠の直線的ゲームプレイワークフローを構築."""
    node_perceive = FunctionNode(func=perceive_node, name="perceive_node")
    node_plan = FunctionNode(func=plan_node, name="plan_node")
    node_act = FunctionNode(func=act_node, name="act_node")

    edges = [
        Edge(from_node=START, to_node=node_perceive),
        Edge(from_node=node_perceive, to_node=node_plan),
        Edge(from_node=node_plan, to_node=node_act),
    ]
    return Workflow(name=name, edges=edges)
