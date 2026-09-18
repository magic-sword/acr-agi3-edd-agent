"""Google ADK 2.0 直線的ゲームプレイワークフローの単体テスト."""

import pytest

from acr_agi3.agent.cognitive_workflow import (
    CognitiveState,
    build_cognitive_workflow,
    perceive_node,
    plan_node,
    act_node,
)
from acr_agi3.agent.workflow_schemas import PlanProposal


def test_workflow_graph_structure():
    """ADK 2.0 直線ワークフローの構造（START -> Perceive -> Plan -> Act）を検証."""
    wf = build_cognitive_workflow("test_gameplay_wf")
    assert wf.name == "test_gameplay_wf"
    assert len(wf.edges) == 3


def test_perceive_and_plan_nodes():
    """知覚・思考ノードが正常に状態を受け渡すこと."""
    state = CognitiveState(step=1, working_memory="Initial inspection")
    s1 = perceive_node(state)
    assert s1.step == 1

    s1.plan_proposal = PlanProposal(
        hypothesis="Blue is key, green is door",
        goal="Collect blue key",
        action="UP",
        reasoning="Step toward blue key",
    )
    s2 = plan_node(s1)
    assert s2.plan_proposal.action == "UP"


def test_act_node_execution_with_game_controller():
    """act_node が game-controller を通じて動的操作力学を解決し ActionDecision を確定すること."""
    state = CognitiveState(
        step=2,
        available_actions=[1, 2, 3, 4],
        dynamics_map={"UP": 3, "DOWN": 4},
        plan_proposal=PlanProposal(
            hypothesis="Testing act node",
            goal="Move UP",
            action="UP",
            reasoning="Testing game-controller resolution",
        ),
    )
    res = act_node(state)
    assert res.final_decision is not None
    assert res.final_decision.action_id == 3
    assert res.final_decision.action_name == "ACTION3"
    assert res.final_decision.loaded_skill == "game-controller"
