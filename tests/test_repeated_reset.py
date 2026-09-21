"""Three positive and three negative contracts for active reset recovery."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveState
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.agent.my_agent import MyAgent, GameState, GameAction
from acr_agi3.harness.game_action_tools import ActionDecision


@pytest.fixture
def player():
    model = LocalTransformersLlm(
        "mock", generation_fn=lambda _: '```json\n{"action":"ACTION6","x":2,"y":2}\n```'
    )
    return ADKGamePlayer(model=model)


def reset_decision():
    return ActionDecision(action_type="RESET", action_name="RESET", action_id=0)


def stub_workflow(monkeypatch, player, action_id):
    async def workflow(**kwargs):
        return ActionDecision(
            action_type="RESET" if action_id == 0 else "STEP",
            action_name="RESET" if action_id == 0 else f"ACTION{action_id}",
            action_id=action_id,
        )
    monkeypatch.setattr(player, "_run_plan_act_workflow", workflow)


def test_positive_first_stagnation_reset_uses_tool(player, monkeypatch):
    grid = np.zeros((8, 8), dtype=int)
    player.stagnation_count = 6
    player.last_grid = grid.copy()
    player.last_action_info = {"action_id": 6, "coordinates": {"x": 2, "y": 2}}
    tool = Mock(wraps=player.action_tools.reset_game)
    monkeypatch.setattr(player.action_tools, "reset_game", tool)
    decision = player.decide_next_action(grid, available_actions=[6])
    assert decision.action_id == 0
    assert decision.coordinates is None
    assert tool.called


def test_positive_reset_observation_clears_plan_preserves_knowledge(player, monkeypatch):
    grid = np.zeros((8, 8), dtype=int)
    player.memory_tools.memory_write("rules.learned", "Keep this rule", title="Rule")
    player.memory_tools.memory_write("plan.active", "Old plan", title="Plan")
    player.taboo_click_coords = [(2, 2)]
    from acr_agi3.agent.execution_evidence import Motion
    player.execution_evidence.samples[1] = (Motion(2, -1, 0), 2)
    player.execution_evidence.trials[player.execution_evidence.key(grid, 0)] = {(6, 2, 2): "no_effect"}
    player.dynamics_map["UP"] = 1
    player.plan_queue = [{"action": "ACTION1"}]
    player.cognitive_state = CognitiveState.EXECUTING
    player.stagnation_count = 80
    player.last_action_info = {"action_id": 0}
    player.last_grid = grid.copy()
    player.planning_tools.guard.action_history = [1, 2, 1, 2]
    probe = Mock(wraps=player.planning_tools.prober.analyze_displacement)
    monkeypatch.setattr(player.planning_tools.prober, "analyze_displacement", probe)
    stub_workflow(monkeypatch, player, 1)
    player.decide_next_action(grid, available_actions=[1, 2])
    assert player.stagnation_count == 0
    assert not player.plan_queue
    assert player.planning_tools.guard.action_history == [1]
    assert player.dynamics_map["UP"] == 1
    assert (2, 2) in player.taboo_click_coords
    assert "Keep this rule" in player.memory_tools.memory_read("rules.learned")
    assert json.loads(player.memory_tools.memory_read("plan.active"))["status"] == "error"
    probe.assert_not_called()


def test_positive_required_lifecycle_resets_are_preserved():
    from acr_agi3.agent.deliberative_player import DeliberativeGamePlayer
    from acr_agi3.agent.llm.local_vlm import LocalQwenVL
    agent = MyAgent.__new__(MyAgent)
    agent.player = DeliberativeGamePlayer(model=LocalQwenVL("mock", generate_fn=lambda *_: ""))
    agent.step_count = 0
    for state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        assert agent.choose_action([], SimpleNamespace(state=state)) == GameAction.RESET
        assert agent.player.actions.pending_decision.action_id == 0


def test_negative_80_unchanged_frames_do_not_repeat_reset(player):
    grid = np.zeros((8, 8), dtype=int)
    actions = [player.decide_next_action(grid, available_actions=[6]).action_id for _ in range(80)]
    assert actions.count(0) == 1
    assert set(actions) == {0, 6}
    assert actions[-1] == 6


def test_negative_model_reset_requests_use_untried_click(player, monkeypatch):
    grid = np.zeros((8, 8), dtype=int)
    stub_workflow(monkeypatch, player, 0)
    assert player.decide_next_action(grid, available_actions=[6]).action_id == 0
    player.execution_evidence.trials[player.execution_evidence.key(grid, 0)] = {(6, 0, 0): "no_effect"}
    decision = player.decide_next_action(grid, available_actions=[6])
    assert decision.action_id == 6
    assert decision.coordinates != {"x": 0, "y": 0}
    assert decision.metadata["reset_suppressed"]


def test_negative_changed_reset_destination_cannot_reset_again(player, monkeypatch):
    grid = np.zeros((8, 8), dtype=int)
    stub_workflow(monkeypatch, player, 0)
    assert player.decide_next_action(grid, available_actions=[1, 2]).action_id == 0
    destination = grid.copy()
    destination[4, 4] = 3
    assert player.decide_next_action(destination, available_actions=[1, 2]).action_id in (1, 2)
    # Returning to the pre-reset board also must not restart the reset loop.
    assert player.decide_next_action(grid, available_actions=[1, 2]).action_id in (1, 2)
