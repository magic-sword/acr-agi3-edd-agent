"""Integration contracts for bounded execution and perception reuse."""
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.execution_evidence import Motion
from acr_agi3.agent.llm.local_model import LocalTransformersLlm


def make_player(action="ACTION4"):
    model = LocalTransformersLlm("mock", generation_fn=lambda _: action)
    return ADKGamePlayer(model=model)


def frame(col=3):
    grid = np.zeros((12, 12), dtype=int)
    grid[5, col] = 2
    grid[5, 10] = 3
    return grid


@pytest.mark.parametrize("action", ["ACTION4", "RIGHT", "Move RIGHT"])
def test_positive_route_is_bounded_and_first_action_matches(action):
    player = make_player(action)
    player.execution_evidence.samples[4] = (Motion(2, 0, 1), 2)
    decision = player.decide_next_action(frame(), available_actions=[4])
    assert decision.action_id == 4
    assert len(player.plan_queue) == 2
    assert decision.metadata["prediction_armed"]
    next_decision = player.decide_next_action(frame(4), available_actions=[4])
    assert next_decision.metadata["fast_path"]
    assert next_decision.metadata["previous_transition"]["prediction_match"] is True


@pytest.mark.parametrize("case", ["unknown", "wrong_first", "hud_only"])
def test_negative_unverified_or_divergent_plan_never_runs_fast(case):
    player = make_player("ACTION1" if case == "wrong_first" else "ACTION4")
    if case != "unknown":
        player.execution_evidence.samples[4] = (Motion(2, 0, 1), 2)
    decision = player.decide_next_action(frame(), available_actions=[1, 4])
    if case == "hud_only":
        grid = frame()
        grid[0, 0] = 7
        decision = player.decide_next_action(grid, available_actions=[1, 4])
        assert decision.metadata["previous_transition"]["prediction_match"] is False
    assert not player.plan_queue
    assert not decision.metadata.get("fast_path", False)


@pytest.mark.parametrize("case", ["same_frame", "timings", "bounded_reuse"])
def test_positive_perception_reuse_and_timing(case):
    player = make_player("ACTION1")
    grid = np.zeros((12, 12), dtype=int)
    first = player.decide_next_action(grid, available_actions=[1])
    second = player.decide_next_action(grid, available_actions=[1])
    assert not first.metadata["phase_metrics"]["perception_reused"]
    assert second.metadata["phase_metrics"]["perception_reused"]
    if case == "timings":
        assert set(second.metadata["phase_metrics"]) >= {"perceive_ms", "plan_ms", "act_ms"}
    if case == "bounded_reuse":
        player.decide_next_action(grid, available_actions=[1])
        fourth = player.decide_next_action(grid, available_actions=[1])
        assert not fourth.metadata["phase_metrics"]["perception_reused"]


@pytest.mark.parametrize("case", ["board", "actions", "reset"])
def test_negative_changed_context_invalidates_perception_cache(case):
    player = make_player("ACTION1")
    grid = np.zeros((12, 12), dtype=int)
    player.decide_next_action(grid, available_actions=[1])
    actions = [1]
    if case == "board":
        grid[4, 4] = 2
    elif case == "actions":
        actions = [1, 2]
    else:
        player.reset()
    decision = player.decide_next_action(grid, available_actions=actions)
    assert not decision.metadata["phase_metrics"]["perception_reused"]
