"""Evidence contracts: 3 positive + 3 negative cases for each behavior."""
import numpy as np
import pytest

from acr_agi3.agent.execution_evidence import ExecutionEvidence, Motion


def frame(col=2):
    grid = np.zeros((12, 12), dtype=int)
    grid[5, col] = 2
    grid[5, 10] = 3
    return grid


@pytest.mark.parametrize("aid,dr,dc", [(1, 0, 1), (3, 1, 0), (5, 0, -2)])
def test_positive_repeated_translation_confirms_dynamics(aid, dr, dc):
    e = ExecutionEvidence()
    for step in range(3):
        grid = np.zeros((12, 12), dtype=int)
        grid[4 + step * dr, 7 + step * dc] = 7
        e.observe(grid, {"action_id": aid} if step else None)
        assert bool(e.confirmed(aid)) == (step == 2)
    assert e.confirmed(aid) == Motion(7, dr, dc)


@pytest.mark.parametrize("case", ["hud", "click", "ambiguous"])
def test_negative_unreliable_motion_never_confirms(case):
    e = ExecutionEvidence()
    for step in range(3):
        grid = frame(2 + step)
        aid = 6 if case == "click" else 1
        if case == "hud":
            grid = np.zeros((12, 12), dtype=int)
            grid[0, :5-step] = 4
        if case == "ambiguous":
            grid[7, 2 + step] = 5
        e.observe(grid, {"action_id": aid} if step else None)
    assert not e.dynamics()


@pytest.mark.parametrize("case", ["motion", "level", "changed_board"])
def test_positive_progress_and_prediction_are_distinct(case):
    e = ExecutionEvidence()
    e.observe(frame(), None)
    e.samples[1] = (Motion(2, 0, 1), 2)
    e.arm(frame(), 1)
    next_frame = frame(3) if case == "motion" else frame()
    if case == "changed_board":
        next_frame[8, 4:8] = 4
    e.observe(next_frame, {"action_id": 1}, levels=int(case == "level"))
    assert e.meaningful_change
    assert e.level_progress == (case == "level")
    assert e.prediction_match == (case == "motion")


@pytest.mark.parametrize("case", ["wall", "hud", "wrong_direction"])
def test_negative_wrong_prediction_even_if_pixels_change(case):
    e = ExecutionEvidence()
    e.observe(frame(), None)
    e.samples[1] = (Motion(2, 0, 1), 2)
    assert e.arm(frame(), 1)
    grid = frame(1) if case == "wrong_direction" else frame()
    if case == "hud":
        grid[0, 0] = 4
    e.observe(grid, {"action_id": 1})
    assert e.prediction_match is False
    assert e.confirmed(1) is None


@pytest.mark.parametrize("case", ["same", "tiny_delta", "reset"])
def test_positive_click_trials_survive_same_state_and_reset(case):
    e = ExecutionEvidence()
    grid = frame()
    e.observe(grid, None)
    after = grid.copy()
    if case == "tiny_delta":
        after[0, 0] = 4
    e.observe(after, {"action_id": 6, "coordinates": {"x": 3, "y": 3}})
    if case == "reset":
        e.reset_episode()
        e.observe(grid, None)
    assert (3, 3) in e.tried_clicks()


@pytest.mark.parametrize("case", ["new_board", "new_level", "new_game"])
def test_negative_old_state_trials_do_not_ban_new_state(case):
    e = ExecutionEvidence()
    e.observe(frame(), None)
    e.observe(frame(), {"action_id": 6, "coordinates": {"x": 3, "y": 3}})
    grid = frame()
    if case == "new_board":
        grid[7, 2:6] = 4
    if case == "new_game":
        e = ExecutionEvidence()
    e.observe(grid, None, levels=int(case == "new_level"))
    assert not e.tried_clicks()


def test_prediction_and_click_exclusions_clear_on_level_change():
    e = ExecutionEvidence()
    e.observe(frame(), None)
    e.samples[1] = (Motion(2, 0, 1), 2)
    e.arm(frame(), 1)
    e.observe(frame(), {"action_id": 1}, levels=1)
    assert e.level_progress
    assert e.samples[1][1] == 2  # Level transition is not a movement sample.


def test_successful_click_can_be_reused_when_returning_to_state():
    e = ExecutionEvidence()
    grid = frame()
    e.observe(grid, None)
    key = e.state_key
    changed = grid.copy()
    changed[8, 4:8] = 7
    e.observe(changed, {"action_id": 6, "coordinates": {"x": 3, "y": 3}})
    assert e.trials[key][(6, 3, 3)] == "changed"
    e.observe(grid, {"action_id": 5})
    assert (3, 3) not in e.tried_clicks()
