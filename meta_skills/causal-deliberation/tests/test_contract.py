"""Three positive and three negative causal workflow contracts."""

import pytest
from acr_agi3.agent.deliberation import ThoughtMode, ThoughtState


def test_positive_plan_suspends_for_missing_knowledge():
    state = ThoughtState()
    state.ask("Which condition opens the door?")
    assert state.mode == ThoughtMode.CAUSAL
    assert state.questions[-1].return_mode == ThoughtMode.PLAN


def test_positive_causal_inference_requests_discriminating_experiment():
    state = ThoughtState()
    state.ask("Is the tile a switch?")
    state.propose_experiment("Tile opens door", "Door opens", "Only the tile changes")
    assert state.mode == ThoughtMode.EXPERIMENT
    assert state.questions[-1].question == "Is the tile a switch?"


def test_positive_evidence_resumes_suspended_plan():
    state = ThoughtState()
    state.ask("Which button?")
    state.results["r1"] = {
        "outcome": "supported",
        "action": {"question_id": state.questions[-1].question_id},
    }
    state.resolve("Button A", "Door closed", "Door opens", ["r1"])
    assert state.mode == ThoughtMode.PLAN
    assert state.rules


def test_negative_experiment_without_a_question():
    with pytest.raises(ValueError):
        ThoughtState().propose_experiment("guess", "change", "no change")


def test_negative_uncertain_evidence_does_not_close_question():
    state = ThoughtState()
    state.ask("Which control?")
    state.results["r1"] = {"outcome": "inconclusive"}
    with pytest.raises(ValueError):
        state.resolve("Button A", "Door closed", "Door opens", ["r1"])
    assert state.questions


def test_negative_plan_without_conditional_rules():
    with pytest.raises(ValueError):
        ThoughtState().make_plan("Open door", [{"action_id": 1, "expected_result": "Open"}], [])
