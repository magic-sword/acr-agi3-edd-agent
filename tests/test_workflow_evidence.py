"""Evidence gates for the five-mode ADK workflow (positive and negative contracts)."""

import json

import numpy as np
import pytest

from acr_agi3.agent.deliberation import ThoughtMode, ThoughtState
from acr_agi3.agent.deliberative_player import DeliberativeGamePlayer
from acr_agi3.agent.llm.local_vlm import LocalQwenVL

TARGET = {"x": 2, "y": 2, "width": 4, "height": 4, "description": "object and possible destination"}


def prepared():
    p = DeliberativeGamePlayer(model=LocalQwenVL("mock", generate_fn=lambda *_: ""))
    p.screen.publish(np.zeros((8, 8), dtype=int))
    p.observe_screen()
    p.set_goal("reach visible target", "target visible")
    p.available_actions = [1]
    p.actions.set_available_actions([1])
    p.need_causal_knowledge("Does this control move the object?")
    assert "error" not in p.need_experiment(
        "motion", "object moves", "object stays", True, TARGET, "route appears free"
    )
    assert p.step_action(1, "test one control")["scheduled"]
    p.decision = None
    return p


@pytest.mark.parametrize("change", ["target", "hud", "none"])
def test_target_evidence_positive_and_negative(change):
    p = prepared()
    grid = np.zeros((8, 8), dtype=int)
    if change == "target":
        grid[3, 3] = 2
    elif change == "hud":
        grid[0, 0] = 2
    p.screen.publish(grid)
    p.observe_screen(view="both")
    result = p.assess_result("supported", "claimed movement", 1, 2)
    if change == "target":
        assert result["recent_results"]["result_1"]["target_comparison"]["target_changed"]
        assert p.state.hypotheses["hypothesis_1"]["status"] == "supported"
        assert (
            p.resolve_question(
                "moves under this condition", "route free", "object moves", ["result_1"]
            )["mode"]
            == "PLAN"
        )
        steps = [
            {
                "action_id": 1,
                "precondition": "route free",
                "expected_result": "moves again",
                "target": TARGET,
                "expected_visual_change": True,
            }
        ]
        assert p.plan_actions("reach target", steps, ["rule_2"])["mode"] == "EXECUTE"
        assert p.step_action(1, "route still free")["scheduled"]
    else:
        assert "error" in result
        assert p.state.mode == ThoughtMode.REVIEW
        assert not p.state.results
        assert p.assess_result("inconclusive", "movement not established", 1, 2)["mode"] == "CAUSAL"
        assert p.state.hypotheses["hypothesis_1"]["status"] == "inconclusive"


def test_next_experiment_must_account_for_previous_result():
    p = prepared()
    p.screen.publish(np.zeros((8, 8), dtype=int))
    p.observe_screen(view="both")
    p.assess_result("refuted", "no object movement", 1, 2)
    result = p.need_experiment(
        "other meaning", "another effect", "no effect", True, TARGET, "same board"
    )
    assert "error" in result
    assert len(p.state.hypotheses) == 1
    assert p.state.constraints[0]["action_id"] == 1
    assert "error" not in p.need_experiment(
        "other meaning",
        "another effect",
        "no effect",
        True,
        TARGET,
        "same board",
        retry_reason="The refuted movement hypothesis leaves interaction untested",
    )


def test_refuted_rule_cannot_be_used_as_executable_effect():
    p = prepared()
    p.screen.publish(np.zeros((8, 8), dtype=int))
    p.observe_screen(view="both")
    p.assess_result("refuted", "motion failed at this obstacle", 1, 2)
    p.resolve_question("does not move here", "blocked route", "no motion", ["result_1"])
    steps = [
        {
            "action_id": 1,
            "precondition": "blocked",
            "expected_result": "moves",
            "target": TARGET,
            "expected_visual_change": True,
        }
    ]
    assert "error" in p.plan_actions("move", steps, ["rule_2"])
    assert p.state.mode == ThoughtMode.PLAN


def test_crop_outside_target_cannot_authorize_review():
    p = prepared()
    p.screen.publish(np.ones((8, 8), dtype=int))
    p.observe_screen(x=0, y=0, width=1, height=1)
    assert "error" in p.assess_result("supported", "HUD changed", 1, 2)
    p.observe_screen()
    assert "error" not in p.assess_result("supported", "target changed", 1, 2)


def test_same_question_text_reopened_cannot_reuse_old_evidence():
    p = prepared()
    p.screen.publish(np.ones((8, 8), dtype=int))
    p.observe_screen(view="both")
    p.assess_result("supported", "target changed", 1, 2)
    p.resolve_question("motion", "free", "moves", ["result_1"])
    p.need_causal_knowledge("Does this control move the object?")
    assert "error" in p.resolve_question("motion", "free", "moves", ["result_1"])


@pytest.mark.parametrize("mode", [ThoughtMode.CAUSAL, ThoughtMode.EXPERIMENT, ThoughtMode.REVIEW])
def test_no_implicit_promotion_to_execution(mode):
    state = ThoughtState(mode=mode)
    with pytest.raises(ValueError, match="Illegal cognitive transition"):
        state.transition(ThoughtMode.EXECUTE, "tool requested execution")
    assert state.mode == mode
    assert state.revision == 0


def test_shape_change_is_inconclusive_not_success():
    p = prepared()
    p.screen.publish(np.zeros((4, 4), dtype=int))
    p.observe_screen()
    assert "error" in p.assess_result("supported", "shape changed", 1, 2)
    assert "error" not in p.assess_result("inconclusive", "cannot compare layouts", 1, 2)


def test_trace_is_emitted_before_a_model_exception():
    events = []

    def fail(prompt, images=None):
        assert events[-1]["kind"] == "model_io"
        assert events[-1]["model_event"]["phase"] == "input"
        raise RuntimeError("model crashed")

    p = DeliberativeGamePlayer(model=LocalQwenVL("mock", generate_fn=fail))
    p.trace_sink = events.append
    with pytest.raises(RuntimeError, match="model crashed"):
        p.decide_next_action(np.zeros((8, 8), dtype=int), [1])
    assert events[0]["kind"] == "deliberation_start"
    assert events[-1]["kind"] == "deliberation_end"
    assert events[-1]["action_selected"] is False
    assert events[-1]["sequence"] > events[0]["sequence"]
    json.dumps(events)


@pytest.mark.parametrize("outcome", ["supported", "refuted", "inconclusive"])
def test_rule_resolution_disclosure_depends_on_evidence(outcome):
    from types import SimpleNamespace

    p = prepared()
    p._loaded_skills.update({"causal-deliberation", "visual-inspector"})
    p.screen.publish(np.ones((8, 8), dtype=int))
    p.observe_screen(view="both")
    p.assess_result(outcome, "target observation", 1, 2)
    assert p._tool_is_relevant(SimpleNamespace(name="resolve_question"), None) == (
        outcome != "inconclusive"
    )
    if outcome == "inconclusive":
        assert "resolve_question is unavailable" in p._build_mode_guidance()


def test_refutation_of_one_effect_does_not_ban_a_different_supported_effect():
    p = prepared()
    grid = np.zeros((8, 8), dtype=int)
    p.screen.publish(grid)
    p.observe_screen(view="both")
    p.assess_result("refuted", "motion did not occur", 1, 2)
    p.resolve_question("no motion here", "current board", "motion failed", ["result_1"])
    p.need_causal_knowledge("Does the control leave the object unchanged?")
    p.need_experiment(
        "no change",
        "object stays",
        "object moves",
        False,
        TARGET,
        "same board",
        retry_reason="Test the alternative to the refuted movement prediction",
    )
    assert p.step_action(1, "test alternative")["scheduled"]
    p.decision = None
    p.screen.publish(grid)
    p.observe_screen(view="both")
    p.assess_result("supported", "object stayed", 2, 3)
    p.resolve_question("stays here", "same board", "object stays", ["result_3"])
    steps = [
        {
            "action_id": 1,
            "precondition": "same board",
            "expected_result": "object stays",
            "target": TARGET,
            "expected_visual_change": False,
        }
    ]
    assert (
        p.plan_actions("preserve object while operating control", steps, ["rule_4"])["mode"]
        == "EXECUTE"
    )
    assert p.step_action(1, "use supported alternative")["scheduled"]


def test_failed_plan_evidence_is_available_to_its_diagnostic_question():
    state = ThoughtState(mode=ThoughtMode.REVIEW)
    state.pending = {"intent": "plan", "frame_id": 1, "action_id": 1, "expected_result": "moves"}
    state.review("refuted", "blocked at wall", 1, 2)
    assert "result_1" in state.evidence_for_active_question()
    assert state.results["result_1"]["action"].get("question_id") is None
    result = state.resolve(
        "blocked under this condition", "wall adjacent", "movement fails", ["result_1"]
    )
    assert result["mode"] == "PLAN"
    assert state.rules["rule_2"]["evidence_outcome"] == "refuted"
