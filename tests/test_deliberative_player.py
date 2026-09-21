"""Contracts for goal-first thought, tool observation and causal resumption."""

import numpy as np
import pytest
from google.adk.models import LlmRequest
from google.genai.types import Content, Part

from acr_agi3.agent.deliberation import ThoughtMode, ThoughtState
from acr_agi3.agent.deliberative_player import (
    DeliberationBudgetExceededError,
    DeliberativeGamePlayer,
)
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.tools.screen_tools import ScreenTools


def call(name, **args):
    return {"name": name, "args": args}


def experiment_calls(action_id=1):
    return [
        call("load_skill", skill_name="causal-deliberation"),
        call(
            "need_causal_knowledge", question="Which action moves this object towards the target?"
        ),
        call("load_skill", skill_name="visual-inspector"),
        call("observe_screen"),
        call(
            "need_experiment",
            hypothesis="This button moves the object right",
            prediction="Object moves one cell right",
            alternative="Object stays or moves elsewhere",
        ),
        call("load_skill", skill_name="game-controller"),
        call(
            "step_action", action_id=action_id, reasoning="Test the missing movement relation once"
        ),
    ]


def model_script(calls, captures=None):
    queue = iter(calls)

    def generate(prompt, images=None):
        if captures is not None:
            captures.append((prompt, list(images or [])))
        return next(queue)

    return LocalQwenVL("mock", generate_fn=generate)


def board(col=2):
    grid = np.zeros((8, 8), dtype=int)
    grid[4, col] = 2
    grid[4, 6] = 3
    return grid


def test_goal_first_experiment_then_resume_plan_with_real_tool_images():
    captures = []
    review_calls = [
        call("load_skill", skill_name="causal-deliberation"),
        call("load_skill", skill_name="visual-inspector"),
        call("observe_screen", view="both"),
        call(
            "assess_result",
            outcome="supported",
            evidence="The red object moved right; target stayed",
            before_frame_id=1,
            after_frame_id=2,
        ),
        call(
            "resolve_question",
            answer="ACTION1 moves the red object right on open space",
            precondition="An empty cell lies to its right",
            effect="Moves one cell right",
            result_ids=["result_1"],
        ),
        call(
            "plan_actions",
            subgoal="Approach the target",
            rule_ids=["rule_2"],
            steps=[
                {
                    "action_id": 1,
                    "precondition": "Right neighbor is free",
                    "expected_result": "Red object advances one cell towards target",
                }
            ],
        ),
        call("load_skill", skill_name="game-controller"),
        call("step_action", action_id=1, reasoning="The right neighbor is visibly empty"),
    ]
    player = DeliberativeGamePlayer(model=model_script(experiment_calls() + review_calls, captures))
    first = player.decide_next_action(board(), available_actions=[1])
    assert first.action_id == 1
    assert captures[0][1] == []  # No automatic image injection.
    assert any(len(images) == 1 for _, images in captures)
    assert player.state.mode == ThoughtMode.REVIEW
    second = player.decide_next_action(board(3), available_actions=[1])
    assert second.action_id == 1
    assert any(len(images) == 2 for _, images in captures)
    assert player.state.rules["rule_2"]["status"] == "provisional"
    assert not player.state.questions
    assert any(t["from"] == "CAUSAL" and t["to"] == "PLAN" for t in player.state.transitions)
    assert player.state.pending["intent"] == "plan"


def test_observation_is_read_only_and_allowed_in_every_mode():
    player = DeliberativeGamePlayer(model=model_script([]))
    player.screen.publish(board())
    for mode in ThoughtMode:
        player.state.mode = mode
        result = player.screen.observe_screen()
        assert result["screen_observation"]["current_frame_id"] == 1
        assert player.state.mode == mode
        assert player.decision is None
        assert not player.actions.history


def test_no_free_text_action_fallback():
    player = DeliberativeGamePlayer(
        model=LocalQwenVL("mock", generate_fn=lambda prompt, images=None: "ACTION1"),
        max_llm_calls=2,
    )
    with pytest.raises(DeliberationBudgetExceededError):
        player.decide_next_action(board(), available_actions=[1])
    assert not player.actions.history


def test_unknown_action_and_unseen_screen_cannot_execute():
    player = DeliberativeGamePlayer(model=model_script([]))
    player.screen.publish(board())
    player.available_actions = [1]
    assert "error" in player.step_action(1, "go")
    player.screen.observe_screen()
    assert "error" in player.step_action(1, "go")  # PLAN is not execution.
    player.need_causal_knowledge("What does the button do?")
    player.need_experiment("moves", "object changes location", "does nothing")
    assert "error" in player.step_action(7, "invalid")
    assert player.decision is None


def test_unobserved_after_frame_cannot_verify_hypothesis():
    player = DeliberativeGamePlayer(model=model_script(experiment_calls()))
    player.decide_next_action(board(), available_actions=[1])
    player.decision = None
    player.screen.publish(board(3))
    assert "error" in player.assess_result("supported", "it moved", 1, 2)
    player.screen.observe_screen(view="both")
    assert "error" in player.assess_result("supported", "it moved", 99, 2)
    result = player.assess_result("inconclusive", "occluded object", 1, 2)
    assert result["mode"] == "CAUSAL"
    assert "error" in player.resolve_question("yes", "open", "moves", ["result_1"])


@pytest.mark.parametrize("view", ["current", "previous", "both"])
def test_screen_tool_delivers_actual_pngs_to_local_vlm(view):
    screen = ScreenTools()
    screen.publish(board())
    screen.publish(board(3))
    result = screen.observe_screen(view=view, x=1, y=3, width=4, height=3)
    request = LlmRequest(
        contents=[
            Content(
                role="user",
                parts=[Part.from_function_response(name="observe_screen", response=result)],
            )
        ]
    )
    model = LocalQwenVL("mock", generate_fn=lambda *_: "")
    prompt, images = model._extract_text_and_images(request)
    assert len(images) == (2 if view == "both" else 1)
    assert images[0].size == (48, 36)
    assert "iVBOR" not in prompt
    assert "frame_id" in prompt


@pytest.mark.parametrize("kwargs", [dict(view="unknown"), dict(x=-1), dict(width=99)])
def test_invalid_observation_is_not_marked_viewed(kwargs):
    screen = ScreenTools()
    screen.publish(board())
    assert "error" in screen.observe_screen(**kwargs)
    assert not screen.viewed


def test_nested_question_resumes_causal_parent_before_plan():
    state = ThoughtState()
    state.ask("How to open the door?")
    state.ask("Which object is the switch?")
    state.results["e1"] = {"outcome": "supported"}
    state.resolve("The left tile is a switch", "tile pressed", "door changes", ["e1"])
    assert state.mode == ThoughtMode.CAUSAL
    assert state.questions[-1].question == "How to open the door?"
    state.resolve("Press the switch", "door closed", "door opens", ["e1"])
    assert state.mode == ThoughtMode.PLAN


def test_rule_requires_evidence_and_plan_requires_rules():
    state = ThoughtState()
    with pytest.raises(ValueError):
        state.make_plan(
            "goal", [{"action_id": 1, "precondition": "free", "expected_result": "move"}], []
        )
    state.ask("Which control?")
    with pytest.raises(ValueError):
        state.resolve("ACTION1", "free", "move", ["fabricated"])


def test_visible_question_and_known_rule_resume_without_experiment():
    player = DeliberativeGamePlayer(model=model_script([]))
    player.screen.publish(board())
    player.need_causal_knowledge("Where is the target?")
    assert "error" in player.answer_visible_question("Right", "green marker")
    player.screen.observe_screen()
    assert player.answer_visible_question("Right", "green marker")["mode"] == "PLAN"
    player.state.rules["known"] = {"precondition": "free right cell", "effect": "move right"}
    player.need_causal_knowledge("How can I approach it?")
    assert "error" in player.use_known_rules("Right", ["invented"], "right cell free")
    result = player.use_known_rules("Right", ["known"], "right cell free")
    assert result["mode"] == "PLAN"
    assert not player.actions.history


def test_failed_plan_preserves_rule_but_discards_remaining_actions():
    state = ThoughtState()
    state.rules["known"] = {"precondition": "open", "effect": "right"}
    state.plan = [{"action_id": 1}]
    state.mode = ThoughtMode.REVIEW
    state.pending = {"intent": "plan", "frame_id": 1, "action_id": 1}
    state.review("refuted", "The destination is now blocked", 1, 2)
    assert state.mode == ThoughtMode.CAUSAL
    assert "known" in state.rules
    assert not state.plan
    assert state.questions[-1].return_mode == ThoughtMode.PLAN


def test_game_switch_isolates_rules_and_reset_drops_stale_observations():
    player = DeliberativeGamePlayer(model=model_script([]))
    player.switch_game("first")
    player.state.rules["known"] = {"effect": "right"}
    player.state.facts.append({"frame_id": 1, "answer": "target on right"})
    first_id = player.screen.publish(board())
    player.screen.observe_screen()
    player.switch_game("second")
    assert not player.state.rules
    assert not player.screen.viewed
    player.switch_game("first")
    assert "known" in player.state.rules
    assert not player.state.facts
    assert player.screen.publish(board()) > first_id
    assert not player.screen.viewed


def test_reset_repetition_is_rejected_after_gateway_restart():
    player = DeliberativeGamePlayer(model=model_script(experiment_calls()))
    player.screen.publish(board())
    player.screen.observe_screen()
    assert "error" in player.reset_game("", "try another route")
    assert player.reset_game("A required object is trapped", "Try another route")["scheduled"]
    player.decide_next_action(board(), available_actions=[1])
    player.decision = None
    player.screen.publish(board(3))
    player.screen.observe_screen(view="both")
    player.assess_result("inconclusive", "Object role remains unknown", 2, 3)
    player.screen.publish(board())
    player.screen.observe_screen()
    assert "error" in player.reset_game("Still trapped", "Try another route")
    assert player.decision is None


def test_one_action_per_frame_and_continue_plan_requires_fresh_observation():
    player = DeliberativeGamePlayer(model=model_script([]))
    player.screen.publish(board())
    player.screen.observe_screen()
    player.available_actions = [1]
    player.actions.set_available_actions([1])
    player.state.rules["known"] = {"effect": "right"}
    steps = [{"action_id": 1, "precondition": "free", "expected_result": "moves right"}] * 2
    assert player.plan_actions("approach goal", steps, ["known"])["mode"] == "EXECUTE"
    assert player.step_action(1, "free right cell")["scheduled"]
    assert "error" in player.step_action(1, "another step")
    assert len(player.state.plan) == 1
    player.decision = None
    player.screen.publish(board(3))
    assert "error" in player.continue_plan("still free")
    player.screen.observe_screen(view="both")
    assert player.assess_result("supported", "Moved right", 1, 2)["mode"] == "PLAN"
    assert player.continue_plan("next right cell visibly free")["mode"] == "EXECUTE"
    assert player.step_action(1, "next right cell free")["scheduled"]
    assert not player.state.plan


def test_skill_name_is_never_reinterpreted_as_an_environment_action():
    model = LocalQwenVL("mock", generate_fn=lambda *_: "")
    name, args = model._detect_tool_call(
        '<tool_call>{"name":"load_skill","arguments":{"skill_name":"ACTION1"}}</tool_call>'
    )
    assert name == "load_skill"
    assert args == {"skill_name": "ACTION1"}


@pytest.mark.parametrize(
    "mode,parent",
    [
        (ThoughtMode.EXECUTE, ThoughtMode.PLAN),
        (ThoughtMode.EXPERIMENT, ThoughtMode.CAUSAL),
    ],
)
def test_new_gap_can_interrupt_an_unissued_action(mode, parent):
    state = ThoughtState(mode=mode)
    state.plan = [{"action_id": 1}] if mode == ThoughtMode.EXECUTE else []
    state.experiment = {"prediction": "moves"} if mode == ThoughtMode.EXPERIMENT else None
    state.ask("Is this tile an obstacle or a switch?")
    assert state.mode == ThoughtMode.CAUSAL
    assert state.questions[-1].return_mode == parent
    assert not state.plan
    assert state.experiment is None
