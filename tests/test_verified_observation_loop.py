"""Live contracts: three positive/negative cases per changed skill, plus input guards."""

import numpy as np
import pytest
from google.adk.models import LlmRequest
from google.genai.types import Content, Part

from acr_agi3.agent.deliberation import ThoughtMode
from acr_agi3.agent.deliberative_player import DeliberativeGamePlayer
from acr_agi3.agent.llm.local_vlm import LocalQwenVL


def player():
    p = DeliberativeGamePlayer(model=LocalQwenVL("mock", generate_fn=lambda *_: ""))
    grid = np.zeros((8, 8), dtype=int)
    grid[3, 4] = 4
    p._loaded_skills.update({"visual-inspector", "causal-deliberation", "game-controller"})
    p.screen.publish(grid)
    p.available_actions = [1, 6]
    p.actions.set_available_actions([1, 6])
    p.actions.set_grid(grid)
    return p


def prepare_experiment(p):
    p.observe_screen()
    p.need_causal_knowledge("Does this switch open the passage?")
    p.need_experiment(
        "switch opens passage",
        "passage opens",
        "switch changes color only",
        expected_visual_change=True,
    )


@pytest.mark.parametrize("view", ["current", "both", "crop"])
def test_controller_positive_verified_click(view):
    p = player()
    prepare_experiment(p)
    p.move_cursor(4, 3, "Aim at the yellow switch")
    assert p.decision is None
    if view == "crop":
        p.observe_screen(x=3, y=2, width=3, height=3)
    else:
        p.observe_screen(view=view)
    assert p.click_at_cursor("Reticle centered on yellow switch", "Test switch")["scheduled"]
    assert p.decision.coordinates == {"x": 4, "y": 3}
    assert len(p.actions.history) == 1


@pytest.mark.parametrize("invalidator", ["no_look", "move_again", "new_frame"])
def test_controller_negative_unverified_click(invalidator):
    p = player()
    prepare_experiment(p)
    p.move_cursor(4, 3, "Aim")
    if invalidator != "no_look":
        p.observe_screen()
    if invalidator == "move_again":
        p.move_cursor(5, 3, "Adjust aim")
    if invalidator == "new_frame":
        p.screen.publish(np.zeros((8, 8), dtype=int))
    assert "error" in p.click_at_cursor("aligned", "test")
    assert p.decision is None
    assert not p.actions.history


@pytest.mark.parametrize("mode", [ThoughtMode.PLAN, ThoughtMode.CAUSAL, ThoughtMode.REVIEW])
def test_visual_positive_on_demand_reticle(mode):
    p = player()
    p.state.mode = mode
    p.move_cursor(4, 3, "Inspect candidate")
    observation = p.observe_screen(x=3, y=2, width=3, height=3)
    img = observation["screen_observation"]["images"][0]
    assert img["cursor"] == [4, 3]
    request = LlmRequest(
        contents=[
            Content(
                role="user",
                parts=[Part.from_function_response(name="observe_screen", response=observation)],
            )
        ]
    )
    messages, images = p.model._build_qwen_messages(request)
    assert len(images) == 1
    assert images[0].size == (36, 36)
    # Reticle center is yellow, not just the unmodified black board.
    assert images[0].getpixel((17, 17)) == (255, 255, 0)
    assert p.state.mode == mode
    assert p.decision is None
    assert not p.actions.history


@pytest.mark.parametrize("view", ["previous", "excluding_crop", "invalid_crop"])
def test_visual_negative_look_does_not_verify_invisible_cursor(view):
    p = player()
    p.screen.publish(np.zeros((8, 8), dtype=int))
    p.move_cursor(4, 3, "Aim")
    if view == "previous":
        p.observe_screen(view="previous")
    elif view == "excluding_crop":
        p.observe_screen(x=0, y=0, width=2, height=2)
    else:
        assert "error" in p.observe_screen(width=99)
    with pytest.raises(ValueError):
        p.screen.require_observed_cursor()


@pytest.mark.parametrize("outcome", ["supported", "refuted", "inconclusive"])
def test_causal_positive_evidence_changes_next_decision(outcome):
    p = player()
    prepare_experiment(p)
    p.step_action(1, "Test control")
    p.decision = None
    p.screen.publish(np.ones((8, 8), dtype=int))
    p.observe_screen(view="both")
    review = p.assess_result(outcome, "Specific observed target relation", 1, 2)
    assert outcome in review["mode_guidance"]
    assert "result_1" in review["mode_guidance"]
    if outcome != "inconclusive":
        result = p.resolve_question(
            "Conditional relation", "switch accessible", "observed effect", ["result_1"]
        )
        assert result["mode"] == "PLAN"
        assert not p.state.questions
    else:
        assert (
            p.need_experiment(
                "different hypothesis", "different effect", "no effect", expected_visual_change=True
            )["mode"]
            == "EXPERIMENT"
        )


@pytest.mark.parametrize("case", ["same_predictions", "invented_result", "unchanged_retry"])
def test_causal_negative_invalid_evidence_or_retry(case):
    p = player()
    p.observe_screen()
    p.need_causal_knowledge("What opens the passage?")
    if case == "same_predictions":
        assert "error" in p.need_experiment(
            "switch", "no change", "no change", expected_visual_change=True
        )
    elif case == "invented_result":
        assert "error" in p.resolve_question("opens", "pressed", "open", ["fake"])
    else:
        p.need_experiment("switch", "open", "closed", expected_visual_change=True)
        p.step_action(1, "test")
        p.decision = None
        p.screen.publish(p.screen.frames[1])
        p.observe_screen(view="both")
        p.assess_result("refuted", "No passage opened", 1, 2)
        p.need_experiment("switch", "open", "closed", expected_visual_change=True)
        assert "error" in p.step_action(1, "repeat unchanged experiment")
        assert p.decision is None


@pytest.mark.parametrize(
    "output",
    [
        "click_at(x=4, y=3)",
        "call step_action(1)",
        "load visual-inspector",
        '<tool_call>{"name":"resolve_question","arguments":{"effect":"moved"}}</tool_call>',
    ],
)
def test_adapter_never_invents_or_reinterprets_calls(output):
    model = LocalQwenVL("mock")
    assert model._detect_tool_call(output, available_tools={"assess_result"}) is None


def test_latest_observation_is_unique_and_survives_long_tool_history():
    p = player()
    contents = [Content(role="user", parts=[Part.from_text(text="Goal and pending question")])]
    for _ in range(2):
        contents.append(
            Content(
                role="user",
                parts=[
                    Part.from_function_response(name="observe_screen", response=p.observe_screen())
                ],
            )
        )
    for i in range(15):
        contents.append(
            Content(
                role="user",
                parts=[Part.from_function_response(name="reason", response={"evidence": i})],
            )
        )
    messages, images = p.model._build_qwen_messages(LlmRequest(contents=contents))
    assert len(images) == 1
    assert len(messages) > 12
    assert messages[0]["content"] == "Goal and pending question"
    assert (
        sum(
            item.get("type") == "image"
            for m in messages
            if isinstance(m["content"], list)
            for item in m["content"]
        )
        == 1
    )
    assert '"frame_id": 1' in messages[-1]["content"][0]["text"]


def test_cursor_cannot_bypass_mode_or_reset():
    p = player()
    p.move_cursor(4, 3, "Aim")
    p.observe_screen()
    assert "error" in p.click_at_cursor("Aligned", "click in PLAN")
    p.reset()
    assert p.screen.cursor is None
    assert p.screen.cursor_observed is None
    assert not hasattr(p, "click_at")


def test_latest_guidance_accompanies_reticle_image():
    p = player()
    first = p.observe_screen()
    prepare_experiment(p)
    latest = p.move_cursor(4, 3, "Aim")
    request = LlmRequest(
        contents=[
            Content(
                role="user",
                parts=[
                    Part.from_function_response(name="observe_screen", response=first),
                    Part.from_function_response(name="move_cursor", response=latest),
                ],
            )
        ]
    )
    messages, _ = p.model._build_qwen_messages(request)
    assert "EXPERIMENT" in messages[-1]["content"][-1]["text"]


def test_repeated_open_question_is_rejected_without_nested_stack_growth():
    p = player()
    p.need_causal_knowledge("Which control opens the door?")
    assert "error" in p.need_causal_knowledge("Which control opens the door?")
    assert len(p.state.questions) == 1
    assert p.state.mode == ThoughtMode.CAUSAL


def test_game_palette_preserves_all_official_color_ids():
    from arc_agi.rendering import COLOR_MAP, hex_to_rgb

    from acr_agi3.dsl.renderer import GAME_COLORS, render_grid_to_image

    assert GAME_COLORS == {key: hex_to_rgb(value) for key, value in COLOR_MAP.items()}
    image = render_grid_to_image(np.arange(16).reshape(1, 16), cell_size=12, palette=GAME_COLORS)
    assert len({image.getpixel((i * 12 + 5, 5)) for i in range(16)}) == 16
    assert image.getpixel((12 * 12 + 5, 5)) == GAME_COLORS[12]
    with pytest.raises(ValueError, match="Unmapped"):
        render_grid_to_image(np.array([[16]]), palette=GAME_COLORS)


def test_reticle_does_not_create_a_false_before_after_change():
    p = player()
    p.screen.publish(p.screen.frames[1])
    p.move_cursor(4, 3, "Aim")
    response = p.observe_screen(view="both")
    images = response["screen_observation"]["images"]
    assert images[0]["data"] == images[1]["data"]
    assert all(item["reticle_is_annotation"] for item in images)
    assert p.screen.frames[2][3, 4] == 4


def test_thought_snapshot_does_not_change_when_later_evidence_arrives():
    p = player()
    snapshot = p.state.snapshot()
    p.state.rules["new"] = {"effect": "later rule"}
    p.state.facts.append({"observation": "later frame"})
    assert snapshot["rules"] == {}
    assert snapshot["visible_facts"] == []


@pytest.mark.parametrize(
    "expected,changed", [(True, True), (False, False), (True, False), (False, True)]
)
def test_review_checks_explicit_visual_prediction(expected, changed):
    p = player()
    p.observe_screen()
    p.need_causal_knowledge("Does this control change the board?")
    p.need_experiment(
        "control effect",
        "predicted visible effect",
        "competing effect",
        expected_visual_change=expected,
    )
    p.step_action(1, "test control")
    p.decision = None
    before = p.screen.frames[1].copy()
    after = np.ones_like(before) if changed else before.copy()
    p.screen.publish(after)
    p.observe_screen(view="both")
    result = p.assess_result("supported", "observed effect", 1, 2)
    if expected == changed:
        assert result["recent_results"]["result_1"]["observed_visual_change"] == changed
        assert p.state.mode == ThoughtMode.CAUSAL
    else:
        assert "error" in result
        assert not p.state.results
        assert not p.state.rules
        assert p.state.mode == ThoughtMode.REVIEW
        assert "error" not in p.assess_result("inconclusive", "Prediction not established", 1, 2)


@pytest.mark.parametrize("expected", ["false", 1, None])
def test_experiment_requires_boolean_prediction(expected):
    p = player()
    p.observe_screen()
    p.need_causal_knowledge("Does control move the object?")
    result = p.need_experiment("movement", "moves", "stays", expected_visual_change=expected)
    assert "error" in result
    assert p.state.mode == ThoughtMode.CAUSAL
    assert p.state.experiment is None
