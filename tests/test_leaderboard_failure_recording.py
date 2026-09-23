"""The ARC evaluation adapter must distinguish a failed inference from a long search."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "leaderboard_under_test",
    Path(__file__).resolve().parents[1] / "scripts/run_local_leaderboard.py",
)
leaderboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaderboard)


def observation():
    return SimpleNamespace(
        frame=[np.zeros((8, 8), dtype=int)],
        state=leaderboard.GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        guid="test",
        available_actions=[1],
        game_id="test",
    )


class Agent:
    def __init__(self, **kwargs):
        self.player = SimpleNamespace(last_trace=[{"kind": "call", "name": "observe_screen"}])

    def choose_action(self, frames, latest_frame):
        raise RuntimeError("failed inference")


def test_inference_failure_preserves_reason_trace_and_frame(monkeypatch):
    monkeypatch.setattr(leaderboard, "FrameData", SimpleNamespace)
    env = SimpleNamespace(reset=observation)
    arcade = SimpleNamespace(make=lambda *args, **kwargs: env)
    result = leaderboard.evaluate_single_environment(arcade, Agent, "test", "TEST", 5, 80)
    assert result["steps"] == 0
    assert result["termination_reason"] == "inference_error"
    assert result["errors"][0]["type"] == "RuntimeError"
    assert "failed inference" in result["errors"][0]["traceback"]
    assert result["failure_trace"][0]["name"] == "observe_screen"
    assert len(result["observations"]) == 1


def test_environment_creation_failure_is_explicit():
    arcade = SimpleNamespace(make=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Failed to create environment"):
        leaderboard.evaluate_single_environment(arcade, Agent, "test", "TEST", 5, 80)


def test_actual_action_budget_exhaustion_is_distinct(monkeypatch):
    monkeypatch.setattr(leaderboard, "FrameData", SimpleNamespace)
    env = SimpleNamespace(reset=observation, step=lambda *args, **kwargs: observation())
    arcade = SimpleNamespace(make=lambda *args, **kwargs: env)

    class WorkingAgent(Agent):
        def choose_action(self, frames, latest_frame):
            return SimpleNamespace(value=1, name="ACTION1", reasoning={})

    result = leaderboard.evaluate_single_environment(arcade, WorkingAgent, "test", "TEST", 5, 2)
    assert result["steps"] == 2
    assert result["termination_reason"] == "action_budget_exhausted"
    assert result["errors"] == []
    assert len(result["observations"]) == 3


def test_workflow_events_are_on_disk_before_inference_failure(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(leaderboard, "FrameData", SimpleNamespace)
    env = SimpleNamespace(reset=observation)
    arcade = SimpleNamespace(make=lambda *args, **kwargs: env)
    journal = tmp_path / "events.jsonl"

    class InterruptedAgent(Agent):
        def choose_action(self, frames, latest_frame):
            self.player.trace_sink({"kind": "model_input", "frame_id": 1})
            assert json.loads(journal.read_text())["frame_id"] == 1
            raise RuntimeError("interrupted after input")

    result = leaderboard.evaluate_single_environment(
        arcade, InterruptedAgent, "test", "TEST", 5, 80, event_log_path=journal
    )
    assert result["termination_reason"] == "inference_error"
    assert json.loads(journal.read_text())["kind"] == "model_input"
