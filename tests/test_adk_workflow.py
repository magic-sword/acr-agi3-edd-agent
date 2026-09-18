"""Tests for Google ADK 2.0 Plan-Act Workflow."""

import json
from typing import Any, Dict, List
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.agent.workflow_schemas import PlanProposal
from acr_agi3.harness.game_action_tools import ActionDecision


def test_plan_proposal_parsing():
    """PlanProposal の堅牢な抽出・パース検証."""
    raw = (
        "Based on the visual board, I observe the player and target.\n"
        "```json\n"
        "{\n"
        '  "hypothesis": "Red dot is player, green is goal",\n'
        '  "goal": "Move toward green goal",\n'
        '  "action": "UP",\n'
        '  "reasoning": "Move up to reduce vertical distance"\n'
        "}\n"
        "```"
    )
    p = PlanProposal.from_text(raw)
    assert p.action == "UP"
    assert "Red dot" in p.hypothesis
    assert "Move toward" in p.goal

    # クリック座標付き
    click_raw = (
        '```json\n{"action": "click_at", "x": 12, "y": 7, "reasoning": "Click switch"}\n```'
    )
    p_click = PlanProposal.from_text(click_raw)
    assert p_click.action == "click_at"
    assert p_click.coordinates == {"x": 12, "y": 7}


def test_adk_workflow_direct_execution():
    """Planner の計画を直接アクションへ実行する単一推論フロー."""
    responses = [
        json.dumps({
            "hypothesis": "Red agent at (5, 5), goal at (5, 2)",
            "goal": "Move up toward goal",
            "action": "UP",
            "reasoning": "Direct path up",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        return responses[0]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_player", app_name="test_app_1")
    grid = np.zeros((10, 10), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert decision.action_id == 1
    assert decision.action_name == "ACTION1"
    assert "Direct path up" in decision.reasoning


def test_adk_workflow_click_coordinate_auto_snap():
    """Planner が座標なしで ACTION6 (Click) を選択した場合、幾何オートスナップで座標が補完されること."""
    responses = [
        json.dumps({
            "hypothesis": "Interactive object detected",
            "goal": "Interact with tile",
            "action": "ACTION6",
            "reasoning": "Click to toggle",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        return responses[0]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_player_click", app_name="test_app_2")
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[3, 4] = 3  # インタラクティブ対象

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4, 6])
    assert decision.action_type == "CLICK"
    assert decision.action_id == 6
    assert decision.coordinates is not None
    assert "x" in decision.coordinates and "y" in decision.coordinates
