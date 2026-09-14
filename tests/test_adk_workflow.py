"""Tests for Google ADK 2.0 Plan-Review-Act Workflow."""

import json
from typing import Any, Dict, List
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.agent.workflow_schemas import PlanProposal, ReviewFeedback
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


def test_review_feedback_parsing():
    """ReviewFeedback の承認・差し戻しパース検証."""
    approved_raw = (
        '```json\n{"status": "APPROVED", "critique": "Plan is clear and directed toward goal"}\n```'
    )
    r_app = ReviewFeedback.from_text(approved_raw)
    assert r_app.is_approved is True
    assert "clear" in r_app.critique

    revise_raw = (
        '```json\n'
        '{\n'
        '  "status": "REVISE",\n'
        '  "critique": "Action is ACTION6 click but coordinates are missing!",\n'
        '  "suggested_fix": "Specify x and y coordinates"\n'
        '}\n'
        '```'
    )
    r_rev = ReviewFeedback.from_text(revise_raw)
    assert r_rev.is_approved is False
    assert "missing" in r_rev.critique
    assert "Specify" in r_rev.suggested_fix


def test_adk_workflow_approved_execution():
    """Planner の妥当な計画を Reviewer が即時承認してアクションを実行する正常系フロー."""
    call_idx = 0
    responses = [
        # Planner: UP 計画
        json.dumps({
            "hypothesis": "Red agent at (5, 5), goal at (5, 2)",
            "goal": "Move up toward goal",
            "action": "UP",
            "reasoning": "Direct path up",
        }),
        # Reviewer: APPROVED
        json.dumps({
            "status": "APPROVED",
            "critique": "Sound logical path",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_player", app_name="test_app_1")
    grid = np.zeros((10, 10), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4])
    assert decision.action_id == 1
    assert decision.action_name == "ACTION1"
    assert "Move up toward goal" in decision.reasoning


def test_adk_workflow_click_coordinate_revision():
    """Reviewer が座標欠落を指摘し、Planner が修正してクリックを実行する自己改善フロー."""
    call_idx = 0
    responses = [
        # 1. Planner: ACTION6 を座標なしで提案
        json.dumps({
            "hypothesis": "Interactive object detected",
            "goal": "Interact with tile",
            "action": "ACTION6",
            "reasoning": "Click to toggle",
        }),
        # 2. Reviewer: REVISE: coordinates missing
        json.dumps({
            "status": "REVISE",
            "critique": "ACTION6 is click but no coordinates provided",
            "suggested_fix": "Add numeric coordinates x and y",
        }),
        # 3. Planner: 修正計画: coordinates {"x": 4, "y": 6} を追加
        json.dumps({
            "hypothesis": "Interactive object at (4, 6)",
            "goal": "Click toggle button",
            "action": "click_at",
            "coordinates": {"x": 4, "y": 6},
            "reasoning": "Targeting switch at (4, 6)",
        }),
    ]

    def mock_fn(prompt: str) -> str:
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_fn)
    player = ADKGamePlayer(model=mock_llm, name="test_player_rev", app_name="test_app_2")
    grid = np.zeros((10, 10), dtype=np.uint8)

    decision = player.decide_next_action(grid=grid, available_actions=[1, 2, 3, 4, 6])
    assert decision.action_type == "CLICK"
    assert decision.action_id == 6
    assert decision.coordinates == {"x": 4, "y": 6}
