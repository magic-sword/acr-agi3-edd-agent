"""Tests for Shared Blackboard (Memory Notebook) Integration with ADKGamePlayer.

Verifies:
1. causality.dynamics auto-sync on movement identification.
2. plan.active auto-sync on multi-step route formulation.
3. Eraser (memory_delete) cleanup on plan completion.
4. Taboo constraint recording (taboo.step_N) and plan invalidation on deviation (0-pixel change).
5. TOC generation in Markdown for low-token prompt injection.
6. Agent reset() cleanly wipes the memory notebook.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from acr_agi3.agent.adk_game_player import ADKGamePlayer, CognitiveState
from acr_agi3.agent.llm.local_model import LocalTransformersLlm


def create_mock_player(action_text: str = "ACTION1") -> ADKGamePlayer:
    def mock_fn(prompt: str) -> str:
        return f"Subgoal: test. Call step_action({action_text}).\n```json\n{{\"action\": \"{action_text}\"}}\n```"

    llm = LocalTransformersLlm("mock", generation_fn=mock_fn)
    return ADKGamePlayer(model=llm)


def test_blackboard_causality_dynamics_sync():
    """Verify causality.dynamics is automatically recorded to memory-notebook upon displacement."""
    player = create_mock_player("ACTION3")

    # Initial grid: player (color 2) at (1, 1)
    grid_0 = np.zeros((5, 5), dtype=np.uint8)
    grid_0[1, 1] = 2

    # Step 1: Execute ACTION3
    _ = player.decide_next_action(grid_0, available_actions=[1, 2, 3, 4], state_str="NOT_FINISHED")

    # Grid 1: player moved right to (1, 2)
    grid_1 = np.zeros((5, 5), dtype=np.uint8)
    grid_1[1, 2] = 2

    # Step 2: Displacement analyzed -> dynamics mapped
    _ = player.decide_next_action(grid_1, available_actions=[1, 2, 3, 4], state_str="NOT_FINISHED")

    grid_2 = np.zeros((5, 5), dtype=np.uint8)
    grid_2[1, 3] = 2
    player.decide_next_action(grid_2, available_actions=[1, 2, 3, 4])

    # Verify memory_tools contains causality.dynamics
    res = player.memory_tools.memory_read("causality.dynamics")
    data = json.loads(res)
    assert data["status"] == "ok"
    assert "causality" in data["tags"] or "dynamics" in data["tags"]
    dynamics = json.loads(data["content"])
    assert dynamics.get("RIGHT") == 3


def test_blackboard_plan_active_sync_and_eraser_on_completion():
    """Verify plan.active is written when A* route is found, and erased when finished."""
    player = create_mock_player("RIGHT")
    from acr_agi3.agent.execution_evidence import Motion
    player.execution_evidence.samples[4] = (Motion(2, 0, 1), 2)

    # Grid with player (2) at (1, 1) and goal (3) at (1, 3)
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[1, 1] = 2  # Agent
    grid[1, 3] = 3  # Goal

    # Step 1: Slow Path -> A* finds path: ['RIGHT', 'RIGHT']
    # actions[0] = 'RIGHT' returned, actions[1:] = ['RIGHT'] enqueued
    dec1 = player.decide_next_action(grid, available_actions=[1, 2, 3, 4])

    assert player.cognitive_state == CognitiveState.EXECUTING
    assert len(player.plan_queue) == 1

    # Verify plan.active exists in blackboard
    res_plan = json.loads(player.memory_tools.memory_read("plan.active"))
    assert res_plan["status"] == "ok"
    plan_content = json.loads(res_plan["content"])
    assert "planned_actions" in plan_content

    # Step 2: Fast Path -> executes remaining 'RIGHT'
    grid_next = np.zeros((5, 5), dtype=np.uint8)
    grid_next[1, 2] = 2
    grid_next[1, 3] = 3
    dec2 = player.decide_next_action(grid_next, available_actions=[1, 2, 3, 4])

    assert dec2.metadata.get("fast_path") is True
    assert len(player.plan_queue) == 0
    assert player.cognitive_state == CognitiveState.PLANNING

    # Verify plan.active was erased (deleted) from blackboard
    res_deleted = json.loads(player.memory_tools.memory_read("plan.active"))
    assert res_deleted["status"] == "error"


def test_blackboard_taboo_barrier_sync_on_deviation():
    """Verify plan.active is erased and taboo.step_N is written on wall bump (0 pixel change)."""
    player = create_mock_player("RIGHT")
    from acr_agi3.agent.execution_evidence import Motion
    player.execution_evidence.samples[4] = (Motion(2, 0, 1), 2)

    # Populate plan_queue
    player.cognitive_state = CognitiveState.EXECUTING
    player.plan_queue = [{"action": "RIGHT", "reasoning": "advance"}]
    player.last_expected_action = "RIGHT"
    player.memory_tools.memory_write(
        section_id="plan.active",
        title="Active Plan",
        content="step towards goal",
        summary="A* plan",
    )

    # Agent encounters 0 pixel change (wall bump)
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[1, 1] = 2
    player.last_grid = grid.copy()
    player.last_action_info = {"action": "RIGHT", "action_id": 1}

    # Step: stagnation_count becomes 1
    _ = player.decide_next_action(grid, available_actions=[1, 2, 3, 4])

    # Verify state transitioned to RECOVERY and queue dropped
    assert player.cognitive_state == CognitiveState.RECOVERY
    assert len(player.plan_queue) == 0

    # Verify plan.active was deleted
    res_plan = json.loads(player.memory_tools.memory_read("plan.active"))
    assert res_plan["status"] == "error"

    # Verify taboo was written
    taboo_res = json.loads(player.memory_tools.memory_read(f"taboo.step_{player.step_index}"))
    assert taboo_res["status"] == "ok"
    assert "taboo" in taboo_res["tags"]
    assert "RIGHT" in taboo_res["content"]


def test_blackboard_toc_and_reset():
    """Verify markdown TOC generation and clear() on agent reset."""
    player = create_mock_player("ACTION1")

    # Write some sections
    player.memory_tools.memory_write("rule.movement", "ACTION1 is UP", title="Movement Rule", summary="Action 1 moves up", tags="rule")
    player.memory_tools.memory_write("hypo.switch", "Blue button opens door", title="Switch Hypothesis", summary="Switch opens door", tags="hypo")

    toc_md = player.memory_tools.memory_toc(as_markdown=True)
    assert "# Memory Notebook TOC" in toc_md
    assert "rule.movement" in toc_md
    assert "hypo.switch" in toc_md

    # Episode Reset (同一ゲームリトライ): 永続ルールは保持される
    player.reset()
    assert player.step_index == 0
    toc_after_ep_reset = player.memory_tools.memory_toc(as_markdown=True)
    assert "rule.movement" in toc_after_ep_reset

    # Full Wipe Reset (環境初期化): ノートが完全に初期化される
    player.reset(full_wipe=True)
    toc_after_full_reset = player.memory_tools.memory_toc(as_markdown=True)
    assert "*(Notebook is currently empty)*" in toc_after_full_reset
