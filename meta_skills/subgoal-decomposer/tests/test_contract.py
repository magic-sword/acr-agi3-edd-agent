"""Contract tests for subgoal-decomposer meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from subgoal_decomposer import SubgoalDecomposerCore


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_precondition_dependency_ordering():
    """Case 1: Orders keys before doors and doors before exit goals."""
    core = SubgoalDecomposerCore()
    entities = [
        {"id": "goal_1", "type": "goal", "x": 9, "y": 9},
        {"id": "door_1", "type": "door", "x": 5, "y": 5},
        {"id": "key_1", "type": "key", "x": 1, "y": 2},
    ]
    res = core.decompose_hierarchical_subgoals(entities)
    assert res["status"] == "ok"
    assert res["subgoal_count"] == 3

    subgoals = res["subgoals"]
    assert subgoals[0]["type"] == "precondition_interaction"  # key first
    assert subgoals[0]["target_entity"] == "key_1"
    assert subgoals[1]["type"] == "barrier_clearance"          # door second
    assert subgoals[1]["target_entity"] == "door_1"
    assert subgoals[2]["type"] == "reach_goal"                 # goal last
    assert subgoals[2]["target_entity"] == "goal_1"


def test_positive_case_2_buffer_staging_subgoal_generation():
    """Case 2: Allocates a buffer staging subgoal when blocks are disordered."""
    core = SubgoalDecomposerCore()
    entities = [
        {"id": "block_red", "type": "block", "color": 2, "x": 3, "y": 3},
        {"id": "block_blue", "type": "block", "color": 1, "x": 4, "y": 3},
    ]
    target_pattern = [
        {"color": 2, "x": 5, "y": 5},
        {"color": 1, "x": 6, "y": 5},
    ]
    res = core.decompose_hierarchical_subgoals(entities, target_pattern=target_pattern)
    assert res["status"] == "ok"
    subgoals = res["subgoals"]
    assert any(s["type"] == "stage_buffer" for s in subgoals)


def test_positive_case_3_sequential_subgoal_advancement():
    """Case 3: Tracks and advances active subgoals until completion."""
    core = SubgoalDecomposerCore()
    entities = [
        {"id": "k1", "type": "key", "x": 1, "y": 1},
        {"id": "g1", "type": "goal", "x": 8, "y": 8},
    ]
    core.decompose_hierarchical_subgoals(entities)
    assert core.get_active_subgoal()["type"] == "precondition_interaction"

    # Advance 1st
    adv1 = core.advance_subgoal("subgoal_1")
    assert adv1["status"] == "ok"
    assert adv1["is_all_completed"] is False
    assert adv1["next_subgoal"]["type"] == "reach_goal"

    # Advance 2nd
    adv2 = core.advance_subgoal("subgoal_2")
    assert adv2["status"] == "ok"
    assert adv2["is_all_completed"] is True
    assert core.get_active_subgoal() is None


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_reject_circular_dependency():
    """Case 4: Rejects entities containing self-referencing circular dependency."""
    core = SubgoalDecomposerCore()
    entities = [
        {"id": "key_loop", "type": "key", "requires": "key_loop"},
    ]
    res = core.decompose_hierarchical_subgoals(entities)
    assert res["status"] == "error"
    assert "Circular dependency" in res["message"]


def test_negative_case_2_empty_entities_error():
    """Case 5: Returns error on empty entity list."""
    core = SubgoalDecomposerCore()
    res = core.decompose_hierarchical_subgoals([])
    assert res["status"] == "error"
    assert "empty" in res["message"]


def test_negative_case_3_advance_exhausted_subgoals():
    """Case 6: Returns exhausted status when attempting to advance past final subgoal."""
    core = SubgoalDecomposerCore()
    entities = [{"id": "g1", "type": "goal", "x": 5, "y": 5}]
    core.decompose_hierarchical_subgoals(entities)
    core.advance_subgoal("subgoal_1")

    # Already completed, advance again
    adv = core.advance_subgoal()
    assert adv["status"] == "exhausted"
