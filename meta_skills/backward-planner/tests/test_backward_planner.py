"""Contract tests for backward-planner meta-skill (3 Positive + 3 Negative)."""

import sys
from pathlib import Path
import pytest

skill_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(skill_root / "scripts"))

from backward_planner import BackwardPlanner


@pytest.fixture
def planner():
    return BackwardPlanner()


# ==========================================
# Positive Contract Tests (3件)
# ==========================================

def test_positive_reversal_puzzle_backward_chaining(planner):
    """Positive 1: 順序逆転パズル（4,3,2,1 -> 1,2,3,4）における待避バッファ付き逆算計画の生成."""
    current = [4, 3, 2, 1]
    target = [1, 2, 3, 4]
    buffers = [[0, 0], [0, 1]]
    res = planner.plan_backward_subgoals(current, target, open_grid_spaces=buffers)

    assert res["success"] is True
    assert res["backward_chaining_applied"] is True
    assert res["buffer_allocated"] is True
    assert res["total_subgoals"] >= 4

    # 第1サブゴールが一時待避であること
    first_sg = res["subgoals"][0]
    assert first_sg["phase"] == "ISOLATE_AND_STAGE"
    assert first_sg["tolerates_temporary_disorder"] is True


def test_positive_already_solved_sequence_bypass(planner):
    """Positive 2: すでに目標と一致している場合に余計な待避を行わず完了サブゴールを出力."""
    current = [1, 2, 3]
    target = [1, 2, 3]
    res = planner.plan_backward_subgoals(current, target)

    assert res["success"] is True
    assert res["is_already_solved"] is True
    assert len(res["subgoals"]) == 1
    assert res["subgoals"][0]["action_type"] == "TRIGGER_COMPLETION"


def test_positive_custom_buffer_coordinate_propagation(planner):
    """Positive 3: 指定された待避バッファ座標が正しく第1サブゴールに伝播すること."""
    current = [3, 2, 1]
    target = [1, 2, 3]
    custom_buffer = [[5, 5]]
    res = planner.plan_backward_subgoals(current, target, open_grid_spaces=custom_buffer)

    assert res["success"] is True
    assert res["subgoals"][0]["staging_buffer_coords"] == [5, 5]


# ==========================================
# Negative Contract Tests (3件)
# ==========================================

def test_negative_empty_target_sequence_handling(planner):
    """Negative 1: 空の目標シーケンスに対して例外を出さずエラーを返却すること."""
    res = planner.plan_backward_subgoals([1, 2], [])

    assert res["success"] is False
    assert "Target sequence cannot be empty" in res["error"]
    assert res["subgoals"] == []


def test_negative_empty_current_sequence_handling(planner):
    """Negative 2: 空の現在シーケンスに対して安全にエラーを返却すること."""
    res = planner.plan_backward_subgoals([], [1, 2])

    assert res["success"] is False
    assert "Current sequence cannot be empty" in res["error"]


def test_negative_none_buffer_spaces_fallback_default(planner):
    """Negative 3: buffer spacesがNoneの場合でも内部デフォルトバッファを割り当てて進行すること."""
    res = planner.plan_backward_subgoals([2, 1], [1, 2], open_grid_spaces=None)

    assert res["success"] is True
    assert res["buffer_allocated"] is True
    assert len(res["staging_buffers"]) >= 1
