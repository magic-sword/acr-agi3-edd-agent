"""Contract tests for backward-planner meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backward_planner import BackwardPlanner


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_open_grid_shortest_path():
    """Case 1: Computes direct shortest path on open unobstructed grid."""
    grid = np.zeros((10, 10), dtype=int)
    # Start at (1, 1), Goal at (1, 4) (3 steps down)
    path, next_dir = BackwardPlanner.plan_path(grid, start_pos=(1, 1), goal_pos=(1, 4))
    assert len(path) == 4
    assert next_dir == "DOWN"
    assert path[0] == (1, 1)
    assert path[-1] == (1, 4)


def test_positive_case_2_obstacle_detour_navigation():
    """Case 2: Finds detour around an impassable horizontal wall barrier."""
    grid = np.zeros((10, 10), dtype=int)
    # Wall (color 1) across row 2, cols 0..7
    grid[2, 0:8] = 1
    # Start at (1, 1) above wall, Goal at (1, 4) below wall
    # Must detour through column 8
    path, next_dir = BackwardPlanner.plan_path(grid, start_pos=(1, 1), goal_pos=(1, 4), impassable_colors=[1])
    assert len(path) > 0
    # Next direction from (1, 1) must be RIGHT to reach column 8
    assert next_dir == "RIGHT"
    assert path[-1] == (1, 4)


def test_positive_case_3_precondition_sequencing():
    """Case 3: Correctly orders prerequisite items (key -> door -> goal)."""
    items = [
        {"type": "goal", "pos": (9, 9)},
        {"type": "door", "pos": (5, 5)},
        {"type": "key", "pos": (1, 2)},
    ]
    ordered = BackwardPlanner.sequence_preconditions(items)
    types = [it["type"] for it in ordered]
    assert types == ["key", "door", "goal"]


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_fully_enclosed_impassable_goal():
    """Case 1: When goal is completely surrounded by impassable walls, returns empty path without hanging."""
    grid = np.zeros((10, 10), dtype=int)
    # Surround (5, 5) with walls
    grid[4, 5] = 1
    grid[6, 5] = 1
    grid[5, 4] = 1
    grid[5, 6] = 1

    path, next_dir = BackwardPlanner.plan_path(grid, start_pos=(0, 0), goal_pos=(5, 5), impassable_colors=[1])
    # Cannot reach because all neighbors are blocked
    assert path == []
    assert next_dir is None


def test_negative_case_2_start_equals_goal():
    """Case 2: Start and goal at identical coordinate returns trivial path with no direction."""
    grid = np.zeros((10, 10), dtype=int)
    path, next_dir = BackwardPlanner.plan_path(grid, start_pos=(3, 3), goal_pos=(3, 3))
    assert path == [(3, 3)]
    assert next_dir is None


def test_negative_case_3_out_of_bounds_coordinates():
    """Case 3: Out-of-bounds start or goal handles gracefully returning empty path."""
    grid = np.zeros((5, 5), dtype=int)
    path, next_dir = BackwardPlanner.plan_path(grid, start_pos=(-1, 2), goal_pos=(10, 10))
    assert path == []
    assert next_dir is None
