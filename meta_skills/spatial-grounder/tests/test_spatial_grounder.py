"""Contract tests for spatial-grounder meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spatial_grounder import SpatialGrounder


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_two_distinct_squares():
    """Case 1: Grid with two distinct colored squares correctly identified with exact centroids."""
    grid = np.zeros((20, 20), dtype=int)
    # Background is 0
    # Red square (2) at cols 2..4, rows 2..4 (center col=3, row=3)
    grid[2:5, 2:5] = 2
    # Green rectangle (3) at cols 10..14, rows 10..12 (center col=12, row=11)
    grid[10:13, 10:15] = 3

    anchors = SpatialGrounder.extract_clickable_anchors(grid)
    assert len(anchors) == 2

    # Verify colors
    colors = {a["color"] for a in anchors}
    assert colors == {2, 3}

    # Verify coordinates
    for a in anchors:
        if a["color"] == 2:
            assert a["x"] == 3 and a["y"] == 3
            assert a["area"] == 9
        elif a["color"] == 3:
            assert a["x"] == 12 and a["y"] == 11
            assert a["area"] == 15


def test_positive_case_2_nearest_snap_within_radius():
    """Case 2: Snaps loose coordinates to closest anchor centroid when within tolerance radius."""
    grid = np.zeros((30, 30), dtype=int)
    grid[10:15, 10:15] = 4  # Yellow box center at (12, 12)

    anchors = SpatialGrounder.extract_clickable_anchors(grid)
    assert len(anchors) == 1

    # Loose click at (13, 11) - within radius
    snapped_x, snapped_y, anchor_id = SpatialGrounder.snap_to_anchor(13, 11, anchors, max_dist=5.0)
    assert snapped_x == 12
    assert snapped_y == 12
    assert anchor_id == anchors[0]["id"]


def test_positive_case_3_formatted_prompt_generation():
    """Case 3: Correctly formats human and LLM readable anchor summary list."""
    grid = np.zeros((15, 15), dtype=int)
    grid[5, 5] = 1  # Blue single pixel
    anchors = SpatialGrounder.extract_clickable_anchors(grid)
    prompt_str = SpatialGrounder.format_anchors_prompt(anchors)

    assert "Clickable Target Anchors" in prompt_str
    assert "Blue" in prompt_str
    assert "col=5, row=5" in prompt_str


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_uniform_blank_grid():
    """Case 1: Completely uniform grid (all background) returns empty list safely without exception."""
    grid = np.full((20, 20), 4, dtype=int)  # All yellow background
    anchors = SpatialGrounder.extract_clickable_anchors(grid)
    assert anchors == []

    prompt = SpatialGrounder.format_anchors_prompt(anchors)
    assert "No distinct clickable anchors detected" in prompt


def test_negative_case_2_empty_or_malformed_grid():
    """Case 2: Empty grid (0x0 or None) handles edge cases gracefully."""
    empty_grid = np.array([[]], dtype=int)
    anchors = SpatialGrounder.extract_clickable_anchors(empty_grid)
    assert anchors == []

    # Snap on empty anchors returns original coordinates
    sx, sy, aid = SpatialGrounder.snap_to_anchor(7, 8, [])
    assert sx == 7 and sy == 8
    assert aid is None


def test_negative_case_3_snap_out_of_radius_returns_original():
    """Case 3: Loose click far beyond max_dist returns original coordinates instead of false snapping."""
    anchors = [{"id": 0, "x": 10, "y": 10, "color": 2, "area": 4}]
    # Click at (25, 25), distance > 20
    sx, sy, aid = SpatialGrounder.snap_to_anchor(25, 25, anchors, max_dist=5.0)
    assert sx == 25 and sy == 25
    assert aid is None
