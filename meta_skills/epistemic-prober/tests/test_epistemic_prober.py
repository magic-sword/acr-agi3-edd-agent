"""Contract tests for epistemic-prober meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from epistemic_prober import EpistemicProber


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_directional_shift_detection():
    """Case 1: Correctly infers RIGHT when an object shifts horizontally rightward."""
    prober = EpistemicProber()
    b = np.zeros((10, 10), dtype=int)
    a = np.zeros((10, 10), dtype=int)
    b[4, 4] = 2  # Red object at (r=4, c=4)
    a[4, 5] = 2  # Red object moves to (r=4, c=5)

    role = prober.analyze_displacement(b, a, action_id=3)
    assert role == "RIGHT"
    assert prober.get_dynamics_map().get("RIGHT") == 3


def test_positive_case_2_probing_recommendation_flow():
    """Case 2: Correctly recommends untested action IDs at episode onset."""
    prober = EpistemicProber()
    avail = [1, 2, 3, 4]
    assert prober.is_probing_needed(step_index=1, available_actions=avail) is True
    rec = prober.recommend_probe_action(avail)
    assert rec == 1


def test_positive_case_3_click_only_skips_directional_probe():
    """Case 3: In click-only environments (available=[6]), automatically registers CLICK and skips directional probe."""
    prober = EpistemicProber()
    assert prober.is_probing_needed(step_index=1, available_actions=[6]) is False
    assert prober.get_dynamics_map().get("CLICK") == 6


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_zero_change_wall_bump():
    """Case 1: When action results in 0 pixel change (wall bump), returns None and does not map."""
    prober = EpistemicProber()
    b = np.zeros((10, 10), dtype=int)
    b[0, 0] = 2
    a = b.copy()  # No change

    role = prober.analyze_displacement(b, a, action_id=1)
    assert role is None
    assert len(prober.get_dynamics_map()) == 0


def test_negative_case_2_shape_mismatch_or_empty_grid():
    """Case 2: Grids with shape mismatch or empty content handled safely without exception."""
    prober = EpistemicProber()
    b = np.zeros((5, 5), dtype=int)
    a = np.zeros((6, 6), dtype=int)

    role = prober.analyze_displacement(b, a, action_id=2)
    assert role is None


def test_negative_case_3_probing_disabled_after_max_steps():
    """Case 3: Probing is safely disabled once step count exceeds onset threshold."""
    prober = EpistemicProber()
    avail = [1, 2, 3, 4]
    # Step 10 is way past onset probe limit (default 4)
    assert prober.is_probing_needed(step_index=10, available_actions=avail) is False
