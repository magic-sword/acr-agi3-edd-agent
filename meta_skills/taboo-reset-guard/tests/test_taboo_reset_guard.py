"""Contract tests for taboo-reset-guard meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from taboo_reset_guard import TabooResetGuard


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_veto_wall_bump_repetition():
    """Case 1: Vetoes repetition of an action that just produced 0 pixel change (wall bump)."""
    guard = TabooResetGuard()
    # Proposed ACTION1, but previous ACTION1 had 0 pixel change
    is_allowed, sanitized, reason = guard.filter_taboo_actions(
        proposed_action=1,
        last_action_effective=False,
        last_action_id=1,
        available_actions=[1, 2, 3, 4],
        stagnation_count=1,
    )
    assert is_allowed is False
    assert sanitized != 1
    assert sanitized in [2, 3, 4]
    assert "wall bump" in reason


def test_positive_case_2_detect_2_step_oscillation():
    """Case 2: Detects alternating 2-step loop [1, 2, 1, 2] and breaks oscillation."""
    guard = TabooResetGuard()
    for a in [1, 2, 1, 2]:
        guard.record_step(a)

    assert guard.is_oscillating() is True
    is_allowed, sanitized, reason = guard.filter_taboo_actions(
        proposed_action=1,
        last_action_effective=True,
        last_action_id=2,
        available_actions=[1, 2, 3, 4],
        stagnation_count=0,
    )
    assert is_allowed is False
    assert "oscillation" in reason


def test_positive_case_3_active_reset_on_prolonged_stagnation():
    """Case 3: Triggers active reset when consecutive stagnation reaches threshold (>= 6)."""
    guard = TabooResetGuard(max_stagnation=6)
    assert guard.should_active_reset(stagnation_count=6) is True
    assert guard.should_active_reset(stagnation_count=10) is True


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_normal_effective_progress():
    """Case 1: Normal effective progress allows the proposed action without modification."""
    guard = TabooResetGuard()
    is_allowed, sanitized, reason = guard.filter_taboo_actions(
        proposed_action=3,
        last_action_effective=True,
        last_action_id=3,
        available_actions=[1, 2, 3, 4],
        stagnation_count=0,
    )
    assert is_allowed is True
    assert sanitized == 3


def test_negative_case_2_single_available_action_bypass():
    """Case 2: Single available action (e.g. only click 6) bypasses taboo to avoid deadlock."""
    guard = TabooResetGuard()
    is_allowed, sanitized, reason = guard.filter_taboo_actions(
        proposed_action=6,
        last_action_effective=False,
        last_action_id=6,
        available_actions=[6],
        stagnation_count=5,
    )
    assert is_allowed is True
    assert sanitized == 6


def test_negative_case_3_no_reset_under_threshold():
    """Case 3: Zero or low stagnation count does not trigger active reset."""
    guard = TabooResetGuard(max_stagnation=6)
    assert guard.should_active_reset(stagnation_count=0) is False
    assert guard.should_active_reset(stagnation_count=4) is False
