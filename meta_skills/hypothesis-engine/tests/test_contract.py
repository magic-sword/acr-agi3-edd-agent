"""Contract tests for hypothesis-engine meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hypothesis_engine import HypothesisEngineCore


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_formulate_valid_hypothesis():
    """Case 1: Formulates a valid single-variable hypothesis successfully."""
    engine = HypothesisEngineCore()
    res = engine.formulate_hypothesis(
        step_index=1,
        claim="Clicking at (10, 15) rotates the blue block.",
        action_name="ACTION6",
        action_id=6,
        coords={"x": 10, "y": 15},
        expected_effect="Rotate block clockwise",
    )
    assert res["status"] == "ok"
    assert engine.active_hypothesis is not None
    assert engine.active_hypothesis["action_id"] == 6
    assert engine.active_hypothesis["coords"] == {"x": 10, "y": 15}


def test_positive_case_2_outcome_verified_on_pixel_change():
    """Case 2: When action produces changes (>0 pixels), hypothesis is verified."""
    engine = HypothesisEngineCore()
    engine.formulate_hypothesis(
        step_index=2,
        claim="Moving UP enters the corridor.",
        action_name="ACTION1",
        action_id=1,
        expected_effect="Displacement UP by 1 cell",
    )
    status, record = engine.evaluate_outcome(
        step_index=2,
        pixels_changed=3,
        is_effective=True,
        actual_notes="Corridor entry confirmed",
    )
    assert status == "VERIFIED"
    assert engine.active_hypothesis is None
    assert len(engine.verified_hypotheses) == 1
    assert record["pixels_changed"] == 3


def test_positive_case_3_outcome_refuted_and_bookmarked_on_zero_change():
    """Case 3: When action produces 0 pixel changes, hypothesis is refuted and bookmarked."""
    engine = HypothesisEngineCore()
    engine.formulate_hypothesis(
        step_index=3,
        claim="Clicking gray background at (0, 1) triggers interaction.",
        action_name="ACTION6",
        action_id=6,
        coords={"x": 0, "y": 1},
    )
    status, record = engine.evaluate_outcome(
        step_index=3,
        pixels_changed=0,
        is_effective=False,
    )
    assert status == "REFUTED"
    assert engine.active_hypothesis is None
    assert "hypothesis.refuted.s3_ACTION6_0_1" in engine.refuted_hypotheses
    
    # Verify lightweight TOC bookmarks
    bookmarks = engine.get_refuted_bookmarks()
    assert len(bookmarks) == 1
    assert "ACTION6 pos=(0,1)" in bookmarks[0] or "ACTION6 pos=(0, 1)" in bookmarks[0]
    assert "refuted" in bookmarks[0]


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_reject_already_refuted_hypothesis():
    """Case 4: Rejects formulating a hypothesis that targets an already-refuted coordinate/action."""
    engine = HypothesisEngineCore()
    # 1. Refute action 6 at (5, 5)
    engine.formulate_hypothesis(
        step_index=1,
        claim="Click (5, 5)",
        action_name="ACTION6",
        action_id=6,
        coords={"x": 5, "y": 5},
    )
    engine.evaluate_outcome(step_index=1, pixels_changed=0)

    # 2. Try to formulate identical hypothesis again
    res = engine.formulate_hypothesis(
        step_index=2,
        claim="Try click (5, 5) again",
        action_name="ACTION6",
        action_id=6,
        coords={"x": 5, "y": 5},
    )
    assert res["status"] == "rejected"
    assert "already been refuted" in res["message"]


def test_negative_case_2_empty_claim_or_action_returns_error():
    """Case 5: Returns error when claim or action_name is empty."""
    engine = HypothesisEngineCore()
    res1 = engine.formulate_hypothesis(
        step_index=1,
        claim="",
        action_name="ACTION1",
        action_id=1,
    )
    assert res1["status"] == "error"

    res2 = engine.formulate_hypothesis(
        step_index=1,
        claim="Valid claim",
        action_name="",
        action_id=1,
    )
    assert res2["status"] == "error"


def test_negative_case_3_evaluate_without_active_hypothesis():
    """Case 6: Gracefully handles evaluate_outcome when no hypothesis is active."""
    engine = HypothesisEngineCore()
    status, record = engine.evaluate_outcome(step_index=5, pixels_changed=0)
    assert status == "NO_ACTIVE_HYPOTHESIS"
    assert "No active hypothesis" in record["message"]
