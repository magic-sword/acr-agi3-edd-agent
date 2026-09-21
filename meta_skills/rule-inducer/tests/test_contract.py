"""Contract tests for rule-inducer meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rule_inducer import RuleInducerCore


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_induce_win_condition_on_level_advance():
    """Case 1: Induces a win condition invariant when level advances."""
    core = RuleInducerCore()
    res = core.induce_rule_from_transition(
        step_index=5,
        action_name="ACTION1",
        action_id=1,
        pixels_changed=10,
        level_before=0,
        level_after=1,
    )
    assert res["status"] == "ok"
    assert res["is_new"] is True
    assert res["rule"]["rule_type"] == "win_condition"
    assert "advances level from 0 to 1" in res["rule"]["statement"]

    win_hypos = core.get_win_condition_hypotheses()
    assert len(win_hypos) == 1
    assert "advances level" in win_hypos[0]


def test_positive_case_2_induce_causal_interaction_rule():
    """Case 2: Induces a causal interaction rule when pixels change."""
    core = RuleInducerCore()
    res = core.induce_rule_from_transition(
        step_index=2,
        action_name="ACTION6",
        action_id=6,
        coords={"x": 5, "y": 7},
        pixels_changed=4,
        level_before=0,
        level_after=0,
    )
    assert res["status"] == "ok"
    assert res["rule"]["rule_type"] == "causal_interaction"
    assert "4 pixels changed" in res["rule"]["effect"]


def test_positive_case_3_induce_barrier_rule_and_query_by_type():
    """Case 3: Induces an invariant barrier rule when 0 pixels change and filters by type."""
    core = RuleInducerCore()
    res = core.induce_rule_from_transition(
        step_index=3,
        action_name="ACTION2",
        action_id=2,
        pixels_changed=0,
        level_before=0,
        level_after=0,
    )
    assert res["status"] == "ok"
    assert res["rule"]["rule_type"] == "invariant_barrier"

    barrier_rules = core.get_known_rules(rule_type="invariant_barrier")
    assert len(barrier_rules) == 1
    assert barrier_rules[0]["action_name"] == "ACTION2"


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_empty_action_name_error():
    """Case 4: Rejects transition with empty action name."""
    core = RuleInducerCore()
    res = core.induce_rule_from_transition(
        step_index=1,
        action_name="",
        action_id=1,
        pixels_changed=1,
    )
    assert res["status"] == "error"
    assert "empty" in res["message"]


def test_negative_case_2_confidence_capped_at_one():
    """Case 5: Repeated observations increase confidence but never exceed 1.0."""
    core = RuleInducerCore()
    for _ in range(10):
        core.induce_rule_from_transition(
            step_index=1,
            action_name="ACTION1",
            action_id=1,
            pixels_changed=2,
        )
    rules = core.get_known_rules(rule_type="causal_interaction")
    assert len(rules) == 1
    assert rules[0]["confidence"] <= 1.0


def test_negative_case_3_nonexistent_rule_type_returns_empty():
    """Case 6: Querying an unknown rule type returns an empty list."""
    core = RuleInducerCore()
    core.induce_rule_from_transition(step_index=1, action_name="ACTION1", action_id=1, pixels_changed=1)
    res = core.get_known_rules(rule_type="nonexistent_type")
    assert res == []
