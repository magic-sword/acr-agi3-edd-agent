"""Contract tests for epistemic-prober meta-skill (3 Positive + 3 Negative)."""

import sys
from pathlib import Path
import pytest

skill_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(skill_root / "scripts"))

from epistemic_prober import EpistemicProber


@pytest.fixture
def prober():
    return EpistemicProber()


# ==========================================
# Positive Contract Tests (3件)
# ==========================================

def test_positive_actuator_probing_selection(prober):
    """Positive 1: アクチュエータ（ピストンレール）存在時の認識論的プローブ行動選定."""
    actions = [10, 20, 30]
    affordances = {"has_actuator_rail": True}
    res = prober.evaluate_and_propose_probe(step_index=1, available_actions=actions, unexplored_affordances=affordances)

    assert res["success"] is True
    assert res["is_epistemic"] is True
    assert res["probe_type"] == EpistemicProber.PROBE_ACTUATOR_REACH
    assert res["recommended_action_id"] == 10
    assert "piston" in res["hypothesis"].lower()


def test_positive_mirrored_linkage_probing(prober):
    """Positive 2: 鏡面連動オブジェクト存在時の連動性検証プローブ選定."""
    actions = [1, 2, 3, 4]
    affordances = {"has_linked_objects": True}
    res = prober.evaluate_and_propose_probe(step_index=2, available_actions=actions, unexplored_affordances=affordances)

    assert res["success"] is True
    assert res["is_epistemic"] is True
    assert res["probe_type"] == EpistemicProber.PROBE_LINKAGE_DYNAMICS
    assert "linkage" in res["target_variable"].lower()


def test_positive_pragmatic_transition_after_exploration(prober):
    """Positive 3: 探索フェーズ完了後（ステップ10以降、未知要素なし）に実利行動へ移行すること."""
    actions = [1, 2, 3]
    affordances = {}
    res = prober.evaluate_and_propose_probe(step_index=10, available_actions=actions, unexplored_affordances=affordances)

    assert res["success"] is True
    assert res["is_epistemic"] is False
    assert res["probe_type"] == EpistemicProber.PRAGMATIC_EXPLOIT
    assert res["epistemic_ratio_recommendation"] <= 0.2


# ==========================================
# Negative Contract Tests (3件)
# ==========================================

def test_negative_no_available_actions_safe_handling(prober):
    """Negative 1: 利用可能な行動リストが空の時、例外を出さずエラーを報告すること."""
    res = prober.evaluate_and_propose_probe(step_index=1, available_actions=[])

    assert res["success"] is False
    assert "No available actions" in res["error"]
    assert res["recommended_action_id"] is None


def test_negative_none_affordances_graceful_default(prober):
    """Negative 2: affordancesがNoneの場合でもデフォルト探索手へ安全に縮退すること."""
    res = prober.evaluate_and_propose_probe(step_index=1, available_actions=[5, 6], unexplored_affordances=None)

    assert res["success"] is True
    assert res["is_epistemic"] is True
    assert res["recommended_action_id"] in [5, 6]


def test_negative_avoid_repeating_tested_action_in_history(prober):
    """Negative 3: 過去に試行済みの行動を避け、未検証の行動IDを選択すること."""
    actions = [1, 2, 3]
    history = [{"action": 1}]
    res = prober.evaluate_and_propose_probe(
        step_index=2,
        available_actions=actions,
        unexplored_affordances={"has_actuator_rail": True},
        transition_history=history,
    )

    assert res["success"] is True
    assert res["recommended_action_id"] == 2  # 1を避けて未試行の2を選択
