"""Contract tests for taboo-reset-guard meta-skill (3 Positive + 3 Negative)."""

import sys
from pathlib import Path
import pytest

skill_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(skill_root / "scripts"))

from taboo_reset_guard import TabooResetGuard


@pytest.fixture
def guard():
    return TabooResetGuard()


# ==========================================
# Positive Contract Tests (3件)
# ==========================================

def test_positive_consecutive_collision_reset_recommendation(guard):
    """Positive 1: 連続無効行動（壁衝突）が閾値を超えた際の能動的リセット推奨と禁忌登録."""
    current_grid = [[1, 2], [3, 4]]
    res = guard.evaluate_state_and_failure(
        recent_trajectory=[],
        current_grid=current_grid,
        consecutive_ineffective_actions=5,
        game_over_occurred=False,
    )

    assert res["success"] is True
    assert res["is_reset_recommended"] is True
    assert res["status"] == TabooResetGuard.STATUS_COLLISION_STAGNANT
    assert res["suggested_action"] == "RESET"
    assert res["new_no_go_constraint"] is not None
    assert res["total_taboo_count"] >= 1


def test_positive_oscillation_loop_detection(guard):
    """Positive 2: 状態振動ループ（A -> B -> A -> B）の検出と能動的リセット."""
    traj = [
        {"grid": [[1, 0]]},
        {"grid": [[0, 1]]},
        {"grid": [[1, 0]]},
        {"grid": [[0, 1]]},
    ]
    res = guard.evaluate_state_and_failure(
        recent_trajectory=traj,
        consecutive_ineffective_actions=0,
        game_over_occurred=False,
    )

    assert res["success"] is True
    assert res["is_reset_recommended"] is True
    assert res["status"] == TabooResetGuard.STATUS_OSCILLATION_LOOP
    assert "Cyclic" in res["attribution_reason"]


def test_positive_game_over_immediate_taboo_registration(guard):
    """Positive 3: GAME OVER 発生時の即時禁忌登録とリセット."""
    traj = [{"grid": [[9, 9]]}]
    res = guard.evaluate_state_and_failure(
        recent_trajectory=traj,
        consecutive_ineffective_actions=0,
        game_over_occurred=True,
    )

    assert res["success"] is True
    assert res["is_reset_recommended"] is True
    assert res["status"] == TabooResetGuard.STATUS_DEADLOCK
    assert "Game over" in res["attribution_reason"]


# ==========================================
# Negative Contract Tests (3件)
# ==========================================

def test_negative_normal_progress_no_premature_reset(guard):
    """Negative 1: 正常進行時に誤って早期リセットを推奨しないこと."""
    traj = [
        {"grid": [[1, 0]]},
        {"grid": [[2, 0]]},
        {"grid": [[3, 0]]},
    ]
    res = guard.evaluate_state_and_failure(
        recent_trajectory=traj,
        consecutive_ineffective_actions=0,
        game_over_occurred=False,
    )

    assert res["success"] is True
    assert res["is_reset_recommended"] is False
    assert res["status"] == TabooResetGuard.STATUS_OK
    assert res["suggested_action"] == "CONTINUE"


def test_negative_empty_trajectory_safe_handling(guard):
    """Negative 2: 軌跡が空の場合でも安全にCONTINUEを返却すること."""
    res = guard.evaluate_state_and_failure(
        recent_trajectory=[],
        consecutive_ineffective_actions=1,
        game_over_occurred=False,
    )

    assert res["success"] is True
    assert res["is_reset_recommended"] is False
    assert res["status"] == TabooResetGuard.STATUS_OK


def test_negative_none_grid_taboo_hash_graceful_fallback(guard):
    """Negative 3: グリッドデータがNoneでもクラッシュせず文字列キーで処理すること."""
    res = guard.evaluate_state_and_failure(
        recent_trajectory=[],
        current_grid=None,
        consecutive_ineffective_actions=5,
        game_over_occurred=False,
    )

    assert res["success"] is True
    assert res["is_reset_recommended"] is True
