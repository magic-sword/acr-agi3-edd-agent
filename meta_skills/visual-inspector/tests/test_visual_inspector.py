"""Contract tests for visual-inspector meta-skill (3 Positive + 3 Negative)."""

import sys
from pathlib import Path
import pytest
import numpy as np

# scripts/ ディレクトリをモジュール検索パスに追加
skill_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(skill_root / "scripts"))

from visual_inspector import VisualInspector


@pytest.fixture
def inspector():
    return VisualInspector()


# ==========================================
# Positive Contract Tests (3件)
# ==========================================

def test_positive_step_zero_pause_enforcement(inspector):
    """Positive 1: ステップ0での目視点検休止（Visual Inspection Pause）が要求されること."""
    grid = [[0, 0, 0], [1, 2, 3], [0, 0, 0]]
    target = [3, 2, 1]
    res = inspector.inspect_board(grid, target, step_index=0)

    assert res["success"] is True
    assert res["pause_required"] is True
    assert res["recommended_action"] == "NO_OP"
    assert res["diff_type"] == VisualInspector.ALIGNMENT_REVERSAL_REORDER
    assert any("buffer" in inv.lower() for inv in res["invariants_hypothesized"])


def test_positive_reversal_reorder_classification(inspector):
    """Positive 2: 順序逆転（Reversal Reorder）の正確な検出とバッファ待避不変量の導出."""
    grid = [[0, 0, 0, 0], [4, 3, 2, 1], [0, 0, 0, 0]]
    target = [1, 2, 3, 4]
    res = inspector.inspect_board(grid, target, step_index=1)

    assert res["success"] is True
    assert res["pause_required"] is False
    assert res["recommended_action"] == "PROCEED"
    assert res["diff_type"] == VisualInspector.ALIGNMENT_REVERSAL_REORDER
    assert res["current_sequence"] == [4, 3, 2, 1]
    assert res["target_sequence"] == [1, 2, 3, 4]


def test_positive_interleaved_alternating_detection(inspector):
    """Positive 3: 交互配列（Interleaved Alternating: 赤-青-赤）の検出."""
    grid = [[0, 0, 0], [1, 1, 2], [0, 0, 0]]
    target = [1, 2, 1]  # 赤-青-赤
    res = inspector.inspect_board(grid, target, step_index=1)

    assert res["success"] is True
    assert res["diff_type"] == VisualInspector.ALIGNMENT_INTERLEAVED
    assert any("bypass" in inv.lower() or "two" in inv.lower() for inv in res["invariants_hypothesized"])


# ==========================================
# Negative Contract Tests (3件)
# ==========================================

def test_negative_empty_observation_handling(inspector):
    """Negative 1: 空の観測グリッドが渡された際に例外を出さず安全にフォールバックすること."""
    empty_grid = []
    res = inspector.inspect_board(empty_grid, [1, 2, 3], step_index=0)

    assert res["success"] is False
    assert "Empty" in res["error"]
    assert res["recommended_action"] == "NO_OP"


def test_negative_missing_target_graceful_degradation(inspector):
    """Negative 2: ターゲット指定がNoneの場合でもUNKNOWNとしてクラッシュしないこと."""
    grid = [[0, 0], [1, 2]]
    res = inspector.inspect_board(grid, None, step_index=1)

    assert res["success"] is True
    assert res["diff_type"] == VisualInspector.ALIGNMENT_UNKNOWN
    assert res["target_sequence"] == []


def test_negative_single_dimension_or_degenerate_grid(inspector):
    """Negative 3: 1次元または縮退グリッドの入力に対する自動形状正規化."""
    degenerate_grid = [1, 2, 0, 3]
    res = inspector.inspect_board(degenerate_grid, [1, 2, 3], step_index=1)

    assert res["success"] is True
    assert res["grid_dimensions"] == [1, 4]
