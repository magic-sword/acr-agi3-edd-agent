#!/usr/bin/env python3
"""EDD Contract Tests for Episodic Memory Meta-Skill.

Requirements:
- Minimum 3 positive test cases
- Minimum 3 negative test cases
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from episodic_memory import EpisodicMemoryManager, Episode


# ==============================================================================
# Positive Test Cases (正常系 3件)
# ==============================================================================

def test_positive_record_and_summary_generation():
    """正例 1: ステップ記録と高密度 Working Memory 要約の生成."""
    mgr = EpisodicMemoryManager(capacity=10)
    mgr.record_step(
        step_index=0,
        action_name="UP",
        action_id=1,
        pixels_changed=3,
        is_effective=True,
        reflection="Player moved up into clear corridor",
        rule_hypothesis="Cyan block moves with UP action",
    )
    summary = mgr.get_working_memory_summary(max_recent=3)
    assert "Step 00: Action `UP`" in summary
    assert "ΔPixels: 3" in summary
    assert "Effective" in summary
    assert "Cyan block moves with UP action" in summary


def test_positive_causal_query_filtering():
    """正例 2: 条件指定（アクション名・有効性・キーワード）による過去エピソード検索."""
    mgr = EpisodicMemoryManager(capacity=10)
    mgr.record_step(step_index=0, action_name="UP", action_id=1, pixels_changed=2, is_effective=True)
    mgr.record_step(step_index=1, action_name="LEFT", action_id=3, pixels_changed=0, is_effective=False, reflection="Hit wall")
    mgr.record_step(step_index=2, action_name="UP", action_id=1, pixels_changed=1, is_effective=True)

    # 1. アクション名検索
    up_episodes = mgr.query_episodes(action_name="UP")
    assert len(up_episodes) == 2

    # 2. 有効アクションのみ検索
    eff_episodes = mgr.query_episodes(effective_only=True)
    assert len(eff_episodes) == 2

    # 3. キーワード検索
    wall_episodes = mgr.query_episodes(keyword="wall")
    assert len(wall_episodes) == 1
    assert wall_episodes[0].action_name == "LEFT"


def test_positive_serialization_roundtrip():
    """正例 3: 辞書シリアライズとデシリアライズの完全復元."""
    mgr = EpisodicMemoryManager(capacity=20)
    mgr.record_step(
        step_index=0,
        action_name="ACTION6",
        action_id=6,
        pixels_changed=5,
        is_effective=True,
        rule_hypothesis="Clicking yellow target activates door",
    )

    data = mgr.to_dict()
    new_mgr = EpisodicMemoryManager()
    new_mgr.from_dict(data)

    assert len(new_mgr.episodes) == 1
    assert new_mgr.episodes[0].action_name == "ACTION6"
    assert new_mgr.learned_rules == ["Clicking yellow target activates door"]


# ==============================================================================
# Negative Test Cases (異常系 3件)
# ==============================================================================

def test_negative_invalid_step_index_rejected():
    """負例 1: 負の step_index の登録拒否."""
    mgr = EpisodicMemoryManager()
    with pytest.raises(ValueError, match="non-negative"):
        mgr.record_step(
            step_index=-1,
            action_name="UP",
            action_id=1,
            pixels_changed=0,
            is_effective=False,
        )


def test_negative_empty_memory_summary():
    """負例 2: 記憶が存在しない場合のセーフティフォールバック要約."""
    mgr = EpisodicMemoryManager()
    summary = mgr.get_working_memory_summary()
    assert summary == "No previous steps in working memory."
    assert mgr.query_episodes(action_name="UNKNOWN") == []


def test_negative_corrupted_data_restoration_handling():
    """負例 3: 不正な型やキーを持つ破損辞書データからの安全な復元."""
    mgr = EpisodicMemoryManager()
    with pytest.raises(ValueError, match="must be a dictionary"):
        mgr.from_dict(["not", "a", "dict"])

    # 一部レコードが破損している場合でも例外でクラッシュせず健全なレコードを読み込む
    corrupted_data = {
        "capacity": 50,
        "learned_rules": ["Valid Rule"],
        "episodes": [
            {"step_index": 0, "action_name": "UP", "action_id": 1, "pixels_changed": 1, "is_effective": True, "state_before": "A", "state_after": "B", "levels_completed": 0},
            {"invalid_field": True},  # 破損レコード
        ]
    }
    mgr.from_dict(corrupted_data)
    assert len(mgr.episodes) == 1
    assert mgr.episodes[0].action_name == "UP"
    assert mgr.learned_rules == ["Valid Rule"]
