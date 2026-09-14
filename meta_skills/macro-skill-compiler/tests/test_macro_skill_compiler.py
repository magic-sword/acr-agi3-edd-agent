"""Contract tests for macro-skill-compiler meta-skill (3 Positive + 3 Negative)."""

import sys
from pathlib import Path
import pytest

skill_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(skill_root / "scripts"))

from macro_skill_compiler import MacroSkillCompiler


@pytest.fixture
def compiler():
    return MacroSkillCompiler()


# ==========================================
# Positive Contract Tests (3件)
# ==========================================

def test_positive_isolate_and_stage_compilation(compiler):
    """Positive 1: Isolate-and-Stage（一時待避）マクロスキルのコンパイルと契約出力."""
    subgoal = {
        "subgoal_index": 1,
        "phase": "ISOLATE_AND_STAGE",
        "description": "Move blue block to buffer siding.",
        "target_colors": [4],
    }
    res = compiler.compile_macro_skill(subgoal)

    assert res["success"] is True
    assert res["archetype_applied"] == MacroSkillCompiler.ARCHETYPE_ISOLATE_AND_STAGE
    assert len(res["compiled_action_ids"]) >= 3
    assert "buffer" in res["contract_specification"].lower()


def test_positive_pair_and_carry_compilation(compiler):
    """Positive 2: Pair-and-Carry（剛体結合運搬）マクロスキルのコンパイル."""
    subgoal = {
        "subgoal_index": 3,
        "phase": "INCREMENTAL_CHAIN",
        "description": "Connect red and orange into a rigid pair.",
        "target_colors": [1, 2],
    }
    res = compiler.compile_macro_skill(subgoal)

    assert res["success"] is True
    assert res["archetype_applied"] == MacroSkillCompiler.ARCHETYPE_PAIR_AND_CARRY
    assert "bond" in res["contract_specification"].lower()


def test_positive_custom_action_id_mapping(compiler):
    """Positive 3: 環境固有のアクションIDマッピング（UP=10, DOWN=20等）の正しい反映."""
    custom_map = {"UP": 10, "DOWN": 20, "LEFT": 30, "RIGHT": 40, "EXTEND": 50, "RETRACT": 60}
    subgoal = {"phase": "ISOLATE_AND_STAGE"}
    res = compiler.compile_macro_skill(subgoal, available_action_mapping=custom_map)

    assert res["success"] is True
    # EXTEND (50) -> DOWN (20) -> RETRACT (60)
    assert res["compiled_action_ids"] == [50, 20, 60]


# ==========================================
# Negative Contract Tests (3件)
# ==========================================

def test_negative_empty_subgoal_graceful_fallback(compiler):
    """Negative 1: 空のサブゴール辞書に対しても例外を出さずデフォルトアーキタイプへ縮退すること."""
    res = compiler.compile_macro_skill({})

    assert res["success"] is True
    assert res["archetype_applied"] == MacroSkillCompiler.ARCHETYPE_LANE_SPLITTING
    assert len(res["compiled_action_ids"]) > 0


def test_negative_missing_keys_in_action_mapping(compiler):
    """Negative 2: 一部のアクションキーが欠損したマッピングでも安全なデフォルトIDで補完されること."""
    sparse_map = {"UP": 99}
    subgoal = {"phase": "ISOLATE_AND_STAGE"}
    res = compiler.compile_macro_skill(subgoal, available_action_mapping=sparse_map)

    assert res["success"] is True
    assert len(res["compiled_action_ids"]) == 3


def test_negative_unknown_phase_safe_routing(compiler):
    """Negative 3: 未知のフェーズ指定でもクラッシュせずレーン分割アーキタイプとして安全に解釈されること."""
    subgoal = {"phase": "TOTALLY_UNKNOWN_ACTION_PHASE"}
    res = compiler.compile_macro_skill(subgoal)

    assert res["success"] is True
    assert res["archetype_applied"] == MacroSkillCompiler.ARCHETYPE_LANE_SPLITTING
