"""FailureDiagnoser メタスキルの抽象故障診断テスト."""

import pytest

from acr_agi3.meta.diagnoser import FailureDiagnoser


@pytest.fixture
def diagnoser():
    return FailureDiagnoser()


def test_diagnose_contract_violation(diagnoser):
    """契約違反 (choose_action 不在) の抽象診断テスト."""
    res = diagnoser.diagnose(error="Function 'choose_action' or 'act' not defined in policy code.")
    assert res["category"] == "ContractViolation"
    assert "CONTRACT FIX" in res["directive"]
    assert res["severity"] == "CRITICAL"


def test_diagnose_numpy_array_ambiguity(diagnoser):
    """NumPy 配列比較真偽値エラーの抽象診断テスト."""
    err = (
        "Policy execution error at step 1: The truth value of an array with more than one"
        " element is ambiguous. Use a.any() or a.all()"
    )
    res = diagnoser.diagnose(error=err)
    assert res["category"] == "ArrayComparisonAmbiguity"
    assert "np.argwhere" in res["directive"]
    assert res["severity"] == "HIGH"


def test_diagnose_indentation_and_syntax_error(diagnoser):
    """構文エラー・インデントエラーの抽象診断テスト."""
    res1 = diagnoser.diagnose(error="SyntaxError: 'return' outside function (<string>, line 1)")
    assert res1["category"] == "IndentationOrScopeError"
    assert "4 spaces" in res1["directive"]

    res2 = diagnoser.diagnose(error="SyntaxError: invalid syntax (<string>, line 1)")
    assert res2["category"] == "SyntaxError"
    assert "SYNTAX FIX" in res2["directive"]


def test_diagnose_behavioral_stagnation(diagnoser):
    """振る舞い停滞・タイムアウトの抽象診断テスト."""
    res = diagnoser.diagnose(error="None", steps_taken=40)
    assert res["category"] == "BehavioralStagnation"
    assert "stagnating in loops" in res["directive"]
    assert res["severity"] == "MEDIUM"


def test_diagnose_safety_invariant_breach(diagnoser):
    """即死トラップ・安全不変量破綻の抽象診断テスト."""
    res = diagnoser.diagnose(error="Action stepped into lethal hazard trap.")
    assert res["category"] == "SafetyInvariantBreach"
    assert "SAFETY FIX" in res["directive"]
    assert res["severity"] == "CRITICAL"
