"""環境非依存の抽象故障診断エンジン (Abstract Failure Diagnoser).

ポリシー実行失敗（契約違反・構文例外・振る舞い停滞・安全不変量破綻）を
環境非依存の視点で解析し、LLM のコンテキストを浪費しない簡潔な修復ディレクティブへ蒸留します。
"""

from __future__ import annotations

import re
from typing import Any, Dict


class FailureDiagnoser:
    """抽象故障診断メタスキル."""

    def diagnose(
        self,
        error: str | None = None,
        steps_taken: int = 0,
        code: str = "",
        raw_verification: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """エラーと実行履歴から抽象故障カテゴリと修復ディレクティブを生成."""
        err_msg = (error or "").strip()
        if not err_msg and raw_verification:
            err_msg = str(raw_verification.get("error", "")).strip()

        # 1. 契約インターフェース違反 (Contract Violation)
        if "not defined in policy code" in err_msg or (
            code and "def choose_action" not in code and "def act" not in code
        ):
            return {
                "category": "ContractViolation",
                "root_cause": "Policy function entry point missing or misnamed.",
                "directive": (
                    "CONTRACT FIX: Define 'def choose_action(obs: np.ndarray, info: dict | None ="
                    " None) -> Action:' returning a valid Action enum (UP, DOWN, LEFT, RIGHT,"
                    " WAIT)."
                ),
                "severity": "CRITICAL",
            }

        # 1-B. Action Enum 契約違反 (ActionEnumViolation)
        if (
            ("has no attribute" in err_msg and "Action" in err_msg)
            or "is not a valid Action" in err_msg
            or ("Action" in err_msg and "not defined" in err_msg)
        ):
            return {
                "category": "ActionEnumViolation",
                "root_cause": (
                    "Returned invalid Action enum value, coordinate tuple, or non-existent action"
                    " method."
                ),
                "directive": (
                    "CONTRACT FIX: Return ONLY a valid Action enum (Action.UP, Action.DOWN,"
                    " Action.LEFT, Action.RIGHT, Action.WAIT). Do NOT invent custom methods (e.g."
                    " Action.MOVE_TO) or return coordinate tuples."
                ),
                "severity": "CRITICAL",
            }

        # 2. 言語構文・実行例外 (Language / Runtime Exception)
        if "The truth value of an array with more than one element is ambiguous" in err_msg:
            return {
                "category": "ArrayComparisonAmbiguity",
                "root_cause": "Direct array boolean evaluation instead of scalar extraction.",
                "directive": (
                    "SYNTAX FIX: Do not use 'if (obs == color):'. Use 'indices = np.argwhere(obs"
                    " == color)' and check 'if len(indices) > 0:' then extract coordinates."
                ),
                "severity": "HIGH",
            }

        if "IndentationError" in err_msg or "SyntaxError: 'return' outside function" in err_msg:
            return {
                "category": "IndentationOrScopeError",
                "root_cause": "Function body indentation or return statement placement invalid.",
                "directive": (
                    "SYNTAX FIX: Ensure exactly 4 spaces indentation for the entire function"
                    " body. All return statements must be inside 'def choose_action'."
                ),
                "severity": "HIGH",
            }

        if "SyntaxError" in err_msg or "unterminated string literal" in err_msg:
            return {
                "category": "SyntaxError",
                "root_cause": f"Malformed Python syntax: {err_msg}",
                "directive": (
                    "SYNTAX FIX: Clean up incomplete syntax or unclosed quotes. Output ONLY valid"
                    " Python code within ```python ``` blocks."
                ),
                "severity": "HIGH",
            }

        if "NameError" in err_msg:
            missing_var = re.findall(r"name '(\w+)' is not defined", err_msg)
            var_name = missing_var[0] if missing_var else "variable"
            return {
                "category": "UndefinedSymbol",
                "root_cause": f"Reference to undefined symbol: {var_name}",
                "directive": (
                    f"RUNTIME FIX: Symbol '{var_name}' is not imported or defined. Import needed"
                    " modules (e.g. Action, np) or define variables before use."
                ),
                "severity": "HIGH",
            }

        # 3. 振る舞い停滞・目標未達 (Behavioral Stagnation / Timeout)
        if steps_taken >= 40 or "timeout" in err_msg.lower() or err_msg == "None":
            return {
                "category": "BehavioralStagnation",
                "root_cause": (
                    f"Policy executed for {steps_taken} steps without satisfying clear conditions."
                ),
                "directive": (
                    "POLICY FIX: Agent is stagnating in loops or failing to make progress."
                    " Prioritize actions that reduce distance to intermediate milestones or"
                    " unblock progression."
                ),
                "severity": "MEDIUM",
            }

        # 4. 安全不変量破綻 (Safety Invariant Breach / Trap)
        if "hazard" in err_msg.lower() or "trap" in err_msg.lower() or "fatal" in err_msg.lower():
            return {
                "category": "SafetyInvariantBreach",
                "root_cause": "Action moved agent into lethal or irreversible penalty state.",
                "directive": (
                    "SAFETY FIX: Avoid lethal cells or hazard colors. Check target neighbor cell"
                    " safety before issuing directional move."
                ),
                "severity": "CRITICAL",
            }

        # 汎用フォールバック
        return {
            "category": "GeneralFailure",
            "root_cause": err_msg or "Episode failed to satisfy clearance conditions.",
            "directive": (
                "POLICY FIX: Re-evaluate step sequence and preconditions. Ensure state changes"
                " toward active milestone."
            ),
            "severity": "LOW",
        }
