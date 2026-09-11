#!/usr/bin/env python3
"""Failure Diagnoser - Core CLI Tool for Abstract Failure Analysis (ACR-AGI-3).

環境非依存の視点でポリシーコードの契約違反・構文例外・振る舞い停滞・不変量破綻を
抽象的に診断し、小規模ローカル LLM でも確実に解釈可能な簡潔修復ディレクティブへ蒸留します。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict


def diagnose_failure(raw_data: Any) -> Dict[str, Any]:
    """エラー情報または実行レポートを解析し、抽象故障モードと修復指示を生成."""
    if isinstance(raw_data, str):
        try:
            data = json.loads(raw_data)
        except Exception:
            data = {"error": raw_data}
    elif isinstance(raw_data, dict):
        data = raw_data
    else:
        data = {"error": str(raw_data)}

    err_msg = str(data.get("error", "")).strip()
    steps = data.get("steps_taken", data.get("steps", 0))
    code = data.get("code", "")

    # 1. 契約インターフェース違反 (Contract Violation)
    if "not defined in policy code" in err_msg or (
        code and "def choose_action" not in code and "def act" not in code
    ):
        return {
            "failure_category": "ContractViolation",
            "root_cause": "Policy function entry point missing or misnamed.",
            "directive": (
                "CONTRACT FIX: Define 'def choose_action(obs: np.ndarray, info: dict | None = None)"
                " -> Action:' returning a valid Action enum (UP, DOWN, LEFT, RIGHT, WAIT)."
            ),
            "severity": "CRITICAL",
        }

    # 2. 言語構文・実行例外 (Language / Runtime Exception)
    if "The truth value of an array with more than one element is ambiguous" in err_msg:
        return {
            "failure_category": "ArrayComparisonAmbiguity",
            "root_cause": "Direct array boolean evaluation instead of scalar extraction.",
            "directive": (
                "SYNTAX FIX: Do not use 'if (obs == color):'. Use 'indices = np.argwhere(obs =="
                " color)' and check 'if len(indices) > 0:' then extract coordinates."
            ),
            "severity": "HIGH",
        }

    if "IndentationError" in err_msg or "SyntaxError: 'return' outside function" in err_msg:
        return {
            "failure_category": "IndentationOrScopeError",
            "root_cause": "Function body indentation or return statement placement invalid.",
            "directive": (
                "SYNTAX FIX: Ensure exactly 4 spaces indentation for the entire function body."
                " All return statements must be inside 'def choose_action'."
            ),
            "severity": "HIGH",
        }

    if "SyntaxError" in err_msg or "unterminated string literal" in err_msg:
        return {
            "failure_category": "SyntaxError",
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
            "failure_category": "UndefinedSymbol",
            "root_cause": f"Reference to undefined symbol: {var_name}",
            "directive": (
                f"RUNTIME FIX: Symbol '{var_name}' is not imported or defined. Import needed"
                " modules (e.g. Action, np) or define variables before use."
            ),
            "severity": "HIGH",
        }

    # 3. 振る舞い停滞・目標未達 (Behavioral Stagnation / Timeout)
    if steps >= 40 or "timeout" in err_msg.lower() or err_msg == "None":
        return {
            "failure_category": "BehavioralStagnation",
            "root_cause": (
                f"Policy executed for {steps} steps without terminating or reaching objective."
            ),
            "directive": (
                "POLICY FIX: Agent is stagnating in loops or failing to make progress. Prioritize"
                " actions that reduce distance to intermediate milestones or unblock progression."
            ),
            "severity": "MEDIUM",
        }

    # 4. 安全不変量破綻 (Safety Invariant Breach / Trap)
    if "hazard" in err_msg.lower() or "trap" in err_msg.lower() or "fatal" in err_msg.lower():
        return {
            "failure_category": "SafetyInvariantBreach",
            "root_cause": "Action moved agent into lethal or irreversible penalty state.",
            "directive": (
                "SAFETY FIX: Avoid lethal cells or hazard colors. Check target neighbor cell"
                " safety before emitting directional move."
            ),
            "severity": "CRITICAL",
        }

    # 汎用フォールバック
    return {
        "failure_category": "GeneralFailure",
        "root_cause": err_msg or "Episode failed to satisfy clearance conditions.",
        "directive": (
            "POLICY FIX: Re-evaluate step sequence and preconditions. Ensure state changes toward"
            " active milestone."
        ),
        "severity": "LOW",
    }


def run(input_val: str | None = None) -> str:
    """Core task execution."""
    if not input_val:
        diag = diagnose_failure({"error": "No input provided"})
    else:
        diag = diagnose_failure(input_val)
    output_str = json.dumps(diag, indent=2, ensure_ascii=False)
    print(output_str)
    return output_str


def main():
    parser = argparse.ArgumentParser(description="Failure Diagnoser execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input argument")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    run(input_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
