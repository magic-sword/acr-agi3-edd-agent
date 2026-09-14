#!/usr/bin/env python3
"""Taboo Reset Guard - Unified Failure Diagnostics, No-Go Constraints & Active Reset (ACR-AGI-3).

Consolidates all failure recovery and constraint learning:
1. Failure Diagnosis (Exceptions, Stagnation, Wall Collisions, Syntax/Execution errors).
2. Taboo / Constraint Learning (Hazard coordinates, lethal colors, irreversible transitions).
3. Active Reset Evaluation (Evaluating repair cost vs fresh run cost to break deadlocks).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class TabooResetGuard:
    """ACR-AGI-3 統合失敗診断・禁忌制約学習・能動的リセット判定エンジン."""

    STATUS_OK = "OK"
    STATUS_DEADLOCK = "DEADLOCK"
    STATUS_OSCILLATION_LOOP = "OSCILLATION_LOOP"
    STATUS_COLLISION_STAGNANT = "COLLISION_STAGNANT"
    STATUS_EXCEPTION = "EXCEPTION"

    def __init__(self, loop_threshold: int = 3, collision_threshold: int = 4) -> None:
        self.loop_threshold = loop_threshold
        self.collision_threshold = collision_threshold
        self.persistent_taboo_states: Set[str] = set()
        self.taboo_colors: Set[int] = set()

    def evaluate_state_and_failure(
        self,
        recent_trajectory: List[Dict[str, Any]],
        current_grid: Optional[Any] = None,
        consecutive_ineffective_actions: int = 0,
        game_over_occurred: bool = False,
        error_trace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """直近の推移履歴およびエラーから手詰まり・例外を包括診断し、No-Go制約とリセット要否を判定."""
        # 1. 例外・エラートレースの診断
        if error_trace:
            return {
                "success": True,
                "status": self.STATUS_EXCEPTION,
                "is_reset_recommended": True,
                "attribution_reason": f"Execution exception occurred: {error_trace[:200]}",
                "new_no_go_constraint": None,
                "total_taboo_count": len(self.persistent_taboo_states),
                "suggested_action": "RESET_OR_REPAIR",
            }

        # 2. GAME OVER 時の即時学習とリセット
        if game_over_occurred:
            last_state = recent_trajectory[-1] if recent_trajectory else {}
            taboo_repr = self._make_state_key(last_state.get("grid") or current_grid)
            if taboo_repr:
                self.persistent_taboo_states.add(taboo_repr)
            return {
                "success": True,
                "status": self.STATUS_DEADLOCK,
                "is_reset_recommended": True,
                "attribution_reason": "Game over triggered. Immediate taboo recorded and reset enforced.",
                "new_no_go_constraint": taboo_repr,
                "total_taboo_count": len(self.persistent_taboo_states),
                "suggested_action": "RESET",
            }

        # 3. 衝突・無効行動の連続発生チェック
        if consecutive_ineffective_actions >= self.collision_threshold:
            taboo_repr = self._make_state_key(current_grid)
            if taboo_repr:
                self.persistent_taboo_states.add(taboo_repr)
            return {
                "success": True,
                "status": self.STATUS_COLLISION_STAGNANT,
                "is_reset_recommended": True,
                "attribution_reason": f"Detected {consecutive_ineffective_actions} consecutive ineffective steps (wall/deadlock).",
                "new_no_go_constraint": taboo_repr,
                "total_taboo_count": len(self.persistent_taboo_states),
                "suggested_action": "RESET",
            }

        # 4. 振動ループ（Oscillation Loop）の検出
        has_loop, loop_states = self._detect_loop(recent_trajectory)
        if has_loop:
            for s in loop_states:
                self.persistent_taboo_states.add(s)
            return {
                "success": True,
                "status": self.STATUS_OSCILLATION_LOOP,
                "is_reset_recommended": True,
                "attribution_reason": "Cyclic oscillation detected in state trajectory without progress.",
                "new_no_go_constraint": loop_states[-1] if loop_states else None,
                "total_taboo_count": len(self.persistent_taboo_states),
                "suggested_action": "RESET",
            }

        return {
            "success": True,
            "status": self.STATUS_OK,
            "is_reset_recommended": False,
            "attribution_reason": "Progressing normally within exploration bounds.",
            "new_no_go_constraint": None,
            "total_taboo_count": len(self.persistent_taboo_states),
            "suggested_action": "CONTINUE",
        }

    def infer_constraints(
        self,
        transitions: List[Dict[str, Any]],
        failure_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """後方互換性API: ConstraintLearner としての制約抽出."""
        for t in transitions:
            if t.get("reward", 0) < 0 or t.get("done") and not t.get("won"):
                color = t.get("stepped_color")
                if color is not None:
                    self.taboo_colors.add(int(color))

        return {
            "taboo_colors": list(self.taboo_colors),
            "persistent_taboo_count": len(self.persistent_taboo_states),
            "hazard_type": "LETHAL_OR_BLOCKING",
            "rule": "Avoid stepping on learned taboo colors and cyclic deadlocked cells",
        }

    def diagnose_failure(
        self,
        error_context: Any,
    ) -> Dict[str, Any]:
        """後方互換性API: FailureDiagnoser としての失敗原因診断."""
        if isinstance(error_context, str):
            err_str = error_context
        elif isinstance(error_context, dict):
            err_str = error_context.get("error", "Unknown failure")
        else:
            err_str = str(error_context)

        return {
            "failure_type": "RUNTIME_OR_DEADLOCK",
            "root_cause": err_str,
            "recommended_repair": "Roll back to last verified state or invoke active reset.",
            "should_reset": True,
        }

    def _make_state_key(self, grid_data: Optional[Any]) -> Optional[str]:
        if grid_data is None:
            return None
        try:
            arr = np.array(grid_data, dtype=int)
            return f"grid_hash_{hash(arr.tobytes())}"
        except Exception:
            return str(grid_data)

    def _detect_loop(
        self, trajectory: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        if len(trajectory) < 4:
            return False, []

        keys = []
        for step in trajectory[-8:]:
            g = step.get("grid") or step.get("observation")
            k = self._make_state_key(g)
            if k:
                keys.append(k)

        if len(keys) >= 4:
            if keys[-1] == keys[-3] and keys[-2] == keys[-4]:
                return True, [keys[-1], keys[-2]]
            if len(keys) >= 6 and keys[-1] == keys[-4] and keys[-2] == keys[-5] and keys[-3] == keys[-6]:
                return True, [keys[-1], keys[-2], keys[-3]]

        return False, []


# 後方互換性エイリアス
ConstraintLearner = TabooResetGuard
FailureDiagnoser = TabooResetGuard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Taboo Reset Guard - Unified Diagnostics, Taboo Constraints & Active Reset CLI"
    )
    parser.add_argument("--consecutive-ineffective", type=int, default=0, help="Consecutive blocked steps")
    parser.add_argument("--game-over", action="store_true", help="Flag indicating game over")
    parser.add_argument("--trajectory", type=str, default="[]", help="JSON list of past transitions")
    parser.add_argument("--input", type=str, help="Input data string (compatibility)")
    parser.add_argument("--file", type=str, help="Path to input JSON file")
    args = parser.parse_args()

    guard = TabooResetGuard()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            traj = data.get("trajectory", [])
            ineffective = data.get("consecutive_ineffective", 0)
            go = data.get("game_over", False)
            grid = data.get("current_grid")
    else:
        traj = json.loads(args.trajectory)
        ineffective = args.consecutive_ineffective
        go = args.game_over
        grid = None

    res = guard.evaluate_state_and_failure(
        traj, current_grid=grid, consecutive_ineffective_actions=ineffective, game_over_occurred=go
    )
    print(json.dumps(res, indent=2))


def diagnose_failure(raw_data: Any) -> Dict[str, Any]:
    """エラー情報または実行レポートを解析し、抽象故障モードと修復指示を生成 (後方互換用)."""
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

    # 1-B. Action Enum 契約違反 (ActionEnumViolation)
    if (
        ("has no attribute" in err_msg and "Action" in err_msg)
        or "is not a valid Action" in err_msg
        or ("Action" in err_msg and "not defined" in err_msg)
    ):
        return {
            "failure_category": "ActionEnumViolation",
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

    # 2-B. 空配列参照例外 (Empty Coordinates Index Error)
    if "out of bounds for axis" in err_msg or "IndexError" in err_msg:
        return {
            "failure_category": "EmptyCoordinatesIndexError",
            "root_cause": (
                "Accessing index [0] on an empty coordinate array when target/player is not found."
            ),
            "directive": (
                "RUNTIME FIX: Always verify `if len(coords) > 0:` before indexing `coords[0]`."
                " If not found, return a default safe Action (e.g. Action.WAIT or Action.RIGHT)."
            ),
            "severity": "HIGH",
        }

    # 3. 振る舞い停滞・目標未達 (Behavioral Stagnation / Timeout)
    if steps >= 30 or "timeout" in err_msg.lower() or err_msg == "None":
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


class FailureDiagnoser:
    """抽象故障診断・修復エンジン (後方互換用)."""

    def diagnose(
        self,
        error: Any = None,
        steps_taken: int = 0,
        code: str = "",
        raw_verification: Any = None,
    ) -> Dict[str, Any]:
        data = {
            "error": error,
            "steps_taken": steps_taken,
            "code": code,
            "raw_verification": raw_verification,
        }
        res = diagnose_failure(data)
        res["category"] = res.get("failure_category", "GeneralFailure")
        return res


if __name__ == "__main__":
    main()
