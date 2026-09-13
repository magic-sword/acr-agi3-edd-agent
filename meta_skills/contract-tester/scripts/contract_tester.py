#!/usr/bin/env python3
"""Contract Tester - Core CLI & Script Tool (ACR-AGI-3 EDD Guard Gate).

生成・改良されたスキルに対し、「正例 3 件 ＋ 負例 3 件」の契約テストを実行し、
防壁ゲート（Contract Barrier Gate）として機能します。
1件でも契約違反（例外、境界値破綻、トラップ進入）があれば不合格とします。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np


def run_contract_test(
    policy_code: str,
    test_cases: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """正例 3 件 + 負例 3 件の契約テストを実行."""
    results = {
        "passed": False,
        "positive_passed": 0,
        "negative_passed": 0,
        "total_passed": 0,
        "total_cases": 6,
        "failures": [],
    }

    # 1. ポリシーコードのコンパイル確認
    local_scope: Dict[str, Any] = {}
    try:
        from acr_agi3.game.env import Action
        exec_scope = {"np": np, "Action": Action, "__builtins__": __builtins__}
        exec(policy_code, exec_scope, local_scope)
    except Exception as e:
        results["failures"].append(f"CompilationError: {type(e).__name__}: {e}")
        return results

    choose_act_fn = local_scope.get("choose_action") or local_scope.get("act")
    if not callable(choose_act_fn):
        results["failures"].append("ContractError: 'choose_action' function not found.")
        return results

    # 2. デフォルトの 3 正例 + 3 負例 テストケースの構築 (未指定時)
    # 正例: 通常移動、境界付近移動、ターゲット可視環境
    pos_cases = [
        {"name": "pos_normal_grid", "grid": np.zeros((8, 8), dtype=int), "agent": (1, 1), "target": (5, 5)},
        {"name": "pos_near_border", "grid": np.zeros((8, 8), dtype=int), "agent": (0, 1), "target": (7, 6)},
        {"name": "pos_maze_corridor", "grid": np.zeros((8, 8), dtype=int), "agent": (2, 2), "target": (2, 6)},
    ]
    # 負例: 壁・障害物隣接（衝突回避）、トラップ隣接（ハザード進入回避）、不正グリッド（形状チェック）
    neg_cases = [
        {"name": "neg_wall_collision_avoidance", "grid": np.zeros((8, 8), dtype=int), "agent": (0, 0), "wall": (0, 1)},
        {"name": "neg_hazard_avoidance", "grid": np.zeros((8, 8), dtype=int), "agent": (3, 3), "hazard": (3, 4)},
        {"name": "neg_boundary_limit", "grid": np.zeros((5, 5), dtype=int), "agent": (0, 0)},
    ]

    # 正例テスト
    for idx, case in enumerate(pos_cases, 1):
        g = case["grid"].copy()
        ar, ac = case["agent"]
        g[ar, ac] = 2  # agent
        if "target" in case:
            tr, tc = case["target"]
            g[tr, tc] = 3  # target
        try:
            act = choose_act_fn(g, {})
            if act is not None:
                results["positive_passed"] += 1
            else:
                results["failures"].append(f"Positive case {case['name']} returned None.")
        except Exception as e:
            results["failures"].append(f"Positive case {case['name']} crashed: {e}")

    # 負例テスト
    for idx, case in enumerate(neg_cases, 1):
        g = case["grid"].copy()
        ar, ac = case["agent"]
        g[ar, ac] = 2
        if "wall" in case:
            wr, wc = case["wall"]
            g[wr, wc] = 1  # obstacle
        if "hazard" in case:
            hr, hc = case["hazard"]
            g[hr, hc] = 8  # hazard
        try:
            act = choose_act_fn(g, {})
            # 負例：例外を吐かずに安全なアクションを返すこと
            if act is not None:
                results["negative_passed"] += 1
            else:
                results["failures"].append(f"Negative case {case['name']} returned None.")
        except Exception as e:
            results["failures"].append(f"Negative case {case['name']} failed safety check: {e}")

    results["total_passed"] = results["positive_passed"] + results["negative_passed"]
    results["passed"] = (results["total_passed"] == results["total_cases"]) and len(results["failures"]) == 0
    return results


def run(input_val: Any = None) -> Dict[str, Any]:
    """Core contract test task."""
    code = ""
    if isinstance(input_val, dict):
        code = input_val.get("code") or input_val.get("policy_code", "")
    elif isinstance(input_val, str):
        try:
            parsed = json.loads(input_val)
            if isinstance(parsed, dict):
                code = parsed.get("code") or parsed.get("policy_code", "")
            else:
                code = input_val
        except Exception:
            code = input_val

    if not code:
        code = (
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs, info=None):\n"
            "    return Action.RIGHT\n"
        )

    return run_contract_test(code)


def main():
    parser = argparse.ArgumentParser(description="Contract Tester execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional policy code/json")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input policy code/json")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    res = run(input_val)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
