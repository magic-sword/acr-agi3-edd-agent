#!/usr/bin/env python3
"""Constraint Learner - Core CLI & Script Tool (ACR-AGI-3).

試行履歴や失敗診断ログから、環境制約・禁忌状態（タブルール、即死色、衝突セル）を
抽出・学習し、安全不変量（Safety Invariants）として更新します。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import numpy as np


def learn_constraints(history: Dict[str, Any]) -> Dict[str, Any]:
    """遷移履歴と診断情報から禁忌セルと危険色を抽出."""
    taboo_cells: List[List[int]] = []
    taboo_colors: Set[int] = set()
    rules: List[str] = []

    failed_transitions = history.get("failed_transitions", [])
    diagnostics = history.get("diagnostics", {})

    for tr in failed_transitions:
        pos = tr.get("pos")
        color = tr.get("color")
        reason = tr.get("reason", "collision")
        if pos:
            taboo_cells.append(list(pos))
        if color is not None and reason in ("hazard", "game_over", "trap"):
            taboo_colors.add(int(color))

    cat = diagnostics.get("failure_category", "")
    if cat == "SafetyInvariantBreach":
        rules.append("Avoid entering detected fatal hazard zones.")
    elif cat == "BehavioralStagnation":
        rules.append("Blacklist unproductive local loops.")

    if taboo_colors:
        rules.append(f"Never step on lethal colors: {sorted(list(taboo_colors))}")

    return {
        "taboo_cells": taboo_cells,
        "taboo_colors": sorted(list(taboo_colors)),
        "safety_rules": rules,
        "constraint_count": len(taboo_cells) + len(taboo_colors),
    }


def run(input_val: Any = None) -> Dict[str, Any]:
    """Core constraint learning task."""
    history: Dict[str, Any] = {}
    if isinstance(input_val, dict):
        history = input_val
    elif isinstance(input_val, str):
        try:
            parsed = json.loads(input_val)
            if isinstance(parsed, dict):
                history = parsed
        except Exception:
            pass

    return learn_constraints(history)


def main():
    parser = argparse.ArgumentParser(description="Constraint Learner execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input JSON")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input JSON")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    res = run(input_val)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
