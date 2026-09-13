#!/usr/bin/env python3
"""Subgoal Decomposer - Core CLI & Script Tool (ACR-AGI-3).

タスクの観測状態から、検証可能な中間マイルストーン (Subgoals) のシーケンスを生成します。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from acr_agi3.meta.decomposer import SubgoalDecomposer
from acr_agi3.meta.observer import MetaObserver


def run(input_val: Any = None) -> Dict[str, Any]:
    """Core subgoal decomposition task."""
    grid = None
    goal_desc = ""

    if input_val is not None:
        if isinstance(input_val, dict):
            grid_raw = input_val.get("grid") or input_val.get("observation")
            if grid_raw is not None:
                grid = np.array(grid_raw, dtype=int)
            goal_desc = input_val.get("goal_description", "")
        elif isinstance(input_val, str):
            try:
                parsed = json.loads(input_val)
                if isinstance(parsed, dict):
                    grid_raw = parsed.get("grid") or parsed.get("observation")
                    if grid_raw is not None:
                        grid = np.array(grid_raw, dtype=int)
                    goal_desc = parsed.get("goal_description", "")
                elif isinstance(parsed, list):
                    grid = np.array(parsed, dtype=int)
            except Exception:
                pass
        elif isinstance(input_val, (list, np.ndarray)):
            grid = np.array(input_val, dtype=int)

    if grid is None:
        grid = np.zeros((10, 10), dtype=int)
        grid[1, 1] = 2
        grid[8, 8] = 3

    observer = MetaObserver()
    decomposer = SubgoalDecomposer(observer=observer)
    plan = decomposer.decompose_game(grid, goal_description=goal_desc)
    return plan.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Subgoal Decomposer execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input JSON/grid")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input JSON/grid")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    res = run(input_val)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
