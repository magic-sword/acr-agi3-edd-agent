#!/usr/bin/env python3
"""Env Observer - Core CLI & Script Tool (ACR-AGI-3).

ゲーム観測フレームから、背景色、自機位置、静的障害物、ゴール候補、
インタラクタブルなどのアフォーダンスを完全自律抽出します。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

# プロジェクトルートの追加
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from acr_agi3.meta.observer import MetaObserver


def run(input_val: Any = None) -> Dict[str, Any]:
    """Core affordance extraction task."""
    grid = None
    recent_action = None

    if input_val is not None:
        if isinstance(input_val, dict):
            grid_raw = input_val.get("grid") or input_val.get("observation")
            if grid_raw is not None:
                grid = np.array(grid_raw, dtype=int)
            recent_action = input_val.get("recent_action")
        elif isinstance(input_val, str):
            try:
                parsed = json.loads(input_val)
                if isinstance(parsed, dict):
                    grid_raw = parsed.get("grid") or parsed.get("observation")
                    if grid_raw is not None:
                        grid = np.array(grid_raw, dtype=int)
                    recent_action = parsed.get("recent_action")
                elif isinstance(parsed, list):
                    grid = np.array(parsed, dtype=int)
            except Exception:
                pass
        elif isinstance(input_val, (list, np.ndarray)):
            grid = np.array(input_val, dtype=int)

    if grid is None:
        grid = np.zeros((10, 10), dtype=int)
        grid[1, 1] = 2  # default agent
        grid[8, 8] = 3  # default target

    observer = MetaObserver()
    report = observer.analyze_frame(grid=grid, recent_action=recent_action)

    result = {
        "grid_shape": list(report.grid_shape),
        "background_color": int(report.background_color),
        "agent_pos": list(report.agent_pos) if report.agent_pos else None,
        "agent_color": int(report.agent_object.color) if report.agent_object else None,
        "obstacles_count": len(report.obstacles),
        "target_candidates": [
            {
                "color": int(t.color),
                "pos": [int(round(t.center_r)), int(round(t.center_c))],
                "size": int(t.size),
                "score": float(t.target_score),
            }
            for t in report.target_candidates[:5]
        ],
        "controllable_verified": report.controllable_verified,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Env Observer execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input JSON/grid")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input JSON/grid")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    res = run(input_val)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
