#!/usr/bin/env python3
"""Game Style Intuitor - Core CLI Tool for Visual Gestalt Analysis (ACR-AGI-3).

画面全体のテクスチャ、外周の開放度、障害物密度、対称性を解析し、
ゲームジャンル（画面外探索型、閉鎖迷路型、アイテム連鎖型など）を直感的に分類します。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import numpy as np


def analyze_game_style(obs: np.ndarray) -> Dict[str, Any]:
    """観測グリッドの幾何学的・視覚的ゲシュタルトを解析し、ゲームスタイルを分類."""
    h, w = obs.shape
    unique_colors = np.unique(obs)
    num_colors = len(unique_colors)

    counts = np.bincount(obs.flatten(), minlength=10)
    bg_color = int(np.argmax(counts))

    top_edge = obs[0, :]
    bottom_edge = obs[h - 1, :]
    left_edge = obs[:, 0]
    right_edge = obs[:, w - 1]

    border_cells = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    border_open_ratio = float(np.mean(border_cells == bg_color))
    has_edge_exit = border_open_ratio > 0.25

    non_bg_mask = obs != bg_color
    obstacle_density = float(np.mean(non_bg_mask))

    if h > 2 and w > 2:
        inner_obs = obs[1 : h - 1, 1 : w - 1]
        h_sym = float(np.mean(inner_obs == np.fliplr(inner_obs)))
        v_sym = float(np.mean(inner_obs == np.flipud(inner_obs)))
        symmetry_score = max(h_sym, v_sym)
        inner_non_bg = np.mean(inner_obs != bg_color)
    else:
        symmetry_score = 0.0
        inner_non_bg = 0.0

    isolated_items = []
    for c in unique_colors:
        if c == bg_color:
            continue
        coords = np.argwhere(obs == c)
        if 1 <= len(coords) <= 3:
            isolated_items.append(int(c))

    features = {
        "grid_shape": [h, w],
        "background_color": bg_color,
        "color_count": num_colors,
        "border_open_ratio": round(border_open_ratio, 3),
        "has_edge_exit": has_edge_exit,
        "obstacle_density": round(obstacle_density, 3),
        "symmetry_score": round(symmetry_score, 3),
        "isolated_item_colors": isolated_items,
    }

    if len(isolated_items) >= 2:
        return {
            "style": "ITEM_TRIGGER_PUZZLE",
            "description": (
                "Multiple isolated colored objects detected. Sequential trigger or key-lock"
                " mechanics active."
            ),
            "features": features,
            "recommended_approach": (
                "INTERACTION FIRST: Route to isolated item entities to alter game state"
                " before exit."
            ),
            "recommended_domain": "inventory_puzzle",
        }

    if symmetry_score > 0.85 and inner_non_bg > 0.20:
        return {
            "style": "SYMMETRIC_PATTERN",
            "description": "Board exhibits high spatial symmetry. Geometric alignment game.",
            "features": features,
            "recommended_approach": "Preserve or manipulate spatial balance across symmetry axes.",
            "recommended_domain": "symmetry_pattern",
        }

    if has_edge_exit and obstacle_density < 0.40:
        return {
            "style": "OPEN_EXPLORATION",
            "description": (
                "Outer borders are largely unblocked. Goal is likely off-screen or involves"
                " traversing outside initial view."
            ),
            "features": features,
            "recommended_approach": (
                "EXPLORATION FIRST: Head towards open perimeter edges to expand field of view."
            ),
            "recommended_domain": "exploration",
        }

    if obstacle_density >= 0.20 and not has_edge_exit:
        return {
            "style": "CLOSED_MAZE",
            "description": (
                "Enclosed boundary with internal wall corridors. Traditional shortest path or"
                " obstacle avoidance."
            ),
            "features": features,
            "recommended_approach": (
                "PATHFINDING FIRST: Execute deterministic detour search around internal barriers."
            ),
            "recommended_domain": "navigation",
        }

    return {
        "style": "GENERAL_GRID_GAME",
        "description": "Standard discrete grid environment without extreme structural bias.",
        "features": features,
        "recommended_approach": (
            "BALANCED: Explore forward while avoiding obstacles and observing reward signals."
        ),
        "recommended_domain": "general",
    }


def run(input_val: str | None = None) -> str:
    """CLI 実行エントリポイント."""
    if not input_val:
        grid = np.zeros((8, 8), dtype=int)
        grid[1:7, 1:7] = 1
    else:
        try:
            parsed = json.loads(input_val)
            if isinstance(parsed, list):
                grid = np.array(parsed, dtype=int)
            elif isinstance(parsed, dict) and "grid" in parsed:
                grid = np.array(parsed["grid"], dtype=int)
            else:
                grid = np.zeros((8, 8), dtype=int)
        except Exception:
            grid = np.zeros((8, 8), dtype=int)

    res = analyze_game_style(grid)
    output_str = json.dumps(res, indent=2, ensure_ascii=False)
    print(output_str)
    return output_str


def main():
    parser = argparse.ArgumentParser(description="Game Style Intuitor execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input argument")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    run(input_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
