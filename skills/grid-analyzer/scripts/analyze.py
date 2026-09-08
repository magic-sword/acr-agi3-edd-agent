"""グリッド静的解析スクリプト (決定論的ツール)."""

import argparse
import json
from typing import Any, Dict, List
import numpy as np


def analyze_grid(grid_list: List[List[int]]) -> Dict[str, Any]:
    """グリッドの形状、色、対称性を解析する."""
    grid = np.array(grid_list, dtype=int)
    if grid.ndim != 2:
        raise ValueError(f"グリッドは2次元である必要があります (入力次元: {grid.ndim})")

    h, w = grid.shape
    unique_colors, counts = np.unique(grid, return_counts=True)
    color_counts = {int(c): int(cnt) for c, cnt in zip(unique_colors, counts)}

    # 対称性チェック
    is_horizontal_sym = bool(np.array_equal(grid, np.flipud(grid)))
    is_vertical_sym = bool(np.array_equal(grid, np.fliplr(grid)))
    is_diagonal_sym = bool(np.array_equal(grid, grid.T)) if h == w else False

    return {
        "shape": [int(h), int(w)],
        "num_colors": len(unique_colors),
        "colors": [int(c) for c in unique_colors],
        "color_counts": color_counts,
        "symmetry": {
            "horizontal": is_horizontal_sym,
            "vertical": is_vertical_sym,
            "diagonal": is_diagonal_sym,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC Grid Analyzer")
    parser.add_argument("--input", type=str, required=True, help="JSON形式の2次元配列文字列")
    args = parser.parse_args()

    grid = json.loads(args.input)
    result = analyze_grid(grid)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
