"""ARC グリッド操作のための Domain Specific Language (DSL) プリミティブ関数群."""

from typing import List, Tuple

import numpy as np

Grid = np.ndarray


def rot90(grid: Grid, k: int = 1) -> Grid:
    """グリッドを反時計回りに90*k度回転する."""
    return np.rot90(grid, k=k)


def rot180(grid: Grid) -> Grid:
    """グリッドを180度回転する."""
    return np.rot90(grid, k=2)


def rot270(grid: Grid) -> Grid:
    """グリッドを反時計回りに270度回転する."""
    return np.rot90(grid, k=3)


def fliplr(grid: Grid) -> Grid:
    """グリッドを左右反転する."""
    return np.fliplr(grid)


def flipud(grid: Grid) -> Grid:
    """グリッドを上下反転する."""
    return np.flipud(grid)


def replace_color(grid: Grid, src_color: int, dst_color: int) -> Grid:
    """特定の色を別の色に置き換える."""
    result = grid.copy()
    result[grid == src_color] = dst_color
    return result


def crop(grid: Grid, r1: int, c1: int, r2: int, c2: int) -> Grid:
    """指定された矩形領域 [r1:r2, c1:c2] を切り出す."""
    return grid[r1:r2, c1:c2].copy()


def pad(grid: Grid, pad_width: int, fill_value: int = 0) -> Grid:
    """グリッドの外周を指定色でパディングする."""
    return np.pad(grid, pad_width, mode="constant", constant_values=fill_value)


def get_unique_colors(grid: Grid) -> List[int]:
    """グリッド内に存在するユニークな色一覧（昇順）を返す."""
    return sorted(int(c) for c in np.unique(grid))


def find_bounding_box(grid: Grid, background: int = 0) -> Tuple[int, int, int, int]:
    """背景色以外の非ゼロ画素を含むバウンディングボックス (r1, c1, r2, c2) を計算する."""
    mask = grid != background
    if not np.any(mask):
        return 0, 0, grid.shape[0], grid.shape[1]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return int(r1), int(c1), int(r2 + 1), int(c2 + 1)
