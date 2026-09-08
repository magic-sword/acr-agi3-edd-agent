import importlib.util
from pathlib import Path

import pytest

# skills/grid-analyzer/scripts/analyze.py を動的ロード
script_path = Path(__file__).resolve().parent.parent / "scripts" / "analyze.py"
spec = importlib.util.spec_from_file_location("analyze", script_path)
analyze_module = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(analyze_module)  # type: ignore
analyze_grid = analyze_module.analyze_grid


# --- 正例テスト (Positive Cases) ---


def test_positive_square_symmetric() -> None:
    """正例1: 対角・垂直・水平対称な 3x3 グリッド."""
    grid = [
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1],
    ]
    res = analyze_grid(grid)
    assert res["shape"] == [3, 3]
    assert res["num_colors"] == 3
    assert res["colors"] == [1, 2, 3]
    assert res["symmetry"]["horizontal"] is True
    assert res["symmetry"]["vertical"] is True
    assert res["symmetry"]["diagonal"] is True


def test_positive_asymmetric_rectangle() -> None:
    """正例2: 非対称な 2x4 長方形グリッド."""
    grid = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    res = analyze_grid(grid)
    assert res["shape"] == [2, 4]
    assert res["num_colors"] == 8
    assert res["symmetry"]["horizontal"] is False
    assert res["symmetry"]["vertical"] is False
    assert res["symmetry"]["diagonal"] is False


def test_positive_monochrome_grid() -> None:
    """正例3: 単一色の 2x2 グリッド."""
    grid = [
        [5, 5],
        [5, 5],
    ]
    res = analyze_grid(grid)
    assert res["num_colors"] == 1
    assert res["color_counts"] == {5: 4}
    assert res["symmetry"]["horizontal"] is True
    assert res["symmetry"]["vertical"] is True
    assert res["symmetry"]["diagonal"] is True


# --- 負例テスト (Negative / Error Boundary Cases) ---


def test_negative_1d_array() -> None:
    """負例1: 1次元配列が渡された場合に ValueError が発生すること."""
    grid = [1, 2, 3]  # type: ignore
    with pytest.raises(ValueError, match="2次元"):
        analyze_grid(grid)


def test_negative_jagged_array() -> None:
    """負例2: 各行の長さが揃っていないジャグ配列の場合に例外が発生すること."""
    grid = [
        [1, 2],
        [3],
    ]
    with pytest.raises((ValueError, Exception)):
        analyze_grid(grid)


def test_negative_empty_grid() -> None:
    """負例3: 空の配列が渡された場合に例外または 0 形状になること."""
    grid: list = []
    with pytest.raises((ValueError, IndexError, Exception)):
        analyze_grid(grid)
