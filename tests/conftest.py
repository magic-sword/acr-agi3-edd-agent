import sys
from pathlib import Path
from typing import Any, Dict
import pytest

# src ディレクトリを sys.path に追加 (ローカル・CI双方の互換性確保)
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


@pytest.fixture
def sample_arc_task() -> Dict[str, Any]:
    """回転変換を行うサンプルの ARC タスクデータ."""
    return {
        "train": [
            {
                "input": [
                    [1, 2],
                    [3, 4],
                ],
                "output": [
                    [3, 1],
                    [4, 2],
                ],  # rot270
            },
            {
                "input": [
                    [5, 6],
                    [7, 8],
                ],
                "output": [
                    [7, 5],
                    [8, 6],
                ],  # rot270
            },
        ],
        "test": [
            {
                "input": [
                    [0, 1],
                    [2, 0],
                ],
                "output": [
                    [2, 0],
                    [0, 1],
                ],  # rot270
            }
        ],
    }
