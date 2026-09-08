"""ARC 評価指標 (Metrics) モジュール."""

from typing import List

import numpy as np


def exact_match(prediction: np.ndarray, ground_truth: np.ndarray) -> bool:
    """予測グリッドと正解グリッドが形状・画素値ともに完全一致するか判定する."""
    if prediction.shape != ground_truth.shape:
        return False
    return bool(np.array_equal(prediction, ground_truth))


def compute_pass_at_k(results: List[List[bool]], k: int = 2) -> float:
    """Pass@k 指標を計算する.

    各タスクにおいて、上位 k 個の予測のうち少なくとも 1 つが正解であれば正解とみなす.
    """
    if not results:
        return 0.0
    solved_count = 0
    for task_predictions in results:
        candidates = task_predictions[:k]
        if any(candidates):
            solved_count += 1
    return solved_count / len(results)
