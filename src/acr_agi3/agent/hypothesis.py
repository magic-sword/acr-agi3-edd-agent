"""ARC タスクに対する変換ルール仮説生成モジュール."""

from typing import Any, Dict, List
import numpy as np


class HypothesisGenerator:
    """入出力ペアのパターンから適用可能な DSL プログラム候補を生成する."""

    def __init__(self) -> None:
        pass

    def generate_candidates(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[List[Dict[str, Any]]]:
        """訓練ペアから一貫性のある変換候補プログラムのリストを生成する."""
        candidates: List[List[Dict[str, Any]]] = []

        # 恒等変換 (ベースライン)
        candidates.append([])

        # 幾何変換候補の探索
        first_in = train_pairs[0]["input"]
        first_out = train_pairs[0]["output"]

        # 回転・反転で形状が一致するかチェック
        for op in ["rot90", "rot180", "rot270", "fliplr", "flipud"]:
            candidates.append([{"op": op}])

        # 色置換候補の探索
        in_colors = np.unique(first_in)
        out_colors = np.unique(first_out)
        if len(in_colors) == len(out_colors) and first_in.shape == first_out.shape:
            for c_src in in_colors:
                for c_dst in out_colors:
                    if c_src != c_dst:
                        candidates.append([
                            {"op": "replace_color", "src_color": int(c_src), "dst_color": int(c_dst)}
                        ])

        return candidates
