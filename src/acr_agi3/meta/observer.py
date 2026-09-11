"""環境不変量およびアフォーダンス同定エンジン (Meta-Observer).

ARC グリッドペアから、背景色、形状比率、静的障害物、移動可能オブジェクトなどの
不変量 (Invariants) とアフォーダンス (Affordances) を自律抽出します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import label


@dataclass
class AffordanceObject:
    """同定されたオブジェクトの属性情報."""

    color: int
    size: int
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    is_static: bool = True
    role: str = "obstacle"  # 'agent', 'interactable', 'obstacle', 'target'


@dataclass
class ObservationReport:
    """環境観察結果の構造化レポート."""

    in_shape: Tuple[int, int]
    out_shape: Tuple[int, int]
    shape_ratio: Tuple[float, float]
    background_color: int
    preserved_colors: Set[int] = field(default_factory=set)
    new_colors: Set[int] = field(default_factory=set)
    removed_colors: Set[int] = field(default_factory=set)
    objects: List[AffordanceObject] = field(default_factory=list)
    transformation_hint: str = "unknown"


class MetaObserver:
    """環境の不変量とアフォーダンスを自動抽出するメタ認知エンジン."""

    def __init__(self) -> None:
        pass

    def extract_objects(
        self, grid: np.ndarray, background_color: int
    ) -> List[AffordanceObject]:
        """グリッドから背景色以外の連結成分オブジェクトを抽出."""
        objects: List[AffordanceObject] = []
        unique_colors = [c for c in np.unique(grid) if c != background_color]

        for color in unique_colors:
            mask = grid == color
            labeled, num_features = label(mask)
            for feat_id in range(1, num_features + 1):
                feat_mask = labeled == feat_id
                indices = np.argwhere(feat_mask)
                if len(indices) == 0:
                    continue
                min_r, min_c = indices.min(axis=0)
                max_r, max_c = indices.max(axis=0)
                objects.append(
                    AffordanceObject(
                        color=int(color),
                        size=int(len(indices)),
                        bounding_box=(int(min_r), int(min_c), int(max_r), int(max_c)),
                        is_static=True,
                        role="interactable",
                    )
                )
        return objects

    def analyze_pair(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> ObservationReport:
        """単一の入出力ペアから不変量とアフォーダンスを分析."""
        inp = np.array(input_grid, dtype=int)
        out = np.array(output_grid, dtype=int)

        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        ratio = (out_h / in_h, out_w / in_w)

        # 最頻度色を背景色と推定
        in_counts = np.bincount(inp.flatten(), minlength=10)
        bg_color = int(np.argmax(in_counts))

        in_colors = set(int(c) for c in np.unique(inp))
        out_colors = set(int(c) for c in np.unique(out))

        preserved = in_colors.intersection(out_colors)
        new_colors = out_colors - in_colors
        removed_colors = in_colors - out_colors

        # オブジェクト抽出
        in_objects = self.extract_objects(inp, bg_color)
        out_objects = self.extract_objects(out, bg_color)

        # 形状・位置差分による役割推定
        hint = "geometric_or_color"
        if ratio == (1.0, 1.0):
            if in_colors == out_colors:
                # 色が同じで形状も同じなら移動または回転
                if len(in_objects) == len(out_objects):
                    hint = "translation_or_rotation"
                else:
                    hint = "pattern_repetition_or_split"
            else:
                hint = "color_replacement"
        elif ratio[0] > 1.0 or ratio[1] > 1.0:
            hint = "scaling_or_tiling"
        elif ratio[0] < 1.0 or ratio[1] < 1.0:
            hint = "cropping_or_extraction"

        return ObservationReport(
            in_shape=(in_h, in_w),
            out_shape=(out_h, out_w),
            shape_ratio=ratio,
            background_color=bg_color,
            preserved_colors=preserved,
            new_colors=new_colors,
            removed_colors=removed_colors,
            objects=in_objects,
            transformation_hint=hint,
        )

    def analyze_task(
        self, train_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """タスク全体の全 Train ペアに共通する不変量を集約."""
        if not train_pairs:
            return {}

        reports = [
            self.analyze_pair(p["input"], p["output"]) for p in train_pairs
        ]

        # 全ペアで共通の形状比率か
        ratios = [r.shape_ratio for r in reports]
        consistent_ratio = ratios[0] if all(r == ratios[0] for r in ratios) else None

        bg_colors = [r.background_color for r in reports]
        consistent_bg = bg_colors[0] if all(b == bg_colors[0] for b in bg_colors) else None

        hints = [r.transformation_hint for r in reports]
        dominant_hint = max(set(hints), key=hints.count)

        return {
            "consistent_shape_ratio": consistent_ratio,
            "consistent_background": consistent_bg,
            "transformation_hint": dominant_hint,
            "sample_objects_count": len(reports[0].objects),
            "num_train_pairs": len(train_pairs),
        }
