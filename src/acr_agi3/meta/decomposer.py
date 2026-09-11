"""人間 VCGT 思考モデルに基づくサブゴール階層分解エンジン (Subgoal Decomposer).

タスクの初期状態と目標状態のギャップを解析し、
検証可能な中間マイルストーン (Subgoals) のシーケンスを生成します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from acr_agi3.meta.observer import MetaObserver, ObservationReport


@dataclass
class Subgoal:
    """中間マイルストーン (Subgoal)."""

    index: int
    name: str
    objective: str
    reasoning: str  # VCGT 型の思考理由
    expected_operation: str  # 幾何・論理操作 (e.g. 'resize', 'color_map', 'translate')
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionPlan:
    """階層分解されたサブゴール計画."""

    task_hint: str
    subgoals: List[Subgoal]
    total_steps: int
    constraints: List[str] = field(default_factory=list)
    reasoning_trace: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_hint": self.task_hint,
            "total_steps": self.total_steps,
            "constraints": self.constraints,
            "reasoning_trace": self.reasoning_trace,
            "subgoals": [
                {
                    "step": s.index,
                    "name": s.name,
                    "objective": s.objective,
                    "reasoning": s.reasoning,
                    "operation": s.expected_operation,
                    "params": s.parameters,
                }
                for s in self.subgoals
            ],
        }


class SubgoalDecomposer:
    """VCGT 思考モデルに準拠したサブゴール分解エンジン."""

    def __init__(self, observer: MetaObserver = None) -> None:
        self.observer = observer or MetaObserver()

    def decompose(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> DecompositionPlan:
        """入出力ペアから階層的な中間サブゴール列を策定."""
        report: ObservationReport = self.observer.analyze_pair(input_grid, output_grid)
        subgoals: List[Subgoal] = []
        step_idx = 1

        # 1. 形状の整合性サブゴール
        if report.shape_ratio != (1.0, 1.0):
            ratio_h, ratio_w = report.shape_ratio
            if ratio_h >= 1.0 and ratio_w >= 1.0:
                subgoals.append(
                    Subgoal(
                        index=step_idx,
                        name="AdjustGridDimensions",
                        objective=(
                            f"Expand grid canvas from {report.in_shape} to {report.out_shape} "
                            f"(scale: {ratio_h:.1f}x{ratio_w:.1f})"
                        ),
                        reasoning=(
                            "The target grid is larger than the input. The canvas must be scaled "
                            "or tiled before positioning elements."
                        ),
                        expected_operation="scale_or_tile",
                        parameters={"target_shape": report.out_shape, "ratio": report.shape_ratio},
                    )
                )
            else:
                subgoals.append(
                    Subgoal(
                        index=step_idx,
                        name="CropOrExtractRegion",
                        objective=f"Crop relevant sub-region matching shape {report.out_shape}",
                        reasoning=(
                            "Target grid is smaller than input. We must extract the bounding "
                            "region or sub-pattern."
                        ),
                        expected_operation="crop",
                        parameters={"target_shape": report.out_shape},
                    )
                )
            step_idx += 1

        # 2. 色パレットの整合性サブゴール
        if report.new_colors:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="IntroduceNewColors",
                    objective=f"Map input colors to new target colors: {report.new_colors}",
                    reasoning=(
                        "New colors appear in the output that are not in the input. "
                        "Identify the trigger condition (e.g. enclosed area, collision) "
                        "to apply new colors."
                    ),
                    expected_operation="conditional_color_mapping",
                    parameters={"new_colors": list(report.new_colors)},
                )
            )
            step_idx += 1

        # 3. 幾何対称性・配置サブゴール
        if report.transformation_hint in ["translation_or_rotation", "scaling_or_tiling"]:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="ApplyGeometricTransform",
                    objective="Align spatial objects according to symmetry or relative motion",
                    reasoning=(
                        "Object identities are preserved while coordinates or orientations change. "
                        "Apply rigid 2D transformation (rot90, fliplr, or translation)."
                    ),
                    expected_operation="spatial_alignment",
                    parameters={"hint": report.transformation_hint},
                )
            )
            step_idx += 1

        # 4. 単純変換の場合のフォールバックサブゴール
        if not subgoals:
            subgoals.append(
                Subgoal(
                    index=1,
                    name="DirectTransformation",
                    objective="Transform cell values in-place",
                    reasoning="Canvas dimensions and palette are identical. Apply local rule.",
                    expected_operation="elementwise_rule",
                )
            )

        return DecompositionPlan(
            task_hint=report.transformation_hint,
            subgoals=subgoals,
            total_steps=len(subgoals),
        )
