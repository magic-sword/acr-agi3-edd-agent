"""ARC DSL プログラムのインタープリタと実行器."""

from typing import Any, Callable, Dict, List

import numpy as np

from acr_agi3.dsl import primitives


class DSLInterpreter:
    """DSL 変換パイプラインを順次実行するインタープリタ."""

    def __init__(self) -> None:
        self.operations: Dict[str, Callable[..., np.ndarray]] = {
            "rot90": primitives.rot90,
            "rot180": primitives.rot180,
            "rot270": primitives.rot270,
            "fliplr": primitives.fliplr,
            "flipud": primitives.flipud,
            "replace_color": primitives.replace_color,
            "crop": primitives.crop,
            "pad": primitives.pad,
        }

    def execute_step(self, grid: np.ndarray, op_name: str, **kwargs: Any) -> np.ndarray:
        """単一の DSL プリミティブ操作を適用する."""
        if op_name not in self.operations:
            raise ValueError(f"未定義の DSL 操作: {op_name}")
        func = self.operations[op_name]
        return func(grid, **kwargs)

    def execute_program(self, grid: np.ndarray, program: List[Dict[str, Any]]) -> np.ndarray:
        """複数の操作を連続適用してグリッドを変換する."""
        current = grid.copy()
        for step in program:
            op_name = step["op"]
            kwargs = {k: v for k, v in step.items() if k != "op"}
            current = self.execute_step(current, op_name, **kwargs)
        return current
