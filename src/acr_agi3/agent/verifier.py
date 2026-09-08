"""仮説プログラムの検証および整合性スコア算出モジュール."""

from typing import Any, Dict, List, Optional
import numpy as np
from acr_agi3.dsl.interpreter import DSLInterpreter
from acr_agi3.eval.metrics import exact_match


class ProgramVerifier:
    """生成された仮説が与えられた訓練例をすべて満たすか検証する."""

    def __init__(self, interpreter: Optional[DSLInterpreter] = None) -> None:
        self.interpreter = interpreter or DSLInterpreter()

    def verify_program(
        self,
        program: List[Dict[str, Any]],
        train_pairs: List[Dict[str, np.ndarray]],
    ) -> bool:
        """すべての訓練例で入力から正解出力が完全に生成できるかを検証する."""
        for pair in train_pairs:
            inp = pair["input"]
            expected = pair["output"]
            try:
                actual = self.interpreter.execute_program(inp, program)
                if not exact_match(actual, expected):
                    return False
            except Exception:
                return False
        return True

    def filter_valid_programs(
        self,
        candidates: List[List[Dict[str, Any]]],
        train_pairs: List[Dict[str, np.ndarray]],
    ) -> List[List[Dict[str, Any]]]:
        """全訓練例を満たすプログラムのみを抽出して返す."""
        valid: List[List[Dict[str, Any]]] = []
        for prog in candidates:
            if self.verify_program(prog, train_pairs):
                valid.append(prog)
        return valid
