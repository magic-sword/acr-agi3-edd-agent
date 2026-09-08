"""ARC-AGI-3 推論統括エージェント (Orchestrator)."""

from typing import Dict, List, Optional

import numpy as np

from acr_agi3.agent.hypothesis import HypothesisGenerator
from acr_agi3.agent.verifier import ProgramVerifier
from acr_agi3.dsl.interpreter import DSLInterpreter


class ARCOrchestrator:
    """仮説生成、検証、テストケースへの適用を一貫して統括するオーケストレーター."""

    def __init__(
        self,
        hypothesis_gen: Optional[HypothesisGenerator] = None,
        verifier: Optional[ProgramVerifier] = None,
        interpreter: Optional[DSLInterpreter] = None,
    ) -> None:
        self.interpreter = interpreter or DSLInterpreter()
        self.hypothesis_gen = hypothesis_gen or HypothesisGenerator()
        self.verifier = verifier or ProgramVerifier(interpreter=self.interpreter)

    def solve(
        self,
        train_pairs: List[Dict[str, np.ndarray]],
        test_input: np.ndarray,
        max_attempts: int = 2,
    ) -> List[np.ndarray]:
        """与えられた訓練ペアからルールを推論し、テスト入力に対する予測グリッドを返す."""
        # 1. 変換ルールの仮説候補を生成
        candidates = self.hypothesis_gen.generate_candidates(train_pairs)

        # 2. 全訓練例を満たす有効なプログラムを検証・抽出
        valid_programs = self.verifier.filter_valid_programs(candidates, train_pairs)

        predictions: List[np.ndarray] = []

        # 3. 有効なプログラムが見つかった場合はそれを適用
        for prog in valid_programs:
            try:
                pred = self.interpreter.execute_program(test_input, prog)
                # 重複予測の排除
                if not any(np.array_equal(pred, p) for p in predictions):
                    predictions.append(pred)
                if len(predictions) >= max_attempts:
                    break
            except Exception:
                continue

        # 4. 有効なプログラムがない場合のフォールバック（恒等変換）
        if not predictions:
            predictions.append(test_input.copy())

        return predictions
