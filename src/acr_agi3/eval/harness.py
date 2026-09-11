"""ARC タスク評価ベンチマークハーネス."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from acr_agi3.eval.metrics import exact_match


class BenchmarkHarness:
    """ARC タスク群を順次評価し、正解率と実行メトリクスを集計するクラス."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = data_path

    @staticmethod
    def load_task_file(file_path: Path) -> Dict[str, Any]:
        """JSON 形式の ARC タスクファイルを読み込む."""
        with open(file_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
            return data

    def evaluate_task(
        self,
        task: Dict[str, Any],
        solver_fn: Callable[[List[Dict[str, np.ndarray]], np.ndarray], List[np.ndarray]],
        k: int = 2,
    ) -> Dict[str, Any]:
        """単一タスクに対する評価を実施する.

        Args:
            task: 'train' および 'test' ペアを含むタスク辞書
            solver_fn: (train_pairs, test_input) -> candidates (List[np.ndarray])
            k: 許可される最大予測数 (ARC Prize 2026 では通常 2 試行)
        """
        train_pairs = [
            {"input": np.array(pair["input"]), "output": np.array(pair["output"])}
            for pair in task["train"]
        ]

        task_solved = True
        test_results = []

        for test_case in task["test"]:
            test_in = np.array(test_case["input"])
            test_gt = np.array(test_case["output"]) if "output" in test_case else None

            # ソルバーから予測候補を取得
            predictions = solver_fn(train_pairs, test_in)[:k]

            case_solved = False
            if test_gt is not None:
                for pred in predictions:
                    if exact_match(pred, test_gt):
                        case_solved = True
                        break

            if not case_solved:
                task_solved = False

            test_results.append(
                {
                    "predictions_count": len(predictions),
                    "is_correct": case_solved,
                }
            )

        return {
            "solved": task_solved,
            "test_cases": test_results,
        }

    def evaluate_game(
        self,
        env: Any,
        agent: Any,
        max_steps: int = 50,
        task_id: str = "eval_game",
    ) -> Dict[str, Any]:
        """ゲーム環境に対するエージェントの解法実行とクリア成否・ステップ数評価."""
        if hasattr(agent, "solve_game"):
            res = agent.solve_game(env, max_steps=max_steps, task_id=task_id)
        elif hasattr(agent, "solve"):
            res = agent.solve(env, max_steps=max_steps)
        else:
            raise ValueError(f"Agent {agent} does not support game solve interface.")

        return {
            "solved": res.get("is_solved", False),
            "steps_taken": res.get("steps_taken", 0),
            "task_id": task_id,
            "details": res,
        }
