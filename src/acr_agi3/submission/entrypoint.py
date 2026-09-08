"""Kaggle / ARC Prize 2026 提出環境での推論エントリポイント."""

import json
from pathlib import Path
from typing import Any, Dict
from acr_agi3.agent.orchestrator import ARCOrchestrator
from acr_agi3.eval.harness import BenchmarkHarness


def run_submission(
    test_challenges_path: Path,
    output_submission_path: Path,
) -> None:
    """テスト課題 JSON を読み込み、予測結果を提出形式で保存する."""
    with open(test_challenges_path, "r", encoding="utf-8") as f:
        challenges: Dict[str, Any] = json.load(f)

    orchestrator = ARCOrchestrator()
    submission_output: Dict[str, Any] = {}

    for task_id, task in challenges.items():
        train_pairs = BenchmarkHarness.load_task_file(Path(task_id))["train"] if isinstance(task_id, Path) else task["train"]
        test_cases = task["test"]

        task_predictions = []
        for test_case in test_cases:
            test_in = test_case["input"]
            preds = orchestrator.solve(
                train_pairs=[{"input": p["input"], "output": p["output"]} for p in train_pairs],
                test_input=test_in,
                max_attempts=2,
            )
            # JSON シリアライズ可能な形式に変換
            task_predictions.append({
                f"attempt_{i+1}": p.tolist() if hasattr(p, "tolist") else p
                for i, p in enumerate(preds)
            })

        submission_output[task_id] = task_predictions

    with open(output_submission_path, "w", encoding="utf-8") as f:
        json.dump(submission_output, f, indent=2)
