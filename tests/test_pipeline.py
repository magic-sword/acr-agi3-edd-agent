"""ARC 推論および評価パイプライン全体の疎通・統合テスト."""

from typing import Any, Dict

import numpy as np

from acr_agi3.agent.orchestrator import ARCOrchestrator
from acr_agi3.dsl import primitives
from acr_agi3.eval.harness import BenchmarkHarness
from acr_agi3.eval.metrics import compute_pass_at_k, exact_match


def test_dsl_primitives() -> None:
    """DSL プリミティブの基本動作を検証する."""
    grid = np.array([[1, 2], [3, 4]])
    assert np.array_equal(primitives.rot180(grid), np.array([[4, 3], [2, 1]]))
    assert np.array_equal(primitives.fliplr(grid), np.array([[2, 1], [4, 3]]))
    assert np.array_equal(primitives.flipud(grid), np.array([[3, 4], [1, 2]]))
    assert np.array_equal(primitives.replace_color(grid, 1, 9), np.array([[9, 2], [3, 4]]))


def test_orchestrator_solves_sample_task(sample_arc_task: Dict[str, Any]) -> None:
    """Orchestrator がサンプルの ARC タスクを解けることを検証する."""
    orchestrator = ARCOrchestrator()
    train_pairs = [
        {"input": np.array(p["input"]), "output": np.array(p["output"])}
        for p in sample_arc_task["train"]
    ]
    test_in = np.array(sample_arc_task["test"][0]["input"])
    expected_out = np.array(sample_arc_task["test"][0]["output"])

    preds = orchestrator.solve(train_pairs, test_in, max_attempts=2)

    assert len(preds) > 0
    # 候補のいずれかが正解と一致すること
    assert any(exact_match(p, expected_out) for p in preds)


def test_benchmark_harness_integration(sample_arc_task: Dict[str, Any]) -> None:
    """BenchmarkHarness による評価フローが正常に集計されることを検証する."""
    harness = BenchmarkHarness()
    orchestrator = ARCOrchestrator()

    res = harness.evaluate_task(
        task=sample_arc_task,
        solver_fn=lambda trains, test_in: orchestrator.solve(trains, test_in, max_attempts=2),
        k=2,
    )

    assert res["solved"] is True
    assert len(res["test_cases"]) == 1
    assert res["test_cases"][0]["is_correct"] is True


def test_metrics_pass_at_k() -> None:
    """Pass@k 計算ロジックを検証する."""
    # 2タスク中1タスクが解けた場合
    results = [
        [False, True],  # 2試行目で正解
        [False, False],  # 不正解
    ]
    assert compute_pass_at_k(results, k=2) == 0.5
    assert compute_pass_at_k(results, k=1) == 0.0
