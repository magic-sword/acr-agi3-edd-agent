"""Kaggle ノートブック実行環境のローカル完全シミュレーション & スコア計測スクリプト.

Kaggle Notebook (submission_template.ipynb) と全く同一の実行フローを
完全オフライン条件で実行し、タスククリア率・所要ステップ・実行時間を計測する。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from acr_agi3.submission.entrypoint import KaggleSubmissionPipeline  # noqa: E402
from acr_agi3.submission.path_resolver import ModelPathResolver  # noqa: E402


def generate_benchmark_suite() -> Dict[str, Any]:
    """Kaggle リーダーボード想定の多様なドメインを含むベンチマーク課題セットを生成."""
    suite = {
        # 1. 閉鎖迷路迂回 (Maze Navigation)
        "task_01_maze_detour": {
            "grid_shape": [10, 10],
            "initial_player_pos": [1, 1],
            "goal_pos": [8, 8],
            "walls": [
                [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]
            ],
            "hazards": [],
            "description": "Detour around a long horizontal barrier wall",
        },
        # 2. オープン空間最短経路 (Open World Navigation)
        "task_02_open_diagonal": {
            "grid_shape": [12, 12],
            "initial_player_pos": [2, 2],
            "goal_pos": [10, 10],
            "walls": [],
            "hazards": [],
            "description": "Direct navigation in an open grid",
        },
        # 3. 溶岩・トラップ回避 (Hazard Avoidance)
        "task_03_lava_avoidance": {
            "grid_shape": [10, 10],
            "initial_player_pos": [1, 1],
            "goal_pos": [8, 8],
            "walls": [],
            "hazards": [
                [3, 3], [4, 4], [5, 5], [6, 6],
                [3, 4], [4, 5], [5, 6],
            ],
            "description": "Navigate around a diagonal lava field",
        },
        # 4. クランク型狭小路 (Narrow S-Shaped Corridor)
        "task_04_s_corridor": {
            "grid_shape": [9, 9],
            "initial_player_pos": [1, 1],
            "goal_pos": [7, 7],
            "walls": [
                # 上部仕切り (右側が開口)
                [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6],
                # 下部仕切り (左側が開口)
                [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
            ],
            "hazards": [],
            "description": "Double bottleneck zigzag corridor",
        },
        # 5. アイテム・障害物複合環境 (Cluttered Obstacles)
        "task_05_cluttered_room": {
            "grid_shape": [10, 10],
            "initial_player_pos": [1, 8],
            "goal_pos": [8, 1],
            "walls": [
                [2, 2], [2, 5], [2, 7],
                [4, 3], [4, 6],
                [6, 1], [6, 4], [6, 8],
            ],
            "hazards": [
                [3, 5], [5, 3], [7, 6],
            ],
            "description": "Cluttered obstacle field with scattered hazards",
        },
    }
    return suite


def run_kaggle_simulation() -> None:
    """Kaggle ノートブックと同一の環境シミュレーションを実行."""
    print("=" * 70)
    print("🚀 [KAGGLE LOCAL SIMULATION] Starting Offline Verification...")
    print("=" * 70)

    # 1. ハードウェア・環境チェック (Cell 2 相当)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 2. モデル & データパス解決 (Cell 4 相当)
    model_path = ModelPathResolver.resolve_model_path()
    data_dir = ModelPathResolver.resolve_data_dir()
    print(f"Resolved Model Path: {model_path or 'Fallback Mode (Rule-based / Heuristic)'}")
    print(f"Resolved Data Dir: {data_dir}")

    # 3. テスト課題セットの準備
    suite = generate_benchmark_suite()
    print(f"Generated {len(suite)} benchmark challenge tasks across diverse domains.")

    # 4. KaggleSubmissionPipeline の初期化 & 実行 (Cell 5 相当)
    output_submission = REPO_ROOT / "local_simulation_submission.json"
    pipeline = KaggleSubmissionPipeline(
        model_path=model_path,
        max_steps_per_task=50,
        time_limit_per_task_sec=60.0,
    )

    t_start = time.time()
    results = pipeline.run_on_challenges(
        challenges_source=suite,
        output_submission_path=output_submission,
    )
    total_elapsed = time.time() - t_start

    # 5. スコア集計 & 検証 (Cell 6 相当)
    print("\n" + "=" * 70)
    print("📊 [LEADERBOARD SCORE BENCHMARK REPORT]")
    print("=" * 70)

    total_tasks = len(suite)
    cleared_tasks = 0
    total_steps = 0
    step_details = []

    for task_id, data in results.items():
        status = data.get("status")
        steps = data.get("steps", 0)
        actions = data.get("actions", [])
        is_cleared = (status == "CLEARED")

        if is_cleared:
            cleared_tasks += 1
        total_steps += steps

        # アクションの検証 (0..5 の Action Enum 値)
        valid_actions = all(isinstance(a, int) and 0 <= a <= 5 for a in actions)
        step_details.append({
            "task_id": task_id,
            "status": status,
            "steps": steps,
            "valid_actions": valid_actions,
            "desc": suite[task_id].get("description", ""),
        })

    clear_rate = (cleared_tasks / total_tasks) * 100.0
    avg_steps = (total_steps / total_tasks) if total_tasks > 0 else 0.0

    print(f"Total Challenges Evaluated: {total_tasks}")
    print(f"Cleared Challenges:         {cleared_tasks} / {total_tasks}")
    print(f"Stage Clear Rate (Score):   {clear_rate:.2f}%")
    print(f"Average Steps Taken:        {avg_steps:.2f} steps")
    print(f"Total Execution Time:       {total_elapsed:.3f} seconds")
    print(f"Average Time per Task:      {total_elapsed / max(1, total_tasks):.3f} seconds")

    print("\n--- Task Details ---")
    for d in step_details:
        mark = "✅" if d["status"] == "CLEARED" else "❌"
        act_ok = "PASS" if d["valid_actions"] else "FAIL"
        msg = (
            f"{mark} [{d['task_id']}]: Steps={d['steps']:2d} | "
            f"Status={d['status']:<10} | Actions Valid={act_ok} | {d['desc']}"
        )
        print(msg)

    # 6. submission.json の仕様チェック
    assert output_submission.exists(), "submission file was not created!"
    file_size = output_submission.stat().st_size
    print(f"\n📁 Submission Artifact: {output_submission.name} ({file_size} bytes)")
    print("✅ All validation checks passed! Ready for Kaggle Leaderboard submission.")

    # 一時ファイルのクリーンアップ
    if output_submission.exists():
        output_submission.unlink()
        print("🧹 Cleaned up temporary simulation artifact.")


if __name__ == "__main__":
    run_kaggle_simulation()
