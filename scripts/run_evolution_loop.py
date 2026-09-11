#!/usr/bin/env python3
"""ARC-AGI-3 自己改善・推論ループランナー.

ローカル LLM (Google ADK 2.0 ベース) を用い、
タスクごとの仮説生成・コード検証・自己修正 (Self-Correction) を実行します。
実行状況は logs/evolution.log にリアルタイムで出力されます。
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

# プロジェクトルートを Python パスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acr_agi3.agent.llm import LocalTransformersLlm  # noqa: E402
from acr_agi3.agent.llm.arc_tools import (  # noqa: E402
    execute_and_verify_code,
    extract_python_code,
)
from acr_agi3.agent.llm_agent import LLMProgramSynthesisAgent  # noqa: E402


class RealtimeLogger:
    """コンソールとログファイルに即時 flush 書き込みを行うロガー."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8")

    def log(self, message: str, level: str = "INFO") -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{now}] [{level}] {message}"
        print(formatted, flush=True)
        self.log_file.write(formatted + "\n")
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def run_evolution_loop(
    model_path: str,
    challenges_path: Path,
    solutions_dir: Path,
    log_path: Path,
    num_tasks: int = 3,
    max_iterations: int = 3,
    device: str = "cuda",
) -> None:
    """自己改善ループを実行する."""
    logger = RealtimeLogger(log_path)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 60)
    logger.log("🚀 ARC-AGI-3 自己改善・推論ループ開始")
    logger.log(f"🧠 モデルパス: {model_path}")
    logger.log(f"🎯 対象タスク数: {num_tasks} | 1タスク最大試行: {max_iterations} 回")
    logger.log("=" * 60)

    # 1. モデルの初期化
    logger.log("ローカル推論モデルを初期化中...")
    try:
        local_llm = LocalTransformersLlm(
            model_name_or_path=model_path,
            device=device,
        )
        agent = LLMProgramSynthesisAgent(model=local_llm)
        logger.log("✅ Google ADK 2.0 エージェント初期化完了")
    except Exception as e:
        logger.log(f"❌ モデル初期化エラー: {e}", level="ERROR")
        return

    # 2. タスクデータの読み込み
    with open(challenges_path, "r", encoding="utf-8") as f:
        challenges: Dict[str, Any] = json.load(f)

    task_ids = list(challenges.keys())[:num_tasks]
    logger.log(f"実行対象タスク: {task_ids}")

    solved_count = 0
    total_attempts = 0

    for task_idx, task_id in enumerate(task_ids, 1):
        task_data = challenges[task_id]
        train_pairs = task_data["train"]
        test_cases = task_data["test"]

        logger.log("-" * 60)
        logger.log(f"📌 [Task {task_idx}/{num_tasks}] ID: {task_id} (Train例: {len(train_pairs)}件)")

        feedback = None
        best_code = None
        task_solved = False

        for iteration in range(1, max_iterations + 1):
            total_attempts += 1
            logger.log(f"--- 試行 {iteration}/{max_iterations} ---")

            prompt = agent._build_task_prompt(train_pairs, feedback=feedback)
            if feedback:
                logger.log(f"[自己修正フィードバック送信]\n{feedback}", level="FEEDBACK")

            logger.log("LLM に推論コードを生成要請中...")
            try:
                # ADK Runner 経由で推論実行
                response_text = agent.runner.agent.model._build_prompt  # 検証用
                # solve_async 内の 1 ターンを実行
                turn_response = ""
                import asyncio
                turn_response = asyncio.run(
                    agent._run_agent_turn(prompt, session_id=f"sess_{task_id}_{iteration}")
                )
            except Exception as e:
                logger.log(f"推論実行時エラー: {e}", level="ERROR")
                break

            generated_code = extract_python_code(turn_response)
            logger.log(f"生成されたコード:\n{generated_code}", level="CODE")

            # 検証器でチェック
            verification = execute_and_verify_code(generated_code, train_pairs)
            passed = verification["passed_count"]
            total = verification["total_count"]

            if verification["is_valid"]:
                logger.log(
                    f"🎉 【正解】 全 {total}/{total} 件の Train ペアをパスしました！",
                    level="SUCCESS",
                )
                best_code = generated_code
                task_solved = True
                solved_count += 1

                # 解法コードを保存
                solution_file = solutions_dir / f"{task_id}.py"
                with open(solution_file, "w", encoding="utf-8") as sf:
                    sf.write(f"# Task ID: {task_id}\n")
                    sf.write(f"# Solved at iteration {iteration}\n\n")
                    sf.write(best_code)
                logger.log(f"💾 解法コードを保存: {solution_file}")
                break
            else:
                failures = verification.get("failures", [])
                error_msg = verification.get("error", "")
                if failures:
                    reason_summary = "; ".join(
                        f"Pair {f['pair_index']}: {f['reason']}" for f in failures[:2]
                    )
                    feedback = (
                        f"Failed {len(failures)}/{total} pairs. Details: {reason_summary}. "
                        "Re-analyze the pattern and fix the function."
                    )
                else:
                    feedback = f"Execution error: {error_msg}. Please fix the syntax or runtime error."

                logger.log(
                    f"⚠️ 不正解 (合格 {passed}/{total} 件). 理由: {feedback}",
                    level="FAIL",
                )

        if not task_solved:
            logger.log(f"❌ Task {task_id} は最大試行回数 ({max_iterations}) 内で解けませんでした。")

        # テスト入力に対する予測を実行（正解コードがあればそれを使用）
        test_in = np.array(test_cases[0]["input"], dtype=int)
        if best_code:
            local_scope: Dict[str, Any] = {"np": np}
            exec(best_code, {"np": np, "__builtins__": __builtins__}, local_scope)
            pred = local_scope["transform"](test_in.copy())
            logger.log(f"Test 予測完了: 出力形状 {np.array(pred).shape}")
        else:
            logger.log("恒等変換フォールバックを適用")

    logger.log("=" * 60)
    logger.log("📊 ループ完了サマリー")
    logger.log(f"正解タスク数: {solved_count}/{num_tasks} ({solved_count / num_tasks * 100:.1f}%)")
    logger.log(f"総試行ターン数: {total_attempts}")
    logger.log(f"ログファイル: {log_path.resolve()}")
    logger.log("=" * 60)
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 Evolution Loop")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-Coder-1.5B-Instruct",
        help="Local model path or 'mock'",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help="Number of tasks to evaluate (default: 3)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum self-correction iterations per task (default: 3)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/evolution.log"),
        help="Log file destination",
    )
    parser.add_argument(
        "--challenges-file",
        type=Path,
        default=Path("data/raw/arc-agi_training_challenges.json"),
        help="Challenges JSON file",
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("data/solutions"),
        help="Directory to save solved Python codes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    run_evolution_loop(
        model_path=args.model_path,
        challenges_path=args.challenges_file,
        solutions_dir=args.solutions_dir,
        log_path=args.log_file,
        num_tasks=args.num_tasks,
        max_iterations=args.max_iterations,
        device=args.device,
    )
