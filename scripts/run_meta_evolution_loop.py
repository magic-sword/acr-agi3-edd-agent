#!/usr/bin/env python3
"""ARC-AGI-3 メタスキル駆動型自己改善・推論ループランナー.

MetaObserver, SubgoalDecomposer, Human VCGT, FailureDiagnoser を統合したエージェントにより、
未知タスクに対する仮説生成・検証・自己修復（Self-Correction）ループを実行し、効果を検証します。
実行状況は logs/meta_evolution.log にリアルタイム出力されます。
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートを Python パスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acr_agi3.agent.llm import LocalTransformersLlm  # noqa: E402
from acr_agi3.agent.meta_agent import MetaSkillDrivenAgent  # noqa: E402


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


def load_tasks(data_dir: Path, num_tasks: int = 10) -> Dict[str, Any]:
    """data/raw からタスクをロード."""
    challenges_file = data_dir / "arc-agi_training_challenges.json"
    if challenges_file.exists():
        with open(challenges_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
            if len(tasks) >= num_tasks:
                return {k: tasks[k] for k in list(tasks.keys())[:num_tasks]}

    # 個別 JSON ファイルから収集
    tasks = {}
    for json_file in sorted(data_dir.glob("*.json")):
        if "challenges" in json_file.name or "solutions" in json_file.name:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                if "train" in task_data:
                    tasks[json_file.stem] = task_data
                    if len(tasks) >= num_tasks:
                        break
        except Exception:
            continue
    return tasks


def run_meta_evolution_experiment(
    model_path: str,
    data_dir: Path,
    solutions_dir: Path,
    log_path: Path,
    vcgt_path: Path,
    num_tasks: int = 10,
    max_iterations: int = 3,
    device: str = "cuda",
) -> None:
    """メタスキル駆動型の自己改善ループ実験を実行."""
    logger = RealtimeLogger(log_path)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 65)
    logger.log("🧠 ARC-AGI-3 メタスキル駆動型 自己改善ループ実験 (10 Sessions)")
    logger.log(f"📌 モデル: {model_path} | デバイス: {device}")
    logger.log(f"🎯 対象タスク数: {num_tasks} | 1タスク最大試行: {max_iterations} 回")
    logger.log(f"📚 VCGT 思考データ: {vcgt_path}")
    logger.log("=" * 65)

    # 1. モデルとメタスキルエージェントの初期化
    logger.log("エージェントおよびメタスキル群（Observer, Decomposer, Diagnoser）を初期化中...")
    try:
        local_llm = LocalTransformersLlm(
            model_name_or_path=model_path,
            device=device,
        )
        agent = MetaSkillDrivenAgent(
            model=local_llm,
            vcgt_path=vcgt_path if vcgt_path.exists() else None,
        )
        logger.log("✅ メタスキルエージェント初期化完了")
    except Exception as e:
        logger.log(f"❌ 初期化エラー: {e}", level="ERROR")
        logger.close()
        return

    # 2. タスク読み込み
    tasks = load_tasks(data_dir, num_tasks=num_tasks)
    task_ids = list(tasks.keys())[:num_tasks]
    logger.log(f"実行対象タスク ({len(task_ids)}件): {task_ids}")

    solved_count = 0
    total_turns = 0
    iteration_solved_stats = {i: 0 for i in range(1, max_iterations + 1)}

    for idx, task_id in enumerate(task_ids, 1):
        task_data = tasks[task_id]
        train_pairs = task_data["train"]

        logger.log("-" * 65)
        logger.log(f"📌 [Session {idx}/{num_tasks}] Task: {task_id} (Train: {len(train_pairs)}件)")

        # メタ観察とサブゴール計画のログ
        obs_report, plan = agent.analyze_task(train_pairs)
        logger.log(
            f"🔍 [MetaObserver] 形状: {obs_report.in_shape}->{obs_report.out_shape} | "
            f"仮説: {obs_report.transformation_hint} | 物体数: {len(obs_report.objects)}"
        )
        logger.log(
            f"📋 [SubgoalDecomposer] 目標: {plan.task_hint} "
            f"(全 {plan.total_steps} ステップ)"
        )
        for s in plan.subgoals:
            logger.log(f"    - Step {s.index}: {s.name} ({s.expected_operation})")

        # 自己改善ループ実行
        res = agent.solve(train_pairs, max_iterations=max_iterations, task_id=task_id)
        total_turns += len(res["history"])

        for h in res["history"]:
            it = h["iteration"]
            v = h["verification"]
            pass_cnt = v.get("passed_count", 0)
            tot_cnt = v.get("total_count", 0)
            if v.get("is_valid"):
                logger.log(
                    f"  [Turn {it}] 🎉 正解！ 全 {pass_cnt}/{tot_cnt} ペア通過",
                    level="SUCCESS",
                )
            else:
                err_hint = v.get("error") or (v.get("failures", [{}])[0].get("reason", "mismatch"))
                logger.log(
                    f"  [Turn {it}] ⚠️ 不正解 ({pass_cnt}/{tot_cnt} 通過) - {err_hint}",
                    level="FAIL",
                )

        if res["is_solved"]:
            solved_it = res["iterations"]
            solved_count += 1
            iteration_solved_stats[solved_it] = iteration_solved_stats.get(solved_it, 0) + 1
            logger.log(f"🏆 Task {task_id} 解決成功 (Turn {solved_it})")

            # 解法コード保存
            sol_path = solutions_dir / f"{task_id}.py"
            with open(sol_path, "w", encoding="utf-8") as sf:
                sf.write(f"# Solved by MetaSkillDrivenAgent at Turn {solved_it}\n")
                sf.write(f"# Task: {task_id}\n\n")
                sf.write(res["code"] or "")
        else:
            logger.log(f"❌ Task {task_id} 未解決 (最大試行 {max_iterations} 到達)")

    # 最終集計サマリー
    pct = solved_count / len(task_ids) * 100 if task_ids else 0.0
    avg_turns = total_turns / len(task_ids) if task_ids else 0.0
    logger.log("=" * 65)
    logger.log("📊 メタスキル駆動型 自己改善ループ実験 最終サマリー")
    logger.log(f"総セッション数: {len(task_ids)}")
    logger.log(f"解決成功数: {solved_count}/{len(task_ids)} ({pct:.1f}%)")
    logger.log(f"総試行ターン数: {total_turns} (平均 {avg_turns:.1f} ターン/タスク)")
    logger.log("各ターンでの解決数:")
    for it, count in iteration_solved_stats.items():
        logger.log(f"  - Turn {it} で解決: {count} 件")
    logger.log(f"ログファイル: {log_path.resolve()}")
    logger.log("=" * 65)
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Meta-Skill Evolution Loop Experiment")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--solutions-dir", type=str, default="data/solutions")
    parser.add_argument("--log-path", type=str, default="logs/meta_evolution.log")
    parser.add_argument("--vcgt-path", type=str, default="data/human_vcgt/sample_vcgt.json")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_meta_evolution_experiment(
        model_path=args.model_path,
        data_dir=Path(args.data_dir),
        solutions_dir=Path(args.solutions_dir),
        log_path=Path(args.log_path),
        vcgt_path=Path(args.vcgt_path),
        num_tasks=args.num_tasks,
        max_iterations=args.max_iterations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
