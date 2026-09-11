#!/usr/bin/env python3
"""ARC-AGI-3 EDD 駆動型サブゴールスキル自律開発・検証ループランナー.

ローカル LLM が EDD ツール (edd_init_skill, edd_write_skill_code, edd_validate_skill,
edd_run_contract_test, edd_execute_skill) を自律的に利用し、
サブゴールごとの具象スキルを量産・検証・実行するループを実証します。
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

# プロジェクトルートを Python パスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acr_agi3.agent.llm import LocalTransformersLlm  # noqa: E402
from acr_agi3.agent.meta_agent import MetaSkillDrivenAgent  # noqa: E402


class RealtimeLogger:
    """即時 flush ログ."""

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


def run_edd_experiment(
    model_path: str,
    data_dir: Path,
    log_path: Path,
    num_tasks: int = 3,
    max_retries: int = 2,
    device: str = "cuda",
) -> None:
    """EDD ツールを用いたサブゴールスキル自律開発ループ."""
    logger = RealtimeLogger(log_path)
    logger.log("=" * 65)
    logger.log("🚀 ARC-AGI-3 EDD 駆動型 サブゴールスキル自律開発・検証ループ")
    logger.log(f"🧠 モデル: {model_path} | デバイス: {device}")
    logger.log(f"🎯 対象タスク数: {num_tasks} | サブゴール最大試行: {max_retries} 回")
    logger.log("=" * 65)

    try:
        local_llm = LocalTransformersLlm(model_name_or_path=model_path, device=device)
        agent = MetaSkillDrivenAgent(model=local_llm)
        logger.log("✅ EDD ツールバインド済み MetaSkillDrivenAgent 初期化完了")
    except Exception as e:
        logger.log(f"❌ エージェント初期化エラー: {e}", level="ERROR")
        logger.close()
        return

    # タスク読み込み
    task_files = sorted(data_dir.glob("*.json"))[:num_tasks]
    total_subgoals = 0
    passed_subgoals = 0

    for idx, t_file in enumerate(task_files, 1):
        task_id = t_file.stem
        try:
            with open(t_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)
        except Exception:
            continue

        train_pairs = t_data.get("train", [])
        if not train_pairs:
            continue

        logger.log("-" * 65)
        logger.log(f"📌 [Task {idx}/{num_tasks}] ID: {task_id} (Train: {len(train_pairs)}件)")

        # 1. メタ観察とサブゴール分解
        obs, plan = agent.analyze_task(train_pairs)
        logger.log(
            f"🔍 [MetaObserver] 形状: {obs.in_shape}->{obs.out_shape} | "
            f"仮説: {obs.transformation_hint}"
        )
        logger.log(
            f"📋 [SubgoalDecomposer] 計画: {plan.task_hint} "
            f"(全 {plan.total_steps} サブゴール)"
        )

        # 2. 各サブゴールに対する EDD スキル開発
        for subgoal in plan.subgoals:
            total_subgoals += 1
            logger.log(
                f"  👉 サブゴール開発: Step {subgoal.index} "
                f"[{subgoal.name}] ({subgoal.expected_operation})"
            )
            logger.log(f"     目的: {subgoal.objective}")

            skill_res = agent.synthesize_subgoal_skill(
                subgoal=subgoal,
                train_pairs=train_pairs,
                task_id=task_id,
                max_retries=max_retries,
            )

            s_name = skill_res["skill_name"]
            val_info = skill_res.get("validation", {})
            test_info = skill_res.get("test_result", {})

            if skill_res["success"]:
                passed_subgoals += 1
                p_cnt = test_info.get("passed_count")
                t_cnt = test_info.get("total_count")
                logger.log(
                    f"     🎉 [EDD 合格] '{s_name}' が契約テスト全勝通過！ "
                    f"(規約適合: {val_info.get('is_valid')}, 通過: {p_cnt}/{t_cnt})",
                    level="SUCCESS",
                )
            else:
                logger.log(
                    f"     ⚠️ [EDD 未達] '{s_name}' は試行 {skill_res['attempt']} 回で未達 "
                    f"(規約適合: {val_info.get('is_valid')})",
                    level="FAIL",
                )

    pass_pct = passed_subgoals / total_subgoals * 100 if total_subgoals else 0.0
    logger.log("=" * 65)
    logger.log("📊 EDD サブゴールスキル開発実験 最終サマリー")
    logger.log(f"総サブゴール数: {total_subgoals}")
    logger.log(f"EDD 契約テスト合格スキル数: {passed_subgoals}/{total_subgoals} ({pass_pct:.1f}%)")
    logger.log(f"ログファイル: {log_path.resolve()}")
    logger.log("=" * 65)
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDD Subgoal Skill Evolution")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--log-path", type=str, default="logs/edd_evolution.log")
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_edd_experiment(
        model_path=args.model_path,
        data_dir=Path(args.data_dir),
        log_path=Path(args.log_path),
        num_tasks=args.num_tasks,
        max_retries=args.max_retries,
        device=args.device,
    )


if __name__ == "__main__":
    main()
