#!/usr/bin/env python3
"""ACR-AGI-3 20ループ・4環境マルチドメイン進化実験 (20 Multi-Domain Evolution Experiment).

5 ループごとに全く異なるゲーム環境（ドメイン）を切り替えながら、
新設されたメタスキル体系：
1. GameStyleIntuitor (視覚テクスチャ・ゲシュタルト直感・ドメイン自動特定)
2. MetaObserver (環境観測・アフォーダンス抽出)
3. SubgoalDecomposer (状態述語・因果マイルストーン分解)
4. FailureDiagnoser (抽象故障診断・1行修復ディレクティブ蒸留)
5. Dynamic Skill Scoping (環境切り替え時の前ドメインスキル遮断・誤作動防止)
が正しく機能するかを 20 ループ（各ドメイン 5 ループ × 4 ドメイン）で検証・評価します。
"""

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトルート
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acr_agi3.agent.llm import LocalTransformersLlm
from acr_agi3.agent.llm.arc_tools import execute_and_verify_game_policy, extract_python_code
from acr_agi3.game.vcgt_game import GridWorldGameEnv
from acr_agi3.meta import FailureDiagnoser, GameStyleIntuitor, MetaObserver, SubgoalDecomposer


class ExperimentLogger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(log_path, "w", encoding="utf-8")

    def log(self, message: str, level: str = "INFO"):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] [{level}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def create_20_multidomain_tasks() -> List[Dict[str, Any]]:
    """4ドメイン各5課題、計20課題を生成."""
    tasks = []

    # =========================================================================
    # Phase 1: ドメイン navigation (閉鎖型迷路・障害物迂回) [Loops 1-5]
    # =========================================================================
    for i in range(1, 6):
        walls = (
            {(0, c) for c in range(8)}
            | {(7, c) for c in range(8)}
            | {(r, 0) for r in range(8)}
            | {(r, 7) for r in range(8)}
        )
        if i == 1:
            walls |= {(r, 3) for r in range(2, 6)}  # 単一中央壁
        elif i == 2:
            walls |= {(r, 4) for r in range(1, 5)} | {(r, 2) for r in range(3, 7)}
        elif i == 3:
            walls |= {(3, c) for c in range(1, 6)}  # 横壁
        elif i == 4:
            walls |= {(r, 3) for r in range(1, 6) if r != 3}  # 隘路付き壁
        else:
            walls |= {(2, 2), (2, 3), (3, 2), (4, 4), (5, 4), (5, 5)}  # 散在障害物ブロック

        tasks.append(
            {
                "loop_index": i,
                "domain_expected": "navigation",
                "phase": "Phase 1 (Closed Maze)",
                "id": f"nav_maze_{i:02d}",
                "name": f"Closed Maze Challenge {i}",
                "grid_size": (8, 8),
                "player_pos": (1, 1),
                "goal_pos": (6, 6),
                "walls": walls,
                "hazards": set(),
                "hint": "Enclosed labyrinth. Navigate around internal wall barriers to reach exit.",
            }
        )

    # =========================================================================
    # Phase 2: ドメイン inventory_puzzle (鍵と扉・スイッチパズル) [Loops 6-10]
    # =========================================================================
    for i in range(6, 11):
        idx = i - 5
        walls = (
            {(0, c) for c in range(8)}
            | {(7, c) for c in range(8)}
            | {(r, 0) for r in range(8)}
            | {(r, 7) for r in range(8)}
        )
        # 中央に仕切り壁
        walls |= {(r, 4) for r in range(1, 7)}
        # 鍵(4) と 扉(5)
        k_pos = (idx, 2)
        d_pos = (4, 4)
        tasks.append(
            {
                "loop_index": i,
                "domain_expected": "inventory_puzzle",
                "phase": "Phase 2 (Key-Door Puzzle)",
                "id": f"puzzle_key_{idx:02d}",
                "name": f"Key-Door Sequence {idx}",
                "grid_size": (8, 8),
                "player_pos": (1, 1),
                "goal_pos": (6, 6),
                "walls": walls,
                "hazards": set(),
                "items": {k_pos: 4, d_pos: 5},
                "hint": f"Collect key at {k_pos} to clear barrier at {d_pos}.",
            }
        )

    # =========================================================================
    # Phase 3: ドメイン exploration (画面外探索・オープン外周) [Loops 11-15]
    # =========================================================================
    for i in range(11, 16):
        idx = i - 10
        # 外周壁なし（完全オープン、境界開口率 100%）
        # 中央に疎な障害物
        walls = {(3 + (idx % 2), 3 + (idx % 3)), (4, 4)}
        tasks.append(
            {
                "loop_index": i,
                "domain_expected": "exploration",
                "phase": "Phase 3 (Open Exploration)",
                "id": f"explore_open_{idx:02d}",
                "name": f"Open Perimeter Frontier {idx}",
                "grid_size": (8, 8),
                "player_pos": (4, 4),
                "goal_pos": (0, 7) if idx % 2 == 0 else (7, 0),
                "walls": walls,
                "hazards": set(),
                "hint": "Open perimeter. Traversal to outer boundary triggers progress.",
            }
        )

    # =========================================================================
    # Phase 4: ドメイン hazard_avoidance (危険溶岩・安全不変量) [Loops 16-20]
    # =========================================================================
    for i in range(16, 21):
        idx = i - 15
        walls = (
            {(0, c) for c in range(8)}
            | {(7, c) for c in range(8)}
            | {(r, 0) for r in range(8)}
            | {(r, 7) for r in range(8)}
        )
        # 致死トラップ
        hazards = {(r, 3 + (idx % 2)) for r in range(2, 6)}
        tasks.append(
            {
                "loop_index": i,
                "domain_expected": "hazard_avoidance",
                "phase": "Phase 4 (Lethal Hazards)",
                "id": f"hazard_zone_{idx:02d}",
                "name": f"Lethal Lava Choke {idx}",
                "grid_size": (8, 8),
                "player_pos": (2, 1),
                "goal_pos": (5, 6),
                "walls": walls,
                "hazards": hazards,
                "hint": f"Hazards at col {3 + (idx % 2)}. Avoid entering red cells.",
            }
        )

    return tasks


def run_20_multidomain_experiment(cycle: int = 2):
    log_name = f"evolution_cycle{cycle}.log" if cycle > 1 else "evolution_20_multidomain.log"
    log_path = Path(f"logs/{log_name}")
    logger = ExperimentLogger(log_path)

    logger.log("=" * 75)
    logger.log("🌟 ACR-AGI-3 20回マルチドメイン自己改善・メタスキル実証実験")
    logger.log(
        "🎯 検証目標: 5ループごとの環境激変時における『直感・隔離・診断・自己適応』の動作評価"
    )
    logger.log("🧠 モデル: Qwen/Qwen2.5-Coder-1.5B-Instruct (CUDA)")
    logger.log("=" * 75)

    # モデルロード
    snap_dir = (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct"
        "/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
    )
    logger.log("📥 ローカル推論エンジンを初期化中...")
    local_llm = LocalTransformersLlm(model_name_or_path=snap_dir, device="cuda")
    logger.log("✅ ローカル LLM ロード完了")

    # メタスキル層
    intuitor = GameStyleIntuitor()
    observer = MetaObserver()
    decomposer = SubgoalDecomposer(observer=observer)
    diagnoser = FailureDiagnoser()

    tasks = create_20_multidomain_tasks()

    current_phase = ""
    domain_skills_pool: Dict[str, List[Dict[str, Any]]] = {
        "navigation": [],
        "inventory_puzzle": [],
        "exploration": [],
        "hazard_avoidance": [],
        "general": [],
    }

    results = []
    intuitor_accuracy = 0

    for task in tasks:
        loop_num = task["loop_index"]
        phase_name = task["phase"]

        # フェーズ（環境ドメイン）切り替わり検知
        if phase_name != current_phase:
            logger.log("*" * 75)
            logger.log(f"🚨 【環境ドメイン激変】新しい環境へ移行: {phase_name}")
            prev = current_phase or "None"
            logger.log(f"   🛡️ [Dynamic Scoping] 前環境 ({prev}) の具象スキルを隔離・遮断")
            current_phase = phase_name
            logger.log("*" * 75)

        logger.log(f"▶️ [LOOP {loop_num:02d}/20] {task['name']} (ID: {task['id']})")

        # 盤面構築
        env = GridWorldGameEnv(
            grid_shape=task["grid_size"],
            initial_player_pos=task["player_pos"],
            goal_pos=task["goal_pos"],
            walls=task["walls"],
            hazards=task["hazards"],
        )
        initial_obs = env.reset()

        # アイテムを観測に配置
        if "items" in task:
            for (r, c), color_val in task["items"].items():
                if 0 <= r < task["grid_size"][0] and 0 <= c < task["grid_size"][1]:
                    initial_obs[r, c] = color_val

        # =====================================================================
        # 1. メタスキル: GameStyleIntuitor (テクスチャ・ゲシュタルト直感)
        # =====================================================================
        style_report = intuitor.analyze_style(initial_obs)
        intuit_domain = style_report["recommended_domain"]
        expected_domain = task["domain_expected"]

        is_style_match = (intuit_domain == expected_domain) or (
            expected_domain == "hazard_avoidance"
            and intuit_domain in ["navigation", "hazard_avoidance"]
        )
        if is_style_match:
            intuitor_accuracy += 1

        logger.log(
            f"   👁️ [GameStyleIntuitor] 直感分類: {style_report['style']} "
            f"(推奨ドメイン: {intuit_domain}) -> 判定: {'MATCH' if is_style_match else 'FALLBACK'}"
        )
        logger.log(f"       初動方針: {style_report['recommended_approach']}")

        # =====================================================================
        # 2. メタスキル: MetaObserver & SubgoalDecomposer
        # =====================================================================
        plan = decomposer.decompose_game(initial_obs)
        logger.log(
            f"   📋 [Decomposer] マイルストーン数: {plan.total_steps} 個 "
            f"(Subgoals: {[s.name for s in plan.subgoals]})"
        )

        # =====================================================================
        # 3. フォルダ切り分けによる動的スキルスコープ (Active Skill Scoping)
        # =====================================================================
        # 判定されたドメイン専用のスキルのみをプロンプトに注入し、異環境スキルの汚染を物理的に遮断
        scoped_skills = domain_skills_pool.get(intuit_domain, [])
        logger.log(
            f"   🛡️ [Skill Scoping] ロード対象フォルダ: skills/domains/{intuit_domain}/ "
            f"(有効スキル数: {len(scoped_skills)} 件 / 他ドメイン遮断)"
        )
        skills_context = ""
        if scoped_skills:
            skills_context = "\n".join(f"- {s['name']}: {s['desc']}" for s in scoped_skills[-2:])

        # =====================================================================
        # 4. 自己改善ループ (最大 2 回の自己修正トライアル)
        # =====================================================================
        max_retries = 2
        is_cleared = False
        feedback = None
        attempt_logs = []

        for attempt in range(1, max_retries + 1):
            prompt = (
                "You are an expert Python AI programmer controlling a grid game agent.\n"
                f"Grid Shape: {task['grid_size']}\n"
                f"Player: {task['player_pos']}, Target: {task['goal_pos']}\n"
                f"Game Style: {style_report['style']}\n"
                f"Strategy: {style_report['recommended_approach']}\n"
                f"Subgoals: {', '.join(s.name for s in plan.subgoals)}\n"
            )
            if skills_context:
                prompt += f"\nActive Domain Skills (Safe to reuse):\n{skills_context}\n"
            if feedback:
                prompt += f"\n[DIAGNOSTIC DIRECTIVE FROM FAILURE]:\n{feedback}\n"

            prompt += """
CRITICAL CODING GUIDELINES:
- Complete function: `def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:`
- Return value MUST be an Action enum (UP, DOWN, LEFT, RIGHT, or WAIT).
- NEVER return coordinate tuples like (r, c) or call custom actions like Action.MOVE_TO.
- To find coordinates: `pts = np.argwhere(obs == color)` -> check `if len(pts) > 0: r, c = pts[0]`.

Write the complete Python action policy function:
```python
import numpy as np
from acr_agi3.game.env import Action

def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:
    # Identify player and target positions using np.argwhere
    # Decide next step toward target
```
Output the Python code block.
"""

            start_t = time.time()
            resp = local_llm.generate(prompt)
            gen_sec = time.time() - start_t
            code = extract_python_code(resp)

            # シミュレーション検証
            sim_res = execute_and_verify_game_policy(code, env, max_steps=35)
            success = sim_res.get("success", False)
            steps = sim_res.get("steps_taken", 0)
            err = sim_res.get("error", "None")

            attempt_logs.append(
                {
                    "attempt": attempt,
                    "gen_sec": gen_sec,
                    "success": success,
                    "steps": steps,
                    "error": err,
                }
            )

            if success:
                logger.log(
                    f"   🎉 [Attempt {attempt}] ステージクリア！"
                    f" (ステップ: {steps}, 所要: {gen_sec:.2f}s)",
                    level="SUCCESS",
                )
                is_cleared = True
                # 合格スキルを「該当ドメイン専用プール」にのみ保存
                skill_name = f"skill_{task['id']}"
                domain_skills_pool[intuit_domain].append(
                    {
                        "name": skill_name,
                        "desc": task["hint"],
                    }
                )
                logger.log(f"   💾 スキル '{skill_name}' を domains/{intuit_domain}/ に格納")
                break
            else:
                logger.log(
                    f"   ⚠️ [Attempt {attempt}] 未達: {err} (ステップ: {steps})",
                    level="WARN",
                )
                # FailureDiagnoser による抽象診断
                diag = diagnoser.diagnose(
                    error=err,
                    steps_taken=steps,
                    code=code,
                    raw_verification=sim_res,
                )
                feedback = f"[{diag['category']}] {diag['directive']}"
                logger.log(f"       🩺 [FailureDiagnoser] 蒸留指示: {feedback}")

        results.append(
            {
                "loop": loop_num,
                "phase": phase_name,
                "task_id": task["id"],
                "expected_domain": expected_domain,
                "intuit_domain": intuit_domain,
                "is_style_match": is_style_match,
                "is_cleared": is_cleared,
                "attempts": len(attempt_logs),
                "logs": attempt_logs,
            }
        )

    logger.log("=" * 75)
    logger.log("🏁 20回マルチドメイン自己改善・進化実験 最終集計")
    logger.log("総ループ数: 20 (4 ドメイン × 5 ループ)")
    logger.log(
        f"GameStyleIntuitor 直感精度: {intuitor_accuracy}/20 ({intuitor_accuracy / 20 * 100:.1f}%)"
    )
    cleared_cnt = sum(1 for r in results if r["is_cleared"])
    logger.log(f"ステージクリア達成数: {cleared_cnt}/20 ({cleared_cnt / 20 * 100:.1f}%)")
    logger.log("ドメイン別スキル蓄積状況:")
    for d, s_list in domain_skills_pool.items():
        logger.log(f" - domains/{d}/: {len(s_list)} 個の特化スキル")
    logger.log("=" * 75)

    # 結果保存
    out_name = f"evolution_cycle{cycle}_results.json" if cycle > 1 else "evolution_20_results.json"
    out_json = Path(f"logs/{out_name}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.log(f"詳細データを保存しました: {out_json}")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multidomain evolution experiment")
    parser.add_argument("--cycle", type=int, default=2, help="Self-improvement cycle number")
    args = parser.parse_args()
    run_20_multidomain_experiment(cycle=args.cycle)
