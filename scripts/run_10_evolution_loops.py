#!/usr/bin/env python3
"""ACR-AGI-3 ローカル LLM による 10 回自己改善・進化ループ実験 (10-Evolution Loops).

ローカル LLM (Qwen2.5-Coder-1.5B-Instruct) と Google ADK 2.0 / EDD メタスキル層を連携し、
10 回の異なるインタラクティブゲーム課題に対して自律的に：
1. 環境観測 (MetaObserver)
2. サブゴール分解 (SubgoalDecomposer)
3. 過去スキルの取得・合成 (Skill Retrieval & Composition)
4. ローカル LLM によるポリシー合成 (choose_action)
5. シミュレーション検証・契約テスト (Contract Testing)
6. 失敗時の自己修正 (Failure Diagnosis & Self-Correction)
7. 合格スキルのライブラリ登録 (Verified Skill Registration)
を実行し、自己改善の履歴と進化内容を詳細に記録・分析します。
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
from acr_agi3.agent.llm.edd_tools import (
    edd_init_skill,
    edd_list_skills,
    edd_register_verified_skill,
    edd_write_skill_code,
)
from acr_agi3.game.vcgt_game import GridWorldGameEnv
from acr_agi3.meta import FailureDiagnoser, MetaObserver, SubgoalDecomposer


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


def create_10_game_tasks() -> List[Dict[str, Any]]:
    """10 個の段階的ゲーム環境定義."""
    tasks = []

    # 1. 直線ナビゲーション
    tasks.append(
        {
            "id": "loop_01_straight_nav",
            "name": "Straight Line Navigation",
            "grid_size": (8, 8),
            "player_pos": (1, 1),
            "goal_pos": (1, 6),
            "walls": set(),
            "keys": set(),
            "doors": set(),
            "hazards": set(),
            "hint": "Move player directly to the goal on the same row.",
        }
    )

    # 2. 単一壁の迂回
    walls_2 = {(r, 3) for r in range(1, 5)}
    tasks.append(
        {
            "id": "loop_02_single_wall_detour",
            "name": "Single Vertical Wall Detour",
            "grid_size": (8, 8),
            "player_pos": (2, 1),
            "goal_pos": (2, 6),
            "walls": walls_2,
            "keys": set(),
            "doors": set(),
            "hazards": set(),
            "hint": "Vertical wall at column 3 blocks path; detour via row 0 or row 5.",
        }
    )

    # 3. U字型袋小路の脱出
    walls_3 = {(1, 4), (2, 4), (3, 4), (4, 4), (4, 2), (4, 3)}
    tasks.append(
        {
            "id": "loop_03_u_turn_avoidance",
            "name": "U-turn Corner Navigation",
            "grid_size": (8, 8),
            "player_pos": (1, 1),
            "goal_pos": (5, 5),
            "walls": walls_3,
            "keys": set(),
            "doors": set(),
            "hazards": set(),
            "hint": "Wall pocket blocks direct southeast motion; navigate around the lower corner.",
        }
    )

    # 4. 鍵取得と扉の開錠
    walls_4 = {(r, 4) for r in range(8) if r != 3}  # row 3 is door
    tasks.append(
        {
            "id": "loop_04_key_door_causal",
            "name": "Key-Door Sequential Gate",
            "grid_size": (8, 8),
            "player_pos": (1, 1),
            "goal_pos": (6, 6),
            "walls": walls_4,
            "keys": {(1, 3)},  # key on left side
            "doors": {(3, 4)},
            "hazards": set(),
            "hint": "Full vertical barrier at col 4. Collect key at (1, 3) to unlock door at (3, 4).",
        }
    )

    # 5. 複数障害物迷路 (スキル合成)
    walls_5 = {(r, 2) for r in range(0, 6)} | {(r, 5) for r in range(2, 8)}
    tasks.append(
        {
            "id": "loop_05_zigzag_maze",
            "name": "Zigzag Maze Obstacle Course",
            "grid_size": (8, 8),
            "player_pos": (1, 1),
            "goal_pos": (6, 6),
            "walls": walls_5,
            "keys": set(),
            "doors": set(),
            "hazards": set(),
            "hint": "Two interleaving vertical walls creating S-curve corridor.",
        }
    )

    # 6. 即死トラップ回避
    hazards_6 = {(2, 3), (3, 3), (4, 3)}
    tasks.append(
        {
            "id": "loop_06_hazard_avoidance",
            "name": "Hazardous Lava Pit Avoidance",
            "grid_size": (8, 8),
            "player_pos": (3, 1),
            "goal_pos": (3, 6),
            "walls": set(),
            "keys": set(),
            "doors": set(),
            "hazards": hazards_6,
            "hint": "Lethal hazard cells at column 3, rows 2-4. Detour around hazard zone.",
        }
    )

    # 7. 鍵近傍トラップパズル
    walls_7 = {(r, 3) for r in range(1, 7)}
    hazards_7 = {(1, 4), (2, 4)}
    tasks.append(
        {
            "id": "loop_07_trap_guarded_passage",
            "name": "Trap-Guarded Passage",
            "grid_size": (8, 8),
            "player_pos": (1, 1),
            "goal_pos": (5, 6),
            "walls": walls_7,
            "keys": set(),
            "doors": set(),
            "hazards": hazards_7,
            "hint": "Central wall at col 3 with hazards guarding upper exit; use lower exit at row 7.",
        }
    )

    # 8. 逆方向 (右下から左上) ナビゲーション
    walls_8 = {(r, 4) for r in range(3, 8)}
    tasks.append(
        {
            "id": "loop_08_reverse_navigation",
            "name": "Reverse Direction Detour",
            "grid_size": (8, 8),
            "player_pos": (6, 6),
            "goal_pos": (1, 1),
            "walls": walls_8,
            "keys": set(),
            "doors": set(),
            "hazards": set(),
            "hint": "Starting at bottom-right (6, 6), navigate to top-left (1, 1) avoiding wall barrier.",
        }
    )

    # 9. 狭隘通路と障害物回避
    walls_9 = {(r, c) for r in range(1, 7) for c in range(1, 7) if not (r == 3 or c == 3)}
    tasks.append(
        {
            "id": "loop_09_cross_choke_point",
            "name": "Cross Corridor Choke Navigation",
            "grid_size": (8, 8),
            "player_pos": (3, 0),
            "goal_pos": (0, 3),
            "walls": {(1, 1), (1, 2), (2, 1), (2, 2), (4, 4), (5, 5)},
            "keys": set(),
            "doors": set(),
            "hazards": set(),
            "hint": "Navigate through interconnected narrow corridors without colliding with wall blocks.",
        }
    )

    # 10. 複合ダンジョン (鍵・扉・壁・トラップ)
    walls_10 = {(r, 4) for r in range(8) if r != 4}
    hazards_10 = {(2, 2), (5, 2)}
    tasks.append(
        {
            "id": "loop_10_complex_dungeon",
            "name": "Grand Dungeon Milestone Trial",
            "grid_size": (8, 8),
            "player_pos": (0, 0),
            "goal_pos": (7, 7),
            "walls": walls_10,
            "keys": {(1, 2)},
            "doors": {(4, 4)},
            "hazards": hazards_10,
            "hint": "Collect key at (1, 2) avoiding hazards, unlock door at (4, 4), reach exit at (7, 7).",
        }
    )

    return tasks


def run_10_loops():
    log_path = Path("logs/evolution_10_loops.log")
    logger = ExperimentLogger(log_path)

    logger.log("=" * 70)
    logger.log("🚀 ACR-AGI-3 ローカル LLM 10回自己改善・進化ループ実験開始")
    logger.log("🧠 モデル: Qwen/Qwen2.5-Coder-1.5B-Instruct (GPU: CUDA)")
    logger.log("🛠️ 基盤: Google ADK 2.0 / EDD メタスキル層 (Observer, Decomposer, EDD)")
    logger.log("=" * 70)

    # 1. モデルロード
    model_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
    logger.log(f"📥 ローカル推論エンジンを初期化中: {model_path} ...")
    start_load = time.time()
    local_llm = LocalTransformersLlm(model_name_or_path=model_path, device="cuda")
    logger.log(f"✅ ローカル LLM ロード完了 (所要時間: {time.time() - start_load:.2f}s)")

    # メタスキル層
    observer = MetaObserver()
    decomposer = SubgoalDecomposer(observer=observer)

    tasks = create_10_game_tasks()
    logger.log(f"📋 生成された実験課題数: {len(tasks)} 件")

    loop_results = []
    total_retries_needed = 0

    for loop_idx, task_info in enumerate(tasks, 1):
        t_id = task_info["id"]
        t_name = task_info["name"]
        logger.log("-" * 70)
        logger.log(f"🔄 [LOOP {loop_idx:02d}/10] 課題: {t_name} (ID: {t_id})")

        # 環境初期化
        env = GridWorldGameEnv(
            grid_shape=task_info["grid_size"],
            initial_player_pos=task_info["player_pos"],
            goal_pos=task_info["goal_pos"],
            walls=task_info["walls"],
            hazards=task_info["hazards"],
        )
        initial_obs = env.reset()

        # 1. メタスキル: 環境観測
        aff = observer.analyze_frame(initial_obs)
        logger.log(
            f"   👁️ [Observer] 盤面: {aff.grid_shape}, プレイヤー: {aff.player_pos}, ゴール: {aff.goal_pos}"
        )
        logger.log(
            f"       壁数: {len(aff.obstacles)}, 鍵: {len(task_info['keys'])}, 扉: {len(task_info['doors'])}, トラップ: {len(aff.hazards)}"
        )

        # 2. メタスキル: サブゴール分解
        plan = decomposer.decompose_game(initial_obs)
        logger.log(f"   📋 [Decomposer] マイルストーン数: {plan.total_steps} 個")
        for sg in plan.subgoals:
            logger.log(f"       - Subgoal {sg.index}: {sg.name} -> {sg.objective}")

        # 3. 過去スキルの取得 (Skill Retrieval)
        verified_skills = edd_list_skills(verified_only=True)
        logger.log(f"   📚 [Skill Library] 利用可能な蓄積スキル数: {len(verified_skills)}")
        skills_summary = ""
        if verified_skills:
            skills_summary = "\n".join(
                f"- {s['name']}: {s.get('description', '')}" for s in verified_skills[-3:]
            )
            logger.log(f"       (直近の獲得スキル: {[s['name'] for s in verified_skills[-3:]]})")

        # 4. 自己改善ループ (最大 3 回の自己修正トライアル)
        max_retries = 3
        is_cleared = False
        feedback = None
        attempt_history = []
        diagnoser = FailureDiagnoser()

        for attempt in range(1, max_retries + 1):
            logger.log(f"   ⚡ [Attempt {attempt}/{max_retries}] ポリシー合成プロンプト作成中...")

            prompt = (
                "You are an expert Python AI programmer controlling a discrete grid game agent.\n"
                f"Grid Shape: {task_info['grid_size']}\n"
                f"Player Start: {task_info['player_pos']}, Goal Target: {task_info['goal_pos']}\n"
                f"Objective: {task_info['hint']}\n\n"
                "Subgoals:\n" + "\n".join(f"- {sg.name}: {sg.objective}" for sg in plan.subgoals)
            )

            if task_info["walls"]:
                w_list = sorted(list(task_info["walls"]))[:15]
                prompt += f"\nStatic Walls (Impassable): {w_list}"
            if task_info["hazards"]:
                prompt += f"\nLethal Hazards (Death on touch): {sorted(list(task_info['hazards']))}"
            if task_info["keys"]:
                k_list = sorted(list(task_info["keys"]))
                d_list = sorted(list(task_info["doors"]))
                prompt += f"\nKeys: {k_list}, Doors: {d_list}"

            if verified_skills:
                prompt += (
                    f"\n\nPreviously verified skills in library:\n{skills_summary}\n"
                    "You may reuse ideas from these skills."
                )

            if feedback:
                prompt += (
                    f"\n\n[CRITICAL DIAGNOSTIC FEEDBACK]:\n{feedback}\n"
                    "Please fix the logic to avoid this failure mode!"
                )

            prompt += """
Write the complete Python action policy function:
```python
import numpy as np
from acr_agi3.game.env import Action

def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:
    # return Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, or Action.WAIT
```
Output only the Python code inside ```python ```.
"""

            # LLM 生成
            start_gen = time.time()
            llm_response = local_llm.generate(prompt)
            gen_time = time.time() - start_gen
            code = extract_python_code(llm_response)

            # シミュレーション検証 (EDD 契約テスト)
            sim_res = execute_and_verify_game_policy(code, env, max_steps=40)
            success = sim_res.get("success", False)
            steps = sim_res.get("steps_taken", 0)
            reward = sim_res.get("final_reward", 0.0)
            err = sim_res.get("error", "None")

            attempt_history.append(
                {
                    "attempt": attempt,
                    "gen_time": gen_time,
                    "success": success,
                    "steps": steps,
                    "reward": reward,
                    "error": err,
                }
            )

            if success:
                logger.log(
                    f"   🎉 [Attempt {attempt}] 成功！ (ステップ: {steps}, 報酬: {reward},"
                    f" 生成: {gen_time:.2f}s)",
                    level="SUCCESS",
                )
                is_cleared = True

                # 5. 合格したスキルを EDD ライブラリに正式登録
                skill_name = f"skill_{t_id}"
                edd_init_skill(skill_name)
                edd_write_skill_code(skill_name, code)
                edd_register_verified_skill(
                    name=skill_name,
                    description=task_info["hint"],
                    tags=["game_policy", t_id],
                )
                logger.log(f"   💾 スキル '{skill_name}' を EDD ライブラリに保存完了！")
                break
            else:
                total_retries_needed += 1
                logger.log(
                    f"   ⚠️ [Attempt {attempt}] 失敗: {err} (実行ステップ: {steps})",
                    level="WARN",
                )
                # FailureDiagnoser による環境非依存の抽象診断
                diag = diagnoser.diagnose(
                    error=err,
                    steps_taken=steps,
                    code=code,
                    raw_verification=sim_res,
                )
                feedback = (
                    f"Category: {diag['category']}\n"
                    f"Root Cause: {diag['root_cause']}\n"
                    f"Directive: {diag['directive']}"
                )

        loop_results.append(
            {
                "loop_index": loop_idx,
                "task_id": t_id,
                "task_name": t_name,
                "is_cleared": is_cleared,
                "attempts_used": len(attempt_history),
                "attempt_history": attempt_history,
            }
        )

    logger.log("=" * 70)
    logger.log("🏁 10回自己改善・進化ループ実験完了サマリー")
    cleared_total = sum(1 for r in loop_results if r["is_cleared"])
    logger.log(f"総ループ数: {len(loop_results)}")
    logger.log(
        f"クリア数: {cleared_total}/{len(loop_results)} ({cleared_total / len(loop_results) * 100:.1f}%)"
    )
    logger.log(f"総自己修正リトライ回数: {total_retries_needed}")
    logger.log("=" * 70)

    # 結果 JSON 保存
    result_json_path = Path("logs/evolution_10_results.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(loop_results, f, indent=2, ensure_ascii=False)
    logger.log(f"詳細解析データを保存しました: {result_json_path}")
    logger.close()


if __name__ == "__main__":
    run_10_loops()
