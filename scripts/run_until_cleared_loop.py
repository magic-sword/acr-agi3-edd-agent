"""ACR-AGI-3 1ゲーム完了まで自律完走する自己改善ループスクリプト.

ステージクリア (is_solved == True) を達成するまで、
ローカル LLM (Qwen2.5-Coder-1.5B / CUDA) が試行・失敗診断・自己修復ループを継続します。
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from acr_agi3.agent.llm.arc_tools import execute_and_verify_game_policy, extract_python_code
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.game.vcgt_game import GridWorldGameEnv
from acr_agi3.meta.decomposer import SubgoalDecomposer
from acr_agi3.meta.diagnoser import FailureDiagnoser
from acr_agi3.meta.intuitor import GameStyleIntuitor
from acr_agi3.meta.observer import MetaObserver


class LoopLogger:
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


def run_until_cleared(max_attempts: int = 15):
    log_path = Path("logs/until_cleared_experiment.log")
    logger = LoopLogger(log_path)

    logger.log("=" * 75)
    logger.log("🎮 ACR-AGI-3 単一ゲーム完了まで完走する自己改善ループ実験")
    logger.log(f"🎯 目標: ステージクリア達成まで最大 {max_attempts} 回自己修復ループを継続")
    logger.log("🧠 モデル: Qwen/Qwen2.5-Coder-1.5B-Instruct (CUDA)")
    logger.log("=" * 75)

    # 1. ゲーム環境定義 (迂回壁のある 8x8 ナビゲーション迷路)
    # (1, 1) から (6, 6) を目指すが、列 3 に壁があり (5, 3) が抜け道
    walls = (
        {(0, c) for c in range(8)}
        | {(7, c) for c in range(8)}
        | {(r, 0) for r in range(8)}
        | {(r, 7) for r in range(8)}
        | {(1, 3), (2, 3), (3, 3), (4, 3)}  # 列3の遮断壁、行5以降が開通
    )
    player_pos = (1, 1)
    goal_pos = (6, 6)

    env = GridWorldGameEnv(
        grid_shape=(8, 8),
        initial_player_pos=player_pos,
        goal_pos=goal_pos,
        walls=walls,
    )
    initial_obs = env.reset()

    # 2. メタスキル初期化 & 初期観測解析
    intuitor = GameStyleIntuitor()
    observer = MetaObserver()
    decomposer = SubgoalDecomposer(observer=observer)
    diagnoser = FailureDiagnoser()

    style_report = intuitor.analyze_style(initial_obs)
    plan = decomposer.decompose_game(initial_obs)

    logger.log(f"👁️ [GameStyleIntuitor] 直感スタイル: {style_report['style']}")
    logger.log(f"   方針: {style_report['recommended_approach']}")
    logger.log(f"📋 [SubgoalDecomposer] マイルストーン: {[s.name for s in plan.subgoals]}")

    # 3. ローカルモデルのロード
    snap_dir = (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct"
        "/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
    )
    logger.log("📥 ローカル推論エンジンをロード中...")
    local_llm = LocalTransformersLlm(model_name_or_path=snap_dir, device="cuda")
    logger.log("✅ ローカル LLM ロード完了")
    logger.log("-" * 75)

    # 4. 自己改善ループ実行
    attempt = 0
    is_cleared = False
    feedback_history: List[str] = []
    attempt_records: List[Dict[str, Any]] = []
    cleared_code = ""

    while not is_cleared and attempt < max_attempts:
        attempt += 1
        logger.log(f"▶️ [Attempt {attempt}/{max_attempts}] ポリシー生成 & 自己修復試行")

        feedback_section = ""
        if feedback_history:
            recent_feedbacks = "\n".join(f"- {fb}" for fb in feedback_history[-3:])
            feedback_section = (
                f"\n[CRITICAL FAILURE DIAGNOSIS & REPAIR INSTRUCTIONS]:\n{recent_feedbacks}\n"
            )

        prompt = f"""You are an expert Python AI programmer controlling a grid game agent.
Grid Shape: (8, 8)
Player: {player_pos} (color 2), Goal: {goal_pos} (color 3), Walls: color 1
Game Style: {style_report['style']}
Strategy: {style_report['recommended_approach']}
Subgoals: {', '.join(s.name for s in plan.subgoals)}
{feedback_section}
CRITICAL CODING GUIDELINES:
- Complete function: `def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:`
- Return value MUST be an Action enum (UP, DOWN, LEFT, RIGHT, or WAIT).
- NEVER return coordinate tuples like (r, c) or call custom actions like Action.MOVE_TO.
- To find coordinates:
  `p_pts = np.argwhere(obs == 2)`
  `g_pts = np.argwhere(obs == 3)`
  `if len(p_pts) == 0 or len(g_pts) == 0: return Action.WAIT`
  `pr, pc = p_pts[0]`
  `gr, gc = g_pts[0]`
- Avoid walls: If target neighbor `obs[nr, nc] == 1`, pick another non-wall direction.

Write the complete Python action policy function:
```python
import numpy as np
from acr_agi3.game.env import Action

def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:
    # Identify positions, check adjacent walls, step toward goal
```
Output the Python code block.
"""

        start_t = time.time()
        raw_resp = local_llm.generate(prompt)
        gen_sec = time.time() - start_t
        code = extract_python_code(raw_resp)

        # シミュレーション検証
        sim_res = execute_and_verify_game_policy(code, env, max_steps=40)
        success = sim_res.get("success", False)
        steps = sim_res.get("steps_taken", 0)
        err = sim_res.get("error", "None")

        record = {
            "attempt": attempt,
            "gen_sec": gen_sec,
            "success": success,
            "steps": steps,
            "error": err,
            "code_snippet": code[:150] + "..." if len(code) > 150 else code,
        }

        if success:
            logger.log(
                f"   🎉 [Attempt {attempt}] ステージクリア達成！！"
                f" (所要ステップ: {steps}, 生成時間: {gen_sec:.2f}s)",
                level="SUCCESS",
            )
            is_cleared = True
            cleared_code = code
            attempt_records.append(record)
            break
        else:
            # FailureDiagnoser による抽象診断
            diag = diagnoser.diagnose(error=err, steps_taken=steps, code=code)
            diag_cat = diag.get("category", "GeneralFailure")
            directive = diag.get("directive", "")
            logger.log(
                f"   ⚠️ [Attempt {attempt}] 未達 (ステップ: {steps}, エラー: {err[:60]})",
                level="WARN",
            )
            logger.log(f"       🩺 [FailureDiagnoser] 分類: {diag_cat} | 指示: {directive}")
            feedback_history.append(f"[{diag_cat}] {directive}")
            record["diagnosis_category"] = diag_cat
            record["directive"] = directive
            attempt_records.append(record)

    logger.log("=" * 75)
    logger.log(f"🏁 実験完了: クリア達成={is_cleared} (試行回数: {attempt}/{max_attempts})")
    logger.log("=" * 75)

    # 結果JSON保存
    out_json = Path("logs/until_cleared_results.json")
    summary = {
        "is_cleared": is_cleared,
        "total_attempts": attempt,
        "task": {
            "grid_size": [8, 8],
            "player_pos": list(player_pos),
            "goal_pos": list(goal_pos),
            "walls_count": len(walls),
        },
        "cleared_code": cleared_code if is_cleared else None,
        "attempts": attempt_records,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.log(f"詳細データを保存しました: {out_json}")
    logger.close()


if __name__ == "__main__":
    run_until_cleared(max_attempts=15)
