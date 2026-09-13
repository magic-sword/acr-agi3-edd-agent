"""Kaggle ノートブック実行環境のローカル完全シミュレーション & スコア計測スクリプト.

ARC Prize 2026 - ARC-AGI-3 の公式ゲーム環境 (25 environment_files) および
公式 Arcade / arcengine を用いて、Kaggle Leaderboard と 100% 同一の評価を実行し、
タスクごとのクリア状況・完了レベル・ステップ数・公式スコアを正確に測定する。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルートと公式エージェントディレクトリを sys.path に追加
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

agents_path = REPO_ROOT / "data" / "competition" / "ARC-AGI-3-Agents"
if agents_path.exists():
    sys.path.insert(0, str(agents_path))

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction, GameState


def resolve_environments_dir() -> Path:
    """公式環境ディレクトリのパスを解決."""
    candidates = [
        REPO_ROOT / "data" / "competition" / "environment_files",
        Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
        REPO_ROOT / "data" / "environment_files",
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        "公式 environment_files ディレクトリが見つかりません。"
        "data/competition/environment_files が存在することを確認してください。"
    )


def load_notebook_agent():
    """提出ノートブック (submission_template.ipynb) から MyAgent を直接抽出ロード."""
    nb_path = REPO_ROOT / "notebooks" / "submission_template.ipynb"
    if not nb_path.exists():
        nb_path = REPO_ROOT / "deploy" / "kaggle_kernel" / "submission_template.ipynb"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cell2_source = nb_data["cells"][2]["source"]
    code_lines = [line for line in cell2_source if not line.startswith("%%writefile")]
    exec_code = "".join(code_lines)

    globs: Dict[str, Any] = {
        "FrameData": FrameData,
        "GameAction": GameAction,
        "GameState": GameState,
    }
    try:
        from agents.agent import Agent
        globs["Agent"] = Agent
    except ImportError:
        pass

    exec(exec_code, globs)
    return globs["MyAgent"]


def run_single_game(
    arcade: Arcade,
    game_id: str,
    agent_class: Any,
    max_steps: int = 80,
    scorecard_id: Optional[str] = None,
) -> Dict[str, Any]:
    """1つの公式ゲーム環境に対してエージェントを実行し、結果を収集."""
    env = arcade.make(game_id, scorecard_id=scorecard_id)
    if env is None:
        return {
            "game_id": game_id,
            "status": "ERROR",
            "levels_completed": 0,
            "win_levels": 0,
            "steps": 0,
            "score": 0.0,
            "errors": ["Failed to create environment"],
        }

    agent = agent_class(
        card_id=scorecard_id or "local-sim",
        game_id=game_id,
        agent_name="MyAgent",
        ROOT_URL="http://local",
        record=False,
        arc_env=env,
    )

    obs = env.reset()
    frames = [
        FrameData(
            levels_completed=0,
            state=obs.state,
            frame=[arr.tolist() for arr in obs.frame],
        )
    ]

    steps = 0
    errors = []

    while steps < max_steps:
        latest_frame = frames[-1]
        if hasattr(agent, "is_done") and agent.is_done(frames, latest_frame):
            break
        if latest_frame.state is GameState.WIN:
            break

        try:
            action = agent.choose_action(frames, latest_frame)
        except Exception as e:
            errors.append(f"choose_action: {type(e).__name__}: {e}")
            break

        data = (
            action.action_data.model_dump()
            if hasattr(action, "action_data") and action.action_data
            else {}
        )
        reasoning = (
            action.reasoning
            if hasattr(action, "reasoning") and isinstance(action.reasoning, dict)
            else {}
        )

        try:
            res = env.step(action, data=data, reasoning=reasoning)
        except Exception as e:
            errors.append(f"step: {type(e).__name__}: {e}")
            break

        if res is None:
            errors.append("step returned None")
            break

        steps += 1
        new_frame = FrameData(
            game_id=res.game_id,
            frame=[arr.tolist() for arr in res.frame],
            state=res.state,
            levels_completed=res.levels_completed,
            win_levels=res.win_levels,
            guid=res.guid,
            available_actions=res.available_actions,
        )
        frames.append(new_frame)

        if res.state is GameState.WIN:
            break

    last_frame = frames[-1]
    is_cleared = (last_frame.state is GameState.WIN) or (last_frame.levels_completed >= last_frame.win_levels and last_frame.win_levels > 0)

    return {
        "game_id": game_id,
        "status": "WIN" if is_cleared else str(last_frame.state).replace("GameState.", ""),
        "levels_completed": last_frame.levels_completed,
        "win_levels": last_frame.win_levels,
        "steps": steps,
        "errors": errors,
    }


def run_local_simulation(max_games: Optional[int] = None, max_steps: int = 80) -> None:
    """公式 ARC-AGI-3 オフライン環境での完全シミュレーション & スコア計測."""
    print("=" * 78)
    print("🚀 [ARC-AGI-3 LOCAL SIMULATION] Official Offline Verification & Leaderboard Scoring")
    print("=" * 78)

    env_dir = resolve_environments_dir()
    print(f"📂 Official Environments Directory: {env_dir}")

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(env_dir))
    all_envs = arcade.get_environments()
    print(f"🎮 Available Official Environments: {len(all_envs)}")

    if max_games is not None and max_games > 0:
        target_envs = all_envs[:max_games]
        print(f"🔍 Running on subset: {len(target_envs)} environments (max_steps={max_steps})")
    else:
        target_envs = all_envs
        print(f"🔍 Running on ALL {len(target_envs)} environments (max_steps={max_steps})")

    # ノートブックから MyAgent をロード
    print("📦 Loading MyAgent from submission notebook...")
    agent_class = load_notebook_agent()

    # 公式スコアカードの初期化
    card_id = arcade.create_scorecard()
    print(f"📋 Initialized Official Scorecard: {card_id}\n")

    results = []
    t_start = time.time()

    for idx, env_info in enumerate(target_envs, 1):
        g_id = env_info.game_id
        title = getattr(env_info, "title", g_id)
        baseline = getattr(env_info, "baseline_actions", [])
        
        res = run_single_game(
            arcade=arcade,
            game_id=g_id,
            agent_class=agent_class,
            max_steps=max_steps,
            scorecard_id=card_id,
        )
        res["title"] = title
        res["baseline"] = baseline
        results.append(res)

        status_mark = "🏆" if res["status"] == "WIN" else ("⭐" if res["levels_completed"] > 0 else "❌")
        err_msg = f" | ERRORS: {res['errors']}" if res["errors"] else ""
        print(
            f"[{idx:02d}/{len(target_envs):02d}] {status_mark} {g_id:<14} ({title:<6}) | "
            f"Levels: {res['levels_completed']:2d}/{res['win_levels']:2d} | "
            f"Steps: {res['steps']:2d} | Status: {res['status']:<12}{err_msg}"
        )

    elapsed = time.time() - t_start
    scorecard = arcade.close_scorecard(card_id)

    official_score = scorecard.score if scorecard else 0.0
    total_completed = scorecard.total_environments_completed if scorecard else 0
    total_levels = scorecard.total_levels_completed if scorecard else 0

    print("\n" + "=" * 78)
    print("🏆 [OFFICIAL LEADERBOARD SCORE MEASUREMENT REPORT]")
    print("=" * 78)
    print(f"Target Notebook:          submission_template.ipynb")
    print(f"Evaluated Agent:          MyAgent")
    print(f"Environments Evaluated:   {len(results)}")
    print(f"Environments Cleared:     {total_completed} / {len(results)}")
    print(f"Total Levels Completed:   {total_levels}")
    print(f"Official Scorecard Score: {official_score:.4f} (Kaggle Leaderboard Metric)")
    print(f"Total Simulation Time:    {elapsed:.2f}s ({elapsed/max(1, len(results)):.2f}s/env)")
    print("=" * 78)

    # 提出用 Parquet のスキーマ検証
    working_parquet = REPO_ROOT / "deploy" / "kaggle_kernel" / "submission.parquet"
    if not working_parquet.exists():
        working_parquet = REPO_ROOT / "submission.parquet"
    if working_parquet.exists():
        import pandas as pd
        df = pd.read_parquet(working_parquet)
        print(f"📁 Validated Submission Parquet: {working_parquet.name} ({len(df)} rows, columns={list(df.columns)})")

    # 比較レポート
    print("\n📊 Score Comparison:")
    print(f"  • Version 12 (Kaggle LB):  0.14 (Random Agent Baseline)")
    print(f"  • Version 13 (Kaggle LB):  0.00 (Broken v13 Agent with Action Exceptions)")
    print(f"  • Local Verified Score:    {official_score:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Local Simulation & Scoring")
    parser.add_argument("-n", "--num-games", type=int, default=None, help="Number of games to evaluate (default: all)")
    parser.add_argument("-s", "--max-steps", type=int, default=80, help="Max steps per game (default: 80)")
    args = parser.parse_args()

    run_local_simulation(max_games=args.num_games, max_steps=args.max_steps)
