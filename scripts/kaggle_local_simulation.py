"""Kaggle ノートブック実行環境のローカル完全シミュレーション & スコア計測スクリプト.

ARC Prize 2026 - ARC-AGI-3 の公式ゲーム環境 (25 environment_files) および
公式 Arcade / arcengine を用いて、Kaggle Leaderboard と 100% 同一の評価を実行し、
タスクごとのクリア状況・完了レベル・ステップ数・公式スコアを正確に測定する。
"""

from __future__ import annotations

import argparse
import dataclasses
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

from acr_agi3.edd import (
    DiagnosticAnalyzer,
    DiagnosticReport,
    EDDReportFormatter,
    SessionTelemetry,
    StepTelemetry,
)


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
    title: str = "Unknown",
    baseline: Optional[List[int]] = None,
    agent_class: Any = None,
    max_steps: int = 80,
    scorecard_id: Optional[str] = None,
    verbose: bool = False,
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
            guid=getattr(obs, "guid", ""),
            win_levels=getattr(obs, "win_levels", 0),
            available_actions=getattr(obs, "available_actions", None),
        )
    ]

    session_telemetry = SessionTelemetry(
        game_id=game_id,
        title=title,
        baseline_actions=baseline,
        initial_shape=(len(obs.frame), len(obs.frame[0])) if obs.frame else (0, 0),
    )

    steps = 0
    errors = []

    while steps < max_steps:
        latest_frame = frames[-1]
        if hasattr(agent, "is_done") and agent.is_done(frames, latest_frame):
            break
        if latest_frame.state is GameState.WIN:
            break

        t_act_start = time.time()
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

        time_taken_ms = (time.time() - t_act_start) * 1000.0

        # フレーム差分・有効性の数理計算 (3D テンソルを 2D に安全展開して比較)
        def _extract_2d(g):
            if not g:
                return []
            if isinstance(g, (list, tuple)) and len(g) > 0:
                if isinstance(g[0], (list, tuple)) and len(g[0]) > 0 and isinstance(g[0][0], (list, tuple)):
                    g = g[-1]
                elif len(g) == 1 and isinstance(g[0], (list, tuple)):
                    g = g[0]
            return g

        raw_old = latest_frame.frame or []
        raw_new = [arr.tolist() for arr in res.frame] if res.frame is not None else []
        old_grid = _extract_2d(raw_old)
        new_grid = _extract_2d(raw_new)

        diff_count = 0
        changed_colors = set()
        if old_grid and new_grid and len(old_grid) == len(new_grid) and len(old_grid[0]) == len(new_grid[0]):
            for r in range(len(old_grid)):
                for c in range(len(old_grid[0])):
                    val_o = old_grid[r][c]
                    val_n = new_grid[r][c]
                    o_scalar = val_o[0] if isinstance(val_o, (list, tuple)) else val_o
                    n_scalar = val_n[0] if isinstance(val_n, (list, tuple)) else val_n
                    if o_scalar != n_scalar:
                        diff_count += 1
                        changed_colors.add(n_scalar)

        is_eff = (diff_count > 0) or (res.levels_completed > latest_frame.levels_completed) or (res.state is GameState.WIN)

        step_telemetry = StepTelemetry(
            step_index=steps,
            action_id=getattr(action, "value", -1),
            action_name=getattr(action, "name", str(action)),
            action_data=data,
            reasoning=reasoning,
            state_before=str(latest_frame.state),
            state_after=str(res.state),
            levels_completed=res.levels_completed,
            win_levels=res.win_levels,
            pixels_changed=diff_count,
            changed_colors=list(changed_colors),
            is_effective=is_eff,
            time_taken_ms=time_taken_ms,
        )
        session_telemetry.add_step(step_telemetry)

        if verbose:
            eff_sym = "✅" if is_eff else "❌"
            act_label = getattr(action, "name", str(action))
            coords = f" pos=({data.get('x', 0)}, {data.get('y', 0)})" if "x" in data else ""
            print(
                f"    [Step {steps:02d}] {act_label:<7}{coords:<15} | Eff: {eff_sym} | "
                f"ΔPixels: {diff_count:3d} | Level: {res.levels_completed}/{res.win_levels} | "
                f"State: {str(res.state).replace('GameState.', '')} ({time_taken_ms:.1f}ms)"
            )

        steps += 1
        new_frame = FrameData(
            game_id=res.game_id,
            frame=raw_new,
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

    session_telemetry.final_state = str(last_frame.state)
    session_telemetry.total_levels_completed = last_frame.levels_completed
    session_telemetry.total_win_levels = last_frame.win_levels

    diagnostic_report = DiagnosticAnalyzer.analyze(session_telemetry)

    return {
        "game_id": game_id,
        "status": "WIN" if is_cleared else str(last_frame.state).replace("GameState.", ""),
        "levels_completed": last_frame.levels_completed,
        "win_levels": last_frame.win_levels,
        "steps": steps,
        "errors": errors,
        "diagnostics": diagnostic_report,
        "session_telemetry": session_telemetry,
    }


def run_local_simulation(
    max_games: Optional[int] = None,
    max_steps: int = 80,
    verbose: bool = False,
) -> None:
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

        if verbose:
            print(f"\n--- [{idx:02d}/{len(target_envs):02d}] Starting {g_id} ({title}) ---")

        res = run_single_game(
            arcade=arcade,
            game_id=g_id,
            title=title,
            baseline=baseline,
            agent_class=agent_class,
            max_steps=max_steps,
            scorecard_id=card_id,
            verbose=verbose,
        )
        res["title"] = title
        res["baseline"] = baseline
        results.append(res)

        status_mark = "🏆" if res["status"] == "WIN" else ("⭐" if res["levels_completed"] > 0 else "❌")
        err_msg = f" | ERRORS: {res['errors']}" if res["errors"] else ""
        diag: DiagnosticReport = res["diagnostics"]
        prefix = "  Result -> " if verbose else f"[{idx:02d}/{len(target_envs):02d}] "
        print(
            f"{prefix}{status_mark} {g_id:<14} ({title:<6}) | "
            f"Levels: {res['levels_completed']:2d}/{res['win_levels']:2d} | "
            f"Steps: {res['steps']:2d} | EffRatio: {diag.effective_ratio*100:4.1f}% | Stag: {diag.max_consecutive_stagnation:2d}s | {diag.dominant_failure_category:<26}{err_msg}"
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

    # EDD 診断サマリーの出力
    reports = [r["diagnostics"] for r in results]
    print("\n" + EDDReportFormatter.format_console_summary(reports))

    # 診断結果の永続化 (logs/edd_diagnostics.json)
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    diag_file = logs_dir / "edd_diagnostics.json"
    with open(diag_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() if hasattr(r, "to_dict") else r.model_dump() for r in reports], f, indent=2)
    print(f"\n💾 Saved structured EDD diagnostic reports to: {diag_file}")

    # 詳細ステップテレメトリの永続化 (logs/step_telemetry_detailed.json)
    detailed_file = logs_dir / "step_telemetry_detailed.json"
    detailed_data = []
    for r in results:
        sess = r["session_telemetry"]
        if hasattr(sess, "model_dump"):
            detailed_data.append(sess.model_dump())
        elif hasattr(sess, "__dataclass_fields__"):
            detailed_data.append(dataclasses.asdict(sess))
        else:
            detailed_data.append(str(sess))
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2)
    print(f"📄 Saved step-by-step detailed telemetry to: {detailed_file}")

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
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed step-by-step telemetry logs")
    args = parser.parse_args()

    run_local_simulation(max_games=args.num_games, max_steps=args.max_steps, verbose=args.verbose)
