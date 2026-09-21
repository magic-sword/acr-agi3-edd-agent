#!/usr/bin/env python3
"""ARC-AGI-3 Official Local Leaderboard Evaluation Pipeline.

Kaggle 本番の提出枠を消費せずに、公式採点システム (arc_agi ScorecardManager)
に 100% 準拠したリーダーボードスコアをローカルで高速測定・記録する。

使用方法:
  # 高速評価モード (5環境、約3秒)
  docker exec arc-agi3-dev python3 /workspace/scripts/run_local_leaderboard.py --fast

  # 本番完全同等モード (全25環境、各環境ベースライン×2倍ステップ)
  docker exec arc-agi3-dev python3 /workspace/scripts/run_local_leaderboard.py --full

  # 特定のゲームのみテスト
  docker exec arc-agi3-dev python3 /workspace/scripts/run_local_leaderboard.py -g tu93 -s 150 -v
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルートの設定
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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
    candidates = [
        REPO_ROOT / "data" / "competition" / "environment_files",
        Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
        REPO_ROOT / "data" / "environment_files",
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError("公式 environment_files ディレクトリが見つかりません。")


def load_submission_agent():
    """提出ノートブック (submission_template.ipynb) から MyAgent を直接抽出ロード."""
    nb_path = REPO_ROOT / "notebooks" / "submission_template.ipynb"
    if not nb_path.exists():
        nb_path = REPO_ROOT / "deploy" / "kaggle_kernel" / "submission_template.ipynb"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    agent_globals: Dict[str, Any] = {"__name__": "__main__"}
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # マジックコマンドやシェルコマンドを除外
        code_lines = [
            line for line in src.splitlines(keepends=True)
            if not line.startswith("%%") and not line.strip().startswith("!")
        ]
        clean_code = "".join(code_lines)
        if clean_code.strip():
            try:
                exec(clean_code, agent_globals)
            except Exception:
                pass

        if "MyAgent" in agent_globals and callable(agent_globals["MyAgent"]):
            break

    agent_class = agent_globals.get("MyAgent")
    if agent_class is None:
        from acr_agi3.agent.my_agent import MyAgent
        agent_class = MyAgent
    return agent_class


def _extract_2d_grid(g: Any) -> List[List[int]]:
    """3次元テンソル (N, H, W) またはリストを最新フレームの 2D リストへ正規化."""
    if not g:
        return []
    if isinstance(g, (list, tuple)) and len(g) > 0:
        if isinstance(g[0], (list, tuple)) and len(g[0]) > 0 and isinstance(g[0][0], (list, tuple)):
            g = g[-1]
        elif len(g) == 1 and isinstance(g[0], (list, tuple)):
            g = g[0]
    return g


def evaluate_single_environment(
    arcade: Arcade,
    agent_class: Any,
    game_id: str,
    title: str,
    baseline_actions: int,
    max_steps: int,
    scorecard_id: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """1つの公式ゲーム環境に対してエージェントを実行."""
    env = arcade.make(game_id, scorecard_id=scorecard_id)
    if env is None:
        return {
            "game_id": game_id,
            "title": title,
            "status": "ERROR",
            "levels_completed": 0,
            "win_levels": 0,
            "steps": 0,
            "score": 0.0,
            "errors": ["Failed to make environment"],
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
    raw_obs_frame = [arr.tolist() for arr in obs.frame] if obs.frame else []
    frames = [
        FrameData(
            levels_completed=0,
            state=obs.state,
            frame=raw_obs_frame,
            guid=getattr(obs, "guid", ""),
            win_levels=getattr(obs, "win_levels", 0),
            available_actions=getattr(obs, "available_actions", None),
        )
    ]

    session_telemetry = SessionTelemetry(
        game_id=game_id,
        title=title,
        baseline_actions=baseline_actions,
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

        # フレーム差分計算 (2D 正規化)
        old_2d = _extract_2d_grid(latest_frame.frame or [])
        raw_new = [arr.tolist() for arr in res.frame] if res.frame is not None else []
        new_2d = _extract_2d_grid(raw_new)

        diff_count = 0
        changed_colors = set()
        if old_2d and new_2d and len(old_2d) == len(new_2d) and len(old_2d[0]) == len(new_2d[0]):
            for r in range(len(old_2d)):
                for c in range(len(old_2d[0])):
                    val_o = old_2d[r][c]
                    val_n = new_2d[r][c]
                    o_s = val_o[0] if isinstance(val_o, (list, tuple)) else val_o
                    n_s = val_n[0] if isinstance(val_n, (list, tuple)) else val_n
                    if o_s != n_s:
                        diff_count += 1
                        changed_colors.add(n_s)

        is_eff = (diff_count > 0) or (res.levels_completed > latest_frame.levels_completed) or (res.state is GameState.WIN)
        # Keep upstream pixel-effectiveness semantics intact; report actual
        # ARC game progress and prediction outcomes separately in reasoning.
        reasoning = dict(reasoning)
        outcome = {
            "screen_changed": diff_count > 0,
            "level_progress": res.levels_completed > latest_frame.levels_completed,
            "win": res.state is GameState.WIN,
            "prediction_match": None,
        }
        player = getattr(agent, "player", None)
        evidence = getattr(player, "execution_evidence", None)
        if evidence is not None and evidence.expected is not None:
            import numpy as np
            color, expected_mask = evidence.expected
            actual = np.asarray(new_2d)
            outcome["prediction_match"] = bool(
                actual.shape == expected_mask.shape
                and np.array_equal(actual == color, expected_mask)
            )
        reasoning["execution_outcome"] = outcome

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
            skill = reasoning.get("loaded_skill") if isinstance(reasoning, dict) else None
            skill_str = f" 🔮[{skill}]" if skill else ""

            # 自己改善 (Reviewer 介入) の可視化バッジ
            was_rev = reasoning.get("was_revised", False) if isinstance(reasoning, dict) else False
            orig_act = reasoning.get("original_action") if isinstance(reasoning, dict) else None
            rev_str = f" 🔄[Rev: {orig_act}->{act_label}]" if was_rev and orig_act and orig_act != act_label else (" 🔄[Rev]" if was_rev else "")

            strat = reasoning.get("strategy", "") if isinstance(reasoning, dict) else str(reasoning)
            strat_str = f" | {strat[:45]}..." if strat else ""
            print(
                f"    [Step {steps:02d}] {act_label:<7}{coords:<14}{skill_str}{rev_str} | Eff: {eff_sym} | "
                f"ΔPixels: {diff_count:3d} | Level: {res.levels_completed}/{res.win_levels} | "
                f"State: {str(res.state).replace('GameState.', '')} ({time_taken_ms:.1f}ms){strat_str}",
                flush=True
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
    is_cleared = (last_frame.state is GameState.WIN) or (
        last_frame.levels_completed >= last_frame.win_levels and last_frame.win_levels > 0
    )

    session_telemetry.final_state = str(last_frame.state)
    session_telemetry.total_levels_completed = last_frame.levels_completed
    session_telemetry.total_win_levels = last_frame.win_levels

    diagnostic_report = DiagnosticAnalyzer.analyze(session_telemetry)
    outcomes = [
        (step.reasoning or {}).get("execution_outcome", {})
        for step in session_telemetry.steps
    ]
    predictions = [o["prediction_match"] for o in outcomes if o.get("prediction_match") is not None]
    execution_metrics = {
        "prediction_checks": len(predictions),
        "prediction_matches": sum(predictions),
        "prediction_match_rate": sum(predictions) / len(predictions) if predictions else None,
        "level_progress_steps": sum(o.get("level_progress", False) for o in outcomes),
        "perception_cache_hits": sum(
            (step.reasoning or {}).get("phase_metrics", {}).get("perception_reused", False)
            for step in session_telemetry.steps
        ),
        "repeated_click_trials": sum(
            (step.reasoning or {}).get("repeated_click_trial", False)
            for step in session_telemetry.steps
        ),
    }

    return {
        "game_id": game_id,
        "title": title,
        "status": "WIN" if is_cleared else str(last_frame.state).replace("GameState.", ""),
        "levels_completed": last_frame.levels_completed,
        "win_levels": last_frame.win_levels,
        "steps": steps,
        "errors": errors,
        "diagnostics": diagnostic_report,
        "telemetry": session_telemetry,
        "execution_metrics": execution_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Official Local Leaderboard Pipeline")
    parser.add_argument("--fast", action="store_true", help="高速評価モード: 代表5環境×最大100ステップ (~3秒)")
    parser.add_argument("--full", action="store_true", help="本番完全同等モード: 全25環境×ベースラインステップ (~20秒)")
    parser.add_argument("-n", "--num-envs", type=int, default=None, help="評価する環境数 (デフォルト: 全25環境)")
    parser.add_argument("-s", "--max-steps", type=int, default=None, help="1環境あたりの最大ステップ数")
    parser.add_argument("-g", "--game-id", type=str, default=None, help="特定のゲーム ID (例: tu93, ft09)")
    parser.add_argument("-v", "--verbose", action="store_true", help="ステップごとの詳細ログを表示")
    parser.add_argument("--allow-cpu", action="store_true", help="GPU 利用不能時の CPU 実行を明示的に許可 (デフォルト: 不可・即時エラー停止)")
    args = parser.parse_args()
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    print("=" * 78)
    print("🏆 [ARC-AGI-3 LOCAL LEADERBOARD] Official Offline Evaluation Pipeline")
    print("=" * 78)

    # ハードウェア加速 (GPU / CUDA) 事前健全性チェック
    import torch
    print("🔍 [PREFLIGHT] Hardware Acceleration Check...")
    if not torch.cuda.is_available():
        err_msg = (
            "❌ [FATAL ERROR] GPU (CUDA) が検出されませんでした (torch.cuda.is_available() == False)！\n"
            "   NVML エラーまたは GPU パススルー未設定の可能性があります。\n"
            "   CPU 推論では 1 回の評価に 10 時間以上要し、研究・実験サイクルを著しく阻害するため即時中断します。\n"
            "   【対策】\n"
            "     1. コンテナを再起動してください: docker restart arc-agi3-dev\n"
            "     2. ホスト側で nvidia-smi が動作しているか確認してください。\n"
            "   ※ 意図的に CPU で極小ステップ動作を確認したい場合のみ `--allow-cpu` を指定してください。"
        )
        if not args.allow_cpu:
            print(err_msg, file=sys.stderr)
            sys.exit(1)
        else:
            print("⚠️ [WARNING] GPU 未検出ですが、--allow-cpu が指定されたため CPU で続行します (大幅な遅延にご注意ください)。")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ [PREFLIGHT] GPU Detected: {gpu_name} (VRAM: {vram_gb:.1f} GB, CUDA {torch.version.cuda})")

    envs_dir = resolve_environments_dir()
    arcade = Arcade(environments_dir=envs_dir)

    all_envs = arcade.get_environments()
    if not all_envs:
        print("❌ 環境が見つかりませんでした。")
        sys.exit(1)

    # 評価環境リストの選定
    if args.game_id:
        target_envs = [
            e for e in all_envs
            if args.game_id.lower() in getattr(e, "game_id", "").lower()
            or args.game_id.lower() in getattr(e, "title", "").lower()
        ]
    elif args.fast:
        # 代表的な 5 環境 (移動系 TU93, S5I5, LS20 + クリック系 FT09, SB26)
        rep_ids = ["tu93", "ft09", "s5i5", "ls20", "sb26"]
        target_envs = [
            e for e in all_envs
            if any(r in getattr(e, "game_id", "").lower() for r in rep_ids)
        ]
        if not target_envs:
            target_envs = all_envs[:5]
    elif args.num_envs:
        target_envs = all_envs[:args.num_envs]
    else:
        target_envs = all_envs

    # ステップ数の決定
    if args.max_steps:
        default_steps = args.max_steps
    elif args.fast:
        default_steps = 80
    elif args.full:
        default_steps = 200
    else:
        default_steps = 100

    print(f"📦 Loading submission agent from notebooks/submission_template.ipynb...")
    agent_class = load_submission_agent()

    card_id = arcade.create_scorecard()
    print(f"📋 Initialized Official Scorecard: {card_id}")
    print(f"🎮 Target Environments: {len(target_envs)} (max_steps={default_steps})")
    print("-" * 78)

    results = []
    t_start = time.time()

    for idx, env_info in enumerate(target_envs, 1):
        g_id = getattr(env_info, "game_id", str(env_info))
        title = getattr(env_info, "title", g_id[:4].upper())
        baseline = getattr(env_info, "baseline_actions", [100])
        baseline_val = baseline[0] if isinstance(baseline, list) and baseline else 100

        if args.verbose:
            print(f"\n--- [{idx:02d}/{len(target_envs):02d}] Starting {g_id} ({title}) ---")

        res = evaluate_single_environment(
            arcade=arcade,
            agent_class=agent_class,
            game_id=g_id,
            title=title,
            baseline_actions=baseline_val,
            max_steps=default_steps,
            scorecard_id=card_id,
            verbose=args.verbose,
        )
        results.append(res)

        diag: DiagnosticReport = res["diagnostics"]
        status_sym = "🏆 WIN" if res["status"] == "WIN" else (f"⭐ L{res['levels_completed']}" if res["levels_completed"] > 0 else "❌ FAIL")
        prefix = "  Result -> " if args.verbose else f"[{idx:02d}/{len(target_envs):02d}] "
        print(
            f"{prefix}{status_sym:<7} | {g_id:<14} ({title:<6}) | "
            f"Levels: {res['levels_completed']:2d}/{res['win_levels']:2d} | "
            f"Steps: {res['steps']:3d} | Eff: {diag.effective_ratio*100:5.1f}% | "
            f"Stag: {diag.max_consecutive_stagnation:2d}s | {diag.dominant_failure_category}"
        )

    elapsed = time.time() - t_start
    scorecard = arcade.close_scorecard(card_id)

    official_score = scorecard.score if scorecard else 0.0
    total_completed = scorecard.total_environments_completed if scorecard else 0
    total_levels = scorecard.total_levels_completed if scorecard else 0

    print("\n" + "=" * 78)
    print("🏆 [OFFICIAL LEADERBOARD SCORE REPORT]")
    print("=" * 78)
    print(f"Evaluated Agent:          MyAgent (from notebooks/submission_template.ipynb)")
    print(f"Environments Evaluated:   {len(results)}")
    print(f"Environments Fully Won:   {total_completed} / {len(results)}")
    print(f"Total Levels Cleared:     {total_levels}")
    print(f"Official Scorecard Score: {official_score:.4f} (Kaggle Leaderboard Equivalent)")
    print(f"Total Evaluation Time:    {elapsed:.2f}s ({elapsed/max(1, len(results)):.3f}s/env)")
    print("=" * 78)

    # 履歴への自動保存 (logs/leaderboard_history.json)
    history_file = REPO_ROOT / "logs" / "leaderboard_history.json"
    history_file.parent.mkdir(exist_ok=True)

    history = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "fast" if args.fast else ("full" if args.full else "custom"),
        "score": official_score,
        "environments_cleared": total_completed,
        "levels_cleared": total_levels,
        "environments_count": len(results),
        "steps_per_env": default_steps,
        "total_time_seconds": round(elapsed, 2),
        "results": [
            {
                "game_id": r["game_id"],
                "title": r["title"],
                "levels_completed": r["levels_completed"],
                "win_levels": r["win_levels"],
                "status": r["status"],
                "eff_ratio": round(r["diagnostics"].effective_ratio, 3),
                "execution_metrics": r["execution_metrics"],
            }
            for r in results
        ],
    }
    history.append(record)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 詳細ステップテレメトリおよび EDD 診断レポートの保存
    import dataclasses
    step_telemetry_file = REPO_ROOT / "logs" / "step_telemetry_detailed.json"
    diag_file = REPO_ROOT / "logs" / "edd_diagnostics.json"

    detailed_telemetries = [dataclasses.asdict(r["telemetry"]) for r in results if "telemetry" in r]
    with open(step_telemetry_file, "w", encoding="utf-8") as f:
        json.dump(detailed_telemetries, f, indent=2)

    detailed_diags = [dataclasses.asdict(r["diagnostics"]) for r in results if "diagnostics" in r]
    with open(diag_file, "w", encoding="utf-8") as f:
        json.dump(detailed_diags, f, indent=2)

    print(f"\n📈 Saved benchmark result to: {history_file}")
    print(f"📊 Saved detailed step telemetry to: {step_telemetry_file}")
    print(f"🔬 Saved EDD diagnostic reports to: {diag_file}")
    if len(history) > 1:
        prev = history[-2]
        delta_score = official_score - prev.get("score", 0.0)
        delta_levels = total_levels - prev.get("levels_cleared", 0)
        sym = "+" if delta_score >= 0 else ""
        print(f"📊 Delta from previous run: {sym}{delta_score:.4f} score | {sym}{delta_levels} levels")


if __name__ == "__main__":
    main()
