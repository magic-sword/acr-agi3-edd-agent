#!/usr/bin/env python3
"""Subgoal Decomposer - Core CLI & Script Tool (ACR-AGI-3).

タスクの観測状態から、検証可能な中間マイルストーン (Subgoals) のシーケンスを生成します。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Subgoal:
    """中間マイルストーン (Subgoal)."""

    index: int
    name: str
    objective: str
    reasoning: str
    expected_operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionPlan:
    """階層分解されたサブゴール計画."""

    task_hint: str
    subgoals: List[Subgoal]
    total_steps: int
    constraints: List[str] = field(default_factory=list)
    reasoning_trace: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_hint": self.task_hint,
            "total_steps": self.total_steps,
            "constraints": self.constraints,
            "reasoning_trace": self.reasoning_trace,
            "subgoals": [
                {
                    "step": s.index,
                    "name": s.name,
                    "objective": s.objective,
                    "reasoning": s.reasoning,
                    "operation": s.expected_operation,
                    "params": s.parameters,
                }
                for s in self.subgoals
            ],
        }


def _get_meta_observer_class():
    """env-observer スキルから MetaObserver を取得."""
    curr_dir = Path(__file__).resolve().parent
    env_script = curr_dir.parent.parent / "env-observer" / "scripts" / "env_observer.py"
    if not env_script.exists():
        # Kaggle 展開時フォールバック
        env_script = Path("/kaggle/working/meta_skills/env-observer/scripts/env_observer.py")

    if env_script.exists():
        spec = importlib.util.spec_from_file_location("skill_env_observer", env_script)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod.MetaObserver
    return None


class SubgoalDecomposer:
    """VCGT 思考モデルに準拠したサブゴール分解エンジン."""

    def __init__(self, observer: Any = None) -> None:
        if observer is not None:
            self.observer = observer
        else:
            obs_cls = _get_meta_observer_class()
            self.observer = obs_cls() if obs_cls else None

    def decompose(
        self,
        obs: np.ndarray,
        goal_description: str = "",
        known_roles: Optional[Dict[str, int]] = None,
    ) -> DecompositionPlan:
        return self.decompose_game(obs, goal_description, known_roles)

    def decompose_game(
        self,
        obs: np.ndarray,
        goal_description: str = "",
        known_roles: Optional[Dict[str, int]] = None,
    ) -> DecompositionPlan:
        if self.observer is None:
            obs_cls = _get_meta_observer_class()
            self.observer = obs_cls() if obs_cls else None

        aff = self.observer.analyze_frame(obs, known_roles=known_roles) if self.observer else None
        subgoals: List[Subgoal] = []
        step_idx = 1
        constraints: List[str] = [
            "Avoid impassable obstacle walls at all costs.",
            "Do not step outside grid boundaries.",
        ]

        player_pos = getattr(aff, "player_pos", (1, 1)) if aff else (1, 1)
        goal_pos = getattr(aff, "goal_pos", None) if aff else None
        obstacles = getattr(aff, "obstacles", set()) if aff else set()
        interactables = getattr(aff, "interactables", {}) if aff else {}

        # 1. アイテム/鍵の収集サブゴール
        for item_name, item_pos in interactables.items():
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name=f"Acquire_{item_name}",
                    objective=f"Navigate to item {item_name} at coordinate {item_pos} to unlock downstream path",
                    reasoning=f"Acquire item {item_name} required for progression.",
                    expected_operation="acquire_item",
                    parameters={"item_name": item_name, "item_pos": item_pos},
                )
            )
            step_idx += 1

        # 2. 中央障害物壁の迂回サブゴール
        if len(obstacles) > 0 and goal_pos:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="BypassCentralObstacle",
                    objective="Navigate around obstacle barriers to reach open corridor",
                    reasoning="Circumvent walls to establish a continuous trajectory towards goal.",
                    expected_operation="bypass_obstacle",
                )
            )
            step_idx += 1

        # 3. ゴール到達サブゴール
        if goal_pos:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="ReachGoalAndClearStage",
                    objective=f"Navigate to exit goal at coordinate {goal_pos} to complete stage",
                    reasoning="Enter the goal cell to trigger stage completion.",
                    expected_operation="reach_goal",
                    parameters={"goal_pos": goal_pos},
                )
            )
        else:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="ExploreUnseenTerritory",
                    objective="Explore unvisited open cells to discover goal or interactable target",
                    reasoning="Goal location is not yet visible in the immediate observation field.",
                    expected_operation="explore",
                )
            )

        task_hint = goal_description or f"Navigate from {player_pos} to {goal_pos} avoiding {len(obstacles)} obstacles."

        return DecompositionPlan(
            task_hint=task_hint,
            subgoals=subgoals,
            total_steps=len(subgoals),
            constraints=constraints,
            reasoning_trace="Game environment decomposition based on affordances and obstacles.",
        )


def run(input_val: Any = None) -> Dict[str, Any]:
    """Core subgoal decomposition task."""
    grid = None
    goal_desc = ""

    if input_val is not None:
        if isinstance(input_val, dict):
            grid_raw = input_val.get("grid") or input_val.get("observation")
            if grid_raw is not None:
                grid = np.array(grid_raw, dtype=int)
            goal_desc = input_val.get("goal_description", "")
        elif isinstance(input_val, str):
            try:
                parsed = json.loads(input_val)
                if isinstance(parsed, dict):
                    grid_raw = parsed.get("grid") or parsed.get("observation")
                    if grid_raw is not None:
                        grid = np.array(grid_raw, dtype=int)
                    goal_desc = parsed.get("goal_description", "")
                elif isinstance(parsed, list):
                    grid = np.array(parsed, dtype=int)
            except Exception:
                pass
        elif isinstance(input_val, (list, np.ndarray)):
            grid = np.array(input_val, dtype=int)

    if grid is None:
        grid = np.zeros((10, 10), dtype=int)
        grid[1, 1] = 2
        grid[8, 8] = 3

    decomposer = SubgoalDecomposer()
    plan = decomposer.decompose_game(grid, goal_description=goal_desc)
    return plan.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Subgoal Decomposer execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input JSON/grid")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input JSON/grid")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    res = run(input_val)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
