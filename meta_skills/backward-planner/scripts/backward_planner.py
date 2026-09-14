#!/usr/bin/env python3
"""Backward Planner - Unified Milestone Decomposition, Topological Prerequisites & Buffer Staging (ACR-AGI-3).

Consolidates all goal planning capabilities:
1. Backward Chaining from terminal target state with Staging Buffer allocation.
2. Topological Prerequisite Ordering (Keys before doors, switches before barriers).
3. Provides DecompositionPlan and Subgoal dataclasses for downstream agent execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
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


class BackwardPlanner:
    """ACR-AGI-3 統合目標分解・逆算プランニング・待避バッファ管理エンジン."""

    def __init__(self) -> None:
        pass

    def plan_backward_subgoals(
        self,
        current_sequence: List[int],
        target_sequence: List[int],
        open_grid_spaces: Optional[List[List[int]]] = None,
        anchor_preference: str = "LAST_PIECE",
    ) -> Dict[str, Any]:
        """目標シーケンスから逆算し、待避バッファを伴う中間サブゴール列を生成."""
        if not target_sequence:
            return {
                "success": False,
                "error": "Target sequence cannot be empty",
                "subgoals": [],
                "buffer_allocated": False,
            }

        if not current_sequence:
            return {
                "success": False,
                "error": "Current sequence cannot be empty",
                "subgoals": [],
                "buffer_allocated": False,
            }

        if current_sequence == target_sequence:
            return {
                "success": True,
                "is_already_solved": True,
                "subgoals": [
                    {
                        "subgoal_index": 1,
                        "action_type": "TRIGGER_COMPLETION",
                        "description": "Sequence already matches target. Trigger clearance.",
                        "target_colors": target_sequence,
                    }
                ],
                "buffer_allocated": False,
            }

        subgoals: List[Dict[str, Any]] = []
        staging_buffers = open_grid_spaces or [[0, 0], [0, 1]]

        total_items = len(target_sequence)
        first_target_elem = target_sequence[0]
        blocking_elements = [c for c in current_sequence if c != first_target_elem]

        # ステップ 1: 干渉ピースの一時待避
        subgoals.append({
            "subgoal_index": 1,
            "phase": "ISOLATE_AND_STAGE",
            "action_type": "DISPLACE_TO_BUFFER",
            "description": f"Temporarily move interfering pieces {blocking_elements[:2]} to buffer area to free keystone piece {first_target_elem}.",
            "target_colors": blocking_elements[:2],
            "staging_buffer_coords": staging_buffers[0] if staging_buffers else [0, 0],
            "tolerates_temporary_disorder": True,
        })

        # ステップ 2: キーストーンの確保
        subgoals.append({
            "subgoal_index": 2,
            "phase": "SECURE_KEYSTONE",
            "action_type": "LOCK_ANCHOR",
            "description": f"Move target keystone piece {first_target_elem} into primary assembly lane.",
            "target_colors": [first_target_elem],
            "staging_buffer_coords": None,
            "tolerates_temporary_disorder": False,
        })

        # ステップ 3: 後続ピースの部分結合
        for idx in range(1, total_items):
            current_target_color = target_sequence[idx]
            subgoals.append({
                "subgoal_index": 2 + idx,
                "phase": "INCREMENTAL_CHAIN",
                "action_type": "ATTACH_BLOCK",
                "description": f"Retrieve color {current_target_color} and attach to growing sequence up to index {idx}.",
                "target_colors": [current_target_color],
                "staging_buffer_coords": None,
                "tolerates_temporary_disorder": False,
            })

        # ステップ 4: 最終アライメント
        subgoals.append({
            "subgoal_index": len(subgoals) + 1,
            "phase": "FINAL_ALIGNMENT",
            "action_type": "TRIGGER_COMPLETION",
            "description": "Align full chained sequence with terminal clearance sensor/piston.",
            "target_colors": target_sequence,
            "staging_buffer_coords": None,
            "tolerates_temporary_disorder": False,
        })

        return {
            "success": True,
            "is_already_solved": False,
            "backward_chaining_applied": True,
            "total_subgoals": len(subgoals),
            "buffer_allocated": True,
            "staging_buffers": staging_buffers,
            "subgoals": subgoals,
        }

    def decompose_grid_objective(
        self,
        report: Any,
        grid: Optional[np.ndarray] = None,
    ) -> DecompositionPlan:
        """後方互換性API: アフォーダンスレポートから中間サブゴール計画 (DecompositionPlan) を生成."""
        agent_pos = getattr(report, "agent_pos", None)
        target_candidates = getattr(report, "target_candidates", [])
        interactables = getattr(report, "interactables", {})

        subgoals: List[Subgoal] = []
        step_idx = 1

        # 1. 前提条件（鍵・スイッチ）のトポロジカル解決
        if interactables:
            for name, pos in interactables.items():
                subgoals.append(
                    Subgoal(
                        index=step_idx,
                        name=f"Interact with {name}",
                        objective=f"Navigate to {name} at {pos} and trigger precondition",
                        reasoning="Prerequisite interaction required to unlock downstream path",
                        expected_operation="INTERACT",
                        parameters={"target_pos": pos},
                    )
                )
                step_idx += 1

        # 2. ターゲットの到達サブゴール
        if target_candidates:
            primary_target = target_candidates[0]
            t_pos = (int(round(primary_target.center_r)), int(round(primary_target.center_c)))
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="Reach Primary Target",
                    objective=f"Navigate agent to target object {primary_target.obj_id} at {t_pos}",
                    reasoning="Primary clearance objective milestone",
                    expected_operation="NAVIGATE_TO_TARGET",
                    parameters={"target_pos": t_pos, "target_id": primary_target.obj_id},
                )
            )
            step_idx += 1
        elif agent_pos is not None:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="Exploratory Navigation",
                    objective="Explore unvisited open cells to discover goal or trigger",
                    reasoning="Target not yet identified; proceed with frontier exploration",
                    expected_operation="EXPLORE_FRONTIER",
                )
            )

    def decompose_game(
        self,
        grid: Any,
        known_roles: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> DecompositionPlan:
        """後方互換性API: グリッドから直接サブゴール計画 (DecompositionPlan) を分解・生成."""
        arr = np.array(grid, dtype=int)
        agent_color = known_roles.get("agent") if known_roles else 2
        goal_color = known_roles.get("goal") if known_roles else 3

        subgoals: List[Subgoal] = []
        step_idx = 1

        # 鍵等のインタラクタブルの検出
        for c in np.unique(arr):
            if c not in (0, 1, agent_color, goal_color):
                coords = np.argwhere(arr == c)
                if len(coords) > 0:
                    pos = (int(coords[0, 0]), int(coords[0, 1]))
                    subgoals.append(
                        Subgoal(
                            index=step_idx,
                            name=f"Acquire_item_color_{c}",
                            objective=f"Navigate to item at {pos} before goal",
                            reasoning="Causal prerequisite item must be collected",
                            expected_operation="COLLECT_ITEM",
                            parameters={"target_pos": pos, "color": int(c)},
                        )
                    )
                    step_idx += 1

        # 障害物回避ステップの挿入（障害物が存在する場合）
        if np.any(arr == 1):
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="BypassCentralObstacle",
                    objective="Navigate around obstacle barrier to reach corridor",
                    reasoning="Avoid blocking obstacles to ensure navigation path",
                    expected_operation="BYPASS_OBSTACLE",
                )
            )
            step_idx += 1

        # ゴール到達
        goal_coords = np.argwhere(arr == goal_color)
        if len(goal_coords) > 0:
            g_pos = (int(goal_coords[0, 0]), int(goal_coords[0, 1]))
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="ReachGoalAndClearStage",
                    objective=f"Navigate to goal at {g_pos}",
                    reasoning="Final stage clearance",
                    expected_operation="REACH_GOAL",
                    parameters={"target_pos": g_pos},
                )
            )

        return DecompositionPlan(
            task_hint="Reach goal avoiding obstacles with prerequisites",
            subgoals=subgoals,
            total_steps=len(subgoals),
            constraints=["Avoid impassable obstacle walls", "Collect prerequisites first"],
            reasoning_trace="Decomposed game milestones via BackwardPlanner",
        )



# 後方互換性エイリアス
SubgoalDecomposer = BackwardPlanner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backward Planner - Unified Milestone & Prerequisite Planning CLI"
    )
    parser.add_argument("--current", type=str, default="[4, 3, 2, 1]", help="Current color sequence")
    parser.add_argument("--target", type=str, default="[1, 2, 3, 4]", help="Target color sequence")
    parser.add_argument("--buffers", type=str, default="[[0, 0], [0, 1]]", help="Open buffer coordinates")
    parser.add_argument("--input", type=str, help="Input data string (compatibility)")
    parser.add_argument("--file", type=str, help="Path to input JSON file")
    args = parser.parse_args()

    planner = BackwardPlanner()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            current = data.get("current", [4, 3, 2, 1])
            target = data.get("target", [1, 2, 3, 4])
            buffers = data.get("buffers", [[0, 0]])
    else:
        current = json.loads(args.current)
        target = json.loads(args.target)
        buffers = json.loads(args.buffers)

    res = planner.plan_backward_subgoals(current, target, open_grid_spaces=buffers)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
