"""人間 VCGT 思考モデルに基づくサブゴール階層分解エンジン (Subgoal Decomposer).

タスクの初期状態と目標状態のギャップを解析し、
検証可能な中間マイルストーン (Subgoals) のシーケンスを生成します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from acr_agi3.meta.observer import MetaObserver


@dataclass
class Subgoal:
    """中間マイルストーン (Subgoal)."""

    index: int
    name: str
    objective: str
    reasoning: str  # VCGT 型の思考理由
    expected_operation: str  # 幾何・論理操作 (e.g. 'resize', 'color_map', 'translate')
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


class SubgoalDecomposer:
    """VCGT 思考モデルに準拠したサブゴール分解エンジン."""

    def __init__(self, observer: MetaObserver = None) -> None:
        self.observer = observer or MetaObserver()

    def decompose(
        self,
        obs: np.ndarray,
        goal_description: str = "",
        known_roles: Dict[str, int] = None,
    ) -> DecompositionPlan:
        """ゲーム観測から階層的サブゴール列を策定 (decompose_game のエイリアス)."""
        return self.decompose_game(obs, goal_description, known_roles)



    def decompose_game(
        self,
        obs: np.ndarray,
        goal_description: str = "",
        known_roles: Dict[str, int] = None,
    ) -> DecompositionPlan:
        """ゲーム環境の現在フレーム観測から階層的サブゴール列を策定."""
        aff = self.observer.analyze_frame(obs, known_roles=known_roles)
        subgoals: List[Subgoal] = []
        step_idx = 1
        constraints: List[str] = [
            "Avoid impassable obstacle walls at all costs.",
            "Do not step outside grid boundaries.",
        ]

        if aff.hazards:
            constraints.append("Keep strict distance from lethal hazard cells.")

        # 1. アイテム（鍵・スイッチ等）が存在する場合の先行取得サブゴール
        for item_name, item_pos in aff.interactables.items():
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name=f"Acquire_{item_name}",
                    objective=f"Navigate to {item_name} at coordinate {item_pos} and trigger it",
                    reasoning=(
                        f"Interacting with {item_name} is required to unlock barriers "
                        "or satisfy stage completion preconditions."
                    ),
                    expected_operation="navigate_and_interact",
                    parameters={"target_pos": item_pos, "item": item_name},
                )
            )
            step_idx += 1

        # 2. 壁や障害物による迂回が必要かどうかの判定サブゴール
        if aff.player_pos and aff.goal_pos and aff.obstacles:
            pr, pc = aff.player_pos
            gr, gc = aff.goal_pos
            # 直線上のセルに障害物が存在するか検査
            min_r, max_r = min(pr, gr), max(pr, gr)
            min_c, max_c = min(pc, gc), max(pc, gc)
            obstacles_in_box = [
                (r, c) for (r, c) in aff.obstacles if min_r <= r <= max_r and min_c <= c <= max_c
            ]
            if len(obstacles_in_box) > 2:
                # 迂回サブゴールを挿入
                subgoals.append(
                    Subgoal(
                        index=step_idx,
                        name="BypassCentralObstacle",
                        objective="Navigate along open corridors to circumvent the barrier wall",
                        reasoning=(
                            "Direct straight-line path is occluded by static obstacle walls. "
                            "A detour around the wall's perimeter is necessary."
                        ),
                        expected_operation="bypass_wall",
                        parameters={"obstacle_count": len(obstacles_in_box)},
                    )
                )
                step_idx += 1

        # 3. 最終目標地点（ゴール）への到達サブゴール
        if aff.goal_pos:
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="ReachGoalAndClearStage",
                    objective=(
                        f"Navigate to exit goal at coordinate {aff.goal_pos} to complete stage"
                    ),
                    reasoning=(
                        "With all prerequisites completed and barriers bypassed, "
                        "enter the goal cell to trigger stage completion."
                    ),
                    expected_operation="reach_goal",
                    parameters={"goal_pos": aff.goal_pos},
                )
            )
        else:
            # ゴール座標が未同定の場合の探索サブゴール
            subgoals.append(
                Subgoal(
                    index=step_idx,
                    name="ExploreUnseenTerritory",
                    objective=(
                        "Explore unvisited open cells to discover goal or interactable target"
                    ),
                    reasoning=(
                        "Goal location is not yet visible in the immediate observation field."
                    ),
                    expected_operation="explore",
                )
            )

        task_hint = goal_description or (
            f"Navigate from {aff.player_pos} to {aff.goal_pos} "
            f"avoiding {len(aff.obstacles)} wall cells."
        )

        return DecompositionPlan(
            task_hint=task_hint,
            subgoals=subgoals,
            total_steps=len(subgoals),
            constraints=constraints,
            reasoning_trace="Game environment decomposition based on affordances and obstacles.",
        )
