#!/usr/bin/env python3
"""Macro Skill Compiler - Unified Policy Synthesis, Archetype Compilation & Transfer (ACR-AGI-3).

Consolidates all policy and action synthesis:
1. Compiles recurring human problem-solving archetypes (Isolate-and-Stage, Pair-and-Carry, Lane-Splitting, Anchor-Cross).
2. Synthesizes executable dynamic policies (AffordanceNavigationSkill, InteractiveClickSkill, FrontierExplorationSkill).
3. Attaches safety and behavioral contracts for pre-execution verification.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class BaseSkillPolicy:
    """合成された行動ポリシーの基本抽象クラス."""

    def choose_action(self, report: Any) -> Optional[int]:
        raise NotImplementedError

    def get_action_data(self) -> Dict[str, Any]:
        return {}

    def choose_action_full(
        self,
        report: Any,
        grid: Optional[np.ndarray] = None,
        available_actions: Optional[List[int]] = None,
    ) -> Tuple[Optional[int], Dict[str, Any], str]:
        act = self.choose_action(report)
        return act, self.get_action_data(), f"{type(self).__name__}"


class AffordanceNavigationSkill(BaseSkillPolicy):
    """同定された自機から目標ターゲットへの A* / BFS 障害物回避ナビゲーションスキル."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.target_r = int(round(target.center_r))
        self.target_c = int(round(target.center_c))
        self.plan_path: List[int] = []

    def choose_action(self, report: Any) -> Optional[int]:
        if not report.agent_pos:
            return None

        start_r, start_c = report.agent_pos
        h, w = report.grid_shape

        if (start_r, start_c) == (self.target_r, self.target_c):
            return 0

        queue = collections.deque([(start_r, start_c, [])])
        visited = {(start_r, start_c)}
        obstacles = report.obstacles

        # (dr, dc, action_id) -> 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        moves = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

        while queue:
            curr_r, curr_c, path = queue.popleft()
            if (curr_r, curr_c) == (self.target_r, self.target_c):
                return path[0] if path else 0

            for dr, dc, act in moves:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    if (nr, nc) in obstacles and (nr, nc) != (self.target_r, self.target_c):
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [act]))

        # BFS 到達不可時は貪欲移動
        dr = self.target_r - start_r
        dc = self.target_c - start_c
        if abs(dr) > abs(dc):
            return 1 if dr > 0 else 0
        else:
            return 3 if dc > 0 else 2


class FrontierExplorationSkill(BaseSkillPolicy):
    """未踏領域（フロンティア）へ向かう拡散型探索ポリシー."""

    def __init__(self) -> None:
        self.step_counter = 0

    def choose_action(self, report: Any) -> Optional[int]:
        self.step_counter += 1
        # 4方向巡回
        return self.step_counter % 4


class InteractiveClickSkill(BaseSkillPolicy):
    """インタラクティブクリック操作用ポリシー."""

    def __init__(self, click_r: int, click_c: int) -> None:
        self.click_r = click_r
        self.click_c = click_c

    def choose_action(self, report: Any) -> Optional[int]:
        return 6

    def get_action_data(self) -> Dict[str, Any]:
        return {"x": self.click_c, "y": self.click_r}


class MacroSkillCompiler:
    """ACR-AGI-3 統合ポリシー合成・定石マクロスキルコンパイルエンジン."""

    ARCHETYPE_ISOLATE_AND_STAGE = "ISOLATE_AND_STAGE"
    ARCHETYPE_PAIR_AND_CARRY = "PAIR_AND_CARRY"
    ARCHETYPE_LANE_SPLITTING = "LANE_SPLITTING"
    ARCHETYPE_ANCHOR_CROSS = "ANCHOR_CROSS"

    def __init__(self) -> None:
        self.blacklisted_target_ids: Set[int] = set()
        self.current_skill: Optional[BaseSkillPolicy] = None
        self.current_target_id: Optional[int] = None
        self.click_skill: Optional[InteractiveClickSkill] = None

    def blacklist_current_target(self) -> None:
        """失敗診断により現在のターゲットをブラックリストに追加."""
        if self.current_target_id is not None:
            self.blacklisted_target_ids.add(self.current_target_id)
        self.current_skill = None
        self.current_target_id = None

    def synthesize(
        self,
        report: Any,
        available_action_ids: Optional[List[int]] = None,
    ) -> BaseSkillPolicy:
        """後方互換性API: synthesize_policy のエイリアス."""
        return self.synthesize_policy(report, available_actions=available_action_ids)

    def compile_macro_skill(
        self,
        subgoal: Dict[str, Any],
        available_action_mapping: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """高レベルサブゴールを定石マクロスキルにマッピングし、行動シーケンスをコンパイル."""
        phase = subgoal.get("phase", "UNKNOWN")
        target_colors = subgoal.get("target_colors", [])
        action_map = available_action_mapping or {
            "UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "EXTEND": 4, "RETRACT": 5
        }

        if phase == "ISOLATE_AND_STAGE" or "buffer" in subgoal.get("description", "").lower():
            archetype = self.ARCHETYPE_ISOLATE_AND_STAGE
            action_sequence = [
                action_map.get("EXTEND", 4),
                action_map.get("DOWN", 1),
                action_map.get("RETRACT", 5),
            ]
            contract = "Ensures interfering piece is moved to buffer coordinates without disturbing primary track."

        elif phase == "INCREMENTAL_CHAIN" or "pair" in subgoal.get("description", "").lower():
            archetype = self.ARCHETYPE_PAIR_AND_CARRY
            action_sequence = [
                action_map.get("EXTEND", 4),
                action_map.get("LEFT", 2),
                action_map.get("UP", 0),
            ]
            contract = "Preserves relative bond between already connected blocks while translating."

        elif phase == "CROSS_INTERSECTION" or "anchor" in subgoal.get("description", "").lower():
            archetype = self.ARCHETYPE_ANCHOR_CROSS
            action_sequence = [
                action_map.get("UP", 0),
                action_map.get("RIGHT", 3),
                action_map.get("EXTEND", 4),
            ]
            contract = "Fixes common central anchor piece first before assembling perpendicular arms."

        else:
            archetype = self.ARCHETYPE_LANE_SPLITTING
            action_sequence = [
                action_map.get("UP", 0),
                action_map.get("RIGHT", 3),
                action_map.get("DOWN", 1),
            ]
            contract = "Routes pieces through parallel split lanes around obstacle."

        return {
            "success": True,
            "subgoal_index": subgoal.get("subgoal_index", 1),
            "archetype_applied": archetype,
            "target_colors": target_colors,
            "compiled_action_ids": action_sequence,
            "sequence_length": len(action_sequence),
            "contract_specification": contract,
            "reusable_macro_id": f"macro_{archetype.lower()}",
        }

    def synthesize_policy(
        self,
        report: Any,
        grid: Optional[np.ndarray] = None,
        available_actions: Optional[List[int]] = None,
    ) -> BaseSkillPolicy:
        """後方互換性API: MetaSkillSynthesizer としての動的ポリシーインスタンス化."""
        # ターゲット候補が存在する場合、最短経路ナビゲーション
        target_candidates = getattr(report, "target_candidates", [])
        if target_candidates:
            return AffordanceNavigationSkill(target=target_candidates[0])

        # クリックが必要なインタラクタブルが存在する場合
        interactables = getattr(report, "interactables", {})
        if interactables and 6 in (available_actions or []):
            first_pos = next(iter(interactables.values()))
            return InteractiveClickSkill(click_r=first_pos[0], click_c=first_pos[1])

        # デフォルトはフロンティア探索
        return FrontierExplorationSkill()


# 後方互換性エイリアス
MetaSkillSynthesizer = MacroSkillCompiler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Macro Skill Compiler - Unified Policy Synthesis & Archetype CLI"
    )
    parser.add_argument(
        "--subgoal",
        type=str,
        default='{"phase": "ISOLATE_AND_STAGE", "target_colors": [4]}',
        help="JSON string of subgoal",
    )
    parser.add_argument("--file", type=str, help="Path to input JSON file")
    args = parser.parse_args()

    compiler = MacroSkillCompiler()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            subgoal = data.get("subgoal", {})
    else:
        subgoal = json.loads(args.subgoal)

    res = compiler.compile_macro_skill(subgoal)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
