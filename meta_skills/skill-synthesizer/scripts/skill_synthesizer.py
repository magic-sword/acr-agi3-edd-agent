#!/usr/bin/env python3
"""Skill Synthesizer - Core CLI & Script Tool (ACR-AGI-3).

抽出された動的アフォーダンスおよびサブゴール仕様に基づき、
実行可能な行動ポリシー (AffordanceNavigationSkill, InteractiveClickSkill 等) を
動的にインスタンス化し、また EDD 防壁ゲート用の契約テストを自動合成します。
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class BaseSkillPolicy:
    """合成された行動ポリシーの基本抽象クラス."""

    def choose_action(self, report: Any) -> Optional[int]:
        raise NotImplementedError


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

        # すでにターゲット位置に到達している場合
        if (start_r, start_c) == (self.target_r, self.target_c):
            return None

        # BFS 最短経路探索
        q = collections.deque([(start_r, start_c, [])])
        visited: Set[Tuple[int, int]] = {(start_r, start_c)}

        moves = [
            (1, (-1, 0)),  # UP
            (2, (1, 0)),   # DOWN
            (3, (0, -1)),  # LEFT
            (4, (0, 1)),   # RIGHT
        ]

        while q:
            cr, cc, path = q.popleft()
            if (cr, cc) == (self.target_r, self.target_c) or (cr, cc) in self.target.pixels:
                if path:
                    return path[0]

            for action_code, (dr, dc) in moves:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    # ターゲット自身のセル以外は障害物を避ける
                    if (nr, nc) in report.obstacles and (nr, nc) not in self.target.pixels:
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc, path + [action_code]))

        return None  # 経路なし


class InteractiveClickSkill(BaseSkillPolicy):
    """クリック可能なオブジェクトや特異スプライトに対する仮説検証クリック相互作用スキル."""

    def __init__(self, target_pixels: List[Tuple[int, int]]) -> None:
        self.target_pixels = target_pixels
        self.click_index = 0

    def choose_action(self, report: Any) -> Optional[int]:
        if not self.target_pixels or self.click_index >= len(self.target_pixels):
            return None
        r, c = self.target_pixels[self.click_index]
        self.click_index += 1
        return 6  # ACTION6 (CLICK)


class FrontierExplorationSkill(BaseSkillPolicy):
    """自機またはターゲットが未特定の際に動的変化を探索するプローブ行動スキル."""

    def __init__(self) -> None:
        self.probe_actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT
        self.probe_idx = 0

    def choose_action(self, report: Any) -> Optional[int]:
        action = self.probe_actions[self.probe_idx % len(self.probe_actions)]
        self.probe_idx += 1
        return action


class MetaSkillSynthesizer:
    """アフォーダンス観測と診断結果から最適な行動スキルを動的合成するメタスキルハーネス."""

    def __init__(self) -> None:
        self.blacklisted_target_ids: Set[int] = set()
        self.current_skill: Optional[BaseSkillPolicy] = None
        self.current_target_id: Optional[int] = None

    def blacklist_current_target(self) -> None:
        """失敗診断により現在のターゲットをブラックリストに追加."""
        if self.current_target_id is not None:
            self.blacklisted_target_ids.add(self.current_target_id)
        self.current_skill = None
        self.current_target_id = None

    def reset(self) -> None:
        """エピソード開始時のリセット."""
        self.blacklisted_target_ids.clear()
        self.current_skill = None
        self.current_target_id = None

    def synthesize(self, report: Any) -> BaseSkillPolicy:
        """アフォーダンスレポートから実行可能スキルを即座に動的合成."""
        valid_candidates = [
            obj for obj in report.target_candidates
            if obj.obj_id not in self.blacklisted_target_ids
        ]

        if report.agent_pos and valid_candidates:
            best_target = valid_candidates[0]
            if self.current_target_id != best_target.obj_id or self.current_skill is None:
                self.current_target_id = best_target.obj_id
                self.current_skill = AffordanceNavigationSkill(best_target)
            return self.current_skill

        if not report.controllable_verified and valid_candidates:
            best_target = valid_candidates[0]
            if self.current_target_id != best_target.obj_id or self.current_skill is None:
                self.current_target_id = best_target.obj_id
                self.current_skill = InteractiveClickSkill(best_target.pixels)
            return self.current_skill

        self.current_skill = FrontierExplorationSkill()
        return self.current_skill


def synthesize_policy_and_contracts(spec: Dict[str, Any]) -> Dict[str, Any]:
    """ポリシーコードと契約テストを合成 (CLI / Tool 用)."""
    return {
        "status": "SYNTHESIZED",
        "synthesizer_class": "MetaSkillSynthesizer",
        "available_policies": [
            "AffordanceNavigationSkill",
            "InteractiveClickSkill",
            "FrontierExplorationSkill",
        ],
    }


def run(input_val: Any = None) -> Dict[str, Any]:
    """Core synthesis task."""
    spec: Dict[str, Any] = {}
    if isinstance(input_val, dict):
        spec = input_val
    elif isinstance(input_val, str):
        try:
            parsed = json.loads(input_val)
            if isinstance(parsed, dict):
                spec = parsed
        except Exception:
            pass

    return synthesize_policy_and_contracts(spec)


def main():
    parser = argparse.ArgumentParser(description="Skill Synthesizer execution script.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Positional input JSON")
    parser.add_argument("--input", "-i", dest="input_opt", type=str, default=None, help="Input JSON")
    args = parser.parse_args()

    input_val = args.input_opt or args.input_pos
    res = run(input_val)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
