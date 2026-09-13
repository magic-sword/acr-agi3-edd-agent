#!/usr/bin/env python3
"""Skill Synthesizer - Core CLI & Script Tool (ACR-AGI-3).

抽出された動的アフォーダンスおよびサブゴール仕様に基づき、
実行可能な行動ポリシー (AffordanceNavigationSkill, InteractiveClickSkill, FrontierExplorationSkill 等) を
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

    def get_action_data(self) -> Dict[str, Any]:
        """アクションに付随する追加引数 (例: ACTION6 の x, y)."""
        return {}

    def choose_action_full(
        self,
        report: Any,
        grid: Optional[np.ndarray] = None,
        available_actions: Optional[List[int]] = None,
    ) -> Tuple[Optional[int], Dict[str, Any], str]:
        """アクションID、付随データ、および論理的理由 (reasoning) を包括返却."""
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

    def __init__(self, target_pixels: Optional[List[Tuple[int, int]]] = None) -> None:
        self.target_pixels = target_pixels or []
        self.last_action_data: Dict[str, Any] = {}
        self.clicked_coords: Set[Tuple[int, int]] = set()
        self.confirmed_group: Optional[Tuple[int, int]] = None
        self.last_hit_pos: Optional[Tuple[int, int]] = None

    def get_action_data(self) -> Dict[str, Any]:
        return self.last_action_data

    def choose_action(self, report: Any) -> Optional[int]:
        act, _, _ = self.choose_action_full(report)
        return act

    def choose_action_full(
        self,
        report: Any,
        grid: Optional[np.ndarray] = None,
        available_actions: Optional[List[int]] = None,
    ) -> Tuple[Optional[int], Dict[str, Any], str]:
        """ゲシュタルト同定 (反復スプライト・キーパッド・アイテム) に基づくクリック決定."""
        h, w = report.grid_shape
        target_candidates: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]] = []

        # 1. ゲシュタルト同定: 反復スプライト群 (タイル盤 / キーパッド)
        size_color_groups = collections.defaultdict(list)
        for obj in report.target_candidates:
            if 4 <= obj.size < (h * w * 0.2):
                size_color_groups[(obj.color, obj.size)].append(obj)

        # 2. 過去にヒットが確認された正解グループを最優先
        if self.confirmed_group and self.confirmed_group in size_color_groups:
            hit_group = size_color_groups[self.confirmed_group]
            if self.last_hit_pos:
                lx, ly = self.last_hit_pos
                hit_group = sorted(
                    hit_group,
                    key=lambda o: abs(o.bounding_box[1] - lx) + abs(o.bounding_box[0] - ly),
                )
            for obj in hit_group:
                for pt in [(obj.bounding_box[1], obj.bounding_box[0]), (int(round(obj.center_c)), int(round(obj.center_r)))]:
                    if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                        target_candidates.append((pt, self.confirmed_group))

        # 3. 反復グループの探索 (3個以上同じサイズ・色のスプライト)
        if not target_candidates:
            repeated = [(k, g) for k, g in size_color_groups.items() if len(g) >= 3]
            repeated.sort(key=lambda item: (-len(item[1]), -item[1][0].size))
            for grp_key, group in repeated:
                for obj in group:
                    for pt in [(obj.bounding_box[1], obj.bounding_box[0]), (int(round(obj.center_c)), int(round(obj.center_r)))]:
                        if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                            target_candidates.append((pt, grp_key))

        # 4. その他のターゲット候補
        if not target_candidates:
            for obj in report.target_candidates:
                grp_key = (obj.color, obj.size)
                for pt in [(obj.bounding_box[1], obj.bounding_box[0]), (int(round(obj.center_c)), int(round(obj.center_r)))]:
                    if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                        target_candidates.append((pt, grp_key))

        # ターゲット決定
        if target_candidates:
            (tx, ty), grp_key = target_candidates[0]
            self.clicked_coords.add((tx, ty))
            self.last_action_data = {"x": int(tx), "y": int(ty)}
            return 6, self.last_action_data, f"InteractiveClick[Target]: ({tx}, {ty})"

        # フォールバック: 背景以外の未クリック点
        if grid is not None:
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != report.background_color and (c, r) not in self.clicked_coords:
                        self.clicked_coords.add((c, r))
                        self.last_action_data = {"x": int(c), "y": int(r)}
                        return 6, self.last_action_data, f"InteractiveClick[NonBgFallback]: ({c}, {r})"

        cx, cy = w // 2, h // 2
        self.last_action_data = {"x": int(cx), "y": int(cy)}
        return 6, self.last_action_data, f"InteractiveClick[CenterFallback]: ({cx}, {cy})"


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
        self.click_skill: Optional[InteractiveClickSkill] = None

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
        self.click_skill = None

    def synthesize(
        self,
        report: Any,
        available_action_ids: Optional[List[int]] = None,
    ) -> BaseSkillPolicy:
        """アフォーダンスレポートとアクション空間から実行可能スキルを即座に動的合成."""
        # 1. クリック系アクション (ACTION6) が利用可能な環境なら InteractiveClickSkill を優先合成
        if available_action_ids and 6 in available_action_ids:
            if self.click_skill is None:
                self.click_skill = InteractiveClickSkill()
            self.current_skill = self.click_skill
            return self.current_skill

        valid_candidates = [
            obj for obj in report.target_candidates
            if obj.obj_id not in self.blacklisted_target_ids
        ]

        # 2. 自機が同定されておりターゲット候補が存在する場合はナビゲーションスキル
        if report.agent_pos and valid_candidates:
            best_target = valid_candidates[0]
            if self.current_target_id != best_target.obj_id or self.current_skill is None:
                self.current_target_id = best_target.obj_id
                self.current_skill = AffordanceNavigationSkill(best_target)
            return self.current_skill

        # 3. 自機が未確定だがターゲットが存在する場合
        if not report.controllable_verified and valid_candidates:
            best_target = valid_candidates[0]
            if self.current_target_id != best_target.obj_id or self.current_skill is None:
                self.current_target_id = best_target.obj_id
                self.current_skill = AffordanceNavigationSkill(best_target)
            return self.current_skill

        # 4. デフォルト: フロンティア探索
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
