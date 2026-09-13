"""ACR-AGI-3 未知環境自律適応型 メタスキルハーネス (Meta-Skill Harness).

特定のゲーム環境に特化したハードコードを排し、
1. Meta-Observer: 観測グリッド/画像から動的差分相関による自機・ターゲット・障害物アフォーダンス同定
2. Failure-Diagnoser: 停滞・壁衝突・無効行動の検知と仮説修復 (Blacklisting & Replanning)
3. Skill-Synthesizer: 状況に応じた実行可能ポリシー (Navigation, InteractiveClick, Exploration) の動的合成
を協調動作させる自律エージェント基盤。
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from acr_agi3.meta.skill_harness import SkillHarness


class MetaSkillHarnessPlanner:
    """未知ゲーム環境において自律的にアフォーダンスを同定し、行動スキルを動的合成・修復するハーネス."""

    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.harness = SkillHarness()
        obs_mod = self.harness.get_skill_module("env-observer")
        syn_mod = self.harness.get_skill_module("skill-synthesizer")
        self.syn_mod = syn_mod
        self.observer = obs_mod.MetaObserver()
        self.synthesizer = syn_mod.MetaSkillSynthesizer()

        self.step_index: int = 0
        self.last_action_id: Optional[int] = None
        self.last_action_data: Dict[str, Any] = {}
        self.last_grid: Optional[np.ndarray] = None
        self.consecutive_ineffective: int = 0
        self.clicked_coords: Set[Tuple[int, int]] = set()

        # クリック相互作用の仮説検証メモリ
        self.confirmed_interactive_group: Optional[Tuple[int, int]] = None
        self.last_clicked_group_key: Optional[Tuple[int, int]] = None
        self.last_hit_pos: Optional[Tuple[int, int]] = None

    def decide_action(
        self,
        grid: Any,
        available_action_ids: List[int],
    ) -> Tuple[int, Dict[str, Any], str]:
        """観測からアフォーダンスを同定し、最適な行動スキルを動的合成して実行."""
        self.step_index += 1

        # 1. 3D アニメーションテンソル等の安全な正規化
        arr = np.array(grid, dtype=int)
        if arr.ndim == 3:
            arr = arr[-1]
        elif arr.ndim == 1:
            arr = np.array([arr])

        h, w = arr.shape
        if h == 0 or w == 0:
            return available_action_ids[0], {}, "Fallback: Empty observation"

        # 2. Meta-Observer によるアフォーダンス同定
        report = self.observer.analyze_frame(
            grid=arr,
            recent_action=self.last_action_id,
        )

        # 3. クリック系アクション (ACTION6) が利用可能な場合の特殊合成
        if 6 in available_action_ids:
            act_id, act_data, reasoning = self._handle_interactive_click(report, arr)
            self.last_action_id = act_id
            self.last_action_data = act_data
            self.last_grid = arr.copy()
            return act_id, act_data, reasoning

        # 4. Meta-Skill Synthesizer による行動スキルの動的合成
        active_skill = self.synthesizer.synthesize(report)

        # 5. スキルポリシーによる行動選択
        chosen_action = active_skill.choose_action(report)

        # スキルの行動が利用可能アクションに含まれているか検証
        if chosen_action is not None and chosen_action in available_action_ids:
            self.last_action_id = chosen_action
            self.last_action_data = {}
            self.last_grid = arr.copy()
            skill_name = type(active_skill).__name__
            return chosen_action, {}, f"MetaSkill[{skill_name}]: Step {self.step_index}"

        # 6. スキルが失敗・行き止まりの場合：Failure Diagnoser による自己修復
        self.synthesizer.blacklist_current_target()
        fallback_skill = self.syn_mod.FrontierExplorationSkill()
        fallback_act = fallback_skill.choose_action(report)
        if fallback_act not in available_action_ids:
            fallback_act = available_action_ids[0]

        self.last_action_id = fallback_act
        self.last_action_data = {}
        self.last_grid = arr.copy()
        return fallback_act, {}, f"MetaSkill[DiagnosedReplanning]: Fallback to {fallback_act}"

    def on_feedback(self, is_effective: bool, pixels_changed: int) -> None:
        """環境からの状態フィードバックを受け取り、仮説とスキルを動的修復 (EDD Loop)."""
        if is_effective:
            self.consecutive_ineffective = 0
            # クリックがヒットした場合、そのグループを記憶
            if self.last_action_id == 6 and self.last_clicked_group_key:
                self.confirmed_interactive_group = self.last_clicked_group_key
                if "x" in self.last_action_data and "y" in self.last_action_data:
                    self.last_hit_pos = (self.last_action_data["x"], self.last_action_data["y"])
        else:
            self.consecutive_ineffective += 1
            # 移動系アクションが無効（壁衝突）だった場合、衝突地点を学習してターゲット仮説を破棄
            if self.last_action_id in (1, 2, 3, 4):
                if self.observer.identified_agent_color is not None and self.last_grid is not None:
                    # 自機位置の近傍セルを障害物として登録
                    agent_pos = self.observer.analyze_frame(self.last_grid).agent_pos
                    if agent_pos:
                        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[self.last_action_id]
                        blocked_r, blocked_c = agent_pos[0] + dr, agent_pos[1] + dc
                        self.observer.register_collision(blocked_r, blocked_c)

            # 2回連続で行動が無効なら現在のターゲット仮説をブラックリストに入れて再合成
            if self.consecutive_ineffective >= 2:
                self.synthesizer.blacklist_current_target()

    def _handle_interactive_click(
        self,
        report: DynamicAffordanceReport,
        grid: np.ndarray,
    ) -> Tuple[int, Dict[str, Any], str]:
        """クリック系パズルに対する仮説検証型クリック合成."""
        h, w = report.grid_shape
        target_candidates: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]] = []

        # ゲシュタルト同定: 反復スプライト群（タイル盤/キーパッド）
        size_color_groups = collections.defaultdict(list)
        for obj in report.target_candidates:
            if 4 <= obj.size < (h * w * 0.2):
                size_color_groups[(obj.color, obj.size)].append(obj)

        # 過去にヒットが確認された正解グループを最優先
        if self.confirmed_interactive_group and self.confirmed_interactive_group in size_color_groups:
            hit_group = size_color_groups[self.confirmed_interactive_group]
            if self.last_hit_pos:
                lx, ly = self.last_hit_pos
                hit_group = sorted(
                    hit_group,
                    key=lambda o: abs(o.bounding_box[1] - lx) + abs(o.bounding_box[0] - ly),
                )
            for obj in hit_group:
                for pt in [(obj.bounding_box[1], obj.bounding_box[0]), (int(round(obj.center_c)), int(round(obj.center_r)))]:
                    if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                        target_candidates.append((pt, self.confirmed_interactive_group))

        # 反復グループの探索
        if not target_candidates:
            repeated = [(k, g) for k, g in size_color_groups.items() if len(g) >= 3]
            repeated.sort(key=lambda item: (-len(item[1]), -item[1][0].size))
            for grp_key, group in repeated:
                for obj in group:
                    for pt in [(obj.bounding_box[1], obj.bounding_box[0]), (int(round(obj.center_c)), int(round(obj.center_r)))]:
                        if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                            target_candidates.append((pt, grp_key))

        # その他のターゲット候補
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
            self.last_clicked_group_key = grp_key
            return 6, {"x": int(tx), "y": int(ty)}, f"MetaSkill[ClickInteraction]: ({tx}, {ty})"

        # フォールバック
        for r in range(h):
            for c in range(w):
                if grid[r, c] != report.background_color and (c, r) not in self.clicked_coords:
                    self.clicked_coords.add((c, r))
                    return 6, {"x": int(c), "y": int(r)}, f"MetaSkill[ClickFallback]: ({c}, {r})"

        cx, cy = w // 2, h // 2
        return 6, {"x": int(cx), "y": int(cy)}, f"MetaSkill[ClickCenter]: ({cx}, {cy})"


# 下位互換エイリアス
GestaltVCGTPlanner = MetaSkillHarnessPlanner
