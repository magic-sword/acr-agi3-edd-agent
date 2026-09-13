"""ACR-AGI-3 未知環境自律適応型行動プランナー (MetaSkillHarnessPlanner).

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
    """未知ゲーム環境において自律的にアフォーダンスを同定し、行動スキルを動的合成・修復するプランナー."""

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

        # 3. Meta-Skill Synthesizer による行動スキルの動的合成 (クリック系含む)
        active_skill = self.synthesizer.synthesize(report, available_action_ids=available_action_ids)

        # 4. スキルポリシーによる行動選択 (付随データ x, y を含む)
        chosen_action, act_data, reasoning = active_skill.choose_action_full(
            report=report,
            grid=arr,
            available_actions=available_action_ids,
        )

        # スキルの行動が利用可能アクションに含まれているか検証
        if chosen_action is not None and chosen_action in available_action_ids:
            self.last_action_id = chosen_action
            self.last_action_data = act_data
            self.last_grid = arr.copy()
            skill_name = type(active_skill).__name__
            return chosen_action, act_data, f"MetaSkill[{skill_name}]: {reasoning}"

        # 5. スキルが失敗・行き止まりの場合：Failure Diagnoser による自己修復
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
            if self.last_action_id == 6 and hasattr(self.synthesizer, "click_skill") and self.synthesizer.click_skill:
                if "x" in self.last_action_data and "y" in self.last_action_data:
                    self.synthesizer.click_skill.last_hit_pos = (
                        self.last_action_data["x"],
                        self.last_action_data["y"],
                    )
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


# 下位互換エイリアス
GestaltVCGTPlanner = MetaSkillHarnessPlanner
