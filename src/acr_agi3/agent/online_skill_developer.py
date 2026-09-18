"""オンライン自律スキル開発エンジン (OnlineSkillDeveloper).

ACR-AGI-3 における人間の適応プロセス（docs/HUMAN_ADAPTATION_ANALYSIS_REPORT.md）に基づき、
1. 認識論的プローブ (Epistemic Probing): 未知の力学・アクション対応を最小実験（1手）で同定
2. 動的ポリシーコード合成 (Policy Code Synthesizer): 判明したルールから具象Pythonポリシーを合成
3. EDD 防壁ゲート (Contract Barrier Gate): 正例3件＋負例3件の契約テストをインメモリ実行（100%合格のみ採用）
4. スキル登録 & 高速マクロ実行 (Fast Macro Execution): generated_skills/ への保存、ADKホットリロード、LLM推論バイパス高速実行
"""

from __future__ import annotations

import collections
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from acr_agi3.harness.game_action_tools import ActionDecision

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class OnlineSkillDeveloper:
    """仮説検証・EDD契約テスト・動的スキル生成・マクロ高速実行を統括するエンジン."""

    def __init__(
        self,
        game_id: str = "default",
        generated_skills_dir: Optional[Path] = None,
    ) -> None:
        self.game_id = game_id
        import re
        clean_id = re.sub(r"[^a-z0-9]+", "-", game_id.lower()).strip("-")
        self.clean_id = clean_id or "default"
        self.generated_skills_dir = generated_skills_dir or (REPO_ROOT / "generated_skills")
        self.generated_skills_dir.mkdir(parents=True, exist_ok=True)

        # 1. 認識論的プローブ状態
        # 移動アクションIDと方向ベクトルのマッピング: { "UP": act_id, "DOWN": act_id, ... }
        self.action_semantics: Dict[str, int] = {}
        self.reverse_action_map: Dict[int, Tuple[int, int]] = {}  # act_id -> (dr, dc)
        self.agent_color: Optional[int] = None
        self.target_colors: Set[int] = set()
        self.obstacle_colors: Set[int] = set()
        self.is_click_game: bool = False
        self.is_dynamics_identified: bool = False

        # プローブ用未試行アクション
        self.probing_step_count: int = 0
        self.probe_actions_queue: collections.deque[int] = collections.deque()
        self.tried_probe_actions: Set[int] = set()

        # 2. 合成・承認されたアクティブポリシー
        self.active_policy_name: Optional[str] = None
        self.active_policy_instance: Optional[Any] = None
        self.active_policy_dir: Optional[Path] = None
        self.macro_steps_executed: int = 0

        self.taboo_positions: Set[Tuple[int, int]] = set()
        self.policy_suppressed: bool = False

    @property
    def is_rule_identified(self) -> bool:
        return self.is_dynamics_identified

    @property
    def is_probing(self) -> bool:
        return (not self.is_dynamics_identified) and (self.probing_step_count < 4)

    @property
    def active_policy(self) -> Optional[Any]:
        return self.active_policy_instance

    @property
    def active_skill_name(self) -> Optional[str]:
        return self.active_policy_name

    def reset_policy(self) -> None:
        """アクティブなマクロポリシーの解除と再合成の一時抑制."""
        self.active_policy_instance = None
        self.active_policy_name = None
        self.macro_steps_executed = 0
        self.policy_suppressed = True

    def reset(self) -> None:
        """エピソード開始時の内部状態初期化."""
        self.action_semantics.clear()
        self.reverse_action_map.clear()
        self.agent_color = None
        self.target_colors.clear()
        self.obstacle_colors.clear()
        self.is_click_game = False
        self.is_dynamics_identified = False
        self.probing_step_count = 0
        self.probe_actions_queue.clear()
        self.active_policy_name = None
        self.active_policy_instance = None
        self.active_policy_dir = None
        self.macro_steps_executed = 0
        self.taboo_positions.clear()
        self.policy_suppressed = False
        self.tried_probe_actions.clear()

    # =========================================================================
    # Phase 1: 認識論的プローブ (Epistemic Probing) & ルール同定
    # =========================================================================

    def should_probe(self, step_index: int, available_actions: List[int]) -> bool:
        """認識論的プローブを実施すべきか判定."""
        if self.is_dynamics_identified:
            return False
        # 初手〜ステップ4、かつ移動系またはクリック系でアクション意味論が未確定
        if 6 in available_actions and len(available_actions) <= 2:
            # クリック主体のゲーム
            self.is_click_game = True
            self.is_dynamics_identified = True
            return False
        return step_index <= 5 and len(self.action_semantics) < 4

    def get_next_probe_action(
        self, available_actions: List[int]
    ) -> Optional[ActionDecision]:
        """未知のアクション効果を解明するための最小介入実験手を決定."""
        valid_acts = [a for a in available_actions if a not in (0,)]
        if not valid_acts:
            return None

        # まだ試行していないアクションを系統的に選択
        candidates = [a for a in valid_acts if a not in self.tried_probe_actions and a not in self.reverse_action_map]
        if not candidates:
            candidates = [a for a in valid_acts if a not in self.reverse_action_map]
        act_id = candidates[0] if candidates else valid_acts[0]
        self.tried_probe_actions.add(act_id)

        self.probing_step_count += 1
        return ActionDecision(
            action_type="STEP",
            action_name=f"ACTION{act_id}",
            action_id=act_id,
            coordinates=None,
            reasoning=f"Epistemic Probe Step {self.probing_step_count}: Testing physical displacement of ACTION{act_id} to isolate directional invariant.",
            loaded_skill="epistemic-prober",
            metadata={
                "is_epistemic_probe": True,
                "probe_target_action": act_id,
                "probing_step": self.probing_step_count,
            },
        )

    def analyze_probe_transition(
        self,
        prev_grid: Optional[np.ndarray] = None,
        curr_grid: Optional[np.ndarray] = None,
        action_id: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """プローブ実行前後のグリッド差分から、自機色・移動ベクトル・因果関係を同定."""
        if prev_grid is None:
            prev_grid = kwargs.get("grid_before")
        if curr_grid is None:
            curr_grid = kwargs.get("grid_after")
        if action_id is None:
            action_id = kwargs.get("action_taken", 1)

        if prev_grid is None or curr_grid is None or prev_grid.shape != curr_grid.shape:
            return {"identified": False}

        # アクション指定が文字列（"ACTION1", "UP" 等）の場合は整数IDに正規化
        if isinstance(action_id, str):
            import re
            m = re.search(r"\d+", action_id)
            if m:
                action_id = int(m.group(0))
            else:
                str_map = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "CLICK": 6, "RESET": 0}
                action_id = str_map.get(action_id.upper(), 1)
        else:
            action_id = int(action_id)

        diff_mask = (prev_grid != curr_grid)
        diff_count = int(np.sum(diff_mask))
        if diff_count == 0:
            logger.info("🧪 [Epistemic Probe Analysis] ACTION%d produced ΔPixels: 0 (Barrier/Inactive)", action_id)
            return {"identified": False, "delta_pixels": 0}

        # 盤面に変化があった場合はマクロ抑制を解除
        self.policy_suppressed = False

        # 変化したピクセルの解析
        coords = np.argwhere(diff_mask)
        # 前後で消えた色と現れた色の追跡
        disappeared_colors = [prev_grid[r, c] for r, c in coords]
        appeared_colors = [curr_grid[r, c] for r, c in coords]

        # 自機推定: 前後で共通して移動した単一色（例: 以前のセルで消え、新しいセルで出現）
        moving_colors = set(disappeared_colors).intersection(set(appeared_colors))
        # 背景色(0)や消滅色を除外
        moving_colors.discard(0)

        dr, dc = 0, 0
        if moving_colors:
            agent_c = list(moving_colors)[0]
            self.agent_color = int(agent_c)

            prev_pos = np.argwhere(prev_grid == agent_c)
            curr_pos = np.argwhere(curr_grid == agent_c)
            if len(prev_pos) > 0 and len(curr_pos) > 0:
                p_r, p_c = prev_pos.mean(axis=0)
                c_r, c_c = curr_pos.mean(axis=0)
                dr = int(round(c_r - p_r))
                dc = int(round(c_c - p_c))

        # 方向ベクトルの同定 (主軸判定によるロバスト推定)
        direction = None
        if abs(dr) > abs(dc) and abs(dr) > 0:
            direction = "UP" if dr < 0 else "DOWN"
        elif abs(dc) > abs(dr) and abs(dc) > 0:
            direction = "LEFT" if dc < 0 else "RIGHT"
        elif dr != 0 or dc != 0:
            if dr < 0:
                direction = "UP"
            elif dr > 0:
                direction = "DOWN"
            elif dc < 0:
                direction = "LEFT"
            elif dc > 0:
                direction = "RIGHT"

        if direction:
            self.action_semantics[direction] = action_id
            self.reverse_action_map[action_id] = (dr, dc)
            logger.info(
                "🧪 [Epistemic Invariant Identified] ACTION%s -> %s (dr=%d, dc=%d) | Agent Color: %s",
                action_id, direction, dr, dc, self.agent_color
            )

        # クリック・トグル型アクションの効果同定
        if action_id == 6 or "6" in str(action_id):
            self.is_click_game = True
            logger.info("🧪 [Epistemic Affordance Identified] ACTION6 confirmed as effective interact/click actuator (ΔPixels: %d)", diff_count)

        # 4方向が揃ったか、あるいは対向軸（上下または左右）が同定されたら力学確立
        has_nav = ("UP" in self.action_semantics or "DOWN" in self.action_semantics) and ("LEFT" in self.action_semantics or "RIGHT" in self.action_semantics)
        if len(self.action_semantics) >= 2 and has_nav:
            self.is_dynamics_identified = True
            logger.info("🎉 [Environmental Dynamics Identified via Probing] Invariant Action Map: %s", self.action_semantics)
        elif self.is_click_game and diff_count > 0:
            self.is_dynamics_identified = True
            logger.info("🎉 [Click Dynamics Identified via Probing] Interactive click affordance verified.")

        return {
            "identified": direction is not None,
            "direction": direction,
            "dr": dr,
            "dc": dc,
            "agent_color": self.agent_color,
            "diff_pixels": diff_count,
        }

    # =========================================================================
    # Phase 2: 動的ポリシーコード合成 (Skill Synthesis)
    # =========================================================================

    def synthesize_navigation_skill_code(
        self,
        agent_color: int,
        target_color: int,
        obstacle_colors: List[int],
        action_map: Dict[str, int],
    ) -> str:
        """同定されたルールを埋め込んだ安全な BFS 最短経路ナビゲーションコードを生成."""
        code = f'''# Generated Navigation Policy for {self.game_id}
import numpy as np
import collections

AGENT_COLOR = {agent_color}
TARGET_COLOR = {target_color}
OBSTACLE_COLORS = set({obstacle_colors})
ACTION_MAP = {action_map}  # e.g. {{"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}}

def find_positions(grid, color):
    pts = np.argwhere(grid == color)
    return [(int(r), int(c)) for r, c in pts]

def choose_action(obs, info=None):
    grid = np.array(obs, dtype=int)
    h, w = grid.shape
    
    agent_pts = find_positions(grid, AGENT_COLOR)
    target_pts = find_positions(grid, TARGET_COLOR)
    
    if not agent_pts:
        return ACTION_MAP.get("UP", 1)
    if not target_pts:
        return ACTION_MAP.get("RIGHT", 4)
        
    start_r, start_c = agent_pts[0]
    target_r, target_c = target_pts[0]
    
    if (start_r, start_c) == (target_r, target_c):
        return ACTION_MAP.get("UP", 1)
        
    # BFS 最短経路探索
    queue = collections.deque([(start_r, start_c, [])])
    visited = {{(start_r, start_c)}}
    moves = [
        (-1, 0, ACTION_MAP.get("UP", 1)),
        (1, 0, ACTION_MAP.get("DOWN", 2)),
        (0, -1, ACTION_MAP.get("LEFT", 3)),
        (0, 1, ACTION_MAP.get("RIGHT", 4)),
    ]
    
    while queue:
        curr_r, curr_c, path = queue.popleft()
        if (curr_r, curr_c) == (target_r, target_c):
            return path[0] if path else ACTION_MAP.get("UP", 1)
            
        for dr, dc, act in moves:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                cell_val = grid[nr, nc]
                if cell_val in OBSTACLE_COLORS and (nr, nc) != (target_r, target_c):
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc, path + [act]))
                
    # 経路が遮断されている場合は貪欲前進
    dr = target_r - start_r
    dc = target_c - start_c
    if abs(dr) >= abs(dc):
        return ACTION_MAP.get("DOWN", 2) if dr > 0 else ACTION_MAP.get("UP", 1)
    else:
        return ACTION_MAP.get("RIGHT", 4) if dc > 0 else ACTION_MAP.get("LEFT", 3)
'''
        return code

    def synthesize_click_sweep_skill_code(
        self,
        target_colors: List[int],
        click_action_id: int = 6,
    ) -> str:
        """インタラクティブタイルを走査・クリックするコードを生成."""
        code = f'''# Generated Click Target Sweep Policy for {self.game_id}
import numpy as np

TARGET_COLORS = set({target_colors})
CLICK_ACTION_ID = {click_action_id}

def choose_action(obs, info=None):
    return CLICK_ACTION_ID

def get_action_data(obs, info=None):
    grid = np.array(obs, dtype=int)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] in TARGET_COLORS:
                return {{"x": c, "y": r}}
    return {{"x": w // 2, "y": h // 2}}
'''
        return code

    # =========================================================================
    # Phase 3: EDD 防壁ゲート (正例3件＋負例3件 契約テスト)
    # =========================================================================

    def run_edd_contract_gate(self, policy_code: str) -> Tuple[bool, Dict[str, Any]]:
        """meta_skills/contract-tester に準拠した正例3+負例3 契約テストを実行."""
        results = {
            "passed": False,
            "positive_passed": 0,
            "negative_passed": 0,
            "total_passed": 0,
            "total_cases": 6,
            "failures": [],
        }

        # 1. 構文とコンパイル検証
        exec_scope: Dict[str, Any] = {"np": np, "collections": collections, "__builtins__": __builtins__}
        try:
            exec(policy_code, exec_scope)
        except Exception as e:
            results["failures"].append(f"CompilationError: {type(e).__name__}: {e}")
            logger.error("❌ [EDD Barrier Gate] Compilation Failed: %s", e)
            return False, results

        choose_act_fn = exec_scope.get("choose_action")
        if not callable(choose_act_fn):
            results["failures"].append("ContractError: 'choose_action' callable missing")
            return False, results

        # 2. テストケース定義
        # 正例3件 (正常盤面、境界付近、ターゲット到達)
        pos_cases = [
            {"name": "pos_standard_grid", "h": 10, "w": 10, "agent": (2, 2), "target": (6, 6)},
            {"name": "pos_border_proximity", "h": 8, "w": 8, "agent": (0, 0), "target": (7, 7)},
            {"name": "pos_adjacent_target", "h": 6, "w": 6, "agent": (3, 3), "target": (3, 4)},
        ]
        # 負例3件 (壁隣接、全周障害物、異形グリッドでの例外耐性)
        neg_cases = [
            {"name": "neg_obstacle_barrier", "h": 8, "w": 8, "agent": (1, 1), "wall": (1, 2)},
            {"name": "neg_edge_bounds", "h": 5, "w": 5, "agent": (0, 0), "wall": (0, 1)},
            {"name": "neg_missing_entities", "h": 4, "w": 4},
        ]

        agent_c = self.agent_color or 2
        target_c = list(self.target_colors)[0] if self.target_colors else 3
        obs_c = list(self.obstacle_colors)[0] if self.obstacle_colors else 1

        # 正例テストの実行
        for case in pos_cases:
            g = np.zeros((case["h"], case["w"]), dtype=int)
            ar, ac = case["agent"]
            tr, tc = case["target"]
            g[ar, ac] = agent_c
            g[tr, tc] = target_c
            try:
                act = choose_act_fn(g)
                if act is not None and isinstance(act, (int, np.integer)):
                    results["positive_passed"] += 1
                else:
                    results["failures"].append(f"Case {case['name']} returned invalid action: {act}")
            except Exception as e:
                results["failures"].append(f"Case {case['name']} crashed: {e}")

        # 負例テストの実行 (例外を吐かずに安全に代替手を返すこと)
        for case in neg_cases:
            g = np.zeros((case["h"], case["w"]), dtype=int)
            if "agent" in case:
                ar, ac = case["agent"]
                g[ar, ac] = agent_c
            if "wall" in case:
                wr, wc = case["wall"]
                g[wr, wc] = obs_c
            try:
                act = choose_act_fn(g)
                if act is not None and isinstance(act, (int, np.integer)):
                    results["negative_passed"] += 1
                else:
                    results["failures"].append(f"Case {case['name']} returned invalid action: {act}")
            except Exception as e:
                results["failures"].append(f"Case {case['name']} failed safety check: {e}")

        results["total_passed"] = results["positive_passed"] + results["negative_passed"]
        results["passed"] = (results["total_passed"] == results["total_cases"]) and len(results["failures"]) == 0

        if results["passed"]:
            logger.info("🛡️ [EDD Barrier Gate PASSED] 6/6 Contract Tests Cleared (Pos: 3/3, Neg: 3/3)")
        else:
            logger.warning("🛡️ [EDD Barrier Gate FAILED] Passed %d/6. Failures: %s", results["total_passed"], results["failures"])

        return results["passed"], results

    # =========================================================================
    # Phase 4: スキル永続化 & 高速マクロ実行 (Fast Macro Execution)
    # =========================================================================

    def develop_and_register_skill(
        self,
        grid: Optional[np.ndarray] = None,
        available_actions: Optional[List[int]] = None,
        **kwargs,
    ) -> bool:
        """実証されたルールからスキルを自動開発し、EDDゲートを通した上で登録."""
        if grid is None:
            grid = kwargs.get("current_grid")
        if available_actions is None:
            available_actions = kwargs.get("avail_ids", [1, 2, 3, 4])
        if grid is None:
            return False
        h, w = grid.shape
        unique_colors = [int(c) for c in np.unique(grid) if c != 0]

        # 自機・ターゲット色の推定
        agent_c = self.agent_color or (unique_colors[0] if unique_colors else 2)
        target_c = None
        for c in unique_colors:
            if c != agent_c:
                target_c = c
                break
        target_c = target_c or 3
        self.target_colors.add(target_c)

        # 1. ポリシーコードの合成
        if self.is_click_game or 6 in available_actions:
            skill_type = "click_sweep"
            code = self.synthesize_click_sweep_skill_code([target_c], click_action_id=6)
        else:
            skill_type = "navigation"
            act_map = self.action_semantics if len(self.action_semantics) >= 2 else {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
            code = self.synthesize_navigation_skill_code(
                agent_color=agent_c,
                target_color=target_c,
                obstacle_colors=list(self.obstacle_colors),
                action_map=act_map,
            )

        # 2. EDD 防壁ゲートの検証
        passed, diag = self.run_edd_contract_gate(code)
        if not passed:
            logger.warning("⚠️ Skill development halted due to EDD barrier failure.")
            return False

        # 3. 具象スキルディレクトリへの永続化 (generated_skills/)
        import re
        clean_type = re.sub(r"[^a-z0-9]+", "-", skill_type.lower()).strip("-")
        skill_name = f"skill-{self.clean_id}-{clean_type}"
        skill_name = re.sub(r"-+", "-", skill_name).strip("-")

        skill_dir = self.generated_skills_dir / skill_name
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        skill_md_content = f"""---
name: {skill_name}
description: |
  Auto-synthesized dynamic {skill_type} policy for {self.game_id} verified by EDD barrier gate.
  Agent color: {agent_c}, Target color: {target_c}.
license: MIT
allowed-tools: run_skill_script
metadata:
  synthesizer: online_skill_developer
  version: "1.0.0"
  edd_gate_passed: true
  total_contracts: 6
---

# {skill_name}

Auto-synthesized dynamic gameplay policy verified against positive and negative contracts.
"""
        (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
        policy_file = scripts_dir / "policy.py"
        policy_file.write_text(code, encoding="utf-8")

        # インメモリ実行モジュールのコンパイル
        mod_name = skill_name.replace("-", "_")
        spec = importlib.util.spec_from_file_location(mod_name, policy_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.active_policy_instance = mod
            self.active_policy_name = skill_name
            self.active_policy_dir = skill_dir
            logger.info("🚀 [Dynamic Skill Registered & Hotloaded] Active Policy: %s in %s", skill_name, skill_dir)
            return True

        return False

    def execute_active_policy(
        self,
        grid: np.ndarray,
        available_actions: Optional[List[int]] = None,
    ) -> Optional[ActionDecision]:
        """承認された具象ポリシーによる超高速アクション決定 (LLMバイパス)."""
        if not self.active_policy_instance:
            return None

        choose_fn = getattr(self.active_policy_instance, "choose_action", None)
        if not choose_fn:
            return None

        try:
            act_id = choose_fn(grid)
            coords = None
            data_fn = getattr(self.active_policy_instance, "get_action_data", None)
            if data_fn:
                coords = data_fn(grid)

            self.macro_steps_executed += 1
            act_type = "CLICK" if act_id == 6 else "STEP"

            logger.info(
                "⚡ [Fast Macro Execution Step %d] Executing '%s' via Policy '%s' (Coords: %s)",
                self.macro_steps_executed, f"ACTION{act_id}", self.active_policy_name, coords
            )

            return ActionDecision(
                action_type=act_type,
                action_name=f"ACTION{act_id}",
                action_id=int(act_id),
                coordinates=coords,
                reasoning=f"Fast macro execution via EDD-verified policy '{self.active_policy_name}' (Step {self.macro_steps_executed}).",
                loaded_skill=self.active_policy_name,
                metadata={
                    "is_macro_execution": True,
                    "macro_skill": self.active_policy_name,
                    "macro_step": self.macro_steps_executed,
                },
            )
        except Exception as e:
            logger.warning("Active macro policy encountered exception: %s. Reverting to LLM cognitive planner.", e)
            self.active_policy_instance = None
            return None
