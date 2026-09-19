#!/usr/bin/env python3
"""Visual Inspector - Unified Perception, Gestalt Diff, Affordances & Game Style Discovery (ACR-AGI-3).

Consolidates all visual perception capabilities:
1. Visual Inspection Pause & Gestalt Diff (Target vs Initial Sequence).
2. Complete Affordance & Object Extraction (Player, Target Candidates, Obstacles, Rails).
3. Game Style & Genre Classification (OPEN_EXPLORATION, CLOSED_MAZE, ITEM_TRIGGER_PUZZLE, SYMMETRIC_PATTERN).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclasses.dataclass
class VisualObject:
    """同定されたオブジェクトの属性情報."""

    obj_id: int
    color: int
    pixels: List[Tuple[int, int]]  # (r, c)
    size: int
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    center_r: float
    center_c: float
    is_static: bool = True
    role: str = "unknown"  # 'agent', 'target', 'obstacle', 'button', 'hazard'
    target_score: float = 0.0


@dataclasses.dataclass
class DynamicAffordanceReport:
    """動的ゲーム環境におけるリアルタイムアフォーダンス観測レポート."""

    grid_shape: Tuple[int, int]
    background_color: int
    agent_object: Optional[VisualObject] = None
    agent_pos: Optional[Tuple[int, int]] = None  # (r, c)
    target_candidates: List[VisualObject] = dataclasses.field(default_factory=list)
    obstacles: Set[Tuple[int, int]] = dataclasses.field(default_factory=set)
    all_objects: List[VisualObject] = dataclasses.field(default_factory=list)
    controllable_verified: bool = False
    goal_pos: Optional[Tuple[int, int]] = None
    interactables: Dict[str, Tuple[int, int]] = dataclasses.field(default_factory=dict)
    style: str = "CLOSED_MAZE"
    diff_type: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.goal_pos is None and self.target_candidates:
            valid_targets = [
                t for t in self.target_candidates
                if self.agent_pos is None or (int(round(t.center_r)), int(round(t.center_c))) != self.agent_pos
            ]
            t = valid_targets[0] if valid_targets else self.target_candidates[0]
            self.goal_pos = (int(round(t.center_r)), int(round(t.center_c)))

    @property
    def player_pos(self) -> Optional[Tuple[int, int]]:
        return self.agent_pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_shape": list(self.grid_shape),
            "background_color": int(self.background_color),
            "agent_pos": list(self.agent_pos) if self.agent_pos else None,
            "goal_pos": list(self.goal_pos) if self.goal_pos else None,
            "target_candidates": [
                {"id": t.obj_id, "color": t.color, "center": [t.center_r, t.center_c]}
                for t in self.target_candidates
            ],
            "obstacle_count": len(self.obstacles),
            "controllable_verified": self.controllable_verified,
            "style": self.style,
            "diff_type": self.diff_type,
        }


class VisualInspector:
    """ACR-AGI-3 統合視覚認識・ゲシュタルト差分・アフォーダンス・スタイル解析エンジン."""

    ALIGNMENT_IDENTITY = "IDENTITY"
    ALIGNMENT_TRANSLATION_ONLY = "TRANSLATION_ONLY"
    ALIGNMENT_REVERSAL_REORDER = "REVERSAL_REORDER"
    ALIGNMENT_INTERLEAVED = "INTERLEAVED_ALTERNATING"
    ALIGNMENT_CROSS_INTERSECTION = "CROSS_INTERSECTION"
    ALIGNMENT_UNKNOWN = "UNKNOWN"

    STYLE_OPEN_EXPLORATION = "OPEN_EXPLORATION"
    STYLE_CLOSED_MAZE = "CLOSED_MAZE"
    STYLE_ITEM_TRIGGER_PUZZLE = "ITEM_TRIGGER_PUZZLE"
    STYLE_SYMMETRIC_PATTERN = "SYMMETRIC_PATTERN"

    def __init__(self) -> None:
        self.confirmed_agent_color: Optional[int] = None
        self.identified_agent_color: Optional[int] = None
        self.known_collisions: Set[Tuple[int, int]] = set()

    def register_collision(self, r: int, c: int) -> None:
        """衝突が発生した座標を記憶."""
        self.known_collisions.add((r, c))

    def inspect_board(
        self,
        current_grid: Any,
        target_grid_or_sequence: Optional[Any] = None,
        step_index: int = 0,
    ) -> Dict[str, Any]:
        """観測フレームからアフォーダンス、差分、ゲームスタイルを包括解析."""
        grid = np.array(current_grid, dtype=int)
        if grid.ndim == 3:
            grid = grid[-1]
        elif grid.ndim == 1:
            grid = np.array([grid])

        h, w = grid.shape
        if h == 0 or w == 0:
            return {
                "success": False,
                "error": "Empty observation grid",
            }

        # 構成色と背景色の同定
        unique_colors, counts = np.unique(grid, return_counts=True)
        color_freq = dict(zip(unique_colors.tolist(), counts.tolist()))
        bg_color = max(color_freq, key=color_freq.get)
        fg_colors = [int(c) for c in unique_colors if c != bg_color]

        # ターゲット配列の抽出と差分分類
        target_seq = self._extract_sequence(target_grid_or_sequence)
        current_seq = self._extract_primary_sequence(grid, bg_color)
        diff_type, diff_details = self._classify_gestalt_diff(current_seq, target_seq)

        # アフォーダンスとレール・自機
        affordances = self._detect_affordances(grid, bg_color)

        # ゲームスタイル分類
        style = self._classify_style(grid, bg_color, affordances)

        # 不変量仮説の策定
        invariants = self._infer_initial_invariants(diff_type, affordances, style)

        return {
            "success": True,
            "step_index": step_index,
            "grid_dimensions": [h, w],
            "background_color": int(bg_color),
            "foreground_colors": fg_colors,
            "current_sequence": current_seq,
            "target_sequence": target_seq,
            "diff_type": diff_type,
            "diff_details": diff_details,
            "style": style,
            "affordances": affordances,
            "invariants_hypothesized": invariants,
        }

    def analyze_frame(
        self,
        grid: Any,
        prev_grid: Optional[Any] = None,
        action_taken: Optional[int] = None,
        known_roles: Optional[Dict[str, int]] = None,
        recent_action: Optional[int] = None,
        **kwargs: Any,
    ) -> DynamicAffordanceReport:
        """後方互換性API: MetaObserver と完全同一の DynamicAffordanceReport を出力."""
        arr = np.array(grid, dtype=int)
        if arr.ndim == 3:
            arr = arr[-1]
        elif arr.ndim == 1:
            arr = np.array([arr])

        h, w = arr.shape
        if h == 0 or w == 0:
            return DynamicAffordanceReport(grid_shape=(0, 0), background_color=0)

        unique_colors, counts = np.unique(arr, return_counts=True)
        color_freq = dict(zip(unique_colors.tolist(), counts.tolist()))
        bg_color = int(max(color_freq, key=color_freq.get))

        # オブジェクト抽出 (4連結)
        visited = np.zeros((h, w), dtype=bool)
        objects: List[VisualObject] = []
        obj_id = 0

        for r in range(h):
            for c in range(w):
                if arr[r, c] == bg_color or visited[r, c]:
                    continue
                color = int(arr[r, c])
                pixels: List[Tuple[int, int]] = []
                queue = collections.deque([(r, c)])
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    pixels.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                min_r = min(p[0] for p in pixels)
                max_r = max(p[0] for p in pixels)
                min_c = min(p[1] for p in pixels)
                max_c = max(p[1] for p in pixels)
                size = len(pixels)
                center_r = float(np.mean([p[0] for p in pixels]))
                center_c = float(np.mean([p[1] for p in pixels]))

                objects.append(
                    VisualObject(
                        obj_id=obj_id,
                        color=color,
                        pixels=pixels,
                        size=size,
                        bounding_box=(min_r, min_c, max_r, max_c),
                        center_r=center_r,
                        center_c=center_c,
                    )
                )
                obj_id += 1

        agent_obj: Optional[VisualObject] = None
        targets: List[VisualObject] = []
        obstacles: Set[Tuple[int, int]] = set()
        interactables: Dict[str, Tuple[int, int]] = {}

        agent_color = known_roles.get("agent") if known_roles else None
        goal_color = known_roles.get("goal") if known_roles else None

        for obj in objects:
            if agent_color is not None and obj.color == agent_color:
                agent_obj = obj
            elif goal_color is not None and obj.color == goal_color:
                targets.insert(0, obj)
            elif obj.color == 1 or obj.size > (h * w * 0.20):
                for p in obj.pixels:
                    obstacles.add(p)
            elif obj.size == 1:
                interactables[f"item_color_{obj.color}"] = (int(round(obj.center_r)), int(round(obj.center_c)))
                targets.append(obj)
            else:
                for p in obj.pixels:
                    obstacles.add(p)

        # agent_obj が未指定・未発見の場合、最小の非障害物オブジェクトを候補とする
        if agent_obj is None:
            small_objs = [o for o in objects if o.size <= 4 and o.color != 1 and o not in targets]
            if not small_objs:
                small_objs = [o for o in objects if o.size <= 4 and o.color != 1]
            if small_objs:
                agent_obj = min(small_objs, key=lambda o: o.size)

        if agent_obj is not None:
            targets = [t for t in targets if t.obj_id != agent_obj.obj_id]

        agent_pos = (int(round(agent_obj.center_r)), int(round(agent_obj.center_c))) if agent_obj else None

        # スタイル判定
        affordances = self._detect_affordances(arr, bg_color)
        style = self._classify_style(arr, bg_color, affordances)

        return DynamicAffordanceReport(
            grid_shape=(h, w),
            background_color=bg_color,
            agent_object=agent_obj,
            agent_pos=agent_pos,
            target_candidates=targets,
            obstacles=obstacles,
            all_objects=objects,
            interactables=interactables,
            style=style,
        )

    def analyze_transition(
        self,
        obs_before: Any,
        action: Any,
        obs_after: Any,
        reward: float = 0.0,
        done: bool = False,
        known_roles: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """後方互換性API: 状態遷移前後の差分から自機の移動と衝突を判定."""
        arr_before = np.array(obs_before, dtype=int)
        arr_after = np.array(obs_after, dtype=int)

        rep_before = self.analyze_frame(arr_before, known_roles=known_roles)
        rep_after = self.analyze_frame(arr_after, known_roles=known_roles)

        p_before = rep_before.player_pos
        p_after = rep_after.player_pos

        # グリッド差分から移動オブジェクトの変位を検出
        diff = (arr_before != arr_after)
        if np.any(diff):
            bg_color = rep_before.background_color
            before_changed = arr_before[diff]
            after_changed = arr_after[diff]
            cand_colors = set(before_changed[before_changed != bg_color]) & set(after_changed[after_changed != bg_color])
            if cand_colors:
                color = list(cand_colors)[0]
                b_pts = np.argwhere(arr_before == color)
                a_pts = np.argwhere(arr_after == color)
                if len(b_pts) > 0 and len(a_pts) > 0:
                    p_before = (int(b_pts[0, 0]), int(b_pts[0, 1]))
                    p_after = (int(a_pts[0, 0]), int(a_pts[0, 1]))

        moved = (p_before != p_after) if (p_before and p_after) else False
        displacement = (
            (p_after[0] - p_before[0], p_after[1] - p_before[1])
            if (p_before and p_after)
            else (0, 0)
        )

        hit_obstacle = not moved
        action_name = action.name if hasattr(action, "name") else str(action)

        return {
            "action": action_name,
            "moved": moved,
            "displacement": displacement,
            "hit_obstacle": hit_obstacle,
            "reward": reward,
            "done": done,
            "player_pos_before": p_before,
            "player_pos_after": p_after,
        }


    def _extract_sequence(self, target_data: Optional[Any]) -> List[int]:
        if target_data is None:
            return []
        if isinstance(target_data, list):
            if all(isinstance(x, (int, np.integer)) for x in target_data):
                return [int(x) for x in target_data]
            arr = np.array(target_data, dtype=int)
            return [int(x) for x in arr[arr > 0].tolist()]
        arr = np.array(target_data, dtype=int)
        if arr.ndim > 0:
            return [int(x) for x in arr[arr > 0].tolist()]
        return []

    def _extract_primary_sequence(self, grid: np.ndarray, bg_color: int) -> List[int]:
        fg_coords = np.argwhere(grid != bg_color)
        if len(fg_coords) == 0:
            return []
        colors = [int(grid[y, x]) for y, x in fg_coords]
        compact: List[int] = []
        for c in colors:
            if not compact or compact[-1] != c:
                compact.append(c)
        return compact

    def _classify_gestalt_diff(
        self, current_seq: List[int], target_seq: List[int]
    ) -> Tuple[str, Dict[str, Any]]:
        if not target_seq or not current_seq:
            return self.ALIGNMENT_UNKNOWN, {"reason": "Missing target or current sequence"}

        if current_seq == target_seq:
            return self.ALIGNMENT_IDENTITY, {"reason": "Already in target sequence"}

        if sorted(current_seq) == sorted(target_seq):
            if current_seq == list(reversed(target_seq)):
                return self.ALIGNMENT_REVERSAL_REORDER, {
                    "reason": "Complete reverse order, requires buffer staging for reordering"
                }
            return self.ALIGNMENT_REVERSAL_REORDER, {
                "reason": "Permuted order, requires swap/buffer staging"
            }

        if len(target_seq) >= 3 and target_seq[0] == target_seq[2] and target_seq[0] != target_seq[1]:
            return self.ALIGNMENT_INTERLEAVED, {
                "reason": "Interleaved color sequence, requires multi-lane splitting"
            }

        return self.ALIGNMENT_TRANSLATION_ONLY, {
            "reason": "Subsets match, primarily relative translation and grouping"
        }

    def _detect_affordances(self, grid: np.ndarray, bg_color: int) -> Dict[str, Any]:
        h, w = grid.shape
        horizontal_rails = []
        vertical_rails = []
        for r in range(h):
            row_vals = grid[r, :]
            non_bg = np.where(row_vals != bg_color)[0]
            if len(non_bg) >= w - 2:
                horizontal_rails.append(r)

        for c in range(w):
            col_vals = grid[:, c]
            non_bg = np.where(col_vals != bg_color)[0]
            if len(non_bg) >= h - 2:
                vertical_rails.append(c)

        return {
            "rail_positions": {
                "horizontal": horizontal_rails,
                "vertical": vertical_rails,
            },
            "has_actuator_rail": bool(horizontal_rails or vertical_rails),
            "free_space_ratio": float(np.mean(grid == bg_color)),
        }

    def _classify_style(
        self, grid: np.ndarray, bg_color: int, affordances: Dict[str, Any]
    ) -> str:
        """ゲームスタイル・ジャンルの分類 (旧 game-style-intuitor の統合)."""
        h, w = grid.shape
        # 外枠が空いているか
        perimeter = np.concatenate([
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        ])
        perimeter_open_ratio = float(np.mean(perimeter == bg_color))

        if perimeter_open_ratio > 0.6:
            return self.STYLE_OPEN_EXPLORATION

        # 左右対称性チェック
        left_half = grid[:, : w // 2]
        right_half_flipped = np.fliplr(grid[:, w - w // 2 :])
        if np.array_equal(left_half, right_half_flipped):
            return self.STYLE_SYMMETRIC_PATTERN

        if affordances.get("has_actuator_rail"):
            return self.STYLE_ITEM_TRIGGER_PUZZLE

        return self.STYLE_CLOSED_MAZE

    def _infer_initial_invariants(
        self, diff_type: str, affordances: Dict[str, Any], style: str
    ) -> List[str]:
        invariants = [
            "Background cells represent passable or open travel space.",
        ]
        if affordances.get("has_actuator_rail"):
            invariants.append("Piston or carriage motion is constrained to fixed linear rails.")
        if diff_type == self.ALIGNMENT_REVERSAL_REORDER:
            invariants.append("Reordering requires separation into an intermediate buffer area.")
        elif diff_type == self.ALIGNMENT_INTERLEAVED:
            invariants.append("Interleaving requires two distinct bypass routes around obstacles.")
        if style == self.STYLE_OPEN_EXPLORATION:
            invariants.append("Boundaries are open; goal or key objects may reside at the perimeter.")
        return invariants

    def analyze_style(self, obs: np.ndarray) -> Dict[str, Any]:
        """後方互換性API: GameStyleIntuitor と完全同一のスタイル辞書を生成."""
        return analyze_game_style(obs)

    def intuit(self, obs: np.ndarray) -> Dict[str, Any]:
        """後方互換性API."""
        return analyze_game_style(obs)


def analyze_game_style(obs: np.ndarray) -> Dict[str, Any]:
    """観測グリッドの幾何学的・視覚的ゲシュタルトを解析し、ゲームスタイルを分類 (後方互換用)."""
    obs = np.array(obs, dtype=int)
    h, w = obs.shape
    unique_colors = np.unique(obs)
    num_colors = len(unique_colors)

    counts = np.bincount(obs.flatten(), minlength=10)
    if counts[0] > 0:
        bg_color = 0
    else:
        bg_color = int(np.argmax(counts))

    top_edge = obs[0, :]
    bottom_edge = obs[h - 1, :]
    left_edge = obs[:, 0]
    right_edge = obs[:, w - 1]

    border_cells = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
    border_open_ratio = float(np.mean(border_cells == bg_color))
    has_edge_exit = border_open_ratio > 0.25

    non_bg_mask = obs != bg_color
    obstacle_density = float(np.mean(non_bg_mask))

    if h > 2 and w > 2:
        inner_obs = obs[1 : h - 1, 1 : w - 1]
        fg_mask = inner_obs != bg_color
        fg_count = int(np.sum(fg_mask))
        inner_non_bg = float(np.mean(fg_mask))

        if fg_count >= 4:
            h_sym_match = int(np.sum(fg_mask & np.fliplr(fg_mask)))
            v_sym_match = int(np.sum(fg_mask & np.flipud(fg_mask)))
            h_sym_union = int(np.sum(fg_mask | np.fliplr(fg_mask)))
            v_sym_union = int(np.sum(fg_mask | np.flipud(fg_mask)))

            h_sym = float(h_sym_match / h_sym_union) if h_sym_union > 0 else 0.0
            v_sym = float(v_sym_match / v_sym_union) if v_sym_union > 0 else 0.0
            symmetry_score = max(h_sym, v_sym)
        else:
            symmetry_score = 0.0
    else:
        symmetry_score = 0.0
        inner_non_bg = 0.0

    color_counts = {}
    for c in unique_colors:
        if c == bg_color:
            continue
        coords = np.argwhere(obs == c)
        color_counts[int(c)] = len(coords)

    isolated_items = [c for c, count in color_counts.items() if 1 <= count <= 2]
    hazard_candidates = [c for c, count in color_counts.items() if 3 <= count <= 8]

    features = {
        "grid_shape": [h, w],
        "background_color": bg_color,
        "color_count": num_colors,
        "border_open_ratio": round(border_open_ratio, 3),
        "has_edge_exit": has_edge_exit,
        "obstacle_density": round(obstacle_density, 3),
        "symmetry_score": round(symmetry_score, 3),
        "isolated_item_colors": isolated_items,
        "hazard_colors": hazard_candidates,
    }

    if symmetry_score > 0.80 and inner_non_bg > 0.15:
        return {
            "style": "SYMMETRIC_PATTERN",
            "description": "Board exhibits high spatial symmetry. Geometric alignment game.",
            "features": features,
            "recommended_approach": "Preserve or manipulate spatial balance across symmetry axes.",
            "recommended_domain": "symmetry_pattern",
        }

    if has_edge_exit and obstacle_density < 0.40:
        return {
            "style": "OPEN_EXPLORATION",
            "description": (
                "Outer borders are largely unblocked. Goal is likely off-screen or involves"
                " traversing outside initial view."
            ),
            "features": features,
            "recommended_approach": (
                "EXPLORATION FIRST: Head towards open perimeter edges to expand field of view."
            ),
            "recommended_domain": "exploration",
        }

    if len(hazard_candidates) >= 1 and not has_edge_exit:
        return {
            "style": "HAZARD_AVOIDANCE",
            "description": (
                "Dangerous hazard barrier zones detected. Lethal penalty or game over on"
                " contact."
            ),
            "features": features,
            "recommended_approach": (
                "SAFETY FIRST: Identify and avoid entering fatal hazard cells while navigating"
                " to destination."
            ),
            "recommended_domain": "hazard_avoidance",
        }

    if len(isolated_items) >= 3:
        return {
            "style": "ITEM_TRIGGER_PUZZLE",
            "description": (
                "Multiple isolated colored objects detected. Sequential trigger or key-lock"
                " mechanics active."
            ),
            "features": features,
            "recommended_approach": (
                "INTERACTION FIRST: Route to isolated item entities to alter game state"
                " before exit."
            ),
            "recommended_domain": "inventory_puzzle",
        }

    if obstacle_density >= 0.15 and not has_edge_exit:
        return {
            "style": "CLOSED_MAZE",
            "description": (
                "Enclosed boundary with internal wall corridors. Traditional shortest path or"
                " obstacle avoidance."
            ),
            "features": features,
            "recommended_approach": (
                "PATHFINDING FIRST: Execute deterministic detour search around internal barriers."
            ),
            "recommended_domain": "navigation",
        }

    return {
        "style": "GENERAL_GRID_GAME",
        "description": "Standard discrete grid environment without extreme structural bias.",
        "features": features,
        "recommended_approach": (
            "BALANCED: Explore forward while avoiding obstacles and observing reward signals."
        ),
        "recommended_domain": "general",
    }


class GameStyleIntuitor:
    """ゲームスタイル分類エンジン (後方互換用)."""

    def analyze(self, obs: np.ndarray) -> Dict[str, Any]:
        return analyze_game_style(obs)

    def intuit(self, obs: np.ndarray) -> Dict[str, Any]:
        return analyze_game_style(obs)

    def analyze_style(self, obs: np.ndarray) -> Dict[str, Any]:
        return analyze_game_style(obs)


# 後方互換性エイリアス
MetaObserver = VisualInspector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual Inspector - Unified Perception, Affordance & Style CLI"
    )
    parser.add_argument("--grid", type=str, help="JSON string of current observation grid")
    parser.add_argument("--target", type=str, help="JSON string of target sequence or grid")
    parser.add_argument("--step", type=int, default=0, help="Current step index (default: 0)")
    parser.add_argument("--input", type=str, help="Input data string (compatibility)")
    parser.add_argument("--file", type=str, help="Path to JSON file with input data")
    args = parser.parse_args()

    inspector = VisualInspector()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            grid = data.get("grid") or data.get("current_grid")
            target = data.get("target") or data.get("target_sequence")
            step = data.get("step", 0)
    elif args.grid:
        grid = json.loads(args.grid)
        target = json.loads(args.target) if args.target else None
        step = args.step
    else:
        grid = [[0, 0, 0], [1, 2, 3], [0, 0, 0]]
        target = [3, 2, 1]
        step = 0

    result = inspector.inspect_board(grid, target, step_index=step)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
