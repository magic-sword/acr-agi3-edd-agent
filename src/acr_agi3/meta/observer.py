"""環境不変量およびアフォーダンス同定エンジン (Meta-Observer).

ARC グリッドペアから、背景色、形状比率、静的障害物、移動可能オブジェクトなどの
不変量 (Invariants) とアフォーダンス (Affordances) を自律抽出します。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import label

from acr_agi3.game.env import Action


@dataclass
class AffordanceObject:
    """同定されたオブジェクトの属性情報."""

    color: int
    size: int
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    is_static: bool = True
    role: str = "obstacle"  # 'agent', 'interactable', 'obstacle', 'target', 'hazard'


@dataclass
class GameAffordanceReport:
    """動的ゲーム環境におけるフレーム観測レポート."""

    grid_shape: Tuple[int, int]
    background_color: int
    player_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    obstacles: Set[Tuple[int, int]] = field(default_factory=set)
    hazards: Set[Tuple[int, int]] = field(default_factory=set)
    interactables: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    color_map: Dict[str, int] = field(default_factory=dict)
    all_objects: List[AffordanceObject] = field(default_factory=list)


@dataclass
class ObservationReport:
    """環境観察結果の構造化レポート."""

    in_shape: Tuple[int, int]
    out_shape: Tuple[int, int]
    shape_ratio: Tuple[float, float]
    background_color: int
    preserved_colors: Set[int] = field(default_factory=set)
    new_colors: Set[int] = field(default_factory=set)
    removed_colors: Set[int] = field(default_factory=set)
    objects: List[AffordanceObject] = field(default_factory=list)
    transformation_hint: str = "unknown"


class MetaObserver:
    """環境の不変量とアフォーダンスを自動抽出するメタ認知エンジン."""

    def __init__(self) -> None:
        pass

    def analyze_frame(
        self,
        grid: np.ndarray,
        known_roles: Optional[Dict[str, int]] = None,
    ) -> GameAffordanceReport:
        """単一フレームのグリッドからゲームアフォーダンスを同定.

        known_roles: 例 {'player': 2, 'goal': 3, 'wall': 1, 'hazard': 6, 'key': 4}
        """
        arr = np.array(grid, dtype=int)
        h, w = arr.shape
        counts = np.bincount(arr.flatten(), minlength=10)
        bg_color = int(np.argmax(counts))

        roles = known_roles or {}
        p_color = roles.get("player", 2)
        g_color = roles.get("goal", 3)
        w_color = roles.get("wall", 1)
        h_color = roles.get("hazard", 6)

        player_pos: Optional[Tuple[int, int]] = None
        goal_pos: Optional[Tuple[int, int]] = None
        obstacles: Set[Tuple[int, int]] = set()
        hazards: Set[Tuple[int, int]] = set()
        interactables: Dict[str, Tuple[int, int]] = {}

        # プレイヤー座標
        p_coords = np.argwhere(arr == p_color)
        if len(p_coords) > 0:
            player_pos = (int(p_coords[0][0]), int(p_coords[0][1]))

        # ゴール座標
        g_coords = np.argwhere(arr == g_color)
        if len(g_coords) > 0:
            goal_pos = (int(g_coords[0][0]), int(g_coords[0][1]))

        # 壁・障害物座標
        for r, c in np.argwhere(arr == w_color):
            obstacles.add((int(r), int(c)))

        # 危険物座標
        for r, c in np.argwhere(arr == h_color):
            hazards.add((int(r), int(c)))

        # その他の色（鍵やスイッチ等のインタラクティブ要素）
        for c in np.unique(arr):
            ci = int(c)
            if ci in (bg_color, p_color, g_color, w_color, h_color):
                continue
            coords = np.argwhere(arr == ci)
            if len(coords) > 0:
                interactables[f"item_color_{ci}"] = (int(coords[0][0]), int(coords[0][1]))

        all_objs = self.extract_objects(arr, bg_color)

        return GameAffordanceReport(
            grid_shape=(h, w),
            background_color=bg_color,
            player_pos=player_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            hazards=hazards,
            interactables=interactables,
            color_map=roles,
            all_objects=all_objs,
        )

    def analyze_transition(
        self,
        obs_before: np.ndarray,
        action: Action,
        obs_after: np.ndarray,
        reward: float,
        done: bool,
    ) -> Dict[str, Any]:
        """行動前後の観測遷移から因果規則（移動の成否、壁の衝突、効果）を抽出."""
        rep_before = self.analyze_frame(obs_before)
        rep_after = self.analyze_frame(obs_after)

        p_before = rep_before.player_pos
        p_after = rep_after.player_pos

        moved = (p_before != p_after) if (p_before and p_after) else False
        displacement = (
            (p_after[0] - p_before[0], p_after[1] - p_before[1])
            if (p_before and p_after)
            else (0, 0)
        )

        hit_obstacle = False
        if not moved and action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            hit_obstacle = True

        return {
            "action": action.name if hasattr(action, "name") else str(action),
            "moved": moved,
            "displacement": displacement,
            "hit_obstacle": hit_obstacle,
            "reward": reward,
            "done": done,
            "player_pos_before": p_before,
            "player_pos_after": p_after,
        }

    def extract_objects(self, grid: np.ndarray, background_color: int) -> List[AffordanceObject]:
        """グリッドから背景色以外の連結成分オブジェクトを抽出."""
        objects: List[AffordanceObject] = []
        unique_colors = [c for c in np.unique(grid) if c != background_color]

        for color in unique_colors:
            mask = grid == color
            labeled, num_features = label(mask)
            for feat_id in range(1, num_features + 1):
                feat_mask = labeled == feat_id
                indices = np.argwhere(feat_mask)
                if len(indices) == 0:
                    continue
                min_r, min_c = indices.min(axis=0)
                max_r, max_c = indices.max(axis=0)
                objects.append(
                    AffordanceObject(
                        color=int(color),
                        size=int(len(indices)),
                        bounding_box=(int(min_r), int(min_c), int(max_r), int(max_c)),
                        is_static=True,
                        role="interactable",
                    )
                )
        return objects

    def analyze_pair(self, input_grid: np.ndarray, output_grid: np.ndarray) -> ObservationReport:
        """単一の入出力ペアから不変量とアフォーダンスを分析."""
        inp = np.array(input_grid, dtype=int)
        out = np.array(output_grid, dtype=int)

        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        ratio = (out_h / in_h, out_w / in_w)

        # 最頻度色を背景色と推定
        in_counts = np.bincount(inp.flatten(), minlength=10)
        bg_color = int(np.argmax(in_counts))

        in_colors = set(int(c) for c in np.unique(inp))
        out_colors = set(int(c) for c in np.unique(out))

        preserved = in_colors.intersection(out_colors)
        new_colors = out_colors - in_colors
        removed_colors = in_colors - out_colors

        # オブジェクト抽出
        in_objects = self.extract_objects(inp, bg_color)
        out_objects = self.extract_objects(out, bg_color)

        # 形状・位置差分による役割推定
        hint = "geometric_or_color"
        if ratio == (1.0, 1.0):
            if in_colors == out_colors:
                # 色が同じで形状も同じなら移動または回転
                if len(in_objects) == len(out_objects):
                    hint = "translation_or_rotation"
                else:
                    hint = "pattern_repetition_or_split"
            else:
                hint = "color_replacement"
        elif ratio[0] > 1.0 or ratio[1] > 1.0:
            hint = "scaling_or_tiling"
        elif ratio[0] < 1.0 or ratio[1] < 1.0:
            hint = "cropping_or_extraction"

        return ObservationReport(
            in_shape=(in_h, in_w),
            out_shape=(out_h, out_w),
            shape_ratio=ratio,
            background_color=bg_color,
            preserved_colors=preserved,
            new_colors=new_colors,
            removed_colors=removed_colors,
            objects=in_objects,
            transformation_hint=hint,
        )

    def analyze_task(self, train_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """タスク全体の全 Train ペアに共通する不変量を集約."""
        if not train_pairs:
            return {}

        reports = [self.analyze_pair(p["input"], p["output"]) for p in train_pairs]

        # 全ペアで共通の形状比率か
        ratios = [r.shape_ratio for r in reports]
        consistent_ratio = ratios[0] if all(r == ratios[0] for r in ratios) else None

        bg_colors = [r.background_color for r in reports]
        consistent_bg = bg_colors[0] if all(b == bg_colors[0] for b in bg_colors) else None

        hints = [r.transformation_hint for r in reports]
        dominant_hint = max(set(hints), key=hints.count)

        return {
            "consistent_shape_ratio": consistent_ratio,
            "consistent_background": consistent_bg,
            "transformation_hint": dominant_hint,
            "sample_objects_count": len(reports[0].objects),
            "num_train_pairs": len(train_pairs),
        }
