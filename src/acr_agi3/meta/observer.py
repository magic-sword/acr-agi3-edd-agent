"""ACR-AGI-3 動的ゲーム環境向けアフォーダンス同定エンジン (Meta-Observer).

ゲーム観測フレームから、背景色、自機位置、静的障害物、ゴール、危険物などの
アフォーダンス (Affordances) と状態遷移因果 (Transition Dynamics) を自律抽出します。
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


class MetaObserver:
    """ゲーム環境の不変量とアフォーダンスを自動抽出するメタ認知エンジン."""

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
