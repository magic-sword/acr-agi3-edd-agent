"""ACR-AGI-3 動的ゲーム環境向け汎用アフォーダンス同定エンジン (Meta-Observer).

ゲーム観測フレームから、背景色、自機位置、静的障害物、ゴール、インタラクタブルなどの
アフォーダンス (Affordances) と状態遷移因果 (Transition Dynamics) を完全自律抽出します。
特定ゲーム環境の固定色 (player=2, goal=3 等) に依存せず、動的差分相関とゲシュタルト特徴量で同定します。
"""

from __future__ import annotations

import collections
import dataclasses
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


class MetaObserver:
    """未知のゲーム環境から不変量とアフォーダンスを自律抽出するメタ認知エンジン."""

    def __init__(self) -> None:
        self.background_color: int = 0
        self.known_obstacles: Set[Tuple[int, int]] = set()
        self.identified_agent_color: Optional[int] = None
        self.identified_agent_id: Optional[int] = None
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None

    def analyze_frame(
        self,
        grid: np.ndarray,
        recent_action: Optional[int] = None,
        displaced_pixels: Optional[List[Tuple[int, int]]] = None,
    ) -> DynamicAffordanceReport:
        """単一フレームまたは遷移情報からアフォーダンスを自律同定."""
        arr = np.array(grid, dtype=int)
        if arr.ndim == 3:
            arr = arr[-1]  # アニメーションシーケンスの場合は最新フレーム
        h, w = arr.shape

        # 1. 最頻色を背景色と同定
        counts = np.bincount(arr.flatten(), minlength=10)
        bg_color = int(np.argmax(counts))
        self.background_color = bg_color

        # 2. 4近傍連結成分 (Connected Components) によるオブジェクト分割
        raw_objects = self._extract_components(arr, bg_color)

        # 3. 自機 (Agent) の動的同定
        # 行動後のピクセル差分または過去の同定情報から自機を決定
        agent_obj: Optional[VisualObject] = None
        controllable_verified = False

        if recent_action is not None and self.prev_grid is not None:
            # アクションに伴う差分ピクセル
            diff = np.argwhere(arr != self.prev_grid)
            if len(diff) > 0 and recent_action in (1, 2, 3, 4):  # UP, DOWN, LEFT, RIGHT
                expected_dr, expected_dc = {
                    1: (-1, 0),  # UP
                    2: (1, 0),   # DOWN
                    3: (0, -1),  # LEFT
                    4: (0, 1),   # RIGHT
                }[recent_action]

                # 各オブジェクトの重心移動を評価
                for obj in raw_objects:
                    for prev_obj in self._extract_components(self.prev_grid, bg_color):
                        if prev_obj.color == obj.color and abs(prev_obj.size - obj.size) <= 2:
                            dr = obj.center_r - prev_obj.center_r
                            dc = obj.center_c - prev_obj.center_c
                            # 期待される移動方向と一致しているか
                            if (np.sign(dr) == expected_dr and expected_dr != 0) or \
                               (np.sign(dc) == expected_dc and expected_dc != 0):
                                agent_obj = obj
                                agent_obj.role = "agent"
                                self.identified_agent_color = obj.color
                                controllable_verified = True
                                break
                    if controllable_verified:
                        break

        # 差分から未特定の場合、過去に特定した自機色を照合
        if agent_obj is None and self.identified_agent_color is not None:
            for obj in raw_objects:
                if obj.color == self.identified_agent_color:
                    agent_obj = obj
                    agent_obj.role = "agent"
                    controllable_verified = True
                    break

        # 自機がまだ未特定の場合、ヒューリスティクス（小さく独立したスプライト）で仮説生成
        if agent_obj is None and raw_objects:
            small_objs = [o for o in raw_objects if o.size < (h * w * 0.05)]
            if small_objs:
                agent_obj = min(small_objs, key=lambda o: o.size)
                agent_obj.role = "agent_candidate"

        # 4. 障害物 (Obstacles) の同定
        obstacles: Set[Tuple[int, int]] = set(self.known_obstacles)
        for obj in raw_objects:
            if agent_obj and obj.obj_id == agent_obj.obj_id:
                continue
            is_large = obj.size > (h * w * 0.08)
            aspect_ratio = max(
                (obj.bounding_box[2] - obj.bounding_box[0] + 1) / max(1, (obj.bounding_box[3] - obj.bounding_box[1] + 1)),
                (obj.bounding_box[3] - obj.bounding_box[1] + 1) / max(1, (obj.bounding_box[2] - obj.bounding_box[0] + 1)),
            )
            is_line = aspect_ratio > 4.0 and obj.size > 8
            if is_large or is_line:
                obj.role = "obstacle"
                for r, c in obj.pixels:
                    obstacles.add((r, c))

        # 5. ターゲット/ゴール候補 (Target Candidates) のゲシュタルトスコアリング
        target_candidates: List[VisualObject] = []
        for obj in raw_objects:
            if agent_obj and obj.obj_id == agent_obj.obj_id:
                continue
            if obj.role == "obstacle":
                continue

            # スコア算出
            color_rarity = 1.0 - (counts[obj.color] / (h * w))
            size_compactness = 1.0 / (1.0 + np.log1p(obj.size))
            dist_score = 1.0
            if agent_obj:
                d = abs(obj.center_r - agent_obj.center_r) + abs(obj.center_c - agent_obj.center_c)
                dist_score = 1.0 / (1.0 + d * 0.05)

            obj.target_score = (color_rarity * 2.0) + (size_compactness * 1.5) + dist_score
            obj.role = "target_candidate"
            target_candidates.append(obj)

        target_candidates.sort(key=lambda o: o.target_score, reverse=True)
        self.prev_grid = arr.copy()

        agent_pos = None
        if agent_obj:
            agent_pos = (int(round(agent_obj.center_r)), int(round(agent_obj.center_c)))

        return DynamicAffordanceReport(
            grid_shape=(h, w),
            background_color=bg_color,
            agent_object=agent_obj,
            agent_pos=agent_pos,
            target_candidates=target_candidates,
            obstacles=obstacles,
            all_objects=raw_objects,
            controllable_verified=controllable_verified,
        )

    def register_collision(self, r: int, c: int) -> None:
        """移動に失敗したセルを障害物として動的学習."""
        self.known_obstacles.add((r, c))

    def _extract_components(self, grid: np.ndarray, bg_color: int) -> List[VisualObject]:
        """グリッドから 4 近傍連結成分オブジェクトを高速抽出."""
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        objects: List[VisualObject] = []
        obj_id = 0

        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if color == bg_color or visited[r, c]:
                    continue

                comp_pixels: List[Tuple[int, int]] = []
                q = collections.deque([(r, c)])
                visited[r, c] = True

                min_r, max_r = r, r
                min_c, max_c = c, c

                while q:
                    cr, cc = q.popleft()
                    comp_pixels.append((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)

                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                size = len(comp_pixels)
                center_r = sum(p[0] for p in comp_pixels) / size
                center_c = sum(p[1] for p in comp_pixels) / size

                objects.append(
                    VisualObject(
                        obj_id=obj_id,
                        color=color,
                        pixels=comp_pixels,
                        size=size,
                        bounding_box=(min_r, min_c, max_r, max_c),
                        center_r=center_r,
                        center_c=center_c,
                    )
                )
                obj_id += 1

        return objects
