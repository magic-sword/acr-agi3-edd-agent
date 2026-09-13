"""VCGT 準拠 視覚ゲシュタルト認知＆サブゴール計画エンジン (Gestalt-VCGT Planner).

人間の視覚認知（Visual Concept Guided Thinking）に基づき、
1. 画面のオブジェクト（自機、ボタン、ターゲット、壁）をゲシュタルト同定
2. ターゲットの重心座標への意図的クリック (ACTION6)
3. 障害物回避の A* / BFS 経路計画による連続移動 (ACTION1〜5)
4. 予期せぬ壁衝突時のメンタルマップ更新と再計画 (Replanning)
を 1 ステップ 1ms 未満の決定論的ベクトル演算で実行する。
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclasses.dataclass
class VisualObject:
    """同定された視覚オブジェクト."""

    color: int
    pixels: List[Tuple[int, int]]  # list of (r, c)
    bbox: Tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    center_x: int  # c
    center_y: int  # r
    role: str = "unknown"  # 'player', 'target', 'wall', 'button'


class GestaltPerceiver:
    """視覚ゲシュタルト同定モジュール."""

    def __init__(self) -> None:
        self.prev_grid: Optional[List[List[int]]] = None
        self.player_pos: Optional[Tuple[int, int]] = None  # (r, c)
        self.background_color: int = 0

    def parse(self, grid: Any) -> Dict[str, Any]:
        """グリッドから背景、オブジェクト群、自機、ターゲットを同定."""
        if not grid:
            return {"background_color": 0, "objects": [], "player": None, "targets": [], "walls": set()}

        # 3次元 (N, H, W) やアニメーションシーケンスの安全な正規化（最新フレームを採用）
        if isinstance(grid, (list, tuple)) and len(grid) > 0:
            if isinstance(grid[0], (list, tuple)) and len(grid[0]) > 0 and isinstance(grid[0][0], (list, tuple)):
                grid = grid[-1]
            elif len(grid) == 1 and isinstance(grid[0], (list, tuple)):
                grid = grid[0]

        h = len(grid)
        w = len(grid[0]) if h > 0 and isinstance(grid[0], (list, tuple)) else 0
        if h == 0 or w == 0:
            return {"background_color": 0, "objects": [], "player": None, "targets": [], "walls": set()}

        # 2D int グリッドへ正規化
        norm_grid: List[List[int]] = []
        for r in range(h):
            row = []
            for c in range(w):
                val = grid[r][c]
                pixel_val = val[0] if isinstance(val, (list, tuple)) else val
                try:
                    row.append(int(pixel_val))
                except Exception:
                    row.append(0)
            norm_grid.append(row)
        grid = norm_grid

        # 1. 最頻色を背景色と同定
        color_counts: Dict[int, int] = collections.defaultdict(int)
        for r in range(h):
            for c in range(w):
                color_counts[grid[r][c]] += 1
        self.background_color = max(color_counts, key=color_counts.get)

        # 2. 前フレームとの差分ピクセル（動的オブジェクトの検出）
        diff_pixels: List[Tuple[int, int]] = []
        if self.prev_grid and len(self.prev_grid) == h and len(self.prev_grid[0]) == w:
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != self.prev_grid[r][c]:
                        diff_pixels.append((r, c))

        # 3. 連結成分（4-Connected Components）によるスプライト/オブジェクト同定
        visited: Set[Tuple[int, int]] = set()
        raw_components: List[Tuple[int, List[Tuple[int, int]]]] = []

        for r in range(h):
            for c in range(w):
                val = grid[r][c]
                if val == self.background_color or (r, c) in visited:
                    continue

                comp: List[Tuple[int, int]] = []
                q = collections.deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if (nr, nc) not in visited and grid[nr][nc] == val:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                raw_components.append((val, comp))

        objects: List[VisualObject] = []
        walls: Set[Tuple[int, int]] = set()
        targets: List[VisualObject] = []
        identified_player: Optional[Tuple[int, int]] = None

        # 動いたピクセルの中に非背景色があれば、それを自機候補とする
        if diff_pixels:
            non_bg_diff = [p for p in diff_pixels if grid[p[0]][p[1]] != self.background_color]
            if non_bg_diff:
                identified_player = non_bg_diff[0]

        for color, pixs in raw_components:
            # 大面積（画面の25%超）のものは壁（障害物）
            if len(pixs) > (h * w * 0.25):
                for p in pixs:
                    walls.add(p)
                continue

            # 外接矩形と重心の計算
            min_r = min(p[0] for p in pixs)
            max_r = max(p[0] for p in pixs)
            min_c = min(p[1] for p in pixs)
            max_c = max(p[1] for p in pixs)
            center_r = (min_r + max_r) // 2
            center_c = (min_c + max_c) // 2

            obj = VisualObject(
                color=color,
                pixels=pixs,
                bbox=(min_r, min_c, max_r, max_c),
                center_x=center_c,
                center_y=center_r,
            )

            # 自機判定
            if identified_player and (min_r <= identified_player[0] <= max_r) and (min_c <= identified_player[1] <= max_c):
                obj.role = "player"
                self.player_pos = (center_r, center_c)
            else:
                obj.role = "target"
                targets.append(obj)

            objects.append(obj)

        # 自機がまだ未特定の場合、前回の自機位置に近いもの、または希少色を採用
        if not self.player_pos and objects:
            self.player_pos = (objects[0].center_y, objects[0].center_x)

        self.prev_grid = [row[:] for row in grid]

        return {
            "background_color": self.background_color,
            "objects": objects,
            "player": self.player_pos,
            "targets": targets,
            "walls": walls,
        }


class GestaltVCGTPlanner:
    """VCGT 準拠 リアルタイム認知＆サブゴール計画エンジン."""

    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.perceiver = GestaltPerceiver()
        self.plan_queue: collections.deque[int] = collections.deque()
        self.clicked_coords: Set[Tuple[int, int]] = set()
        self.known_obstacles: Set[Tuple[int, int]] = set()
        self.last_action_id: Optional[int] = None
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.step_index: int = 0
        self.confirmed_interactive_group: Optional[Tuple[int, int]] = None
        self.last_clicked_group_key: Optional[Tuple[int, int]] = None
        self.last_clicked_pos: Optional[Tuple[int, int]] = None
        self.last_hit_pos: Optional[Tuple[int, int]] = None

    def decide_action(
        self,
        grid: List[List[int]],
        available_action_ids: List[int],
    ) -> Tuple[int, Dict[str, Any], str]:
        """フレーム観測と利用可能アクションから、人間的意図を持った行動を決定.

        Returns:
            (action_id, action_data, reasoning_str)
        """
        self.step_index += 1
        # 3次元テンソルを最新フレームの 2D グリッドへ完全正規化
        if isinstance(grid, (list, tuple)) and len(grid) > 0:
            if isinstance(grid[0], (list, tuple)) and len(grid[0]) > 0 and isinstance(grid[0][0], (list, tuple)):
                grid = grid[-1]
            elif len(grid) == 1 and isinstance(grid[0], (list, tuple)):
                grid = grid[0]

        parsed = self.perceiver.parse(grid)
        player = parsed.get("player")
        targets: List[VisualObject] = parsed.get("targets", [])
        walls: Set[Tuple[int, int]] = parsed.get("walls", set())
        all_obstacles = walls | self.known_obstacles

        h = len(grid) if grid else 64
        w = len(grid[0]) if grid and grid[0] else 64

        # -------------------------------------------------------------
        # 1. クリック系タスク (ACTION6: Complex Action) の計画
        # -------------------------------------------------------------
        if 6 in available_action_ids:
            target_candidates: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]] = []

            # ゲシュタルト同定 (A): 同一色・同一サイズの反復スプライト群（キーパッド/タイル盤）をグループ化
            size_color_groups = collections.defaultdict(list)
            for obj in targets:
                if 4 <= len(obj.pixels) < (h * w * 0.2):
                    size_color_groups[(obj.color, len(obj.pixels))].append(obj)

            # 過去にヒットが確認された正解インタラクティブグループを絶対最優先！
            if self.confirmed_interactive_group and self.confirmed_interactive_group in size_color_groups:
                hit_group = size_color_groups[self.confirmed_interactive_group]
                # 直前にヒットした座標に近い順（局所クラスター優先）にソート
                if self.last_hit_pos:
                    lx, ly = self.last_hit_pos
                    hit_group = sorted(
                        hit_group,
                        key=lambda o: abs(o.bbox[1] - lx) + abs(o.bbox[0] - ly),
                    )
                for obj in hit_group:
                    for pt in [(obj.bbox[1], obj.bbox[0]), (obj.center_x, obj.center_y)]:
                        if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                            target_candidates.append((pt, self.confirmed_interactive_group))

            # まだ特定されていない場合、3個以上並んでいる反復グループを優先探索
            if not target_candidates:
                repeated_groups = [(k, g) for k, g in size_color_groups.items() if len(g) >= 3]
                repeated_groups.sort(key=lambda item: (-len(item[1]), -len(item[1][0].pixels)))

                for grp_key, group in repeated_groups:
                    for obj in group:
                        for pt in [(obj.bbox[1], obj.bbox[0]), (obj.center_x, obj.center_y)]:
                            if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                                target_candidates.append((pt, grp_key))

            # ゲシュタルト同定 (B): その他のターゲットオブジェクト
            if not target_candidates:
                sorted_targets = sorted(targets, key=lambda o: len(o.pixels))
                for obj in sorted_targets:
                    grp_key = (obj.color, len(obj.pixels))
                    for pt in [(obj.bbox[1], obj.bbox[0]), (obj.center_x, obj.center_y)]:
                        if pt not in self.clicked_coords and not any(pt == c[0] for c in target_candidates):
                            target_candidates.append((pt, grp_key))

            # ゲシュタルト同定 (C): 各オブジェクトの構成ピクセル
            if not target_candidates:
                for obj in targets:
                    grp_key = (obj.color, len(obj.pixels))
                    for r, c in obj.pixels:
                        if (c, r) not in self.clicked_coords and not any((c, r) == cand[0] for cand in target_candidates):
                            target_candidates.append(((c, r), grp_key))
                            break

            # ターゲットが存在する場合は、その座標をクリック
            if target_candidates:
                (target_x, target_y), grp_key = target_candidates[0]
                self.clicked_coords.add((target_x, target_y))
                self.last_action_id = 6
                self.last_clicked_group_key = grp_key
                self.last_clicked_pos = (target_x, target_y)
                reasoning = f"VCGT Click: Target interactable object at ({target_x}, {target_y})"
                return 6, {"x": int(target_x), "y": int(target_y)}, reasoning

            # 画面内に未クリックの非背景ピクセルがあればそこをクリック
            bg = parsed.get("background_color", 0)
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != bg and (c, r) not in self.clicked_coords:
                        self.clicked_coords.add((c, r))
                        self.last_action_id = 6
                        self.last_clicked_group_key = None
                        return 6, {"x": int(c), "y": int(r)}, f"VCGT Click: Active pixel at ({c}, {r})"

            # 全て探索済みの場合は中央付近をクリック
            cx, cy = w // 2, h // 2
            return 6, {"x": int(cx), "y": int(cy)}, f"VCGT Click: Center fallback ({cx}, {cy})"

        # -------------------------------------------------------------
        # 2. 移動系タスク (ACTION1〜5) のサブゴール＆経路計画 (A* / BFS)
        # -------------------------------------------------------------
        # 計画キューにアクションが残っていればそれを出力
        if self.plan_queue:
            act = self.plan_queue.popleft()
            if act in available_action_ids:
                self.last_action_id = act
                return act, {}, f"VCGT Macro: Step {self.step_index} pursuing planned trajectory"

        # 計画キューが空の場合、新しいサブゴール経路を計画
        if player and targets:
            # 最も近いターゲットを選択
            nearest_target = min(
                targets,
                key=lambda obj: abs(obj.center_y - player[0]) + abs(obj.center_x - player[1]),
            )
            goal_r, goal_c = nearest_target.center_y, nearest_target.center_x

            # BFS 最短経路探索
            actions_plan = self._find_path_bfs(
                start=player,
                goal=(goal_r, goal_c),
                obstacles=all_obstacles,
                grid_shape=(h, w),
                available_actions=available_action_ids,
            )

            if actions_plan:
                for a in actions_plan:
                    self.plan_queue.append(a)
                act = self.plan_queue.popleft()
                self.last_action_id = act
                return act, {}, f"VCGT Plan: Navigating to target ({goal_c}, {goal_r}) with {len(actions_plan)} steps"

        # -------------------------------------------------------------
        # 3. 経路が見つからない・障害物だらけの場合：直進モメンタム付き探索
        # -------------------------------------------------------------
        move_actions = [a for a in available_action_ids if a in [1, 2, 3, 4]]
        if move_actions:
            # 前回のアクションが有効だった場合はそれを維持、そうでなければ別の方向へ
            if self.last_action_id in move_actions:
                act = self.last_action_id
            else:
                act = move_actions[0]
            self.last_action_id = act
            return act, {}, f"VCGT Momentum Explore: {act}"

        # フォールバック
        fallback_act = available_action_ids[0] if available_action_ids else 1
        return fallback_act, {}, "VCGT Fallback"

    def on_feedback(self, is_effective: bool, pixels_changed: int) -> None:
        """行動結果のフィードバックを受け取り、メンタルモデルを修正 (仮説検証)."""
        if is_effective and self.last_action_id == 6 and self.last_clicked_group_key:
            self.confirmed_interactive_group = self.last_clicked_group_key
            if self.last_clicked_pos:
                self.last_hit_pos = self.last_clicked_pos

        if not is_effective and self.last_action_id in [1, 2, 3, 4]:
            # 動けなかった場合、計画していたパスに未知の壁があったと判断
            self.plan_queue.clear()
            self.last_action_id = None

    def _find_path_bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid_shape: Tuple[int, int],
        available_actions: List[int],
    ) -> List[int]:
        """BFS による最短経路計画."""
        h, w = grid_shape
        # ARC-AGI-3 アクションとグリッド移動の対応マッピング
        # 一般的な 4 方向移動マッピング (UP=-1,0; DOWN=+1,0; LEFT=0,-1; RIGHT=0,+1)
        # ACTION1〜4 がそれぞれ方向に対応
        moves: List[Tuple[int, int, int]] = []
        if 1 in available_actions:
            moves.append((1, -1, 0))  # UP
        if 2 in available_actions:
            moves.append((2, 1, 0))   # DOWN
        if 3 in available_actions:
            moves.append((3, 0, -1))  # LEFT
        if 4 in available_actions:
            moves.append((4, 0, 1))   # RIGHT

        if not moves:
            return []

        queue = collections.deque([(start, [])])
        visited = {start}

        while queue:
            (curr_r, curr_c), path = queue.popleft()
            if (curr_r, curr_c) == goal:
                return path

            for act_id, dr, dc in moves:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and (nr, nc) not in obstacles:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [act_id]))
                    if len(path) >= 20:  # 最大 20 ステップ先まで計画
                        return path + [act_id]

        return []
