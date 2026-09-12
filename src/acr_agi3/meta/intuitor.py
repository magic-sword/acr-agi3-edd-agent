"""ゲーム画面の視覚的テクスチャ・ゲシュタルト直感エンジン (Visual Game Style Intuitor).

特定の色や座標の決め打ちに頼らず、画面全体のテクスチャ、境界開放度、
オブジェクト密度、空間的対称性から「どんな種類のゲームか」を直感的に分類し、
ゴールが画面外にある探索型や、動的カラーシフトゲームへの初動アプローチを決定します。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class GameStyleIntuitor:
    """視覚的テクスチャ・雰囲気からゲームジャンルを直感するメタスキル."""

    def analyze_style(self, obs: np.ndarray) -> Dict[str, Any]:
        """観測グリッドの幾何学的・視覚的ゲシュタルトを解析し、ゲームスタイルを分類."""
        if obs.ndim != 2:
            raise ValueError(f"Observation must be 2D array, got shape {obs.shape}")

        h, w = obs.shape
        unique_colors = np.unique(obs)
        num_colors = len(unique_colors)

        # 1. 背景色の特定 (ARC-AGI 仕様として通常は 0 (黒) が背景。0 が不在の場合のみ最頻色)
        counts = np.bincount(obs.flatten(), minlength=10)
        if counts[0] > 0:
            bg_color = 0
        else:
            bg_color = int(np.argmax(counts))

        # 2. 境界（外周 4 辺）の開放度分析 (Border Openness)
        # 上辺、下辺、左辺、右辺で背景色（通路）になっているマスの割合
        top_edge = obs[0, :]
        bottom_edge = obs[h - 1, :]
        left_edge = obs[:, 0]
        right_edge = obs[:, w - 1]

        border_cells = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
        border_open_ratio = float(np.mean(border_cells == bg_color))
        has_edge_exit = border_open_ratio > 0.25  # 外周の25%以上が開いている

        # 3. テクスチャ密度・複雑度分析 (Wall/Obstacle Density)
        non_bg_mask = obs != bg_color
        obstacle_density = float(np.mean(non_bg_mask))

        # 4. 空間的対称性スコア (前景オブジェクトの幾何学的対称性 IoU)
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

        # 5. エンティティとスパースアイテムの検出
        # 各色のピクセル数と役割分析
        color_counts = {}
        for c in unique_colors:
            if c == bg_color:
                continue
            coords = np.argwhere(obs == c)
            color_counts[int(c)] = len(coords)

        # 点オブジェクト (1〜2マス: プレイヤー、ゴール、鍵など)
        isolated_items = [c for c, count in color_counts.items() if 1 <= count <= 2]
        # 中規模危険帯・トラップブロック (3〜8マスの帯状オブジェクト)
        hazard_candidates = [c for c, count in color_counts.items() if 3 <= count <= 8]

        # 6. スタイル分類ロジック (Game Style Taxonomy)
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

        # A. 空間対称型 (Symmetric Pattern)
        # 内部領域が明確な対称形状をなし、かつ内部に十分なオブジェクトがある
        if symmetry_score > 0.80 and inner_non_bg > 0.15:
            return {
                "style": "SYMMETRIC_PATTERN",
                "description": "Board exhibits high spatial symmetry. Geometric alignment game.",
                "features": features,
                "recommended_approach": (
                    "Preserve or manipulate spatial balance across symmetry axes."
                ),
                "recommended_domain": "symmetry_pattern",
            }

        # B. 画面外探索・オープン型 (Open Exploration)
        # 外周が開いており、閉鎖壁がない
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

        # C. 危険回避・致死トラップ帯型 (Hazard Avoidance)
        # 壁(高密度)以外に、帯状のトラップ障害物(3〜8マス)が存在する
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

        # D. アイテム収集・トリガー型パズル (Item Trigger Puzzle)
        # プレイヤー(1マス)とゴール(1マス)に加え、第3以上の孤立アイテム(鍵/スイッチ)が存在する
        if len(isolated_items) >= 3:
            return {
                "style": "ITEM_TRIGGER_PUZZLE",
                "description": (
                    "Multiple isolated colored objects detected. Sequential trigger or key-lock"
                    " mechanics active."
                ),
                "features": features,
                "recommended_approach": (
                    "INTERACTION FIRST: Route to isolated item entities to alter game state before"
                    " exit."
                ),
                "recommended_domain": "inventory_puzzle",
            }

        # E. 閉鎖型迷路・迂回ナビゲーション (Closed Maze)
        # 外周が塞がれており、内部の壁密度が高い
        if obstacle_density >= 0.15 and not has_edge_exit:
            return {
                "style": "CLOSED_MAZE",
                "description": (
                    "Enclosed boundary with internal wall corridors. Traditional shortest path or"
                    " obstacle avoidance."
                ),
                "features": features,
                "recommended_approach": (
                    "PATHFINDING FIRST: Execute deterministic detour search around internal"
                    " barriers."
                ),
                "recommended_domain": "navigation",
            }

        # F. デフォルト (General Navigation)
        return {
            "style": "GENERAL_GRID_GAME",
            "description": "Standard discrete grid environment without extreme structural bias.",
            "features": features,
            "recommended_approach": (
                "BALANCED: Explore forward while avoiding obstacles and observing reward signals."
            ),
            "recommended_domain": "general",
        }
