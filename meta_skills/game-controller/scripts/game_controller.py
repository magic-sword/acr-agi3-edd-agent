#!/usr/bin/env python3
"""Game Controller Meta-Skill Implementation for ARC-AGI-3.

LLM からのゲーム操作要求（移動、座標クリック、リセット）を検証・正規化し、
ARC-AGI-3 ゲーム環境が受け付ける厳密なアクション決定 (ActionDecision) を生成します。

同定された操作力学（Invariant Action Map: UP/DOWN/LEFT/RIGHT）を動的に解決し、
幾何アフォーダンス解析に基づきクリック座標を自動吸着する決定論的実行スキルです。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from acr_agi3.harness.vision_observation import detect_interactable_objects
except ImportError:
    detect_interactable_objects = None

from pathlib import Path
import sys

_SG_DIR = Path(__file__).resolve().parents[2] / "spatial-grounder" / "scripts"
if str(_SG_DIR) not in sys.path and _SG_DIR.exists():
    sys.path.insert(0, str(_SG_DIR))

try:
    from spatial_grounder import SpatialGrounder
except ImportError:
    SpatialGrounder = None


class GameController:
    """ARC-AGI-3 ゲーム操作プロトコル検証・決定論的実行エンジン."""

    # デフォルトの方向マッピング（力学未確定時の初期値）
    DEFAULT_ACTION_MAP: Dict[str, int] = {
        "RESET": 0,
        "ACTION1": 1,
        "UP": 1,
        "ACTION2": 2,
        "DOWN": 2,
        "ACTION3": 3,
        "LEFT": 3,
        "ACTION4": 4,
        "RIGHT": 4,
        "ACTION5": 5,
        "ACTION6": 6,
        "CLICK": 6,
        "CLICK_AT": 6,
        "ACTION7": 7,
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "RESET",
        1: "ACTION1",
        2: "ACTION2",
        3: "ACTION3",
        4: "ACTION4",
        5: "ACTION5",
        6: "ACTION6",
        7: "ACTION7",
    }

    def __init__(
        self,
        available_actions: Optional[List[int]] = None,
        dynamics_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.available_actions: List[int] = available_actions or [1, 2, 3, 4]
        self.dynamics_map: Dict[str, int] = dict(dynamics_map or {})

    def set_available_actions(self, available_actions: List[int]) -> None:
        """現在のターンで利用可能なアクション ID を更新."""
        self.available_actions = available_actions

    def set_dynamics_map(self, dynamics_map: Dict[str, int]) -> None:
        """同定された操作力学マップ (例: {'UP': 3, 'DOWN': 4, ...}) を注入."""
        self.dynamics_map = dict(dynamics_map or {})

    def resolve_action_id(self, action_name: str) -> Tuple[int, str, bool]:
        """アクション名（UP, DOWN, ACTION1 等）を環境のアクション ID へ解決.

        Returns:
            (action_id, canonical_name, is_mapped_via_dynamics)
        """
        act_clean = action_name.strip().upper()

        # 1. 同定済み操作力学マップ (dynamics_map) の参照
        if act_clean in self.dynamics_map:
            mapped_id = self.dynamics_map[act_clean]
            if mapped_id in self.available_actions or mapped_id == 0:
                canonical = self.ACTION_NAMES.get(mapped_id, f"ACTION{mapped_id}")
                return mapped_id, canonical, True

        # 2. デフォルトアクションマップの参照
        if act_clean in self.DEFAULT_ACTION_MAP:
            act_id = self.DEFAULT_ACTION_MAP[act_clean]
            canonical = self.ACTION_NAMES.get(act_id, f"ACTION{act_id}")
            return act_id, canonical, False

        # 3. 数値直接指定 ("1", "2", 等)
        try:
            val = int(act_clean)
            if val in self.ACTION_NAMES:
                return val, self.ACTION_NAMES[val], False
        except ValueError:
            pass

        # 4. 未知のアクション名の場合、-1 を返却
        return -1, f"UNKNOWN({act_clean})", False

    @staticmethod
    def snap_coordinates_to_affordance(
        x: Optional[int],
        y: Optional[int],
        h: int,
        w: int,
        grid: Optional[Any] = None,
        object_id: Optional[int] = None,
    ) -> Tuple[int, int, str]:
        """指定座標 (x, y) または物体ID (object_id) からクリック座標を決定.

        仕様:
        1. object_id が指定された場合:
           検出された該当物体の重心へ正確に自動スナップする。
        2. 明示的に x, y 座標が両方指定された場合 (object_id is None):
           背景色（地面・空きマス）かどうかにかかわらず、指定された生の座標をそのまま尊重する。
           （近隣物体への強制吸着は行わず、地面への移動や任意クリックを保証する）
        3. x, y が未指定 (None) の場合:
           最優先の検出オブジェクト（または盤面中央）へ自動スナップする。
        """
        if grid is None:
            safe_x = x if x is not None else w // 2
            safe_y = y if y is not None else h // 2
            return safe_x, safe_y, ""

        try:
            import numpy as np
            arr = np.array(grid, dtype=int)
            if arr.ndim == 3:
                arr = arr[-1]
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape != (h, w):
                h, w = arr.shape
        except Exception:
            safe_x = x if x is not None else w // 2
            safe_y = y if y is not None else h // 2
            return safe_x, safe_y, ""

        # --- Case 1: 物体 ID (object_id) が指定された場合 ---
        if object_id is not None:
            objs = []
            if SpatialGrounder is not None:
                objs = SpatialGrounder.detect_composite_objects(arr)
            if not objs and detect_interactable_objects is not None:
                objs = detect_interactable_objects(arr)

            target_obj = next((o for o in objs if o.get("id") == object_id), None)
            if target_obj is None and objs:
                # 0-indexed インデックス指定（例: object_id=0 -> 先頭オブジェクト）
                if 0 <= object_id < len(objs):
                    target_obj = objs[object_id]
                # 1-indexed インデックス指定（例: object_id=1 -> 先頭オブジェクト）
                elif 0 <= object_id - 1 < len(objs):
                    target_obj = objs[object_id - 1]

            if target_obj is not None:
                cx = int(target_obj["center"]["x"])
                cy = int(target_obj["center"]["y"])
                obj_type = target_obj.get("type", "OBJECT")
                return cx, cy, f" [auto-snapped to object #{object_id} ({obj_type}) at ({cx}, {cy})]"

            # 指定IDが見つからなかった場合、明示的座標があればそれを使用、なければ最優先物体
            if x is not None and y is not None:
                return x, y, f" [object #{object_id} not found, using specified coords ({x}, {y})]"
            if objs:
                best = objs[0]
                bx, by = int(best["center"]["x"]), int(best["center"]["y"])
                return bx, by, f" [object #{object_id} not found, snapped to object #{best.get('id', 0)} at ({bx}, {by})]"
            return w // 2, h // 2, f" [object #{object_id} not found, defaulted to grid center]"

        # --- Case 2: 明示的に座標 (x, y) が両方指定された場合 ---
        # ユーザー指示: エージェントが明示的に座標を指定した場合はその場所（地面・空セル含む）をクリック可能にする
        if x is not None and y is not None:
            return x, y, ""

        # --- Case 3: 座標が未指定 (None) の場合 ---
        # 最優先の検出オブジェクトへ自動スナップ
        objs = []
        if SpatialGrounder is not None:
            objs = SpatialGrounder.detect_composite_objects(arr)
        if not objs and detect_interactable_objects is not None:
            objs = detect_interactable_objects(arr)

        if objs:
            best = objs[0]
            bx, by = int(best["center"]["x"]), int(best["center"]["y"])
            best_id = best.get("id", 0)
            return bx, by, f" [auto-snapped unspecified click to detected object #{best_id} at ({bx}, {by})]"

        # オブジェクトが見つからない場合はグリッド中心
        return w // 2, h // 2, " [defaulted to grid center]"

    # -------------------------------------------------------------------------
    # ADK 2.0 ツール関数群 (Tool Function Interface)
    # -------------------------------------------------------------------------

    def step_action(self, direction: str, reasoning: str = "") -> Dict[str, Any]:
        """方向キーまたはボタンアクションを環境の操作力学に基づいて実行.

        Args:
            direction: 移動方向 ("UP", "DOWN", "LEFT", "RIGHT") またはアクション名 ("ACTION1"〜"ACTION7")。
            reasoning: この行動を選択した戦略的理由。
        """
        act_id, act_name, is_mapped = self.resolve_action_id(direction)
        note = f" (via dynamic map: {direction}->{act_name})" if is_mapped else ""
        if act_id != 0 and act_id not in self.available_actions:
            return {
                "success": False,
                "action_type": "STEP",
                "action_name": act_name,
                "action_id": act_id,
                "coordinates": None,
                "reasoning": f"{reasoning}{note}".strip(),
                "error": f"Action '{direction}' (resolved to {act_name}/ID:{act_id}) is disabled and not in available actions: {self.available_actions}. Please choose from available actions.",
            }
        return {
            "success": True,
            "action_type": "STEP" if act_id != 0 else "RESET",
            "action_name": act_name,
            "action_id": act_id,
            "coordinates": None,
            "reasoning": f"{reasoning}{note}".strip(),
            "error": None,
        }

    def click_at(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        object_id: Optional[int] = None,
        grid: Optional[Any] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """盤面上の指定座標または特定の検出オブジェクトをクリック (ACTION6).

        Args:
            x: 列インデックス (Column, 0-indexed)。任意座標（地面・空セル含む）をクリックする場合に指定。
            y: 行インデックス (Row, 0-indexed)。任意座標（地面・空セル含む）をクリックする場合に指定。
            object_id: 検出されたオブジェクトのID (0, 1, ...)。特定物体をクリックする場合に指定（重心へ自動スナップ）。
            grid: 盤面グリッド配列。
            grid_shape: 盤面サイズ (h, w)。
            reasoning: このクリックを選択した理由。
        """
        h, w = grid_shape if grid_shape else (30, 30)
        if grid is not None and hasattr(grid, "shape"):
            h, w = grid.shape[:2]

        snap_x, snap_y, note = self.snap_coordinates_to_affordance(
            x=x, y=y, h=h, w=w, grid=grid, object_id=object_id
        )
        reasoning_full = f"{reasoning}{note}".strip()

        if not (0 <= snap_x < w and 0 <= snap_y < h):
            return {
                "success": False,
                "action_type": "CLICK",
                "action_name": "ACTION6",
                "action_id": 6,
                "coordinates": {"x": snap_x, "y": snap_y},
                "reasoning": reasoning_full,
                "error": f"Coordinates ({snap_x}, {snap_y}) out of grid bounds (w={w}, h={h})",
            }

        if 6 not in self.available_actions:
            return {
                "success": False,
                "action_type": "CLICK",
                "action_name": "ACTION6",
                "action_id": 6,
                "coordinates": {"x": snap_x, "y": snap_y},
                "reasoning": reasoning_full,
                "error": f"Click action (ACTION6 / click_at) is disabled and not in available actions: {self.available_actions}. Do NOT use click_at; choose an action from available actions.",
            }

        return {
            "success": True,
            "action_type": "CLICK",
            "action_name": "ACTION6",
            "action_id": 6,
            "coordinates": {"x": snap_x, "y": snap_y},
            "reasoning": reasoning_full,
            "error": None,
        }

    def reset_game(self, reasoning: str = "") -> Dict[str, Any]:
        """現在のステージを手詰まりから能動的にリセット (ACTION0)."""
        return {
            "success": True,
            "action_type": "RESET",
            "action_name": "RESET",
            "action_id": 0,
            "coordinates": None,
            "reasoning": reasoning or "Active reset triggered by agent",
            "error": None,
        }

    # -------------------------------------------------------------------------
    # 構造化・テキストパース＆バリデーション
    # -------------------------------------------------------------------------

    def parse_and_validate(
        self,
        raw_output: Any,
        available_actions: Optional[List[int]] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        grid: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """LLM の出力（JSON, ツール引数, またはテキスト）を解析し検証済みアクションを返却."""
        if available_actions is not None:
            self.set_available_actions(available_actions)
        h, w = grid_shape if grid_shape else (30, 30)
        if grid is not None and hasattr(grid, "shape"):
            h, w = grid.shape[:2]

        parsed_dict = None
        if isinstance(raw_output, dict):
            parsed_dict = raw_output
        elif isinstance(raw_output, str):
            json_match = re.search(r"\{[^{}]*\}", raw_output)
            if json_match:
                try:
                    parsed_dict = json.loads(json_match.group(0))
                except Exception:
                    pass

        if parsed_dict:
            return self._validate_structured_action(parsed_dict, h, w, grid=grid)

        if isinstance(raw_output, str):
            return self._validate_text_action(raw_output, h, w, grid=grid)

        return self.step_action("ACTION1", reasoning="Default fallback on unsupported type")

    def _validate_structured_action(
        self, data: Dict[str, Any], h: int, w: int, grid: Optional[Any] = None
    ) -> Dict[str, Any]:
        """構造化データ (辞書) の検証."""
        act_candidate = (
            data.get("action_name")
            or data.get("direction")
            or data.get("action")
            or data.get("step")
            or ""
        )
        if isinstance(act_candidate, str) and act_candidate.lower() in ("step_action", "action", "step"):
            act_candidate = (
                data.get("direction")
                or data.get("action_name")
                or data.get("name")
                or data.get("value")
                or act_candidate
            )

        act = str(act_candidate or "").upper()
        reasoning = str(data.get("reasoning", "") or "Structured action call")

        # リセット要求
        if act in ("RESET", "RESET_GAME"):
            return self.reset_game(reasoning=reasoning)

        # クリック要求判定:
        # 明示的に CLICK/ACTION6 が指定されているか、または act が明示的ステップでなく座標/物体指定情報がある場合
        is_explicit_click = act in ("CLICK", "CLICK_AT", "ACTION6")
        has_coords = "x" in data or "coordinates" in data or "y" in data or "object_id" in data or "target_id" in data
        is_explicit_step = False
        if act:
            step_id, _, _ = self.resolve_action_id(act)
            if step_id != -1 and step_id != 6:
                is_explicit_step = True

        if is_explicit_click or (not is_explicit_step and has_coords):
            coords = data.get("coordinates") if isinstance(data.get("coordinates"), dict) else None
            x = data.get("x", coords.get("x") if coords else None)
            y = data.get("y", coords.get("y") if coords else None)
            object_id = data.get("object_id", data.get("target_id"))
            if object_id is not None:
                try:
                    object_id = int(object_id)
                except (ValueError, TypeError):
                    object_id = None
            return self.click_at(
                x=x, y=y, object_id=object_id, grid=grid, grid_shape=(h, w), reasoning=reasoning
            )

        # 通常ステップ要求
        return self.step_action(direction=act, reasoning=reasoning)

    def _validate_text_action(
        self, text: str, h: int, w: int, grid: Optional[Any] = None
    ) -> Dict[str, Any]:
        """フリーテキストからのアクション抽出と検証."""
        text_upper = text.upper()

        if "RESET" in text_upper:
            return self.reset_game(reasoning="Extracted reset from text")

        click_match = re.search(r"CLICK.*?(\d+)\s*[,xX\s]\s*(\d+)", text, re.IGNORECASE)
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            return self.click_at(x=x, y=y, grid=grid, grid_shape=(h, w), reasoning="Extracted click from text")

        # 同定済み力学マップのキーワード走査
        for k in self.dynamics_map:
            if re.search(rf"\b{k}\b", text_upper):
                return self.step_action(direction=k, reasoning=f"Extracted {k} via dynamics map from text")

        # デフォルトアクションの走査
        for name in ["UP", "DOWN", "LEFT", "RIGHT", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]:
            if re.search(rf"\b{name}\b", text_upper):
                if name in ("ACTION6", "CLICK"):
                    return self.click_at(grid=grid, grid_shape=(h, w), reasoning="Extracted click from text")
                return self.step_action(direction=name, reasoning=f"Extracted {name} from text")

        # フォールバック (パース失敗時は action_id: -1)
        fb_id = self.available_actions[0] if self.available_actions else 1
        return {
            "success": False,
            "action_type": "STEP",
            "action_name": self.ACTION_NAMES.get(fb_id, f"ACTION{fb_id}"),
            "action_id": -1,
            "coordinates": None,
            "reasoning": "Fallback from unparseable text",
            "error": "No valid action pattern found in text",
        }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ARC-AGI-3 Game Controller Validator CLI")
    parser.add_argument("--action", type=str, required=True, help="Raw action string or JSON")
    parser.add_argument("--available", type=int, nargs="*", default=[1, 2, 3, 4], help="Available action IDs")
    parser.add_argument("--dynamics", type=str, default="{}", help="JSON string of dynamics map")
    parser.add_argument("--width", type=int, default=30, help="Grid width")
    parser.add_argument("--height", type=int, default=30, help="Grid height")
    args = parser.parse_args()

    dyn_map = json.loads(args.dynamics)
    controller = GameController(available_actions=args.available, dynamics_map=dyn_map)
    res = controller.parse_and_validate(
        args.action,
        available_actions=args.available,
        grid_shape=(args.height, args.width),
    )
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["success"] else 1)
