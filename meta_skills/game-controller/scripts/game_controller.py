#!/usr/bin/env python3
"""Game Controller Meta-Skill Implementation for ARC-AGI-3.

LLM からのゲーム操作要求（移動、座標クリック、リセット）を検証・正規化し、
ARC-AGI-3 ゲーム環境が受け付ける厳密なアクション決定 (ActionDecision) を生成します。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class GameController:
    """ARC-AGI-3 ゲーム操作プロトコル検証・実行エンジン."""

    ACTION_MAP: Dict[str, int] = {
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

    def __init__(self, available_actions: Optional[List[int]] = None) -> None:
        self.available_actions = available_actions or [1, 2, 3, 4]

    def set_available_actions(self, available_actions: List[int]) -> None:
        self.available_actions = available_actions

    def parse_and_validate(
        self,
        raw_output: Any,
        available_actions: Optional[List[int]] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """LLM の出力（JSON, ツール引数, またはテキスト）を解析し検証済みアクションを返却.

        Args:
            raw_output: LLM からの出力テキストまたは辞書
            available_actions: 現フレームで利用可能なアクション ID リスト
            grid_shape: 盤面サイズ (H, W)。クリック座標の範囲検査に使用

        Returns:
            検証結果辞書:
                success: bool
                action_type: str ("STEP", "CLICK", "RESET")
                action_name: str ("UP", "ACTION1", "CLICK", "RESET")
                action_id: int (0〜7)
                coordinates: Optional[dict] ({"x": col, "y": row})
                reasoning: str
                error: Optional[str]
        """
        avail = available_actions or self.available_actions
        h, w = grid_shape if grid_shape else (30, 30)

        # 1. 辞書形式または JSON オブジェクトの解析
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
            return self._validate_structured_action(parsed_dict, avail, h, w)

        # 2. テキスト形式の解析
        if isinstance(raw_output, str):
            return self._validate_text_action(raw_output, avail, h, w)

        return {
            "success": False,
            "action_type": "UNKNOWN",
            "action_name": "NONE",
            "action_id": -1,
            "coordinates": None,
            "reasoning": "",
            "error": f"Unsupported input type: {type(raw_output)}",
        }

    def _validate_structured_action(
        self, data: Dict[str, Any], avail: List[int], h: int, w: int
    ) -> Dict[str, Any]:
        """構造化データ (辞書) の検証."""
        # action, action_name, step_action 等の候補からアクションを抽出
        act_candidate = data.get("action_name", None) or data.get("direction", None) or data.get("action", None) or data.get("step", None)
        if isinstance(act_candidate, str) and act_candidate.lower() in ("step_action", "action", "step"):
            # action キーが "step_action" の場合、サブキー (direction, action_name, name, value) を確認
            act_candidate = data.get("direction", None) or data.get("action_name", None) or data.get("name", None) or data.get("value", None) or act_candidate

        act = str(act_candidate or "").upper()
        reasoning = str(data.get("reasoning", "") or "Structured action call")

        # リセット要求
        if act in ("RESET", "RESET_GAME"):
            return {
                "success": True,
                "action_type": "RESET",
                "action_name": "RESET",
                "action_id": 0,
                "coordinates": None,
                "reasoning": reasoning,
                "error": None,
            }

        # クリック要求
        if act in ("CLICK", "CLICK_AT", "ACTION6"):
            x = data.get("x", data.get("col", None))
            y = data.get("y", data.get("row", None))
            if x is None or y is None:
                return {
                    "success": False,
                    "action_type": "CLICK",
                    "action_name": "ACTION6",
                    "action_id": 6,
                    "coordinates": None,
                    "reasoning": reasoning,
                    "error": "Click action requires 'x' (col) and 'y' (row) coordinates",
                }
            try:
                x_val, y_val = int(x), int(y)
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "action_type": "CLICK",
                    "action_name": "ACTION6",
                    "action_id": 6,
                    "coordinates": None,
                    "reasoning": reasoning,
                    "error": f"Coordinates must be integers, got x={x}, y={y}",
                }

            if not (0 <= x_val < w and 0 <= y_val < h):
                return {
                    "success": False,
                    "action_type": "CLICK",
                    "action_name": "ACTION6",
                    "action_id": 6,
                    "coordinates": {"x": x_val, "y": y_val},
                    "reasoning": reasoning,
                    "error": f"Coordinates ({x_val}, {y_val}) out of grid bounds (width={w}, height={h})",
                }

            if 6 not in avail:
                return {
                    "success": False,
                    "action_type": "CLICK",
                    "action_name": "ACTION6",
                    "action_id": 6,
                    "coordinates": {"x": x_val, "y": y_val},
                    "reasoning": reasoning,
                    "error": "Click action (ACTION6) is not in available actions",
                }

            return {
                "success": True,
                "action_type": "CLICK",
                "action_name": "ACTION6",
                "action_id": 6,
                "coordinates": {"x": x_val, "y": y_val},
                "reasoning": reasoning,
                "error": None,
            }

        # 通常ステップ要求
        act_id = self.ACTION_MAP.get(act, None)
        if act_id is None:
            sub_act = str(data.get("action_id", "") or data.get("step", "")).upper()
            act_id = self.ACTION_MAP.get(sub_act, None)

        if act_id is None:
            return {
                "success": False,
                "action_type": "STEP",
                "action_name": act,
                "action_id": -1,
                "coordinates": None,
                "reasoning": reasoning,
                "error": f"Unknown action '{act}'",
            }

        if act_id not in avail and act_id != 0:
            return {
                "success": False,
                "action_type": "STEP",
                "action_name": self.ACTION_NAMES.get(act_id, act),
                "action_id": act_id,
                "coordinates": None,
                "reasoning": reasoning,
                "error": f"Action '{act}' (ID={act_id}) is not in available actions {avail}",
            }

        return {
            "success": True,
            "action_type": "STEP" if act_id != 0 else "RESET",
            "action_name": self.ACTION_NAMES.get(act_id, act),
            "action_id": act_id,
            "coordinates": None,
            "reasoning": reasoning,
            "error": None,
        }

    def _validate_text_action(
        self, text: str, avail: List[int], h: int, w: int
    ) -> Dict[str, Any]:
        """フリーテキストからのアクション抽出と検証."""
        text_upper = text.upper()

        click_match = re.search(r"CLICK.*?(\d+)\s*[,xX\s]\s*(\d+)", text, re.IGNORECASE)
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            if 0 <= x < w and 0 <= y < h:
                if 6 in avail:
                    return {
                        "success": True,
                        "action_type": "CLICK",
                        "action_name": "ACTION6",
                        "action_id": 6,
                        "coordinates": {"x": x, "y": y},
                        "reasoning": f"Extracted click at ({x}, {y})",
                        "error": None,
                    }
                else:
                    return {
                        "success": False,
                        "action_type": "CLICK",
                        "action_name": "ACTION6",
                        "action_id": 6,
                        "coordinates": {"x": x, "y": y},
                        "reasoning": f"Extracted click at ({x}, {y})",
                        "error": "Click action (ACTION6) is not in available actions",
                    }
            else:
                return {
                    "success": False,
                    "action_type": "CLICK",
                    "action_name": "ACTION6",
                    "action_id": 6,
                    "coordinates": {"x": x, "y": y},
                    "reasoning": f"Extracted click at ({x}, {y})",
                    "error": f"Coordinates ({x}, {y}) out of bounds (w={w}, h={h})",
                }

        # アクション名の走査
        matched_unavailable = None
        for name, aid in self.ACTION_MAP.items():
            if re.search(rf"\b{name}\b", text_upper):
                if aid in avail or aid == 0:
                    return {
                        "success": True,
                        "action_type": "STEP" if aid != 0 else "RESET",
                        "action_name": self.ACTION_NAMES.get(aid, name),
                        "action_id": aid,
                        "coordinates": None,
                        "reasoning": f"Extracted {name} from text",
                        "error": None,
                    }
                else:
                    if matched_unavailable is None:
                        matched_unavailable = (name, aid)

        if matched_unavailable:
            name, aid = matched_unavailable
            return {
                "success": False,
                "action_type": "STEP",
                "action_name": self.ACTION_NAMES.get(aid, name),
                "action_id": aid,
                "coordinates": None,
                "reasoning": f"Extracted {name} from text",
                "error": f"Action '{name}' (ID={aid}) is not in available actions {avail}",
            }

        return {
            "success": False,
            "action_type": "UNKNOWN",
            "action_name": "NONE",
            "action_id": -1,
            "coordinates": None,
            "reasoning": "",
            "error": "No valid action pattern found in text",
        }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ARC-AGI-3 Game Controller Validator CLI")
    parser.add_argument("--action", type=str, required=True, help="Raw action string or JSON")
    parser.add_argument("--available", type=int, nargs="*", default=[1, 2, 3, 4], help="Available action IDs")
    parser.add_argument("--width", type=int, default=30, help="Grid width")
    parser.add_argument("--height", type=int, default=30, help="Grid height")
    args = parser.parse_args()

    controller = GameController(available_actions=args.available)
    res = controller.parse_and_validate(
        args.action,
        available_actions=args.available,
        grid_shape=(args.height, args.width),
    )
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["success"] else 1)
