"""Google ADK 2.0 ワークフロー用データスキーマ (Plan-Review-Act).

思考・計画フェーズ (Planner) と レビューフェーズ (Reviewer) の間で
やり取りされる構造化データと安全なパーサー。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanProposal:
    """Planner エージェントが立案する行動計画."""

    hypothesis: str
    goal: str
    action: str
    coordinates: Optional[Dict[str, int]] = None
    reasoning: str = ""
    load_skill: Optional[str] = None
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "hypothesis": self.hypothesis,
            "goal": self.goal,
            "action": self.action,
            "reasoning": self.reasoning,
        }
        if self.coordinates:
            data["coordinates"] = self.coordinates
        if self.load_skill:
            data["load_skill"] = self.load_skill
        return data

    @classmethod
    def from_text(cls, text: str) -> PlanProposal:
        """LLM の出力テキストから JSON ブロックを抽出して PlanProposal を生成."""
        clean_text = text.strip()
        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        candidates = json_blocks if json_blocks else [clean_text]

        for cand in candidates:
            # { ... } の抽出
            match = re.search(r"\{[\s\S]*\}", cand)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, dict) and ("action" in data or "hypothesis" in data):
                        coords = None
                        if "coordinates" in data and isinstance(data["coordinates"], dict):
                            coords = {
                                "x": int(data["coordinates"].get("x", 0)),
                                "y": int(data["coordinates"].get("y", 0)),
                            }
                        elif "x" in data and "y" in data:
                            coords = {"x": int(data["x"]), "y": int(data["y"])}

                        return cls(
                            hypothesis=str(data.get("hypothesis", "")),
                            goal=str(data.get("goal", "")),
                            action=str(data.get("action", "")).strip(),
                            coordinates=coords,
                            reasoning=str(data.get("reasoning", "")),
                            load_skill=data.get("load_skill") or data.get("skill_name"),
                            raw_text=clean_text,
                        )
                except Exception:
                    continue

        # JSON がない場合のフォールバック解析: 自然言語テキストからアクションと座標を抽出
        upper_text = clean_text.upper()

        # 1. クリックパターンの検出 (例: click (5, 8) or click at x=5, y=8)
        coord_match = re.search(r"(?:CLICK|ACTION6)[^\d]*\(?(\d+)[,\s]+(\d+)\)?", clean_text, re.IGNORECASE)
        if not coord_match:
            coord_match = re.search(r"\((\d+)[,\s]+(\d+)\)[^\w]*(?:CLICK|SWITCH|TILE|BUTTON)", clean_text, re.IGNORECASE)
        if coord_match:
            cx, cy = int(coord_match.group(1)), int(coord_match.group(2))
            return cls(
                hypothesis="Detected interactive tile in observation",
                goal=f"Click target at ({cx}, {cy})",
                action="ACTION6",
                coordinates={"x": cx, "y": cy},
                reasoning=clean_text[:200],
                raw_text=clean_text,
            )

        # 2. 移動・基本アクションの検出 (ACTION1..7, UP, DOWN, LEFT, RIGHT, RESET)
        action_patterns = [
            (r"\b(?:ACTION1|UP)\b", "ACTION1"),
            (r"\b(?:ACTION2|DOWN)\b", "ACTION2"),
            (r"\b(?:ACTION3|LEFT)\b", "ACTION3"),
            (r"\b(?:ACTION4|RIGHT)\b", "ACTION4"),
            (r"\bACTION5\b", "ACTION5"),
            (r"\bACTION6\b", "ACTION6"),
            (r"\bACTION7\b", "ACTION7"),
            (r"\bRESET\b", "RESET"),
        ]
        for pat, act in action_patterns:
            if re.search(pat, upper_text):
                return cls(
                    hypothesis="Direct visual observation",
                    goal=f"Execute {act}",
                    action=act,
                    reasoning=clean_text[:200],
                    raw_text=clean_text,
                )

        return cls(
            hypothesis="Direct visual observation",
            goal="Explore environment",
            action="ACTION1",
            reasoning=clean_text[:200],
            raw_text=clean_text,
        )


@dataclass
class ReviewFeedback:
    """Reviewer エージェントによる計画レビュー結果."""

    status: str  # "APPROVED" or "REVISE"
    critique: str
    suggested_fix: Optional[str] = None
    refined_action: Optional[str] = None
    refined_coordinates: Optional[Dict[str, int]] = None
    raw_text: str = ""

    @property
    def is_approved(self) -> bool:
        return self.status.upper() == "APPROVED"

    @classmethod
    def from_text(cls, text: str) -> ReviewFeedback:
        """LLM の出力テキストから ReviewFeedback を抽出."""
        clean_text = text.strip()
        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        candidates = json_blocks if json_blocks else [clean_text]

        for cand in candidates:
            match = re.search(r"\{[\s\S]*\}", cand)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, dict) and "status" in data:
                        status = str(data.get("status", "APPROVED")).upper()
                        coords = None
                        if "refined_coordinates" in data and isinstance(data["refined_coordinates"], dict):
                            coords = {
                                "x": int(data["refined_coordinates"].get("x", 0)),
                                "y": int(data["refined_coordinates"].get("y", 0)),
                            }
                        return cls(
                            status="APPROVED" if "APPROV" in status else "REVISE",
                            critique=str(data.get("critique", "")),
                            suggested_fix=data.get("suggested_fix"),
                            refined_action=data.get("refined_action"),
                            refined_coordinates=coords,
                            raw_text=clean_text,
                        )
                except Exception:
                    continue

        # テキスト内のキーワード判定
        if "REVISE" in clean_text.upper() or "REJECT" in clean_text.upper():
            return cls(
                status="REVISE",
                critique=clean_text[:300],
                raw_text=clean_text,
            )
        return cls(
            status="APPROVED",
            critique="Plan logically sound and actionable.",
            raw_text=clean_text,
        )
