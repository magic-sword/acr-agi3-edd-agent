"""Google ADK 2.0 ワークフロー用データスキーマ (Plan-Act).

思考・計画フェーズ (Planner) から実行フェーズ (Act) に
渡される構造化データと安全なパーサー。
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
    subgoal: str = ""
    method_known: bool = True
    cognitive_intent: str = "DIRECT"  # PLAN | PROBE | SYNTHESIZE | MACRO_EXECUTE | DIRECT
    skill_synthesis_request: Optional[Dict[str, Any]] = None
    load_skill: Optional[str] = None
    load_skill_explicit: bool = False
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "hypothesis": self.hypothesis,
            "goal": self.goal,
            "subgoal": self.subgoal,
            "cognitive_intent": self.cognitive_intent,
            "method_known": self.method_known,
            "action": self.action,
            "reasoning": self.reasoning,
        }
        if self.coordinates:
            data["coordinates"] = self.coordinates
        if self.load_skill:
            data["load_skill"] = self.load_skill
        if self.skill_synthesis_request:
            data["skill_synthesis_request"] = self.skill_synthesis_request
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
                    if isinstance(data, dict) and ("action" in data or "hypothesis" in data or "subgoal" in data):
                        coords = None
                        if "coordinates" in data and isinstance(data["coordinates"], dict):
                            coords = {
                                "x": int(data["coordinates"].get("x", 0)),
                                "y": int(data["coordinates"].get("y", 0)),
                            }
                        elif "x" in data and "y" in data:
                            coords = {"x": int(data["x"]), "y": int(data["y"])}

                        has_skill_key = ("load_skill" in data) or ("skill_name" in data)
                        skill_val = data.get("load_skill") or data.get("skill_name")
                        if isinstance(skill_val, str) and skill_val.lower() in ("none", "null", ""):
                            skill_val = None

                        subgoal_val = str(data.get("subgoal") or data.get("current_subgoal") or data.get("goal") or "")
                        method_known = bool(data.get("method_known", True))
                        if "need_probe" in data and bool(data.get("need_probe")):
                            method_known = False

                        intent_val = str(data.get("cognitive_intent") or "").upper()
                        if not intent_val:
                            if not method_known or data.get("need_probe") or "probe" in data.get("action", "").lower():
                                intent_val = "PROBE"
                            elif "skill_synthesis" in data or "synthesize_skill" in data:
                                intent_val = "SYNTHESIZE"
                            elif skill_val and "navigation" in skill_val:
                                intent_val = "MACRO_EXECUTE"
                            else:
                                intent_val = "DIRECT"

                        synth_req = data.get("skill_synthesis") or data.get("synthesize_skill")
                        if not isinstance(synth_req, dict):
                            synth_req = None

                        return cls(
                            hypothesis=str(data.get("hypothesis", "")),
                            goal=str(data.get("goal", "")),
                            action=str(data.get("action", "")).strip(),
                            coordinates=coords,
                            reasoning=str(data.get("reasoning", "")),
                            subgoal=subgoal_val,
                            method_known=method_known,
                            cognitive_intent=intent_val,
                            skill_synthesis_request=synth_req,
                            load_skill=skill_val,
                            load_skill_explicit=has_skill_key,
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
