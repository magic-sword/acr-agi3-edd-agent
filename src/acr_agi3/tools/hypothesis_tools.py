"""Google ADK 2.0 準拠 Hypothesis Engine 実行ツール (Level 3 Tools).

Plan Agent および Act Agent が因果仮説の立案、検証判定、反証アーカイブ、
および禁忌のない代替仮説の取得を実行するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_HYPO_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "hypothesis-engine" / "scripts"
if str(_HYPO_DIR) not in sys.path and _HYPO_DIR.exists():
    sys.path.insert(0, str(_HYPO_DIR))

try:
    from hypothesis_engine import HypothesisEngineCore
except ImportError:
    HypothesisEngineCore = None

logger = logging.getLogger(__name__)


class HypothesisTools:
    """Hypothesis Engine 向け Level 3 実行ツールセット."""

    def __init__(self, memory_tools: Optional[Any] = None):
        self.engine = HypothesisEngineCore() if HypothesisEngineCore else None
        self.memory_tools = memory_tools
        self.step_index: int = 0
        self.available_actions: List[int] = []

    def set_context(self, step_index: int, available_actions: Optional[List[int]] = None) -> None:
        """各推論ステップ開始時のコンテキスト更新."""
        self.step_index = step_index
        if available_actions is not None:
            self.available_actions = available_actions

    def reset_episode(self) -> None:
        """エピソードリセット時の処理."""
        if self.engine:
            self.engine.reset_episode()

    def formulate_hypothesis(
        self,
        claim: str,
        action_name: str,
        action_id: int,
        coords: Optional[Dict[str, int]] = None,
        expected_effect: str = "",
        reasoning: str = "",
    ) -> str:
        """新しい因果仮説を立案し、有効であればアクティブ仮説として登録します。

        Args:
            claim: 検証したい因果仮説の主張文（例: 'Click at (0, 1) toggles switch'）
            action_name: 行動名（例: 'ACTION6', 'ACTION1'）
            action_id: 行動の数値ID
            coords: 座標辞書 {'x': int, 'y': int}（対象座標がある場合）
            expected_effect: 期待される変化の説明
            reasoning: 立案理由
        """
        if self.engine is None:
            return json.dumps({"status": "error", "message": "HypothesisEngineCore not available."})

        res = self.engine.formulate_hypothesis(
            step_index=self.step_index,
            claim=claim,
            action_name=action_name,
            action_id=action_id,
            coords=coords,
            expected_effect=expected_effect,
            reasoning=reasoning,
        )

        # Memory Notebook にもアクティブ仮説を同期保存
        if res.get("status") == "ok" and self.memory_tools:
            try:
                self.memory_tools.memory_write(
                    section_id="hypothesis.active",
                    title=f"Active Hypothesis at Step {self.step_index}",
                    content=f"Claim: {claim}\nAction: {action_name} (ID: {action_id}) Coords: {coords}\nExpected: {expected_effect}\nReasoning: {reasoning}",
                    summary=claim[:100],
                    tags="hypothesis,active,plan",
                )
            except Exception as e:
                logger.warning("Failed to sync active hypothesis to memory notebook: %s", e)

        return json.dumps(res, ensure_ascii=False)

    def evaluate_hypothesis(
        self,
        pixels_changed: int,
        is_effective: bool = True,
        actual_notes: str = "",
    ) -> str:
        """直前のアクション結果に基づいてアクティブ仮説の検証または反証を行います。

        Args:
            pixels_changed: 直前のアクションで変化したピクセル数
            is_effective: 行動が有効だったかどうか
            actual_notes: 実際の変化に関する補足メモ
        """
        if self.engine is None:
            return json.dumps({"status": "error", "message": "HypothesisEngineCore not available."})

        status, record = self.engine.evaluate_outcome(
            step_index=self.step_index,
            pixels_changed=pixels_changed,
            is_effective=is_effective,
            actual_notes=actual_notes,
        )

        # 反証された場合は Memory Notebook の hypothesis.refuted.* に退避保存
        if status == "REFUTED" and self.memory_tools:
            try:
                refuted_id = record.get("refuted_id", f"hypothesis.refuted.s{self.step_index}")
                act_name = record.get("action_name", "")
                coords = record.get("coords", {})
                coord_str = f"_{coords.get('x')}_{coords.get('y')}" if coords else ""
                refuted_content = (
                    f"Action: {act_name} Coords: {coords}\n"
                    f"Prior Claim: {record.get('original_claim', '')}\n"
                    f"Outcome: {pixels_changed} pixels changed (Action ineffective/collision).\n"
                    f"Lesson: {record.get('lesson', '')}"
                )
                self.memory_tools.memory_write(
                    section_id=refuted_id,
                    title=f"Refuted: {act_name}{coord_str} (0 pixels changed)",
                    content=refuted_content,
                    summary=f"{act_name}{coord_str} produced 0 change",
                    tags="hypothesis,refuted,falsified,constraint",
                )
                self.memory_tools.memory_delete("hypothesis.active")
            except Exception as e:
                logger.warning("Failed to sync refuted hypothesis to memory notebook: %s", e)

        return json.dumps({"status": status, "record": record}, ensure_ascii=False)

    def get_refuted_bookmarks(self) -> str:
        """反証済み仮説の 1 行しおり一覧（TOC）を取得します（極小トークン提示用）。"""
        if self.engine is None:
            return json.dumps([])
        bookmarks = self.engine.get_refuted_bookmarks()
        return json.dumps(bookmarks, ensure_ascii=False)

    def propose_alternative_hypothesis(self, candidate_coords: Optional[List[Tuple[int, int]]] = None) -> str:
        """過去に反証されていない有効な代替アクション・座標候補を提案します。"""
        if self.engine is None:
            return json.dumps({"status": "error", "message": "HypothesisEngineCore not available."})
        res = self.engine.propose_alternative_hypothesis(
            step_index=self.step_index,
            available_actions=self.available_actions,
            candidate_coords=candidate_coords,
        )
        return json.dumps({"status": "ok" if res else "exhausted", "proposal": res}, ensure_ascii=False)

    def is_action_refuted(self, action_id: int, coords: Optional[Dict[str, int]] = None) -> bool:
        """指定のアクションおよび座標がすでに反証済みか判定します。"""
        if self.engine is None:
            return False
        return self.engine.is_refuted(action_id, coords)

    def get_tools(self) -> List[Any]:
        """ADK Agent にバインドする関数ツール一覧を返却."""
        return [
            self.formulate_hypothesis,
            self.evaluate_hypothesis,
            self.get_refuted_bookmarks,
            self.propose_alternative_hypothesis,
        ]
