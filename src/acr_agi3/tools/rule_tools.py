"""Google ADK 2.0 準拠 Rule Inducer 実行ツール (Level 3 Tools).

Plan Agent および Act Agent が状態遷移から因果ルールおよび
不変な勝利条件（Win Condition）を帰納・蓄積するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_RULE_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "rule-inducer" / "scripts"
if str(_RULE_DIR) not in sys.path and _RULE_DIR.exists():
    sys.path.insert(0, str(_RULE_DIR))

try:
    from rule_inducer import RuleInducerCore
except ImportError:
    RuleInducerCore = None

logger = logging.getLogger(__name__)


class RuleTools:
    """Rule Inducer 向け Level 3 実行ツールセット."""

    def __init__(self, memory_tools: Optional[Any] = None):
        self.core = RuleInducerCore() if RuleInducerCore else None
        self.memory_tools = memory_tools

    def induce_rule_from_transition(
        self,
        action_name: str,
        action_id: int,
        pixels_changed: int,
        coords: Optional[Dict[str, int]] = None,
        level_before: int = 0,
        level_after: int = 0,
        notes: str = "",
    ) -> str:
        """状態遷移の結果を評価し、因果ルールや勝利条件を自動帰納します。

        Args:
            action_name: 実行した行動名
            action_id: 行動の数値ID
            pixels_changed: 変化したピクセル数
            coords: 対象座標
            level_before: 行動前のレベル
            level_after: 行動後のレベル
            notes: 補足メモ
        """
        if self.core is None:
            return json.dumps({"status": "error", "message": "RuleInducerCore not available."})

        res = self.core.induce_rule_from_transition(
            step_index=0,
            action_name=action_name,
            action_id=action_id,
            pixels_changed=pixels_changed,
            coords=coords,
            level_before=level_before,
            level_after=level_after,
            notes=notes,
        )

        # 勝利条件または新ルール発見時は Memory Notebook の rules.* に同期永続化
        if res.get("status") == "ok" and res.get("is_new") and self.memory_tools:
            try:
                rule = res.get("rule", {})
                rule_id = rule.get("rule_id", "rule_auto")
                rule_type = rule.get("rule_type", "causal_rule")
                content = rule.get("statement") or rule.get("effect", "")
                self.memory_tools.memory_write(
                    section_id=f"rules.{rule_id}",
                    title=f"Induced Rule: {rule_id}",
                    content=f"Type: {rule_type}\n{content}",
                    summary=content[:100],
                    tags="rules,causality,invariant",
                )
            except Exception as e:
                logger.warning("Failed to sync induced rule to memory notebook: %s", e)

        return json.dumps(res, ensure_ascii=False)

    def get_known_rules(self, rule_type: Optional[str] = None) -> str:
        """帰納された既知のゲームルール一覧を取得します。"""
        if self.core is None:
            return json.dumps([])
        rules = self.core.get_known_rules(rule_type=rule_type)
        return json.dumps(rules, ensure_ascii=False)

    def get_win_condition_hypotheses(self) -> str:
        """発見された勝利条件の仮説文一覧を取得します。"""
        if self.core is None:
            return json.dumps([])
        win_conditions = self.core.get_win_condition_hypotheses()
        return json.dumps(win_conditions, ensure_ascii=False)

    def reset_episode(self) -> None:
        """エピソードリセット時の処理."""
        if self.core:
            self.core.reset_episode()

    def get_tools(self) -> List[Any]:
        """ADK Agent にバインドする関数ツール一覧を返却."""
        return [
            self.induce_rule_from_transition,
            self.get_known_rules,
            self.get_win_condition_hypotheses,
        ]
