"""Google ADK 2.0 準拠 Subgoal Decomposer 実行ツール (Level 3 Tools).

Plan Agent が逆算プランニング、前提条件依存解決（鍵→扉）、および
一時待避バッファ（Staging Buffer）配分を実行するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_SUBGOAL_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "subgoal-decomposer" / "scripts"
if str(_SUBGOAL_DIR) not in sys.path and _SUBGOAL_DIR.exists():
    sys.path.insert(0, str(_SUBGOAL_DIR))

try:
    from subgoal_decomposer import SubgoalDecomposerCore
except ImportError:
    SubgoalDecomposerCore = None

logger = logging.getLogger(__name__)


class SubgoalTools:
    """Subgoal Decomposer 向け Level 3 実行ツールセット."""

    def __init__(self, memory_tools: Optional[Any] = None):
        self.core = SubgoalDecomposerCore() if SubgoalDecomposerCore else None
        self.memory_tools = memory_tools

    def decompose_hierarchical_subgoals(
        self,
        entities: List[Dict[str, Any]],
        target_pattern: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """検出されたエンティティと目標パターンから、前提条件・待避バッファを考慮した順序付きサブゴールを生成します。

        Args:
            entities: 検出されたオブジェクト群（type, x, y, id など）
            target_pattern: 目標とする配置や順序パターン
        """
        if self.core is None:
            return json.dumps({"status": "error", "message": "SubgoalDecomposerCore not available."})

        res = self.core.decompose_hierarchical_subgoals(entities, target_pattern=target_pattern)

        # 記憶ノートブック (memory-notebook) にサブゴール一覧を保存
        if res.get("status") == "ok" and self.memory_tools:
            try:
                subgoals = res.get("subgoals", [])
                subgoal_summary = "\n".join([f"{s['id']}: [{s['type']}] {s['title']}" for s in subgoals])
                self.memory_tools.memory_write(
                    section_id="plan.subgoals",
                    title="Hierarchical Subgoals Sequence",
                    content=subgoal_summary,
                    summary=f"Generated {len(subgoals)} hierarchical subgoals",
                    tags="plan,subgoals,hierarchy",
                )
            except Exception as e:
                logger.warning("Failed to sync subgoals to memory notebook: %s", e)

        return json.dumps(res, ensure_ascii=False)

    def get_active_subgoal(self) -> str:
        """現在実行すべきアクティブなサブゴール情報を取得します。"""
        if self.core is None:
            return json.dumps(None)
        subgoal = self.core.get_active_subgoal()
        return json.dumps(subgoal, ensure_ascii=False)

    def advance_subgoal(self, subgoal_id: Optional[str] = None) -> str:
        """完了したサブゴールを進め、次のサブゴールに移行します。"""
        if self.core is None:
            return json.dumps({"status": "error", "message": "SubgoalDecomposerCore not available."})
        res = self.core.advance_subgoal(subgoal_id)
        return json.dumps(res, ensure_ascii=False)

    def reset_episode(self) -> None:
        """エピソードリセット時の処理."""
        if self.core:
            self.core.reset()

    def get_tools(self) -> List[Any]:
        """ADK Agent にバインドする関数ツール一覧を返却."""
        return [
            self.decompose_hierarchical_subgoals,
            self.get_active_subgoal,
            self.advance_subgoal,
        ]
