"""Google ADK 2.0 準拠 Macro Skill Compiler 実行ツール (Level 3 Tools).

Plan Agent および Act Agent が定石アクションパターンのマクロ化、
インスタンス化、および LLM をバイパスした高速実行を行うためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_MACRO_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "macro-skill-compiler" / "scripts"
if str(_MACRO_DIR) not in sys.path and _MACRO_DIR.exists():
    sys.path.insert(0, str(_MACRO_DIR))

try:
    from macro_skill_compiler import MacroSkillCompilerCore
except ImportError:
    MacroSkillCompilerCore = None

logger = logging.getLogger(__name__)


class MacroTools:
    """Macro Skill Compiler 向け Level 3 実行ツールセット."""

    def __init__(self):
        self.core = MacroSkillCompilerCore() if MacroSkillCompilerCore else None

    def compile_macro_sequence(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
    ) -> str:
        """新しいマクロアクション手順を登録・コンパイルします。

        Args:
            name: マクロ名（例: 'PUSH_AND_STAGE'）
            description: 手順の目的・説明
            steps: 実行するステップ辞書のリスト
        """
        if self.core is None:
            return json.dumps({"status": "error", "message": "MacroSkillCompilerCore not available."})
        res = self.core.register_macro(name, description, steps)
        return json.dumps(res, ensure_ascii=False)

    def instantiate_macro(
        self,
        macro_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """登録済みマクロにパラメータをバインドし、実行キューに読み込みます。"""
        if self.core is None:
            return json.dumps({"status": "error", "message": "MacroSkillCompilerCore not available."})
        res = self.core.instantiate_macro(macro_name, parameters)
        return json.dumps(res, ensure_ascii=False)

    def execute_macro_step(self) -> str:
        """アクティブなマクロキューから次のステップを取り出して返します。"""
        if self.core is None:
            return json.dumps(None)
        step = self.core.pop_next_step()
        return json.dumps(step, ensure_ascii=False)

    def abort_macro(self, reason: str = "Interrupted") -> str:
        """実行中のマクロを即座に破棄・中断します（0変化衝突時など）。"""
        if self.core is None:
            return json.dumps({"status": "error", "message": "MacroSkillCompilerCore not available."})
        res = self.core.abort_macro(reason)
        return json.dumps(res, ensure_ascii=False)

    def list_available_macros(self) -> str:
        """現在利用可能なマクロ手順一覧を返します。"""
        if self.core is None:
            return json.dumps([])
        macros = self.core.list_available_macros()
        return json.dumps(macros, ensure_ascii=False)

    def has_active_macro(self) -> bool:
        """実行待機中のマクロステップが存在するかどうか判定します。"""
        if self.core is None:
            return False
        return len(self.core.active_queue) > 0

    def get_tools(self) -> List[Any]:
        """ADK Agent にバインドする関数ツール一覧を返却."""
        return [
            self.compile_macro_sequence,
            self.instantiate_macro,
            self.execute_macro_step,
            self.list_available_macros,
        ]
