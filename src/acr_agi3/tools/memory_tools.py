"""Google ADK 2.0 準拠 Memory Notebook 実行ツール (Level 3 Tools).

Plan Agent が長期記憶ノート（紙・ペン・消しゴム・しおり）を能動的に操作するためのツール群を提供します。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# meta_skills/memory-notebook または acr_agi3/meta から MemoryNotebook をインポート
try:
    from acr_agi3.meta.memory_notebook import MemoryNotebook
except ImportError:
    _SKILL_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "memory-notebook" / "scripts"
    if str(_SKILL_DIR) not in sys.path:
        sys.path.insert(0, str(_SKILL_DIR))
    from memory_notebook import MemoryNotebook

logger = logging.getLogger(__name__)


class MemoryTools:
    """Plan Agent 向け Level 3 実行ツールセット."""

    def __init__(self, notebook_path: Optional[Path] = None) -> None:
        self.notebook = MemoryNotebook(notebook_path=notebook_path, auto_save=True)
        self.step_index: int = 0

    def set_step(self, step_index: int) -> None:
        """現在のステップ番号を更新."""
        self.step_index = step_index

    def memory_write(
        self,
        section_id: str,
        content: str,
        title: str = "",
        summary: str = "",
        tags: str = "",
    ) -> str:
        """記憶ノートに新しいセクション（仮説、ルール、障害物の位置など）を記録または追記します（ペン）。

        Args:
            section_id: セクションの一意識別子（例: 'rules.movement', 'hypotheses.switches'）。
            content: 記録する本文。
            title: 表示用タイトル。
            summary: 1〜2文の要約（目次検索用）。
            tags: カンマ区切りの分類タグ（例: 'rule,controls'）。
        """
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            node = self.notebook.write(
                section_id=section_id,
                content=content,
                title=title or None,
                summary=summary or None,
                tags=tag_list,
                step=self.step_index,
            )
            return json.dumps(
                {"status": "ok", "message": f"Successfully wrote section '{node.id}'", "title": node.title},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def memory_read(self, section_id: str) -> str:
        """特定のセクション本文を読み出します（選択的開示）。

        Args:
            section_id: 読み出したいセクションIDまたはしおり名。
        """
        try:
            node = self.notebook.read(section_id)
            sec_id = node.get("id") if isinstance(node, dict) else getattr(node, "id", str(section_id))
            title = node.get("title") if isinstance(node, dict) else getattr(node, "title", sec_id)
            content = node.get("content") if isinstance(node, dict) else getattr(node, "content", "")
            summary = node.get("summary") if isinstance(node, dict) else getattr(node, "summary", "")
            tags = node.get("tags") if isinstance(node, dict) else getattr(node, "tags", [])
            return json.dumps(
                {
                    "status": "ok",
                    "section_id": sec_id,
                    "title": title,
                    "content": content,
                    "summary": summary,
                    "tags": tags,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def memory_toc(self, tag: str = "", as_markdown: bool = False) -> str:
        """記憶ノートの目次（TOC）を取得し、どのような情報が保存されているか一覧します（しおり）。

        Args:
            tag: タグでフィルタリングする場合のタグ名（省略時は全件）。
            as_markdown: True の場合、Markdown 形式の目次文字列を返します。
        """
        try:
            toc = self.notebook.get_toc(tag=tag if tag else None, as_markdown=as_markdown)
            if as_markdown:
                return str(toc)
            return json.dumps({"status": "ok", "entries": toc}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def clear(self, tag: Optional[str] = None) -> int:
        """記憶ノートを全消去または指定タグを一括削除します（消しゴム）."""
        return self.notebook.clear(tag=tag)

    def memory_search(self, query: str) -> str:
        """記憶ノート内をキーワード検索します。

        Args:
            query: 検索キーワード。
        """
        try:
            results = self.notebook.search(query)
            return json.dumps({"status": "ok", "matches": results}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def memory_delete(self, section_id: str) -> str:
        """誤った仮説や完了・破棄された古い計画を記憶ノートから削除します（消しゴム）。

        Args:
            section_id: 削除したいセクションIDまたはしおり名。
        """
        try:
            self.notebook.delete(section_id)
            return json.dumps({"status": "ok", "message": f"Successfully deleted section '{section_id}'"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def get_tools(self) -> List[Any]:
        """ADK Agent に渡すための関数ツール一覧を返却."""
        return [self.memory_write, self.memory_read, self.memory_toc, self.memory_search, self.memory_delete]
