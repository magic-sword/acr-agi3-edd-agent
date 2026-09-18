"""Long-Term Memory Notebook for Autonomous Agents.

Provides hierarchical markdown memory management following the metaphor:
- Paper: Structured markdown storage with human-readable formatting and persistence.
- Pen: Writing, appending, and summarizing observations, rules, and hypotheses.
- Eraser: Deleting invalidated hypotheses, obsolete notes, and pruning outdated tags.
- Bookmark & TOC: Index-based selective progressive disclosure to avoid context pollution.

Zero external dependencies: 100% offline compatible (standard library only).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SectionNode:
    """単一の記憶セクションを表すノード."""

    id: str
    title: str
    content: str
    level: int = 2
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    bookmark: Optional[str] = None
    step: int = 0
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """辞書表現に変換."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionNode":
        """辞書から復元."""
        return cls(
            id=data["id"],
            title=data.get("title", data["id"]),
            content=data.get("content", ""),
            level=data.get("level", 2),
            summary=data.get("summary", ""),
            tags=list(data.get("tags", [])),
            bookmark=data.get("bookmark"),
            step=data.get("step", 0),
            updated_at=data.get(
                "updated_at", datetime.now(timezone.utc).isoformat()
            ),
        )


class MemoryNotebook:
    """エージェントの長期記憶ノートブック (紙・ペン・消しゴム・しおり)."""

    def __init__(
        self,
        notebook_path: Optional[Path | str] = None,
        auto_save: bool = True,
    ) -> None:
        self.notebook_path = Path(notebook_path) if notebook_path else None
        self.auto_save = auto_save
        self._sections: Dict[str, SectionNode] = {}
        self._bookmarks: Dict[str, str] = {}  # bookmark_name -> section_id

        if self.notebook_path and self.notebook_path.exists():
            self.load()

    # -------------------------------------------------------------------------
    # ペン (Pen): 書き込み・要約・追記
    # -------------------------------------------------------------------------
    def write(
        self,
        section_id: str,
        content: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        level: int = 2,
        tags: Optional[List[str]] = None,
        bookmark: Optional[str] = None,
        step: int = 0,
        append: bool = False,
    ) -> SectionNode:
        """ノートに情報を書き込む（ペン）.

        Args:
            section_id: セクションの一意識別子（スラッグ形式 e.g. 'rules.movement'）
            content: 本文内容
            title: 表示タイトル（未指定時は section_id をタイトル化）
            summary: 要約（未指定時は content の最初の1文から自動生成）
            level: 見出しレベル (1~4)
            tags: 分類タグリスト
            bookmark: しおり名（例: 'goal', 'key_rule'）
            step: 現在のステップ番号
            append: True の場合、既存の本文の後ろに追記
        """
        sec_id = self._normalize_id(section_id)
        if not sec_id:
            raise ValueError("section_id cannot be empty")

        clean_content = content.strip()
        final_title = (title or sec_id).strip()

        if append and sec_id in self._sections:
            existing = self._sections[sec_id]
            merged_content = f"{existing.content}\n\n{clean_content}".strip()
            final_summary = summary or self._extract_summary(merged_content)
            merged_tags = sorted(list(set(existing.tags + (tags or []))))
            final_bookmark = bookmark if bookmark is not None else existing.bookmark

            node = SectionNode(
                id=sec_id,
                title=final_title or existing.title,
                content=merged_content,
                level=level or existing.level,
                summary=final_summary,
                tags=merged_tags,
                bookmark=final_bookmark,
                step=step,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        else:
            final_summary = summary or self._extract_summary(clean_content)
            final_tags = sorted(list(set(tags or [])))
            node = SectionNode(
                id=sec_id,
                title=final_title,
                content=clean_content,
                level=level,
                summary=final_summary,
                tags=final_tags,
                bookmark=bookmark,
                step=step,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

        self._sections[sec_id] = node

        if node.bookmark:
            self._bookmarks[node.bookmark] = sec_id

        if self.auto_save and self.notebook_path:
            self.save()

        return node

    def summarize(self, section_id: str, new_summary: str) -> SectionNode:
        """指定セクションの要約を更新（ペンによる要約整理）."""
        sec_id = self._normalize_id(section_id)
        if sec_id not in self._sections:
            raise KeyError(f"Section not found: {section_id}")

        node = self._sections[sec_id]
        node.summary = new_summary.strip()
        node.updated_at = datetime.now(timezone.utc).isoformat()

        if self.auto_save and self.notebook_path:
            self.save()
        return node

    # -------------------------------------------------------------------------
    # しおり (Bookmark & TOC): 目次一覧・しおり管理・選択的開示
    # -------------------------------------------------------------------------
    def get_toc(
        self,
        tag: Optional[str] = None,
        as_markdown: bool = False,
    ) -> List[Dict[str, Any]] | str:
        """目次（Table of Contents）を取得（しおり機能）.

        全文を読み込まずに全体の概要・要約を低トークンで把握するための目次。
        """
        entries: List[Dict[str, Any]] = []
        for sec in self._sections.values():
            if tag and tag not in sec.tags:
                continue
            entries.append({
                "id": sec.id,
                "title": sec.title,
                "level": sec.level,
                "summary": sec.summary,
                "bookmark": sec.bookmark,
                "tags": sec.tags,
                "step": sec.step,
                "content_chars": len(sec.content),
            })

        if not as_markdown:
            return entries

        # Markdown 形式の TOC を生成
        lines = [
            "# Memory Notebook TOC",
            f"*Total sections: {len(entries)} | Bookmarks: {len(self._bookmarks)}*",
            "",
        ]
        if not entries:
            lines.append("*(Notebook is currently empty)*")
            return "\n".join(lines)

        for e in entries:
            indent = "  " * max(0, e["level"] - 1)
            bm_tag = f"[🔖 {e['bookmark']}] " if e["bookmark"] else ""
            tag_str = f" `[{', '.join(e['tags'])}]`" if e["tags"] else ""
            step_str = f" (Step {e['step']})" if e["step"] > 0 else ""
            summary_part = f" — *{e['summary']}*" if e["summary"] else ""
            lines.append(
                f"{indent}- {bm_tag}**{e['id']}** ({e['title']}){tag_str}{step_str}{summary_part}"
            )

        return "\n".join(lines)

    def set_bookmark(self, name: str, section_id: str) -> None:
        """セクションにしおりを挟む."""
        sec_id = self._normalize_id(section_id)
        if sec_id not in self._sections:
            raise KeyError(f"Section not found: {section_id}")

        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Bookmark name cannot be empty")

        # 既存の同名ブックマークを更新
        self._bookmarks[clean_name] = sec_id
        self._sections[sec_id].bookmark = clean_name

        if self.auto_save and self.notebook_path:
            self.save()

    def remove_bookmark(self, name: str) -> bool:
        """しおりを外す."""
        clean_name = name.strip()
        if clean_name in self._bookmarks:
            sec_id = self._bookmarks.pop(clean_name)
            if sec_id in self._sections and self._sections[sec_id].bookmark == clean_name:
                self._sections[sec_id].bookmark = None
            if self.auto_save and self.notebook_path:
                self.save()
            return True
        return False

    def get_bookmarks(self) -> Dict[str, str]:
        """現在の全しおり一覧を取得 (bookmark_name -> section_id)."""
        return dict(self._bookmarks)

    # -------------------------------------------------------------------------
    # 読み出し (Read): ピンポイントな選択的読込
    # -------------------------------------------------------------------------
    def read(self, section_id_or_bookmark: str) -> Dict[str, Any]:
        """特定セクションのみをピンポイント読込（段階的開示）."""
        target = section_id_or_bookmark.strip()
        # まず bookmark 名で検索
        if target in self._bookmarks:
            sec_id = self._bookmarks[target]
            return self._sections[sec_id].to_dict()

        norm_id = self._normalize_id(target)
        if norm_id in self._sections:
            return self._sections[norm_id].to_dict()

        raise KeyError(f"Section or bookmark not found: {section_id_or_bookmark}")

    def search(self, query: str) -> List[Dict[str, Any]]:
        """タイトル、本文、要約、タグを検索."""
        q = query.strip().lower()
        if not q:
            return []

        results: List[Dict[str, Any]] = []
        for sec in self._sections.values():
            score = 0
            # マッチ度合いの簡易スコアリング
            if q in sec.id.lower():
                score += 5
            if q in sec.title.lower():
                score += 4
            if any(q in t.lower() for t in sec.tags):
                score += 3
            if q in sec.summary.lower():
                score += 2
            if q in sec.content.lower():
                score += 1

            if score > 0:
                results.append({
                    "score": score,
                    "section_id": sec.id,
                    "title": sec.title,
                    "summary": sec.summary,
                    "bookmark": sec.bookmark,
                    "tags": sec.tags,
                    "preview": self._generate_preview(sec.content, q),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # -------------------------------------------------------------------------
    # 消しゴム (Eraser): 削除・刈り込み
    # -------------------------------------------------------------------------
    def delete(self, section_id_or_bookmark: str) -> bool:
        """指定セクションを削除（消しゴム）."""
        target = section_id_or_bookmark.strip()
        sec_id = self._bookmarks.get(target, self._normalize_id(target))

        if sec_id not in self._sections:
            raise KeyError(f"Cannot delete non-existent section: {section_id_or_bookmark}")

        node = self._sections.pop(sec_id)
        if node.bookmark and node.bookmark in self._bookmarks:
            del self._bookmarks[node.bookmark]

        if self.auto_save and self.notebook_path:
            self.save()
        return True

    def clear(self, tag: Optional[str] = None) -> int:
        """ノートの消去または特定タグの一括刈り込み（消しゴム）."""
        if tag is None:
            count = len(self._sections)
            self._sections.clear()
            self._bookmarks.clear()
            if self.auto_save and self.notebook_path:
                self.save()
            return count

        # 特定タグの削除
        to_delete = [
            sec_id for sec_id, sec in self._sections.items() if tag in sec.tags
        ]
        for sec_id in to_delete:
            self.delete(sec_id)
        return len(to_delete)

    # -------------------------------------------------------------------------
    # 紙 (Paper): Markdown 永続化とシリアライズ
    # -------------------------------------------------------------------------
    def to_markdown(self) -> str:
        """完全な Markdown ドキュメントにシリアライズ."""
        lines = [
            "# Memory Notebook",
            "",
            "<!-- TOC:START -->",
            str(self.get_toc(as_markdown=True)),
            "<!-- TOC:END -->",
            "",
            "---",
            "",
        ]

        for sec in self._sections.values():
            prefix = "#" * sec.level
            lines.append(f"{prefix} {sec.title}")
            meta = {
                "id": sec.id,
                "level": sec.level,
                "summary": sec.summary,
                "tags": sec.tags,
                "bookmark": sec.bookmark,
                "step": sec.step,
                "updated_at": sec.updated_at,
            }
            lines.append(f"<!-- meta: {json.dumps(meta, ensure_ascii=False)} -->")
            if sec.summary:
                lines.append(f"> **Summary**: {sec.summary}")
                lines.append("")
            lines.append(sec.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Optional[Path | str] = None) -> Path:
        """Markdown ファイルとして保存."""
        target_path = Path(path) if path else self.notebook_path
        if not target_path:
            raise ValueError("No notebook_path specified for saving")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.to_markdown(), encoding="utf-8")
        return target_path

    def load(self, path: Optional[Path | str] = None) -> None:
        """Markdown ファイルから復元."""
        target_path = Path(path) if path else self.notebook_path
        if not target_path or not target_path.exists():
            return

        text = target_path.read_text(encoding="utf-8")
        self._parse_markdown(text)

    def _parse_markdown(self, text: str) -> None:
        """Markdown テキストを解析してセクションノード群を構築."""
        self._sections.clear()
        self._bookmarks.clear()

        # <!-- meta: {...} --> のメタデータコメントを抽出
        meta_pattern = re.compile(r"<!-- meta:\s*(\{.*?\})\s*-->")
        parts = re.split(r"\n---\n", text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            match = meta_pattern.search(part)
            if not match:
                continue

            try:
                meta_json = json.loads(match.group(1))
                sec_id = meta_json.get("id")
                if not sec_id:
                    continue

                # 本文の抽出（meta コメント、タイトル見出し、Summary 引用行を除外）
                body = part
                # meta コメント除去
                body = meta_pattern.sub("", body)
                # 見出し行除去 (e.g. ## Title)
                body = re.sub(r"^#{1,6}\s+.*?\n", "", body, flags=re.MULTILINE)
                # Summary 行除去
                body = re.sub(r"^>\s*\*\*Summary\*\*:\s*.*?\n", "", body, flags=re.MULTILINE)
                body = body.strip()

                node = SectionNode(
                    id=sec_id,
                    title=meta_json.get("title", sec_id),
                    content=body,
                    level=meta_json.get("level", 2),
                    summary=meta_json.get("summary", ""),
                    tags=meta_json.get("tags", []),
                    bookmark=meta_json.get("bookmark"),
                    step=meta_json.get("step", 0),
                    updated_at=meta_json.get("updated_at", ""),
                )
                self._sections[sec_id] = node
                if node.bookmark:
                    self._bookmarks[node.bookmark] = sec_id
            except Exception as e:
                logger.warning("Failed to parse section metadata: %s", e)

    # -------------------------------------------------------------------------
    # ユーティリティ
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_id(text: str) -> str:
        """ID を小文字のスラッグ形式に正規化."""
        slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text.strip().lower())
        return slug.strip("._-")

    @staticmethod
    def _extract_summary(content: str, max_chars: int = 120) -> str:
        """本文から最初の1文または概要を抽出."""
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            return ""
        first_line = lines[0]
        # 句点またはピリオドで区切る
        sentence = re.split(r"[。\.!\?]\s*", first_line)[0]
        if len(sentence) > max_chars:
            return sentence[: max_chars - 3] + "..."
        return sentence

    @staticmethod
    def _generate_preview(content: str, query: str, window: int = 60) -> str:
        """検索語の周辺プレビューを生成."""
        lower = content.lower()
        idx = lower.find(query.lower())
        if idx == -1:
            return content[:window] + "..." if len(content) > window else content
        start = max(0, idx - window // 2)
        end = min(len(content), idx + len(query) + window // 2)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        return f"{prefix}{content[start:end]}{suffix}".replace("\n", " ")
