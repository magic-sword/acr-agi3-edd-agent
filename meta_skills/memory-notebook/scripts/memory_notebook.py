#!/usr/bin/env python3
"""Memory Notebook CLI Tool for Google ADK 2.0 Agents.

Long-term memory notebook following the paper, pen, eraser, and bookmark metaphor:
- Paper (Storage): Structured markdown persistence.
- Pen: Writing, appending, summarizing notes.
- Eraser: Deleting entries, pruning tags.
- Bookmark & TOC: Index-based selective progressive disclosure.

Zero external dependencies: 100% offline compatible with standard library only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional


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
        self._bookmarks: Dict[str, str] = {}

        if self.notebook_path and self.notebook_path.exists():
            self.load()

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
        """ノートに情報を書き込む（ペン）."""
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

    def get_toc(
        self,
        tag: Optional[str] = None,
        as_markdown: bool = False,
    ) -> List[Dict[str, Any]] | str:
        """目次（Table of Contents）を取得（しおり機能）."""
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
        """現在の全しおり一覧を取得."""
        return dict(self._bookmarks)

    def read(self, section_id_or_bookmark: str) -> Dict[str, Any]:
        """特定セクションのみをピンポイント読込（段階的開示）."""
        target = section_id_or_bookmark.strip()
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

        to_delete = [
            sec_id for sec_id, sec in self._sections.items() if tag in sec.tags
        ]
        for sec_id in to_delete:
            self.delete(sec_id)
        return len(to_delete)

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

                body = part
                body = meta_pattern.sub("", body)
                body = re.sub(r"^#{1,6}\s+.*?\n", "", body, flags=re.MULTILINE)
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
            except Exception:
                pass

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


_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parents[2]

DEFAULT_NOTEBOOK_PATH = Path(
    os.getenv(
        "MEMORY_NOTEBOOK_PATH",
        str(_REPO_ROOT / "generated_skills" / "memory" / "notebook.md"),
    )
)


def get_notebook(path_str: Optional[str] = None) -> MemoryNotebook:
    """ノートブックインスタンスを初期化."""
    target_path = Path(path_str) if path_str else DEFAULT_NOTEBOOK_PATH
    return MemoryNotebook(notebook_path=target_path, auto_save=True)


def cmd_toc(args: argparse.Namespace) -> int:
    """目次 (TOC) を取得."""
    nb = get_notebook(args.notebook)
    as_markdown = args.format == "markdown"
    result = nb.get_toc(tag=args.tag, as_markdown=as_markdown)

    if as_markdown:
        print(result)
    else:
        print(json.dumps({"status": "ok", "toc": result}, ensure_ascii=False, indent=2))
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    """特定セクションのみを読み出す."""
    nb = get_notebook(args.notebook)
    try:
        data = nb.read(args.section)
        print(json.dumps({"status": "ok", "section": data}, ensure_ascii=False, indent=2))
        return 0
    except KeyError as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1


def cmd_write(args: argparse.Namespace) -> int:
    """セクションを書き込む / 追記する."""
    nb = get_notebook(args.notebook)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    try:
        node = nb.write(
            section_id=args.section,
            content=args.content,
            title=args.title,
            summary=args.summary,
            level=args.level,
            tags=tags,
            bookmark=args.bookmark,
            step=args.step,
            append=args.append,
        )
        print(json.dumps({"status": "ok", "written": node.to_dict()}, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1


def cmd_delete(args: argparse.Namespace) -> int:
    """セクションを削除する (消しゴム)."""
    nb = get_notebook(args.notebook)
    try:
        nb.delete(args.section)
        print(json.dumps({"status": "ok", "deleted": args.section}, ensure_ascii=False, indent=2))
        return 0
    except KeyError as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1


def cmd_bookmark(args: argparse.Namespace) -> int:
    """しおりを設定または解除する."""
    nb = get_notebook(args.notebook)
    try:
        if args.remove:
            removed = nb.remove_bookmark(args.name)
            print(json.dumps({"status": "ok", "removed": removed, "bookmark": args.name}, ensure_ascii=False))
        else:
            if not args.section:
                print(json.dumps({"status": "error", "message": "--section is required to set bookmark"}, ensure_ascii=False), file=sys.stderr)
                return 1
            nb.set_bookmark(name=args.name, section_id=args.section)
            print(json.dumps({"status": "ok", "bookmark": args.name, "section": args.section}, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """ノート内を検索."""
    nb = get_notebook(args.notebook)
    results = nb.search(args.query)
    print(json.dumps({"status": "ok", "query": args.query, "count": len(results), "results": results}, ensure_ascii=False, indent=2))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """ノートをクリアまたはタグ刈り込み (消しゴム)."""
    nb = get_notebook(args.notebook)
    count = nb.clear(tag=args.tag)
    print(json.dumps({"status": "ok", "cleared_count": count, "tag_filter": args.tag}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory Notebook CLI — Long-term memory tool with paper, pen, eraser, and bookmark.",
    )
    parser.add_argument("--notebook", type=str, default=None, help="Path to notebook markdown file")

    subparsers = parser.add_subparsers(dest="command", help="Memory operations")

    p_toc = subparsers.add_parser("toc", help="Get table of contents (TOC) overview")
    p_toc.add_argument("--tag", type=str, default=None, help="Filter TOC by tag")
    p_toc.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")

    p_read = subparsers.add_parser("read", help="Read specific section by ID or bookmark (Selective Disclosure)")
    p_read.add_argument("--section", required=True, help="Section ID or bookmark name")

    p_write = subparsers.add_parser("write", help="Write or append section (Pen)")
    p_write.add_argument("--section", required=True, help="Section ID (slug)")
    p_write.add_argument("--content", required=True, help="Content text")
    p_write.add_argument("--title", type=str, default=None, help="Display title")
    p_write.add_argument("--summary", type=str, default=None, help="Short summary (1-2 sentences)")
    p_write.add_argument("--level", type=int, default=2, help="Heading level (1-4)")
    p_write.add_argument("--tags", type=str, default=None, help="Comma-separated tags")
    p_write.add_argument("--bookmark", type=str, default=None, help="Bookmark name")
    p_write.add_argument("--step", type=int, default=0, help="Step index")
    p_write.add_argument("--append", action="store_true", help="Append to existing content")

    p_del = subparsers.add_parser("delete", help="Delete section (Eraser)")
    p_del.add_argument("--section", required=True, help="Section ID or bookmark name")

    p_bm = subparsers.add_parser("bookmark", help="Manage bookmarks (しおり)")
    p_bm.add_argument("--name", required=True, help="Bookmark name")
    p_bm.add_argument("--section", type=str, default=None, help="Section ID to bookmark")
    p_bm.add_argument("--remove", action="store_true", help="Remove bookmark")

    p_search = subparsers.add_parser("search", help="Search memory notes")
    p_search.add_argument("--query", required=True, help="Search query")

    p_clear = subparsers.add_parser("clear", help="Clear memory or prune tags (Eraser)")
    p_clear.add_argument("--tag", type=str, default=None, help="Only delete sections with this tag")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    handler = {
        "toc": cmd_toc,
        "read": cmd_read,
        "write": cmd_write,
        "delete": cmd_delete,
        "bookmark": cmd_bookmark,
        "search": cmd_search,
        "clear": cmd_clear,
    }.get(args.command)

    if handler:
        return handler(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
