"""EDD Contract Tests for Memory Notebook (memory-notebook).

Contracts:
- Positive Case 1: TOC generation with bookmarks, tags, and summaries (しおり・目次)
- Positive Case 2: Writing, appending, summarizing, and selective reading (ペン・選択的開示)
- Positive Case 3: Bookmark lookup, deletion, and markdown save/load persistence (消しゴム・永続化)
- Negative Case 1: Reading non-existent section or invalid bookmark raises KeyError
- Negative Case 2: Writing with empty section ID or setting empty bookmark raises ValueError
- Negative Case 3: Deleting non-existent section raises KeyError
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

# パス解決
_TEST_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _TEST_DIR.parent
_REPO_ROOT = _SKILL_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from acr_agi3.meta.memory_notebook import MemoryNotebook, SectionNode


class TestMemoryNotebookContract(unittest.TestCase):
    """契約テスト: 正例3件 + 負例3件."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.nb_path = Path(self.temp_dir.name) / "test_notebook.md"
        self.nb = MemoryNotebook(notebook_path=self.nb_path, auto_save=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Positive Cases (正例3件)
    # -------------------------------------------------------------------------
    def test_positive_01_toc_generation_and_bookmarking(self) -> None:
        """正例1: TOC生成としおり・目次機能の検証."""
        # 目次が空の状態
        toc_empty = self.nb.get_toc()
        self.assertEqual(len(toc_empty), 0)

        # 2つのセクションを書き込み
        self.nb.write(
            section_id="rules.controls",
            title="Movement Controls",
            content="ACTION1 is UP. ACTION2 is DOWN. Move carefully.",
            summary="Keybindings for movement",
            tags=["controls", "rule"],
            step=1,
        )
        self.nb.write(
            section_id="goals.primary",
            title="Primary Mission",
            content="Navigate to the green crystal in the center.",
            summary="Main target is green crystal",
            bookmark="mission_goal",
            tags=["goal"],
            step=2,
        )

        toc = self.nb.get_toc()
        self.assertEqual(len(toc), 2)
        sec_map = {item["id"]: item for item in toc}
        self.assertIn("rules.controls", sec_map)
        self.assertIn("goals.primary", sec_map)
        self.assertEqual(sec_map["goals.primary"]["bookmark"], "mission_goal")
        self.assertEqual(sec_map["rules.controls"]["summary"], "Keybindings for movement")

        # Markdown 形式の TOC 出力検証
        md_toc = self.nb.get_toc(as_markdown=True)
        self.assertIn("# Memory Notebook TOC", md_toc)
        self.assertIn("[🔖 mission_goal]", md_toc)
        self.assertIn("rules.controls", md_toc)

    def test_positive_02_pen_writing_append_and_selective_read(self) -> None:
        """正例2: ペンによる書き込み・追記・要約更新とピンポイント選択的読込の検証."""
        # 1. 新規書き込み
        node = self.nb.write(
            section_id="hypotheses.switches",
            title="Switch Mechanics",
            content="Blue switch appears to toggle laser gates.",
            step=3,
        )
        self.assertEqual(node.id, "hypotheses.switches")
        self.assertTrue(len(node.summary) > 0)

        # 2. ピンポイント選択的読込 (Progressive Disclosure)
        read_data = self.nb.read("hypotheses.switches")
        self.assertEqual(read_data["id"], "hypotheses.switches")
        self.assertIn("Blue switch appears", read_data["content"])

        # 3. 追記 (append=True)
        self.nb.write(
            section_id="hypotheses.switches",
            content="Confirmed: Pressing blue switch lowered the gate.",
            append=True,
            step=5,
        )
        read_updated = self.nb.read("hypotheses.switches")
        self.assertIn("Blue switch appears", read_updated["content"])
        self.assertIn("Confirmed: Pressing", read_updated["content"])
        self.assertEqual(read_updated["step"], 5)

        # 4. 要約更新 (summarize)
        self.nb.summarize("hypotheses.switches", "Blue switch reliably toggles laser gates")
        self.assertEqual(self.nb.read("hypotheses.switches")["summary"], "Blue switch reliably toggles laser gates")

    def test_positive_03_eraser_deletion_and_markdown_persistence(self) -> None:
        """正例3: 消しゴムによる削除としおり解除、およびMarkdown永続化・再ロードの検証."""
        self.nb.write(
            section_id="temp_notes.hallway",
            title="Hallway Dead End",
            content="East hallway leads to impassable spikes.",
            tags=["dead_end"],
            bookmark="caution_area",
            step=4,
        )
        self.assertEqual(self.nb.get_bookmarks().get("caution_area"), "temp_notes.hallway")

        # しおり経由での読込
        node_via_bm = self.nb.read("caution_area")
        self.assertEqual(node_via_bm["id"], "temp_notes.hallway")

        # Markdown ファイルとして自動保存されていることを確認
        self.assertTrue(self.nb_path.exists())
        saved_text = self.nb_path.read_text(encoding="utf-8")
        self.assertIn("temp_notes.hallway", saved_text)
        self.assertIn("<!-- meta:", saved_text)

        # 別インスタンスでロード復元できることを確認
        loaded_nb = MemoryNotebook(notebook_path=self.nb_path)
        self.assertEqual(len(loaded_nb.get_toc()), 1)
        self.assertEqual(loaded_nb.get_bookmarks().get("caution_area"), "temp_notes.hallway")

        # 消しゴム (delete) で削除
        deleted = self.nb.delete("temp_notes.hallway")
        self.assertTrue(deleted)
        self.assertEqual(len(self.nb.get_toc()), 0)
        self.assertNotIn("caution_area", self.nb.get_bookmarks())

        # ファイルへも自動反映されることを確認
        reloaded_nb = MemoryNotebook(notebook_path=self.nb_path)
        self.assertEqual(len(reloaded_nb.get_toc()), 0)

    # -------------------------------------------------------------------------
    # Negative Cases (負例3件)
    # -------------------------------------------------------------------------
    def test_negative_01_read_non_existent_section(self) -> None:
        """負例1: 存在しないセクションまたはしおりの読込エラー."""
        with self.assertRaises(KeyError) as ctx:
            self.nb.read("non_existent_section")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_negative_02_invalid_parameters(self) -> None:
        """負例2: 空のセクションID書き込みまたは空しおり設定のエラー."""
        # 空のセクションID
        with self.assertRaises(ValueError):
            self.nb.write(section_id="", content="Invalid content")

        with self.assertRaises(ValueError):
            self.nb.write(section_id="   ", content="Whitespace only")

        # 存在しないセクションへのしおり設定
        with self.assertRaises(KeyError):
            self.nb.set_bookmark(name="my_bm", section_id="unknown_sec")

        # 存在するセクションに対して空のしおり名を設定
        self.nb.write(section_id="valid.sec", content="Some valid content")
        with self.assertRaises(ValueError):
            self.nb.set_bookmark(name="", section_id="valid.sec")

    def test_negative_03_delete_non_existent_section(self) -> None:
        """負例3: 存在しないセクションを消去しようとした場合のエラー."""
        with self.assertRaises(KeyError) as ctx:
            self.nb.delete("does_not_exist")
        self.assertIn("cannot delete non-existent", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
