"""Google ADK 2.0 / Antigravity 仕様準拠 3段階 Progressive Disclosure メタスキルハーネス.

メタスキルおよび生成スキル群を以下の3段階で段階的開示 (Progressive Disclosure) します:
- Level 1 (Metadata): 名前、概要、入出力仕様、許可ツールの軽量カタログ (低トークン消費)
- Level 2 (Instructions): スキルがトリガーされた時に展開される SKILL.md 本文 (ワークフロー、思考プロトコル)
- Level 3 (Execution): スキル配下の scripts/ や allowed-tools を介したオンデマンド実行
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SkillMetadata:
    """Level 1: スキルメタデータ (YAML Frontmatter 由来)."""

    name: str
    description: str
    inputs: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    outputs: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    allowed_tools: List[str] = dataclasses.field(default_factory=list)
    version: str = "2.0.0"
    pattern: str = "workflow"
    skill_dir: Path = dataclasses.field(default_factory=Path)

    def to_catalog_entry(self) -> str:
        """Level 1 用の極小トークンカタログ文字列."""
        # 改行を整形して1行に
        clean_desc = " ".join(self.description.strip().splitlines())
        if len(clean_desc) > 120:
            clean_desc = clean_desc[:117] + "..."
        tools_str = f" [tools: {', '.join(self.allowed_tools)}]" if self.allowed_tools else ""
        return f"- `{self.name}`: {clean_desc}{tools_str}"


class SkillHarness:
    """Progressive Disclosure を管理するスキルハーネス基盤."""

    def __init__(self, search_paths: Optional[List[Path]] = None) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        if search_paths is None:
            self.search_paths = [
                repo_root / "meta_skills",
                repo_root / "generated_skills",
            ]
        else:
            self.search_paths = search_paths

        self._metadata_cache: Dict[str, SkillMetadata] = {}
        self._instructions_cache: Dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        """スキルディレクトリを走査し、Level 1 メタデータを再インデックス化."""
        self._metadata_cache.clear()
        self._instructions_cache.clear()

        for base_path in self.search_paths:
            if not base_path.exists():
                continue

            for skill_dir in sorted(base_path.iterdir()):
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                meta = self._parse_frontmatter(skill_md, skill_dir)
                if meta:
                    self._metadata_cache[meta.name] = meta

    def _parse_frontmatter(self, skill_md: Path, skill_dir: Path) -> Optional[SkillMetadata]:
        """SKILL.md の YAML Frontmatter をパース."""
        try:
            content = skill_md.read_text(encoding="utf-8")
            if not content.startswith("---"):
                return None

            parts = content.split("---", 2)
            if len(parts) < 3:
                return None

            yaml_str = parts[1]
            data = yaml.safe_load(yaml_str)
            if not isinstance(data, dict):
                return None

            name = data.get("name", skill_dir.name)
            description = data.get("description", "")
            allowed_tools = data.get("allowed-tools", "")
            if isinstance(allowed_tools, str):
                allowed_tools_list = [t.strip() for t in allowed_tools.split() if t.strip()]
            elif isinstance(allowed_tools, list):
                allowed_tools_list = [str(t).strip() for t in allowed_tools]
            else:
                allowed_tools_list = []

            meta_block = data.get("metadata", {})
            inputs = meta_block.get("inputs", []) if isinstance(meta_block, dict) else []
            outputs = meta_block.get("outputs", []) if isinstance(meta_block, dict) else []
            version = meta_block.get("version", "2.0.0") if isinstance(meta_block, dict) else "2.0.0"
            pattern = meta_block.get("pattern", "workflow") if isinstance(meta_block, dict) else "workflow"

            return SkillMetadata(
                name=name,
                description=description,
                inputs=inputs,
                outputs=outputs,
                allowed_tools=allowed_tools_list,
                version=version,
                pattern=pattern,
                skill_dir=skill_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to parse frontmatter from {skill_md}: {e}")
            return None

    # =========================================================================
    # Level 1: メタデータカタログの提供 (低コンテキスト消費)
    # =========================================================================
    def list_skills(self) -> List[SkillMetadata]:
        """現在利用可能なスキルメタデータ一覧を取得."""
        return list(self._metadata_cache.values())

    def get_level1_catalog(self) -> str:
        """Google ADK のプロンプトやシステム指示に注入する Level 1 カタログ文字列."""
        if not self._metadata_cache:
            return "No skills currently available."

        lines = [
            "### Available Meta-Skills (Trigger on demand):",
            "To use any skill, request to trigger it by name to load its full workflow (Level 2 instructions).",
        ]
        for meta in self._metadata_cache.values():
            lines.append(meta.to_catalog_entry())

        return "\n".join(lines)

    # =========================================================================
    # Level 2: SKILL.md 本文のオンデマンド展開 (Instructions)
    # =========================================================================
    def load_skill_instructions(self, skill_name: str) -> str:
        """スキルがトリガーされた時に初めて SKILL.md の本文 (Markdown) を開示."""
        if skill_name in self._instructions_cache:
            return self._instructions_cache[skill_name]

        meta = self._metadata_cache.get(skill_name)
        if not meta:
            # ハイフン / アンダースコアの差異を吸収
            alt_name = skill_name.replace("_", "-")
            meta = self._metadata_cache.get(alt_name)
            if not meta:
                alt_name = skill_name.replace("-", "_")
                meta = self._metadata_cache.get(alt_name)

        if not meta:
            raise KeyError(f"Skill '{skill_name}' not found in registry.")

        skill_md = meta.skill_dir / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        parts = content.split("---", 2)
        body = parts[2].strip() if len(parts) >= 3 else content.strip()

        header = (
            f"=== [LOADED LEVEL 2 SKILL: {meta.name}] ===\n"
            f"Description: {meta.description.strip()}\n"
            f"Allowed Tools: {', '.join(meta.allowed_tools)}\n"
            f"===========================================\n\n"
        )
        full_instructions = header + body
        self._instructions_cache[skill_name] = full_instructions
        return full_instructions

    # =========================================================================
    # Level 3: スクリプトおよびツールの動的ディスパッチ (Execution)
    # =========================================================================
    def execute_skill_script(
        self,
        skill_name: str,
        script_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """スキル配下の scripts/<script_name>.py をオンデマンド実行し、構造化結果を取得."""
        meta = self._metadata_cache.get(skill_name)
        if not meta:
            alt_name = skill_name.replace("_", "-")
            meta = self._metadata_cache.get(alt_name)
        if not meta:
            return {"success": False, "error": f"Skill '{skill_name}' not found."}

        scripts_dir = meta.skill_dir / "scripts"
        script_file = scripts_dir / f"{script_name}.py"
        if not script_file.exists():
            # 拡張子付きや同名ファイル確認
            script_file = scripts_dir / script_name
            if not script_file.exists():
                # scripts ディレクトリの先頭 Python ファイル
                py_files = list(scripts_dir.glob("*.py"))
                if py_files:
                    script_file = py_files[0]
                else:
                    return {"success": False, "error": f"No script found in {scripts_dir}"}

        try:
            # Python モジュールとしてインポート実行
            spec = importlib.util.spec_from_file_location(f"skill_{meta.name}_{script_name}", script_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {script_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # run(input_data) または main() 関数を探索
            if hasattr(module, "run"):
                res = module.run(input_data)
                return {"success": True, "result": res}
            elif hasattr(module, "execute"):
                res = module.execute(input_data)
                return {"success": True, "result": res}
            else:
                return {
                    "success": False,
                    "error": f"Script {script_file.name} does not expose 'run' or 'execute' function.",
                }
        except Exception as e:
            logger.error(f"Error executing skill script {script_file}: {e}", exc_info=True)
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
