"""Google ADK 2.0 公式 SkillToolset 準拠 メタスキルハーネス.

Google ADK 2.0 (google.adk.skills, google.adk.tools.skill_toolset) の
ネイティブな 3段階 Progressive Disclosure (L1: Frontmatter, L2: Instructions, L3: Resources/Scripts)
を透過的に利用し、実行環境（Kaggle / ローカル）の自動パス解決と Python 直接利用レイヤーを提供します。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from google.adk.skills import Skill, load_skills_from_dir
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.skill_toolset import SkillToolset

# 後方互換性エイリアス
SkillMetadata = Skill

logger = logging.getLogger(__name__)


def _uri_to_path(uri: str) -> Path:
    """file:// URI を Path に変換."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


class SkillHarness:
    """Google ADK 2.0 公式 SkillToolset を統合管理するスキルハーネス."""

    def __init__(self, search_paths: Optional[List[Path]] = None) -> None:
        if search_paths is None:
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            candidates = [
                # 1. Kaggle 本番 Dataset (Read-Only 直参照)
                Path("/kaggle/input/acr-agi3-agent/meta_skills"),
                Path("/kaggle/input/acr-agi3-source/meta_skills"),
                # 2. ローカル開発環境 / Docker
                repo_root / "meta_skills",
                # 3. 実行時動的生成スキル (Read-Write)
                Path("/kaggle/working/generated_skills"),
                repo_root / "generated_skills",
            ]
            self.search_paths = [p for p in candidates if p.exists()]
            if not self.search_paths:
                self.search_paths = [repo_root / "meta_skills", repo_root / "generated_skills"]
        else:
            self.search_paths = search_paths

        self._skills_map: Dict[str, Skill] = {}
        self._module_cache: Dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        """指定パスから ADK 公式 load_skills_from_dir でスキルを読み込み."""
        self._skills_map.clear()
        for base_path in self.search_paths:
            if not base_path.exists() or not base_path.is_dir():
                continue
            try:
                loaded = load_skills_from_dir(base_path)
                for s in loaded:
                    self._skills_map[s.name] = s
            except Exception as e:
                logger.warning(f"Failed to load skills from {base_path}: {e}")

    @property
    def skills(self) -> List[Skill]:
        """ロードされた ADK Skill オブジェクト一覧."""
        return list(self._skills_map.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """スキル名から ADK Skill オブジェクトを取得."""
        if name in self._skills_map:
            return self._skills_map[name]
        alt = name.replace("_", "-")
        if alt in self._skills_map:
            return self._skills_map[alt]
        alt = name.replace("-", "_")
        return self._skills_map.get(alt)

    def get_toolset(
        self,
        tool_name_prefix: Optional[str] = None,
        additional_tools: Optional[List[Any]] = None,
    ) -> SkillToolset:
        """Google ADK 公式 SkillToolset を生成して返却 (Level 1/2/3 自動提供)."""
        return SkillToolset(
            skills=self.skills,
            tool_name_prefix=tool_name_prefix,
            additional_tools=additional_tools,
        )

    def get_scoped_toolset(
        self,
        skill_names: List[str],
        tool_name_prefix: Optional[str] = None,
        additional_tools: Optional[List[Any]] = None,
        tool_filter: ToolPredicate | list[str] | None = None,
    ) -> SkillToolset:
        """指定されたスキルのみに絞り込んだ最小権限の SkillToolset を生成して返却."""
        scoped_skills: List[Skill] = []
        for name in skill_names:
            skill = self.get_skill(name)
            if skill and skill not in scoped_skills:
                scoped_skills.append(skill)
        return SkillToolset(
            skills=scoped_skills,
            tool_name_prefix=tool_name_prefix,
            additional_tools=additional_tools,
            tool_filter=tool_filter,
        )

    def list_skills(self) -> List[Skill]:
        return self.skills

    def get_level1_catalog(self) -> str:
        """Level 1 カタログ文字列の生成 (デバッグ・確認用)."""
        if not self._skills_map:
            return "No skills currently available."
        lines = [
            "### Available Meta-Skills (Trigger on demand):",
            "To use any skill, request to trigger it by name to load its full workflow (Level 2 instructions).",
        ]
        for s in self._skills_map.values():
            desc = " ".join(s.description.strip().splitlines())
            if len(desc) > 120:
                desc = desc[:117] + "..."
            tools_str = f" [tools: {s.frontmatter.allowed_tools}]" if s.frontmatter.allowed_tools else ""
            lines.append(f"- `{s.name}`: {desc}{tools_str}")
        return "\n".join(lines)

    def read_skill_content(self, skill_name: str) -> str:
        """Level 2: スキル本文（SKILL.md の Instructions）をオンデマンド展開."""
        skill = self.get_skill(skill_name)
        if not skill:
            raise KeyError(f"Skill '{skill_name}' not found.")
        header = (
            f"=== [LOADED LEVEL 2 SKILL: {skill.name}] ===\n"
            f"Description: {skill.description.strip()}\n"
            f"Allowed Tools: {skill.frontmatter.allowed_tools or ''}\n"
            f"===========================================\n\n"
        )
        return header + skill.instructions.strip()

    def get_skill_module(self, skill_name: str, script_name: Optional[str] = None) -> Any:
        """スキルディレクトリ配下の Python モジュールを直接インポートして返却."""
        cache_key = f"{skill_name}:{script_name or 'default'}"
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        skill = self.get_skill(skill_name)
        if not skill:
            raise KeyError(f"Skill '{skill_name}' not found.")

        skill_path = _uri_to_path(skill._uri) if skill._uri else None
        if not skill_path or not skill_path.exists():
            raise FileNotFoundError(f"Skill path not accessible for {skill_name}")

        scripts_dir = skill_path / "scripts"
        target_name = script_name or skill.name.replace("-", "_")
        script_file = scripts_dir / f"{target_name}.py"
        if not script_file.exists():
            py_files = list(scripts_dir.glob("*.py"))
            if py_files:
                script_file = py_files[0]
            else:
                raise FileNotFoundError(f"No python script found in {scripts_dir}")

        spec = importlib.util.spec_from_file_location(f"skill_{skill.name}_{script_file.stem}", script_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {script_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._module_cache[cache_key] = module
        return module

    def execute_skill_script(
        self,
        skill_name: str,
        script_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Level 3: スクリプトを直接呼び出して実行."""
        try:
            mod = self.get_skill_module(skill_name, script_name)
            if hasattr(mod, "run"):
                return {"success": True, "result": mod.run(input_data)}
            elif hasattr(mod, "execute"):
                return {"success": True, "result": mod.execute(input_data)}
            else:
                return {
                    "success": False,
                    "error": f"Script {script_name} does not expose 'run' or 'execute' function.",
                }
        except Exception as e:
            logger.error(f"Error executing skill script {script_name}: {e}", exc_info=True)
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
