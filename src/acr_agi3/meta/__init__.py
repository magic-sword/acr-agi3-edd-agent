"""ARC-AGI-3 メタスキルハーネスパッケージ (Google ADK 2.0 準拠).

本パッケージは Google ADK 2.0 準拠の SkillHarness のみを提供し、
メタスキルの定義・スクリプトはすべて meta_skills/ からオンデマンドでロードされます。
"""

from __future__ import annotations

from acr_agi3.meta.memory_notebook import MemoryNotebook, SectionNode

try:
    from acr_agi3.meta.skill_harness import Skill, SkillHarness, SkillMetadata
except ImportError:
    Skill = None  # type: ignore
    SkillHarness = None  # type: ignore
    SkillMetadata = None  # type: ignore

__all__ = [
    "MemoryNotebook",
    "SectionNode",
    "Skill",
    "SkillHarness",
    "SkillMetadata",
]


