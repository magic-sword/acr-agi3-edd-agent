"""動的スキル合成エンジン (Meta-Skill Synthesizer).

本モジュールは meta_skills/skill-synthesizer/scripts/skill_synthesizer.py の
完全自己完結実装を参照・再エクスポートする互換レイヤーです。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SKILL_SCRIPT = _REPO_ROOT / "meta_skills" / "skill-synthesizer" / "scripts" / "skill_synthesizer.py"

if _SKILL_SCRIPT.exists():
    _spec = importlib.util.spec_from_file_location("skill_synthesizer", _SKILL_SCRIPT)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)

        BaseSkillPolicy = _mod.BaseSkillPolicy
        AffordanceNavigationSkill = _mod.AffordanceNavigationSkill
        InteractiveClickSkill = _mod.InteractiveClickSkill
        FrontierExplorationSkill = _mod.FrontierExplorationSkill
        MetaSkillSynthesizer = _mod.MetaSkillSynthesizer
else:
    raise ImportError(f"Authoritative skill script not found at {_SKILL_SCRIPT}")

__all__ = [
    "BaseSkillPolicy",
    "AffordanceNavigationSkill",
    "InteractiveClickSkill",
    "FrontierExplorationSkill",
    "MetaSkillSynthesizer",
]
