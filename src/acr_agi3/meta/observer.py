"""ACR-AGI-3 動的ゲーム環境向け汎用アフォーダンス同定エンジン (Meta-Observer).

本モジュールは meta_skills/env-observer/scripts/env_observer.py の
完全自己完結実装を参照・再エクスポートする互換レイヤーです。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

# meta_skills/env-observer/scripts/env_observer.py の動的ロード
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SKILL_SCRIPT = _REPO_ROOT / "meta_skills" / "env-observer" / "scripts" / "env_observer.py"

if _SKILL_SCRIPT.exists():
    _spec = importlib.util.spec_from_file_location("skill_env_observer", _SKILL_SCRIPT)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)

        VisualObject = _mod.VisualObject
        DynamicAffordanceReport = _mod.DynamicAffordanceReport
        AffordanceObject = _mod.AffordanceObject
        GameAffordanceReport = _mod.GameAffordanceReport
        MetaObserver = _mod.MetaObserver
else:
    raise ImportError(f"Authoritative skill script not found at {_SKILL_SCRIPT}")

__all__ = [
    "VisualObject",
    "DynamicAffordanceReport",
    "AffordanceObject",
    "GameAffordanceReport",
    "MetaObserver",
]
