"""Google ADK 2.0 向け EDD (Evaluation-Driven Development) ツールバインディング."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from edd_agent_tools.packaging.scaffold import SkillScaffolder
from edd_agent_tools.validation.validator import SkillValidator

from acr_agi3.agent.llm.arc_tools import execute_and_verify_code

logger = logging.getLogger(__name__)

GENERATED_SKILLS_DIR = Path("generated_skills")


def edd_init_skill(name: str) -> str:
    """指定された名前の新しいサブゴールスキル雛形（SKILL.md, scripts, tests）を初期化します.

    Args:
        name: スキル識別名 (例: 'crop_subgrid', 'color_remap')

    Returns:
        初期化完了メッセージまたはエラー
    """
    try:
        norm_name = name.strip().replace(" ", "_").lower()
        target = SkillScaffolder.scaffold(
            skill_name=norm_name,
            output_base_dir=str(GENERATED_SKILLS_DIR),
            pattern="workflow",
        )
        return f"Successfully scaffolded EDD skill '{norm_name}' at {target}"
    except Exception as e:
        logger.error(f"Failed to scaffold skill '{name}': {e}")
        return f"Failed to initialize skill '{name}': {e}"


def edd_validate_skill(name: str) -> dict[str, Any]:
    """スキルの Markdown-First / Progressive Disclosure 規約適合性を静的検証します.

    Args:
        name: スキル識別名

    Returns:
        検証結果辞書 (is_valid, issues)
    """
    norm_name = name.strip().replace(" ", "-").replace("_", "-").lower()
    target_dir = GENERATED_SKILLS_DIR / norm_name
    if not target_dir.exists():
        # アンダースコア版も確認
        norm_name_under = name.strip().replace(" ", "_").lower()
        target_dir = GENERATED_SKILLS_DIR / norm_name_under
        if not target_dir.exists():
            return {
                "is_valid": False,
                "error": f"Skill directory not found at {target_dir}",
                "issues": [],
            }

    try:
        res = SkillValidator.validate_directory(target_dir)
        return {
            "is_valid": res.is_valid,
            "issues": [str(issue) for issue in res.issues],
            "skill_dir": str(target_dir),
        }
    except Exception as e:
        return {"is_valid": False, "error": str(e), "issues": []}


def edd_write_skill_code(name: str, code: str) -> str:
    """スキルの Python 実装コードを generated_skills/<skill_name>/scripts に書き込みます.

    Args:
        name: スキル識別名
        code: Python コード (def transform(grid: np.ndarray) -> np.ndarray: を含む)

    Returns:
        保存完了メッセージ
    """
    norm_name = name.strip().replace(" ", "-").replace("_", "-").lower()
    script_base = name.strip().replace(" ", "_").replace("-", "_").lower()
    target_dir = GENERATED_SKILLS_DIR / norm_name
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = target_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_file = scripts_dir / f"{script_base}.py"

    with script_file.open("w", encoding="utf-8") as f:
        f.write(code)

    return f"Saved skill implementation to {script_file}"


def edd_run_contract_test(
    name: str,
    train_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """スキルの契約テスト（正例・負例）を実行し、EDD 防壁ゲートの合否結果を返します.

    Args:
        name: スキル識別名
        train_pairs: 検証用入出力ペア一覧 [{'input': [[...]], 'output': [[...]]}]

    Returns:
        契約テスト結果 (is_valid, passed_count, total_count, failures)
    """
    norm_name = name.strip().replace(" ", "-").replace("_", "-").lower()
    script_base = name.strip().replace(" ", "_").replace("-", "_").lower()
    script_file = GENERATED_SKILLS_DIR / norm_name / "scripts" / f"{script_base}.py"

    if not script_file.exists():
        return {
            "is_valid": False,
            "error": f"Implementation file not found: {script_file}",
            "passed_count": 0,
            "total_count": len(train_pairs),
        }

    code = script_file.read_text(encoding="utf-8")
    return execute_and_verify_code(code, train_pairs)


def edd_execute_skill(
    name: str,
    input_grid: list[list[int]],
) -> dict[str, Any]:
    """検証済みの具象スキルを実行し、変換後グリッドを取得します.

    Args:
        name: スキル識別名
        input_grid: 入力 2 次元配列

    Returns:
        実行結果 (success: bool, output_grid: list[list[int]], error: str)
    """
    norm_name = name.strip().replace(" ", "-").replace("_", "-").lower()
    script_base = name.strip().replace(" ", "_").replace("-", "_").lower()
    script_file = GENERATED_SKILLS_DIR / norm_name / "scripts" / f"{script_base}.py"

    if not script_file.exists():
        return {"success": False, "error": f"Skill not found: {script_file}"}

    try:
        code = script_file.read_text(encoding="utf-8")
        local_scope: dict[str, Any] = {"np": np}
        exec(code, {"np": np, "__builtins__": __builtins__}, local_scope)
        if "transform" not in local_scope:
            return {"success": False, "error": "Function 'transform' not defined"}

        inp = np.array(input_grid, dtype=int)
        out = local_scope["transform"](inp)
        out_list = np.array(out, dtype=int).tolist()
        return {"success": True, "output_grid": out_list}
    except Exception as e:
        return {"success": False, "error": str(e)}


def edd_run_game_contract_test(
    name: str,
    env: Any,
    max_steps: int = 50,
) -> dict[str, Any]:
    """ゲーム環境シミュレータ上でスキルの契約テスト（ゴール到達、制約回避）を実行.

    Args:
        name: スキル識別名
        env: GameEnvironment インスタンス
        max_steps: 最大許容ステップ数

    Returns:
        契約テスト結果 (is_solved: bool, steps_taken: int, final_reward: float, error: str)
    """
    from acr_agi3.agent.llm.arc_tools import execute_and_verify_game_policy

    norm_name = name.strip().replace(" ", "-").replace("_", "-").lower()
    script_base = name.strip().replace(" ", "_").replace("-", "_").lower()
    script_file = GENERATED_SKILLS_DIR / norm_name / "scripts" / f"{script_base}.py"

    if not script_file.exists():
        return {
            "is_solved": False,
            "error": f"Implementation file not found: {script_file}",
            "steps_taken": 0,
            "final_reward": -1.0,
        }

    code = script_file.read_text(encoding="utf-8")
    return execute_and_verify_game_policy(code, env, max_steps=max_steps)
