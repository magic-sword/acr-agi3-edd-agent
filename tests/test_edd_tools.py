"""Google ADK 2.0 向け EDD ツールバインディングの単体テスト."""

import shutil

from acr_agi3.agent.llm.edd_tools import (
    GENERATED_SKILLS_DIR,
    edd_execute_skill,
    edd_init_skill,
    edd_run_contract_test,
    edd_validate_skill,
    edd_write_skill_code,
)


def test_edd_tools_end_to_end_lifecycle():
    """EDD ツールの初期化、バリデーション、コード保存、契約テスト、実行の完全ライフサイクル検証."""
    test_skill_name = "test_flip_vertical"

    # 1. スキル雛形初期化
    msg = edd_init_skill(test_skill_name)
    assert "Successfully scaffolded" in msg

    # 2. 静的バリデーション
    val_res = edd_validate_skill(test_skill_name)
    assert val_res["is_valid"] is True
    assert len(val_res["issues"]) == 0

    # 3. コード保存
    code = (
        "import numpy as np\n\n"
        "def transform(grid: np.ndarray) -> np.ndarray:\n"
        "    return np.flipud(grid)\n"
    )
    save_msg = edd_write_skill_code(test_skill_name, code)
    assert "Saved skill implementation" in save_msg

    # 4. 契約テスト (正例 3 件 + 負例 3 件)
    train_pairs = [
        {"input": [[1, 2], [3, 4]], "output": [[3, 4], [1, 2]]},
        {"input": [[0, 5], [6, 7]], "output": [[6, 7], [0, 5]]},
        {"input": [[9]], "output": [[9]]},
    ]
    test_res = edd_run_contract_test(test_skill_name, train_pairs)
    assert test_res["is_valid"] is True
    assert test_res["passed_count"] == 3
    assert test_res["total_count"] == 3

    # 5. スキル実行
    exec_res = edd_execute_skill(test_skill_name, [[1, 0], [2, 3]])
    assert exec_res["success"] is True
    assert exec_res["output_grid"] == [[2, 3], [1, 0]]

    # 後片付け
    target_dir = GENERATED_SKILLS_DIR / "test-flip-vertical"
    if target_dir.exists():
        shutil.rmtree(target_dir)
