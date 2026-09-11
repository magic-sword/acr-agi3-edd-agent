"""Google ADK 2.0 向け EDD ツールバインディングの単体テスト."""

import shutil

from acr_agi3.agent.llm.edd_tools import (
    GENERATED_SKILLS_DIR,
    edd_execute_game_skill,
    edd_init_skill,
    edd_run_game_contract_test,
    edd_validate_skill,
    edd_write_skill_code,
)
from acr_agi3.game.vcgt_game import GridWorldGameEnv


def test_edd_tools_end_to_end_lifecycle():
    """EDD ツールの初期化、バリデーション、コード保存、契約テストのライフサイクル検証."""

    test_skill_name = "test_game_mover"
    target_dir = GENERATED_SKILLS_DIR / "test-game-mover"
    if target_dir.exists():
        shutil.rmtree(target_dir)

    # 1. スキル雛形初期化
    msg = edd_init_skill(test_skill_name)
    assert "Successfully scaffolded" in msg

    # 2. 静的バリデーション
    val_res = edd_validate_skill(test_skill_name)
    assert val_res["is_valid"] is True
    assert len(val_res["issues"]) == 0

    # 3. ゲーム行動コード保存
    code = (
        "import numpy as np\n"
        "from acr_agi3.game.env import Action\n\n"
        "def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:\n"
        "    return Action.RIGHT\n"
    )
    save_msg = edd_write_skill_code(test_skill_name, code)
    assert "Saved skill implementation" in save_msg

    # 4. ゲーム契約テスト (ゴール到達テスト)
    env = GridWorldGameEnv(grid_shape=(3, 5), initial_player_pos=(1, 0), goal_pos=(1, 2))
    test_res = edd_run_game_contract_test(test_skill_name, env, max_steps=10)
    assert test_res["is_solved"] is True
    assert test_res["steps_taken"] == 2

    # 5. スキル実行 (単一ステップの推論)
    sample_obs = [[2, 0, 3, 0, 0]]
    exec_res = edd_execute_game_skill(test_skill_name, sample_obs)
    assert exec_res["success"] is True
    assert exec_res["action"] == "RIGHT"

    # 後片付け
    target_dir = GENERATED_SKILLS_DIR / "test-game-mover"
    if target_dir.exists():
        shutil.rmtree(target_dir)


def test_edd_game_skill_library_and_reuse():
    """ゲーム行動スキルの登録、ライブラリ検索、再実行のテスト."""
    from acr_agi3.agent.llm.edd_tools import (
        edd_execute_game_skill,
        edd_list_skills,
        edd_register_verified_skill,
    )

    skill_name = "test_game_step_right"

    # 1. スキル初期化
    edd_init_skill(skill_name)

    # 2. ゲーム行動コード書き込み
    policy_code = (
        "import numpy as np\n"
        "from acr_agi3.game.env import Action\n\n"
        "def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:\n"
        "    return Action.RIGHT\n"
    )
    edd_write_skill_code(skill_name, policy_code)

    # 3. ライブラリ登録
    reg_res = edd_register_verified_skill(
        name=skill_name,
        description="Move right unconditionally",
        tags=["movement", "test"],
    )
    assert reg_res["success"] is True
    assert reg_res["skill_info"]["is_verified"] is True

    # 4. ライブラリ検索
    verified_list = edd_list_skills(verified_only=True)
    names = [s["name"] for s in verified_list]
    assert "test-game-step-right" in names or "test_game_step_right" in names

    # 5. スキル再利用実行 (サブルーチン呼び出し)
    sample_obs = [[0, 2], [0, 3]]
    exec_res = edd_execute_game_skill(skill_name, sample_obs)
    assert exec_res["success"] is True
    assert exec_res["action"] == "RIGHT"

    # 後片付け
    target_dir = GENERATED_SKILLS_DIR / "test-game-step-right"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir_under = GENERATED_SKILLS_DIR / "test_game_step_right"
    if target_dir_under.exists():
        shutil.rmtree(target_dir_under)
