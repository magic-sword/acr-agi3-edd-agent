"""ACR-AGI-3 インタラクティブゲーム環境およびポリシー検証の単体テスト."""

from acr_agi3.agent.llm.arc_tools import execute_and_verify_game_policy
from acr_agi3.game.env import Action
from acr_agi3.game.vcgt_game import GridWorldGameEnv


def test_gridworld_game_env_navigation():
    """GridWorldGameEnv のリセット、ステップ遷移、壁・ゴール判定を検証."""
    env = GridWorldGameEnv(
        grid_shape=(5, 5),
        initial_player_pos=(0, 0),
        goal_pos=(0, 2),
        walls={(0, 1)},  # (0, 1) に壁があるため直進できない
    )

    obs = env.reset()
    assert obs.shape == (5, 5)
    assert env.player_pos == (0, 0)

    # 壁に向かって RIGHT 移動 -> 壁に衝突して移動不可
    res = env.step(Action.RIGHT)
    assert env.player_pos == (0, 0)
    assert res.done is False

    # 迂回ルート: DOWN -> RIGHT -> RIGHT -> UP でゴールへ
    env.step(Action.DOWN)
    assert env.player_pos == (1, 0)

    env.step(Action.RIGHT)
    assert env.player_pos == (1, 1)

    env.step(Action.RIGHT)
    assert env.player_pos == (1, 2)

    res_goal = env.step(Action.UP)
    assert env.player_pos == (0, 2)
    assert res_goal.done is True
    assert res_goal.reward == 1.0
    assert res_goal.info.get("status") == "goal_reached"


def test_execute_and_verify_game_policy_success():
    """行動ポリシーコードのシミュレーション検証テスト (成功ケース)."""
    env = GridWorldGameEnv(
        grid_shape=(3, 3),
        initial_player_pos=(0, 0),
        goal_pos=(0, 1),
    )

    policy_code = """
def choose_action(obs):
    # 常に右へ進むシンプルなポリシー
    return Action.RIGHT
"""
    result = execute_and_verify_game_policy(policy_code, env, max_steps=5)
    assert result["is_solved"] is True
    assert result["steps_taken"] == 1
    assert result["final_reward"] == 1.0


def test_execute_and_verify_game_policy_failure():
    """行動ポリシーコードのシミュレーション検証テスト (タイムアウト/手詰まりケース)."""
    env = GridWorldGameEnv(
        grid_shape=(3, 3),
        initial_player_pos=(0, 0),
        goal_pos=(2, 2),
    )

    policy_code = """
def choose_action(obs):
    # 待機し続けるポリシー (ゴールに到達できない)
    return Action.WAIT
"""
    result = execute_and_verify_game_policy(policy_code, env, max_steps=3)
    assert result["is_solved"] is False
    assert result["status"] == "timeout"
    assert result["steps_taken"] == 3
