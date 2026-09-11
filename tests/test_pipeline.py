"""ARC 推論および評価パイプライン全体の疎通・統合テスト."""

import numpy as np

from acr_agi3.agent.orchestrator import ARCOrchestrator
from acr_agi3.dsl import primitives
from acr_agi3.eval.harness import BenchmarkHarness
from acr_agi3.eval.metrics import compute_pass_at_k


def test_dsl_primitives() -> None:
    """DSL プリミティブの基本動作を検証する."""
    grid = np.array([[1, 2], [3, 4]])
    assert np.array_equal(primitives.rot180(grid), np.array([[4, 3], [2, 1]]))
    assert np.array_equal(primitives.fliplr(grid), np.array([[2, 1], [4, 3]]))
    assert np.array_equal(primitives.flipud(grid), np.array([[3, 4], [1, 2]]))
    assert np.array_equal(primitives.replace_color(grid, 1, 9), np.array([[9, 2], [3, 4]]))


def test_orchestrator_solves_game() -> None:
    """Orchestrator がゲーム環境を解けることを検証する."""
    from acr_agi3.agent.llm.local_model import LocalTransformersLlm
    from acr_agi3.agent.meta_agent import MetaSkillDrivenAgent
    from acr_agi3.game.vcgt_game import GridWorldGameEnv

    def mock_policy_fn(prompt: str) -> str:
        return (
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs, info=None):\n"
            "    return Action.RIGHT\n"
            "```"
        )

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_policy_fn)
    meta_agent = MetaSkillDrivenAgent(model=mock_llm)
    orchestrator = ARCOrchestrator(agent=meta_agent)

    env = GridWorldGameEnv(grid_shape=(3, 3), initial_player_pos=(1, 0), goal_pos=(1, 1))
    res = orchestrator.solve(env)

    assert res["is_solved"] is True
    assert res["policy_code"] is not None


def test_benchmark_harness_game_integration() -> None:
    """BenchmarkHarness によるゲーム環境評価フローが正常に集計されることを検証する."""
    from acr_agi3.agent.llm.local_model import LocalTransformersLlm
    from acr_agi3.agent.meta_agent import MetaSkillDrivenAgent
    from acr_agi3.game.vcgt_game import GridWorldGameEnv

    def mock_policy_fn(prompt: str) -> str:
        return (
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs, info=None):\n"
            "    return Action.RIGHT\n"
            "```"
        )

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_policy_fn)
    agent = MetaSkillDrivenAgent(model=mock_llm)
    harness = BenchmarkHarness()

    env = GridWorldGameEnv(grid_shape=(3, 3), initial_player_pos=(1, 0), goal_pos=(1, 1))
    res = harness.evaluate_game(env=env, agent=agent, task_id="test_stage")
    assert res["solved"] is True
    assert res["task_id"] == "test_stage"


def test_metrics_pass_at_k() -> None:
    """Pass@k 計算ロジックを検証する."""
    # 2タスク中1タスクが解けた場合
    results = [
        [False, True],  # 2試行目で正解
        [False, False],  # 不正解
    ]
    assert compute_pass_at_k(results, k=2) == 0.5
    assert compute_pass_at_k(results, k=1) == 0.0
