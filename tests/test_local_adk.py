"""Google ADK 2.0 × ローカル LLM のゲームプレイ結合・単体テスト."""

import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.arc_tools import (
    execute_and_verify_game_policy,
    extract_python_code,
)
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.agent.llm_agent import LLMGameAgent
from acr_agi3.game.vcgt_game import GridWorldGameEnv


def test_extract_python_code():
    """マークダウンコードブロックからの Python 抽出テスト."""
    text1 = (
        "Here is the code:\n```python\n"
        "from acr_agi3.game.env import Action\n"
        "def choose_action(obs):\n"
        "    return Action.RIGHT\n```\nDone."
    )
    assert "def choose_action" in extract_python_code(text1)

    text2 = "```\ndef choose_action(obs):\n    return Action.UP\n```"
    assert "def choose_action" in extract_python_code(text2)

    text3 = "def choose_action(obs):\n    return Action.DOWN"
    assert extract_python_code(text3) == "def choose_action(obs):\n    return Action.DOWN"


def test_execute_and_verify_game_policy_success():
    """ゲーム環境でゴールに到達する正常系テスト."""
    env = GridWorldGameEnv(
        grid_shape=(5, 5),
        player_pos=(1, 1),
        goal_pos=(1, 3),
        walls=set(),
    )
    code = """
from acr_agi3.game.env import Action
def choose_action(obs):
    return Action.RIGHT
"""
    result = execute_and_verify_game_policy(code, env, max_steps=10)
    assert result["success"] is True
    assert result["steps_taken"] == 2
    assert result["final_reward"] == 1.0


def test_execute_and_verify_game_policy_failure():
    """壁に阻まれてゴールに到達できない失敗テスト."""
    env = GridWorldGameEnv(
        grid_shape=(5, 5),
        player_pos=(1, 1),
        goal_pos=(1, 4),
        walls={(1, 2)},
    )
    code = """
from acr_agi3.game.env import Action
def choose_action(obs):
    return Action.RIGHT
"""
    result = execute_and_verify_game_policy(code, env, max_steps=5)
    assert result["success"] is False
    assert result["steps_taken"] == 5
    assert result["final_reward"] <= 0.0


def test_execute_and_verify_game_policy_syntax_error():
    """構文エラーのハンドリングテスト."""
    env = GridWorldGameEnv(grid_shape=(3, 3), player_pos=(0, 0), goal_pos=(2, 2))
    code = "def choose_action(obs) invalid syntax"
    result = execute_and_verify_game_policy(code, env, max_steps=5)
    assert result["success"] is False
    assert "Syntax/Import error" in result["error"]


def test_local_transformers_llm_with_adk_runner():
    """Google ADK 2.0 Runner と LocalTransformersLlm の連携テスト."""

    async def _run():
        model = LocalTransformersLlm(model_name_or_path="mock")
        agent = Agent(name="test_runner_agent", model=model, instruction="Test instruction.")
        service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="test_app",
            session_service=service,
            auto_create_session=True,
        )

        msg = Content(role="user", parts=[Part.from_text(text="Synthesize action policy")])
        events = []
        async for event in runner.run_async(user_id="u1", session_id="s1", new_message=msg):
            events.append(event)
        return events

    events = asyncio.run(_run())
    assert len(events) > 0


def test_llm_game_agent_solve():
    """LLMGameAgent によるゲーム解決テスト (End-to-End)."""

    def mock_action_generator(prompt: str) -> str:
        return (
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs):\n"
            "    return Action.RIGHT\n"
            "```"
        )

    mock_llm = LocalTransformersLlm(
        model_name_or_path="mock-qwen",
        generation_fn=mock_action_generator,
    )
    agent = LLMGameAgent(llm=mock_llm)
    env = GridWorldGameEnv(grid_shape=(3, 3), initial_player_pos=(1, 0), goal_pos=(1, 1))
    res = agent.solve(env)

    assert res["is_solved"] is True
    assert res["policy_code"] is not None
