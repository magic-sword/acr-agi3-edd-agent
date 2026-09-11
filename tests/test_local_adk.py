"""Google ADK 2.0 × ローカル LLM の結合・単体テスト."""

import asyncio

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.arc_tools import (
    execute_and_verify_code,
    extract_python_code,
)
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.agent.llm_agent import LLMProgramSynthesisAgent


def test_extract_python_code():
    """マークダウンコードブロックからの Python 抽出テスト."""
    text1 = (
        "Here is the code:\n```python\ndef transform(grid):\n    return np.rot90(grid)\n```\nDone."
    )
    assert extract_python_code(text1) == "def transform(grid):\n    return np.rot90(grid)"

    text2 = "```\ndef transform(grid):\n    return grid * 2\n```"
    assert extract_python_code(text2) == "def transform(grid):\n    return grid * 2"

    text3 = "def transform(grid):\n    return grid"
    assert extract_python_code(text3) == "def transform(grid):\n    return grid"


def test_execute_and_verify_code_success():
    """正常に正解するコードの検証テスト."""
    code = """
def transform(grid):
    return np.rot90(grid, k=-1)
"""
    train_pairs = [
        {
            "input": np.array([[1, 2], [3, 4]]),
            "output": np.rot90(np.array([[1, 2], [3, 4]]), k=-1),
        },
        {
            "input": np.array([[5, 6], [7, 8]]),
            "output": np.rot90(np.array([[5, 6], [7, 8]]), k=-1),
        },
    ]

    result = execute_and_verify_code(code, train_pairs)
    assert result["is_valid"] is True
    assert result["passed_count"] == 2
    assert result["total_count"] == 2
    assert len(result["failures"]) == 0


def test_execute_and_verify_code_failure():
    """不正解コードの検証テスト."""
    code = """
def transform(grid):
    return grid  # 恒等変換 (正解ではない)
"""
    train_pairs = [
        {
            "input": np.array([[1, 2], [3, 4]]),
            "output": np.rot90(np.array([[1, 2], [3, 4]]), k=-1),
        }
    ]

    result = execute_and_verify_code(code, train_pairs)
    assert result["is_valid"] is False
    assert result["passed_count"] == 0
    assert len(result["failures"]) == 1
    assert "Grid values mismatch" in result["failures"][0]["reason"]


def test_execute_and_verify_code_syntax_error():
    """構文エラーのハンドリングテスト."""
    code = "def transform(grid) return invalid syntax"
    train_pairs = [{"input": np.array([[1]]), "output": np.array([[1]])}]

    result = execute_and_verify_code(code, train_pairs)
    assert result["is_valid"] is False
    assert "Syntax/Compilation error" in result["error"]


def test_local_transformers_llm_with_adk_runner():
    """Google ADK 2.0 Runner と LocalTransformersLlm の連携テスト."""

    async def _run():

        def mock_generate(prompt: str) -> str:
            return "```python\ndef transform(grid):\n    return np.fliplr(grid)\n```"

        llm = LocalTransformersLlm(
            model_name_or_path="mock-qwen-coder",
            generate_fn=mock_generate,
        )

        agent = Agent(
            name="test_agent",
            model=llm,
            instruction="Solve ARC problem",
        )
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="test_app",
            session_service=session_service,
            auto_create_session=True,
        )

        message = Content(role="user", parts=[Part.from_text(text="Please write transform code.")])
        events = []
        async for event in runner.run_async(
            user_id="tester",
            session_id="session_test_1",
            new_message=message,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    assert len(events) > 0
    # レスポンスに生成テキストが含まれていることを確認
    collected_text = ""
    for ev in events:
        if hasattr(ev, "content") and ev.content:
            for part in getattr(ev.content, "parts", []):
                if hasattr(part, "text") and part.text:
                    collected_text += part.text

    assert "def transform" in collected_text
    assert "np.fliplr" in collected_text


def test_llm_program_synthesis_agent_solve():
    """LLMProgramSynthesisAgent によるタスク解決テスト (End-to-End)."""

    # 左右反転の正解コードを返すモック
    def mock_correct_generator(prompt: str) -> str:
        return "```python\ndef transform(grid):\n    return np.fliplr(grid)\n```"

    mock_llm = LocalTransformersLlm(
        model_name_or_path="mock-qwen",
        generate_fn=mock_correct_generator,
    )

    synthesis_agent = LLMProgramSynthesisAgent(model=mock_llm)

    train_pairs = [
        {
            "input": np.array([[1, 2, 3], [4, 5, 6]]),
            "output": np.array([[3, 2, 1], [6, 5, 4]]),
        }
    ]
    test_input = np.array([[7, 8, 9], [0, 1, 2]])
    expected_output = np.array([[9, 8, 7], [2, 1, 0]])

    predictions = synthesis_agent.solve(train_pairs=train_pairs, test_input=test_input)

    assert len(predictions) == 1
    assert np.array_equal(predictions[0], expected_output)
