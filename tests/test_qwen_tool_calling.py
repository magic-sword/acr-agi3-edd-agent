"""LocalQwenVL アダプターのネイティブ Tool Calling 相互変換テスト."""

import asyncio
import json
import pytest
from google.adk.agents import Agent
from google.adk.models import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    Schema,
    Tool,
    Type,
)

from acr_agi3.agent.llm.local_vlm import LocalQwenVL


def test_declaration_to_schema_conversion():
    """ADK FunctionDeclaration が OpenAI/Qwen 互換の JSON Schema 形式に正しく変換されること."""
    fd = FunctionDeclaration(
        name="inspect_board",
        description="Inspects board geometry and colors.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "detail_level": Schema(type=Type.STRING, description="Inspection detail level: low/high")
            },
            required=["detail_level"],
        ),
    )

    schema = LocalQwenVL._declaration_to_schema(fd)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "inspect_board"
    assert schema["function"]["description"] == "Inspects board geometry and colors."
    assert "detail_level" in schema["function"]["parameters"]["properties"]


def test_format_tools_for_qwen():
    """Tool オブジェクトから Qwen 公式の <tools> ... </tools> XML ブロックが生成されること."""
    fd = FunctionDeclaration(
        name="load_skill",
        description="Loads skill instructions.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={"skill_name": Schema(type=Type.STRING, description="Skill name")},
            required=["skill_name"],
        ),
    )
    tool = Tool(function_declarations=[fd])

    vlm = LocalQwenVL(model_name_or_path="mock")
    prompt_block = vlm._format_tools_for_qwen([tool])

    assert "<tools>" in prompt_block
    assert "</tools>" in prompt_block
    assert "load_skill" in prompt_block
    assert "<tool_call>" in prompt_block


def test_detect_tool_call_from_qwen_xml():
    """Qwen の <tool_call> XML タグからツール名と引数が正しくパースされること."""
    vlm = LocalQwenVL(model_name_or_path="mock")

    raw_output = """
Thinking: I need to load the visual inspector skill first.
<tool_call>
{"name": "load_skill", "arguments": {"skill_name": "visual-inspector"}}
</tool_call>
"""
    result = vlm._detect_tool_call(raw_output)
    assert result is not None
    name, args = result
    assert name == "load_skill"
    assert args == {"skill_name": "visual-inspector"}


def test_qwen_tool_call_generates_adk_function_call_part():
    """LocalQwenVL が <tool_call> を検知した際、ADK の Part.from_function_call を生成すること."""
    async def run_test():
        async def mock_gen(prompt, images=None):
            return '<tool_call>{"name": "inspect_affordances", "arguments": {"mode": "deep"}}</tool_call>'

        vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=mock_gen)

        req = LlmRequest(
            contents=[Content(role="user", parts=[Part.from_text(text="Analyze grid")])],
            config=GenerateContentConfig(),
        )

        responses = []
        async for resp in vlm.generate_content_async(req):
            responses.append(resp)

        assert len(responses) == 1
        content = responses[0].content
        assert content.role == "model"
        part = content.parts[0]
        assert part.function_call is not None
        assert part.function_call.name == "inspect_affordances"
        assert part.function_call.args == {"mode": "deep"}

    asyncio.run(run_test())


def test_end_to_end_react_cycle_with_runner():
    """ADK Runner と連携し、エージェントがツール呼び出し ➔ ツール実行結果受取 ➔ 最終回答を生成する ReAct ループ."""
    async def run_test():
        tool_executed = False

        def dummy_tool(query: str) -> str:
            """A simple test tool."""
            nonlocal tool_executed
            tool_executed = True
            return f"Tool result for {query}"

        # 1手目はツール呼び出し、2手目はツール結果を踏まえた最終回答
        turn_count = 0

        async def mock_vlm_react(prompt, images=None):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                return '<tool_call>{"name": "dummy_tool", "arguments": {"query": "grid_state"}}</tool_call>'
            else:
                return "Final decision: Based on Tool result for grid_state, I choose ACTION1."

        vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=mock_vlm_react)

        agent = Agent(
            name="react_agent",
            model=vlm,
            tools=[dummy_tool],
            instruction="Use dummy_tool before deciding action.",
        )

        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="test_react_app",
            session_service=session_service,
            auto_create_session=True,
        )

        events = []
        async for ev in runner.run_async(
            session_id="s1",
            user_id="u1",
            new_message=Content(role="user", parts=[Part.from_text(text="Start task")]),
        ):
            events.append(ev)

        assert tool_executed is True
        assert turn_count == 2

    asyncio.run(run_test())
