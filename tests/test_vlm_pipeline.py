"""Qwen2.5-VL 画像認識 × SKILL.md パイプラインの単体・結合テスト."""

import asyncio
import io

import numpy as np
from PIL import Image

from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.agent.vlm_agent import VLMProgramSynthesisAgent
from acr_agi3.dsl.renderer import ARC_COLORS, render_grid_to_image, render_task_pair


def test_render_grid_to_image():
    """グリッド配列の画像レンダリングテスト."""
    grid = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ]
    )
    cell_size = 10
    img = render_grid_to_image(grid, cell_size=cell_size, grid_line_width=1)

    assert isinstance(img, Image.Image)
    assert img.size == (3 * cell_size, 2 * cell_size)
    assert img.mode == "RGB"

    # セルの中央ピクセル色を検証
    center_color_0 = img.getpixel((cell_size // 2, cell_size // 2))
    assert center_color_0 == ARC_COLORS[0]

    center_color_1 = img.getpixel((cell_size + cell_size // 2, cell_size // 2))
    assert center_color_1 == ARC_COLORS[1]


def test_render_task_pair():
    """Input-Output ペアの比較画像生成テスト."""
    inp = np.array([[1, 2], [3, 4]])
    out = np.array([[4, 3], [2, 1]])

    pair_img = render_task_pair(inp, out, cell_size=12, spacing=20)
    assert isinstance(pair_img, Image.Image)
    assert pair_img.width > inp.shape[1] * 12 * 2
    assert pair_img.mode == "RGB"


def test_local_qwen_vl_mock():
    """LocalQwenVL のモック推論テスト."""

    def mock_generate(prompt: str, images=None) -> str:
        assert images is not None
        assert len(images) > 0
        return (
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs, info=None):\n"
            "    return Action.RIGHT\n"
            "```"
        )

    vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=mock_generate)

    # ダミー画像を作成
    dummy_img = Image.new("RGB", (20, 20), color="blue")
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")

    from google.adk.models import LlmRequest
    from google.genai.types import Blob, Content, Part

    req = LlmRequest(
        model="mock",
        contents=[
            Content(
                role="user",
                parts=[
                    Part.from_text(text="Analyze image:"),
                    Part(inline_data=Blob(mime_type="image/png", data=buf.getvalue())),
                ],
            )
        ],
    )

    async def _test():
        responses = []
        async for resp in vlm.generate_content_async(req):
            responses.append(resp)
        return responses

    responses = asyncio.run(_test())
    assert len(responses) == 1
    text = responses[0].content.parts[0].text
    assert "def choose_action" in text
    assert "Action.RIGHT" in text


def test_vlm_game_agent_end_to_end():
    """VLMGameAgent によるゲーム環境の視覚認識とポリシー推論解決テスト."""
    from acr_agi3.game.vcgt_game import GridWorldGameEnv

    def mock_generate(prompt: str, images=None) -> str:
        return (
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs, info=None):\n"
            "    return Action.RIGHT\n"
            "```"
        )

    mock_vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=mock_generate)
    agent = VLMProgramSynthesisAgent(model=mock_vlm)

    env = GridWorldGameEnv(grid_shape=(3, 3), initial_player_pos=(1, 0), goal_pos=(1, 1))
    res = agent.solve(env=env, max_steps=10)

    assert res["is_solved"] is True
    assert res["code"] is not None


