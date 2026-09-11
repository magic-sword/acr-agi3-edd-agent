"""Qwen2.5-VL 画像認識 × SKILL.md パイプラインの単体・結合テスト."""

import asyncio
import io
from pathlib import Path

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
        return "```python\ndef transform(grid):\n    return np.rot90(grid)\n```"

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
    assert "def transform" in text
    assert "np.rot90" in text


def test_vlm_program_synthesis_agent_end_to_end(tmp_path: Path):
    """VLMProgramSynthesisAgent の解法テスト (SKILL.md 読み込み + 画像入力)."""
    # 一時的なスキルディレクトリと SKILL.md を作成
    skill_dir = tmp_path / "skills" / "mock-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: mock-skill\n---\nUse np.fliplr to flip horizontally.",
        encoding="utf-8",
    )

    def mock_generate(prompt: str, images=None) -> str:
        # プロンプト内に SKILL.md の記述が含まれているか確認
        return "```python\ndef transform(grid):\n    return np.fliplr(grid)\n```"

    mock_vlm = LocalQwenVL(model_name_or_path="mock", generate_fn=mock_generate)
    agent = VLMProgramSynthesisAgent(
        model=mock_vlm,
        skills_dir=tmp_path / "skills",
    )

    assert "mock-skill" in agent.skills_context

    train_pairs = [
        {
            "input": np.array([[1, 2, 3], [4, 5, 6]]),
            "output": np.array([[3, 2, 1], [6, 5, 4]]),
        }
    ]
    test_input = np.array([[7, 8, 9], [0, 1, 2]])
    expected_output = np.array([[9, 8, 7], [2, 1, 0]])

    predictions = agent.solve(train_pairs=train_pairs, test_input=test_input)
    assert len(predictions) == 1
    assert np.array_equal(predictions[0], expected_output)
