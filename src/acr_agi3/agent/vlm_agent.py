"""Qwen2.5-VL 画像認識 × SKILL.md 連携型 ARC 自律プログラミングエージェント."""

import asyncio
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Blob, Content, Part

from acr_agi3.agent.llm.arc_tools import execute_and_verify_code, extract_python_code
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.dsl.renderer import render_task_pair

logger = logging.getLogger(__name__)


class VLMProgramSynthesisAgent:
    """Qwen2.5-VL によるグリッド画像認識と SKILL.md 連携自律エージェント."""

    def __init__(
        self,
        model: Optional[LocalQwenVL] = None,
        skills_dir: Optional[Path] = None,
        name: str = "arc_vlm_synthesis_agent",
        app_name: str = "arc_vlm_app",
    ) -> None:
        """初期化.

        Args:
            model: ADK BaseLlm 互換のローカル VLM (未指定時はモック)
            skills_dir: SKILL.md を含むスキルディレクトリのパス
            name: エージェント名
            app_name: ADK アプリ名
        """
        self.model = model or LocalQwenVL(model_name_or_path="mock")
        self.skills_dir = skills_dir or (Path(__file__).resolve().parent.parent.parent / "skills")
        self.name = name
        self.app_name = app_name

        # 利用可能な SKILL.md をロード
        self.skills_context = self._load_available_skills()

        # Google ADK 2.0 Agent の初期化
        instruction = (
            "You are a multimodal ARC-AGI solver. "
            "You are provided with images illustrating input-output grid transformations, "
            "along with skill specifications (SKILL.md) and Python primitives. "
            "Carefully analyze the visual geometry, color changes, and symmetry in the images. "
            "Then, synthesize a Python function `def transform(grid: np.ndarray) -> np.ndarray:` "
            "that accurately performs the transformation. Only output valid Python code."
        )
        if self.skills_context:
            instruction += f"\n\n--- AVAILABLE DOMAIN SKILLS (SKILL.md) ---\n{self.skills_context}"

        self.adk_agent = Agent(
            name=self.name,
            model=self.model,
            instruction=instruction,
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent,
            app_name=self.app_name,
            session_service=self.session_service,
            auto_create_session=True,
        )

    def _load_available_skills(self) -> str:
        """skills/ ディレクトリ下の SKILL.md ファイルを再帰的に読み込む."""
        skill_texts: List[str] = []
        if not self.skills_dir.exists():
            return ""

        for skill_md in self.skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_md.read_text(encoding="utf-8")
                skill_texts.append(f"### Skill: {skill_md.parent.name}\n{content}\n")
            except Exception as e:
                logger.warning(f"Failed to read {skill_md}: {e}")

        return "\n".join(skill_texts)

    def _prepare_task_parts(
        self,
        train_pairs: List[Dict[str, Any]],
        feedback: Optional[str] = None,
    ) -> List[Part]:
        """Train ペアの比較画像をインメモリ生成し、ADK Part のリストを作成."""
        parts: List[Part] = []

        prompt_text = "Here are the visual training examples of the grid transformation:\n"
        parts.append(Part.from_text(text=prompt_text))

        for idx, pair in enumerate(train_pairs):
            inp = np.array(pair["input"], dtype=int)
            out = np.array(pair["output"], dtype=int)

            # 横並びの比較カラー画像を生成
            pair_img = render_task_pair(inp, out, cell_size=24)

            # PNG バイトに変換
            buf = io.BytesIO()
            pair_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            parts.append(Part.from_text(text=f"--- Example {idx + 1} (Input -> Output) ---"))
            parts.append(
                Part(
                    inline_data=Blob(
                        mime_type="image/png",
                        data=img_bytes,
                    )
                )
            )

        if feedback:
            parts.append(Part.from_text(text=f"--- PREVIOUS ATTEMPT FEEDBACK ---\n{feedback}\n"))
            parts.append(
                Part.from_text(
                    text="Please fix the code using visual insights and available skills. "
                    "Output the corrected `def transform(grid: np.ndarray) -> np.ndarray:`."
                )
            )
        else:
            parts.append(
                Part.from_text(
                    text=(
                        "Based on the images and SKILL.md, write "
                        "`def transform(grid: np.ndarray) -> np.ndarray:`. "
                        "Provide only the Python code."
                    )
                )
            )

        return parts

    async def _run_agent_turn(self, parts: List[Part], session_id: str) -> str:
        """Google ADK Runner 経由でマルチモーダル推論を実行."""
        message = Content(
            role="user",
            parts=parts,
        )

        response_texts: List[str] = []
        async for event in self.runner.run_async(
            user_id="arc_vlm_evaluator",
            session_id=session_id,
            new_message=message,
        ):
            if hasattr(event, "content") and event.content:
                for part in getattr(event.content, "parts", []):
                    if hasattr(part, "text") and part.text:
                        response_texts.append(part.text)

        return "".join(response_texts)

    async def solve_async(
        self,
        train_pairs: List[Dict[str, Any]],
        test_input: np.ndarray,
        max_iterations: int = 3,
    ) -> List[np.ndarray]:
        """画像とスキルを元に自律的にコード合成・自己修正を実行."""
        session_id = f"vlm_sess_{np.random.randint(100000, 999999)}"
        feedback: Optional[str] = None
        best_code: Optional[str] = None

        for iteration in range(max_iterations):
            parts = self._prepare_task_parts(train_pairs, feedback=feedback)
            response_text = await self._run_agent_turn(parts, session_id=session_id)
            code = extract_python_code(response_text)

            verification = execute_and_verify_code(code, train_pairs)

            if verification["is_valid"]:
                logger.info(f"VLM found valid code at iteration {iteration + 1}!")
                best_code = code
                break
            else:
                failures = verification.get("failures", [])
                error_msg = verification.get("error", "")
                if failures:
                    feedback = f"Failed {len(failures)} pairs: " + "; ".join(
                        f"Pair {f['pair_index']}: {f['reason']}" for f in failures[:2]
                    )
                else:
                    feedback = f"Execution error: {error_msg}"
                logger.debug(f"VLM Iteration {iteration + 1} failed: {feedback}")

        if best_code:
            local_scope: Dict[str, Any] = {"np": np}
            try:
                exec(best_code, {"np": np, "__builtins__": __builtins__}, local_scope)
                pred = local_scope["transform"](test_input.copy())
                return [np.array(pred, dtype=int)]
            except Exception as e:
                logger.warning(f"Failed to execute best code on test input: {e}")

        return [test_input.copy()]

    def solve(
        self,
        train_pairs: List[Dict[str, Any]],
        test_input: np.ndarray,
        max_iterations: int = 3,
    ) -> List[np.ndarray]:
        """同期インターフェース."""
        return asyncio.run(
            self.solve_async(
                train_pairs=train_pairs,
                test_input=test_input,
                max_iterations=max_iterations,
            )
        )
