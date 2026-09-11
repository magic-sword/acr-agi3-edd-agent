"""Qwen2.5-VL 画像認識 × SKILL.md 連携型 ACR-AGI-3 ゲームプレイ自律エージェント."""

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

from acr_agi3.agent.llm.arc_tools import execute_and_verify_game_policy, extract_python_code
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.dsl.renderer import render_grid_to_image
from acr_agi3.game.env import GameEnvironment

logger = logging.getLogger(__name__)


class VLMGameAgent:
    """Qwen2.5-VL によるゲーム盤面画像認識と SKILL.md 連携自律ゲームエージェント."""

    def __init__(
        self,
        model: Optional[LocalQwenVL] = None,
        skills_dir: Optional[Path] = None,
        name: str = "acr_vlm_game_agent",
        app_name: str = "vlm_game_app",
    ) -> None:
        self.model = model or LocalQwenVL(model_name_or_path="mock")
        self.skills_dir = skills_dir or (Path(__file__).resolve().parent.parent.parent / "skills")
        self.name = name
        self.app_name = app_name

        self.adk_agent = Agent(
            name=self.name,
            model=self.model,
            instruction=(
                "You are an expert multimodal AI agent solving ARC-AGI-3 dynamic games. "
                "Analyze the rendered color game observation image, recognize affordances "
                "(agent, goal, obstacles, hazards), and synthesize an action policy "
                "`def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:` "
                "to clear the stage."
            ),
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent,
            app_name=self.app_name,
            session_service=self.session_service,
            auto_create_session=True,
        )

    def _prepare_game_parts(
        self,
        obs: np.ndarray,
        feedback: Optional[str] = None,
    ) -> List[Part]:
        """ゲーム観測グリッド画像を Part オブジェクトに変換."""
        parts: List[Part] = [
            Part.from_text(
                text=(
                    "Below is the current visual game board observation "
                    "rendered in official ARC-AGI colors:"
                )
            )
        ]


        img = render_grid_to_image(obs, cell_size=20)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        parts.append(
            Part(
                inline_data=Blob(
                    mime_type="image/png",
                    data=png_bytes,
                )
            )
        )


        instruction_text = (
            f"\nObservation grid shape: {obs.shape}.\n"
            "Synthesize a robust Python action policy function:\n"
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:\n"
            "    # Navigate safely avoiding obstacles to reach the goal\n"
            "    ...\n"
            "```\n"
            "Output only the Python code block."
        )
        if feedback:
            instruction_text = f"\n[PREVIOUS FAILURE FEEDBACK]:\n{feedback}\n" + instruction_text

        parts.append(Part.from_text(text=instruction_text))
        return parts

    async def _run_agent_turn(self, parts: List[Part], session_id: str) -> str:
        """マルチモーダルメッセージを Runner に送信しレスポンス取得."""
        message = Content(role="user", parts=parts)
        response_texts = []
        async for event in self.runner.run_async(
            user_id="vlm_player",
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
        env: GameEnvironment,
        max_steps: int = 50,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """画像認識を元に自律的に行動ポリシー合成・自己修正を実行."""
        session_id = f"vlm_game_sess_{np.random.randint(100000, 999999)}"
        feedback: Optional[str] = None
        obs = env.reset()

        best_code: Optional[str] = None
        best_verification: Dict[str, Any] = {"success": False}

        for iteration in range(max_iterations):
            parts = self._prepare_game_parts(obs, feedback=feedback)
            response_text = await self._run_agent_turn(parts, session_id=session_id)
            code = extract_python_code(response_text)

            verification = execute_and_verify_game_policy(code, env, max_steps=max_steps)

            if verification["success"]:
                logger.info(f"VLM game agent solved stage at iteration {iteration + 1}!")
                best_code = code
                best_verification = verification
                break
            else:
                err_msg = verification.get("error", "Failed to reach goal within limit.")
                feedback = f"Attempt {iteration + 1} failed: {err_msg}"
                logger.debug(f"VLM Iteration {iteration + 1} failed: {feedback}")

        return {
            "is_solved": best_verification.get("success", False),
            "code": best_code,
            "policy_code": best_code,
            "verification": best_verification,
        }


    def solve(
        self,
        env: GameEnvironment,
        max_steps: int = 50,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """同期インターフェース."""
        return asyncio.run(
            self.solve_async(
                env=env,
                max_steps=max_steps,
                max_iterations=max_iterations,
            )
        )


# 後方互換クラスエイリアス
VLMProgramSynthesisAgent = VLMGameAgent

