"""Google ADK 2.0 をベースにした ACR-AGI-3 ゲームプレイ自律エージェント."""

import asyncio
import logging
from typing import Any, Dict, Optional

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.arc_tools import execute_and_verify_game_policy, extract_python_code
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.game.env import GameEnvironment

logger = logging.getLogger(__name__)


class LLMGameAgent:
    """Google ADK 2.0 とローカル LLM による動的ゲーム行動ポリシー自律開発エージェント."""

    def __init__(
        self,
        model: Optional[LocalTransformersLlm] = None,
        llm: Optional[LocalTransformersLlm] = None,
        name: str = "acr_agi3_game_agent",
        app_name: str = "game_app",
    ) -> None:
        self.model = model or llm or LocalTransformersLlm(model_name_or_path="mock")
        self.name = name
        self.app_name = app_name

        self.adk_agent = Agent(
            name=self.name,
            model=self.model,
            instruction=(
                "You are an expert AI game agent solving ARC-AGI-3 interactive games. "
                "Analyze the environment observation and synthesize an action policy "
                "`def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:` "
                "that navigates safely avoiding obstacles and reaches the goal."
            ),
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent,
            app_name=self.app_name,
            session_service=self.session_service,
            auto_create_session=True,
        )

    async def _run_agent_turn(self, prompt: str, session_id: str) -> str:
        """Google ADK Runner 経由で 1 ターンの推論を実行."""
        message = Content(
            role="user",
            parts=[Part.from_text(text=prompt)],
        )

        response_texts = []
        async for event in self.runner.run_async(
            user_id="game_player",
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
        """ゲーム環境を自律プレイして攻略コードを合成."""
        session_id = f"session_game_{np.random.randint(100000, 999999)}"
        feedback: Optional[str] = None
        obs = env.reset()

        base_prompt = (
            f"Solve ARC-AGI-3 dynamic game: grid shape is {obs.shape}.\n"
            "Write a Python function:\n"
            "`def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:`\n"
            "Actions: Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, "
            "Action.INTERACT, Action.WAIT.\n"
            "Provide only the Python code block."
        )

        best_code: Optional[str] = None
        best_verification: Dict[str, Any] = {"success": False}

        for iteration in range(max_iterations):
            prompt = base_prompt if not feedback else f"{base_prompt}\n\n[FEEDBACK]: {feedback}"
            response_text = await self._run_agent_turn(prompt, session_id=session_id)
            code = extract_python_code(response_text)

            verification = execute_and_verify_game_policy(code, env, max_steps=max_steps)

            if verification["success"]:
                logger.info(f"Solved game at iteration {iteration + 1}!")
                best_code = code
                best_verification = verification
                break
            else:
                err = verification.get("error", "Failed to reach goal within step limit.")
                feedback = (
                    f"Attempt failed: {err}. Please adjust navigation logic to avoid obstacles."
                )
                logger.debug(f"Iteration {iteration + 1} failed: {feedback}")

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
LLMProgramSynthesisAgent = LLMGameAgent
