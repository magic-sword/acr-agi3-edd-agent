"""Google ADK 2.0 をベースにした ARC 自律プログラミングエージェント."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.arc_tools import execute_and_verify_code, extract_python_code
from acr_agi3.agent.llm.local_model import LocalTransformersLlm

logger = logging.getLogger(__name__)


class LLMProgramSynthesisAgent:
    """Google ADK 2.0 とローカル LLM によるプログラム合成・自己修正エージェント."""

    def __init__(
        self,
        model: Optional[LocalTransformersLlm] = None,
        name: str = "arc_code_synthesis_agent",
        app_name: str = "arc_app",
    ) -> None:
        """初期化.

        Args:
            model: ADK BaseLlm 互換のローカル推論モデル (未指定時はモック)
            name: エージェント名
            app_name: ADK アプリケーション名
        """
        self.model = model or LocalTransformersLlm(model_name_or_path="mock")
        self.name = name
        self.app_name = app_name

        # Google ADK 2.0 Agent の初期化
        self.adk_agent = Agent(
            name=self.name,
            model=self.model,
            instruction=(
                "You are an expert Python programmer solving ARC-AGI grid transformation puzzles. "
                "Analyze the given input-output training pairs. "
                "Write a Python function `def transform(grid: np.ndarray) -> np.ndarray:` "
                "that correctly transforms each input grid into its corresponding output grid. "
                "Use numpy (imported as np). Only output valid Python code."
            ),
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent,
            app_name=self.app_name,
            session_service=self.session_service,
            auto_create_session=True,
        )


    def _format_grid(self, grid: np.ndarray) -> str:
        """グリッドを可読性の高い文字列に変換."""
        lines = ["[" + " ".join(f"{val:2d}" for val in row) + "]" for row in grid]
        return "\n".join(lines)

    def _build_task_prompt(
        self,
        train_pairs: List[Dict[str, np.ndarray]],
        feedback: Optional[str] = None,
    ) -> str:
        """タスク情報から LLM 向けプロンプトを構築."""
        lines: List[str] = [
            "Given the following ARC grid transformation examples:",
            "",
        ]

        for i, pair in enumerate(train_pairs):
            lines.append(f"--- Example {i + 1} ---")
            lines.append("Input grid:")
            lines.append(self._format_grid(np.array(pair["input"], dtype=int)))
            lines.append("Output grid:")
            lines.append(self._format_grid(np.array(pair["output"], dtype=int)))
            lines.append("")

        if feedback:
            lines.append("--- PREVIOUS ATTEMPT FEEDBACK ---")
            lines.append(feedback)
            lines.append(
                "Please fix the code so it passes all examples. "
                "Return the updated function `def transform(grid: np.ndarray) -> np.ndarray:`."
            )
        else:
            lines.append(
                "Write a Python function `def transform(grid: np.ndarray) -> np.ndarray:` "
                "that reproduces the transformation. Provide only the Python code."
            )

        return "\n".join(lines)

    async def _run_agent_turn(self, prompt: str, session_id: str) -> str:
        """Google ADK Runner 経由で 1 ターンの推論を実行."""
        message = Content(
            role="user",
            parts=[Part.from_text(text=prompt)],
        )

        response_texts: List[str] = []
        async for event in self.runner.run_async(
            user_id="arc_evaluator",
            session_id=session_id,
            new_message=message,
        ):
            # イベントからモデルのレスポンスを抽出
            if hasattr(event, "content") and event.content:
                for part in getattr(event.content, "parts", []):
                    if hasattr(part, "text") and part.text:
                        response_texts.append(part.text)

        return "".join(response_texts)

    async def solve_async(
        self,
        train_pairs: List[Dict[str, np.ndarray]],
        test_input: np.ndarray,
        max_iterations: int = 3,
    ) -> List[np.ndarray]:
        """タスクを自律的に解き、テスト入力に対する予測を生成."""
        session_id = f"session_arc_{np.random.randint(100000, 999999)}"
        feedback: Optional[str] = None
        best_code: Optional[str] = None

        for iteration in range(max_iterations):
            prompt = self._build_task_prompt(train_pairs, feedback=feedback)
            response_text = await self._run_agent_turn(prompt, session_id=session_id)
            code = extract_python_code(response_text)

            verification = execute_and_verify_code(code, train_pairs)

            if verification["is_valid"]:
                logger.info(
                    f"Found valid transformation code at iteration {iteration + 1}!"
                )
                best_code = code
                break
            else:
                # 失敗要因を分析して次のプロンプトのフィードバックとする (Self-Correction)
                failures = verification.get("failures", [])
                error_msg = verification.get("error", "")
                if failures:
                    feedback = f"Failed on {len(failures)} pairs: " + "; ".join(
                        f"Pair {f['pair_index']}: {f['reason']}" for f in failures
                    )
                else:
                    feedback = f"Execution error: {error_msg}"
                logger.debug(f"Iteration {iteration + 1} failed: {feedback}")

        # 正解コードが見つかった場合はテスト入力に適用
        if best_code:
            local_scope: Dict[str, Any] = {"np": np}
            try:
                exec(best_code, {"np": np, "__builtins__": __builtins__}, local_scope)
                transform_fn = local_scope["transform"]
                pred = transform_fn(test_input.copy())
                return [np.array(pred, dtype=int)]
            except Exception as e:
                logger.warning(f"Failed to execute best code on test input: {e}")

        # フォールバック (恒等変換)
        return [test_input.copy()]

    def solve(
        self,
        train_pairs: List[Dict[str, np.ndarray]],
        test_input: np.ndarray,
        max_iterations: int = 3,
    ) -> List[np.ndarray]:
        """同期インターフェース (ARCOrchestrator 互換)."""
        return asyncio.run(
            self.solve_async(
                train_pairs=train_pairs,
                test_input=test_input,
                max_iterations=max_iterations,
            )
        )
