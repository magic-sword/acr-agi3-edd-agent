"""メタスキル駆動型自律プログラミングエージェント (Meta-Skill Driven Synthesis Agent).

MetaObserver, SubgoalDecomposer, VCGTDataset, FailureDiagnoser を Google ADK 2.0 に統合し、
高精度な自己改善ループ（Self-Correction）を実現します。
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.arc_tools import (
    execute_and_verify_game_policy,
    extract_python_code,
)
from acr_agi3.agent.llm.edd_tools import (
    edd_execute_game_skill,
    edd_init_skill,
    edd_list_skills,
    edd_register_verified_skill,
    edd_run_game_contract_test,
    edd_validate_skill,
    edd_write_skill_code,
)
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.game.env import GameEnvironment
from acr_agi3.meta.decomposer import SubgoalDecomposer
from acr_agi3.meta.human_vcgt import VCGTDataset
from acr_agi3.meta.observer import MetaObserver

logger = logging.getLogger(__name__)


class MetaSkillDrivenAgent:
    """メタスキル（Observer, Decomposer, VCGT, Diagnoser, EDD）を統合した自己改善エージェント."""

    def __init__(
        self,
        model: Optional[LocalTransformersLlm] = None,
        vcgt_path: Optional[Path] = None,
        name: str = "meta_skill_agent",
        app_name: str = "meta_arc_app",
    ) -> None:
        self.model = model or LocalTransformersLlm(model_name_or_path="mock")
        self.name = name
        self.app_name = app_name

        # メタスキル層のコンポーネント
        self.observer = MetaObserver()
        self.decomposer = SubgoalDecomposer(observer=self.observer)

        # EDD ツールセット (ライブラリ検索・実行ツールを含む)
        self.edd_tools = [
            edd_init_skill,
            edd_validate_skill,
            edd_write_skill_code,
            edd_run_game_contract_test,
            edd_list_skills,
            edd_execute_game_skill,
            edd_register_verified_skill,
        ]

        # VCGT データセット（Few-shot 思考例）
        self.vcgt_dataset = None
        if vcgt_path and vcgt_path.exists():
            try:
                self.vcgt_dataset = VCGTDataset.load_from_json(vcgt_path)
            except Exception as e:
                logger.warning(f"Failed to load VCGT dataset from {vcgt_path}: {e}")

        # Google ADK 2.0 Agent (EDD ツールをバインド)
        self.adk_agent = Agent(
            name=self.name,
            model=self.model,
            tools=self.edd_tools,
            instruction=(
                "You are an elite AI researcher and programmer solving ACR-AGI-3 interactive games "
                "guided by Human Visual Concept Guided Thinking (VCGT) and "
                "Evaluation-Driven Development (EDD).\n"
                "You have access to EDD tools: edd_init_skill, edd_write_skill_code, "
                "edd_validate_skill, edd_run_game_contract_test, edd_list_skills, "
                "edd_execute_game_skill, edd_register_verified_skill.\n"
                "Break down game tasks into subgoals, create verified skills for each subgoal, "
                "and compose them to solve the game."
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

    def solve(
        self,
        env: GameEnvironment,
        max_steps: int = 50,
        task_id: str = "game_task",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """ゲーム環境に対する自己解決 (solve_game へのエイリアス)."""
        return self.solve_game(env, max_steps, task_id, max_retries)

    async def _run_agent_turn(self, prompt: str, session_id: str) -> str:
        """Google ADK Runner 経由で推論実行."""
        content = Content(
            role="user",
            parts=[Part.from_text(text=prompt)],
        )
        response_text = ""
        async for event in self.runner.run_async(
            user_id="arc_solver",
            session_id=session_id,
            new_message=content,
        ):
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text

        return response_text


    def solve_game(
        self,
        env: GameEnvironment,
        max_steps: int = 50,
        task_id: str = "game_task",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """ACR-AGI-3 ゲーム環境に対するメタスキル駆動型解決 (スキル再利用・合成ループ対応)."""
        initial_obs = env.reset()
        aff_report = self.observer.analyze_frame(initial_obs)
        plan = self.decomposer.decompose_game(initial_obs)

        logger.info(f"Task {task_id}: Generated {plan.total_steps} subgoals.")
        for sg in plan.subgoals:
            logger.info(f"  [Subgoal {sg.index}] {sg.name}: {sg.objective}")

        # 1. 蓄積された検証済みスキルライブラリの取得
        verified_skills = edd_list_skills(verified_only=True)
        library_section = ""
        if verified_skills:
            library_lines = [
                "## Available Verified Skill Library (You can reuse or compose these):"
            ]
            for vs in verified_skills:
                library_lines.append(f"- Skill '{vs['name']}': {vs['description']}")
            library_lines.append(
                "You can compose these verified skills as subroutines or build on their logic.\n"
            )
            library_section = "\n".join(library_lines)

        base_prompt = (
            f"Solve ARC-AGI-3 dynamic game: {plan.task_hint}\n"
            f"Environment shape: {aff_report.grid_shape}, "
            f"Background: {aff_report.background_color}\n"
            f"{library_section}\n"
            f"Subgoals:\n"
            + "\n".join(
                f"- {s.name}: {s.objective} (Reasoning: {s.reasoning})" for s in plan.subgoals
            )
            + "\n\nConstraints:\n"
            + "\n".join(f"- {c}" for c in plan.constraints)
            + "\n\nWrite a complete Python action policy function:\n"
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:\n"
            "    # Navigate avoiding obstacles and reach the goal\n"
            "    ...\n"
            "```\n"
            "Provide only the Python code block."
        )

        feedback = None
        code = ""
        verification: Dict[str, Any] = {"success": False}

        for attempt in range(1, max_retries + 1):
            prompt = (
                base_prompt
                if not feedback
                else f"{base_prompt}\n\n[DIAGNOSTIC FEEDBACK]:\n{feedback}"
            )
            session_id = f"game_sess_{task_id}_{attempt}"
            resp = asyncio.run(self._run_agent_turn(prompt, session_id=session_id))
            code = extract_python_code(resp)

            # シミュレーション検証
            verification = execute_and_verify_game_policy(code, env, max_steps=max_steps)

            if verification["success"]:
                # 2. 合格したスキルをライブラリに正式登録
                skill_name = f"policy_{task_id}"
                edd_init_skill(skill_name)
                edd_write_skill_code(skill_name, code)
                edd_register_verified_skill(
                    name=skill_name,
                    description=plan.task_hint,
                    tags=["game_policy", "verified_solution"],
                )
                logger.info(f"Registered verified skill '{skill_name}' to skill library.")
                break

            # 失敗診断
            err_msg = verification.get("error", "Policy failed to reach goal within step limit.")
            steps_done = verification.get("steps_taken", 0)
            feedback = (
                f"Attempt {attempt} failed after {steps_done} steps. Error: {err_msg}. "
                "Adjust detour logic, obstacle margin, or step order to avoid obstacles."
            )

        return {
            "task_id": task_id,
            "is_solved": verification["success"],
            "verification": verification,
            "code": code,
            "policy_code": code,
            "steps_taken": verification.get("steps_taken", 0),
            "plan": plan,
            "aff_report": aff_report,
            "available_skills_count": len(verified_skills),
        }

