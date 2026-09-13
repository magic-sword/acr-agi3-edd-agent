"""Google ADK 2.0 準拠 Progressive Disclosure メタスキル駆動型自律エージェント.

3段階の段階的開示 (Progressive Disclosure) によりコンテキスト消費を抑制しつつ、
タスク状況に応じて必要なメタスキル (Observer, Intuitor, Decomposer, Tester, Diagnoser) を
動的にトリガー・ロード・実行して自己修復ループを回します。
"""

from __future__ import annotations

import asyncio
import json
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
from acr_agi3.agent.human_vcgt import VCGTDataset
from acr_agi3.meta.skill_harness import SkillHarness

logger = logging.getLogger(__name__)


class MetaSkillDrivenAgent:
    """Google ADK 2.0 準拠 3段階 Progressive Disclosure メタスキル自己改善エージェント."""

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

        # 3段階 Progressive Disclosure スキルハーネス
        self.harness = SkillHarness()

        # Google ADK 2.0 公式 SkillToolset (Progressive Disclosure L1/L2/L3)
        self.skill_toolset = self.harness.get_toolset()

        # ADK ツール群 (SkillToolset + EDD ツール群)
        self.tools = [
            self.skill_toolset,
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

        # Google ADK 2.0 Agent
        instruction = (
            "You are an elite AI researcher solving ACR-AGI-3 interactive games.\n"
            "You operate under the Progressive Disclosure (3-tier) Skill Architecture powered by Google ADK 2.0:\n"
            "- Level 1: Use `list_skills` to inspect available skills with minimal context consumption.\n"
            "- Level 2: Call `load_skill` to open a skill's SKILL.md body on-demand.\n"
            "- Level 3: Call `run_skill_script` or `load_skill_resource` to execute deterministic scripts or fetch assets.\n"
            "Follow the EDD principle: break tasks into subgoals, create verified skills with 3 positive and 3 negative tests, and compose them."
        )

        self.adk_agent = Agent(
            name=self.name,
            model=self.model,
            tools=self.tools,
            instruction=instruction,
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
        """ACR-AGI-3 ゲーム環境に対するメタスキル駆動型解決 (Progressive Disclosure ライフサイクル)."""
        initial_obs = env.reset()

        # ---------------------------------------------------------------------
        # 1. Level 1: メタデータカタログの取得 & スキルの自律トリガー
        # ---------------------------------------------------------------------
        self.harness.refresh()
        catalog_summary = self.harness.get_level1_catalog()
        logger.info(f"Task {task_id}: Loaded Level 1 Catalog with {len(self.harness.list_skills())} skills.")

        # ---------------------------------------------------------------------
        # 2. Level 2 & 3: env-observer のオンデマンド実行
        # ---------------------------------------------------------------------
        obs_res = self.harness.execute_skill_script(
            skill_name="env-observer",
            script_name="env_observer",
            input_data={"grid": initial_obs.tolist() if isinstance(initial_obs, np.ndarray) else initial_obs},
        )
        aff_report = obs_res.get("result", {})

        # ---------------------------------------------------------------------
        # 3. Level 2 & 3: game-style-intuitor のオンデマンド実行
        # ---------------------------------------------------------------------
        style_res = self.harness.execute_skill_script(
            skill_name="game-style-intuitor",
            script_name="game_style_intuitor",
            input_data={"grid": initial_obs.tolist() if isinstance(initial_obs, np.ndarray) else initial_obs},
        )
        style_report = style_res.get("result", {})
        if isinstance(style_report, str):
            try:
                style_report = json.loads(style_report)
            except Exception:
                style_report = {}

        # ---------------------------------------------------------------------
        # 4. Level 2 & 3: subgoal-decomposer のオンデマンド実行
        # ---------------------------------------------------------------------
        decomp_res = self.harness.execute_skill_script(
            skill_name="subgoal-decomposer",
            script_name="subgoal_decomposer",
            input_data={"grid": initial_obs.tolist() if isinstance(initial_obs, np.ndarray) else initial_obs},
        )
        plan_dict = decomp_res.get("result", {})

        logger.info(
            f"Affordances: agent={aff_report.get('agent_pos')}, "
            f"targets={len(aff_report.get('target_candidates', []))}, "
            f"style={style_report.get('style', 'GENERAL')}"
        )

        # 5. 検証済み具象スキルライブラリの取得
        verified_skills = edd_list_skills(verified_only=True)
        library_section = ""
        if verified_skills:
            library_lines = ["## Reusable Verified Skills:"]
            for vs in verified_skills:
                library_lines.append(f"- `{vs['name']}`: {vs['description']}")
            library_section = "\n".join(library_lines) + "\n"

        # 6. LLM プロンプト構築 (Level 1 カタログ + 構造化観測のみの低コンテキスト構成)
        base_prompt = (
            f"Solve ARC-AGI-3 dynamic game: {plan_dict.get('task_hint', 'task')}\n"
            f"Game Style: {style_report.get('style', 'GENERAL')} - {style_report.get('recommended_approach', '')}\n"
            f"Grid Shape: {aff_report.get('grid_shape', [10, 10])}, Background: {aff_report.get('background_color', 0)}\n"
            f"Agent Pos: {aff_report.get('agent_pos')}\n"
            f"{catalog_summary}\n"
            f"{library_section}\n"
            f"Subgoals:\n"
            + "\n".join(
                f"- Step {sg.get('step')}: {sg.get('name')} - {sg.get('objective')}"
                for sg in plan_dict.get("subgoals", [])
            )
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
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    coro = self._run_agent_turn(prompt, session_id=session_id)
                    resp = loop.run_until_complete(coro)
                except Exception:
                    coro = self._run_agent_turn(prompt, session_id=session_id)
                    resp = asyncio.run(coro)
            else:
                resp = asyncio.run(self._run_agent_turn(prompt, session_id=session_id))

            code = extract_python_code(resp)

            # 7. Level 2 & 3: contract-tester による防壁ゲート事前検証
            contract_res = self.harness.execute_skill_script(
                skill_name="contract-tester",
                script_name="contract_tester",
                input_data={"policy_code": code},
            )
            contract_report = contract_res.get("result", {})
            logger.info(
                f"Attempt {attempt}: Contract Gate Passed={contract_report.get('passed', False)} "
                f"({contract_report.get('total_passed', 0)}/{contract_report.get('total_cases', 6)})"
            )

            # シミュレーション実行
            verification = execute_and_verify_game_policy(code, env, max_steps=max_steps)

            if verification["success"]:
                # 合格したスキルをライブラリに正式登録
                skill_name = f"policy_{task_id}"
                edd_init_skill(skill_name)
                edd_write_skill_code(skill_name, code)
                edd_register_verified_skill(
                    name=skill_name,
                    description=plan_dict.get("task_hint", "solved policy"),
                    tags=["game_policy", "verified_solution"],
                )
                self.harness.refresh()
                logger.info(f"Registered verified skill '{skill_name}' to skill library.")
                break

            # 8. Level 2 & 3: failure-diagnoser による失敗診断と自己修復ディレクティブ生成
            diag_res = self.harness.execute_skill_script(
                skill_name="failure-diagnoser",
                script_name="failure_diagnoser",
                input_data={
                    "error": verification.get("error"),
                    "steps_taken": verification.get("steps_taken", 0),
                    "code": code,
                },
            )
            diag = diag_res.get("result", {})
            if isinstance(diag, str):
                try:
                    diag = json.loads(diag)
                except Exception:
                    diag = {"failure_category": "Unknown", "root_cause": diag, "directive": "Retry"}

            feedback = (
                f"[DIAGNOSIS CATEGORY: {diag.get('failure_category', 'GeneralFailure')}]\n"
                f"Root Cause: {diag.get('root_cause', '')}\n"
                f"Directive: {diag.get('directive', 'Ensure valid action')}"
            )

        return {
            "task_id": task_id,
            "is_solved": verification["success"],
            "verification": verification,
            "code": code,
            "policy_code": code,
            "steps_taken": verification.get("steps_taken", 0),
            "plan": plan_dict,
            "aff_report": aff_report,
            "available_skills_count": len(self.harness.list_skills()),
        }
