"""メタスキル駆動型自律プログラミングエージェント (Meta-Skill Driven Synthesis Agent).

MetaObserver, SubgoalDecomposer, VCGTDataset, FailureDiagnoser を Google ADK 2.0 に統合し、
高精度な自己改善ループ（Self-Correction）を実現します。
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.arc_tools import execute_and_verify_code, extract_python_code
from acr_agi3.agent.llm.edd_tools import (
    edd_execute_skill,
    edd_init_skill,
    edd_run_contract_test,
    edd_validate_skill,
    edd_write_skill_code,
)
from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.meta.decomposer import DecompositionPlan, Subgoal, SubgoalDecomposer
from acr_agi3.meta.human_vcgt import VCGTDataset
from acr_agi3.meta.observer import MetaObserver, ObservationReport

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

        # EDD ツールセット
        self.edd_tools = [
            edd_init_skill,
            edd_validate_skill,
            edd_write_skill_code,
            edd_run_contract_test,
            edd_execute_skill,
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
                "You are an elite AI researcher and programmer solving ARC-AGI puzzles "
                "guided by Human Visual Concept Guided Thinking (VCGT) and "
                "Evaluation-Driven Development (EDD).\n"
                "You have access to EDD tools: edd_init_skill, edd_write_skill_code, "
                "edd_validate_skill, edd_run_contract_test, edd_execute_skill.\n"
                "Break down problems into subgoals, create verified skills for each subgoal, "
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

    def analyze_task(
        self, train_pairs: List[Dict[str, Any]]
    ) -> tuple[ObservationReport, DecompositionPlan]:
        """メタスキル（Observer & Decomposer）によるタスク分析."""
        first_in = np.array(train_pairs[0]["input"], dtype=int)
        first_out = np.array(train_pairs[0]["output"], dtype=int)

        obs_report = self.observer.analyze_pair(first_in, first_out)
        plan = self.decomposer.decompose(first_in, first_out)
        return obs_report, plan

    def build_meta_prompt(
        self,
        train_pairs: List[Dict[str, Any]],
        obs_report: ObservationReport,
        plan: DecompositionPlan,
        feedback: Optional[str] = None,
    ) -> str:
        """メタスキル情報を統合したプロンプトを構築."""
        lines: List[str] = []

        # 1. Few-shot VCGT 思考例
        if self.vcgt_dataset and len(self.vcgt_dataset) > 0:
            lines.append(self.vcgt_dataset.build_few_shot_prompt(max_examples=1))
            lines.append("\n" + "=" * 40 + "\n")

        # 2. Meta-Observer による環境不変量・アフォーダンス分析
        lines.append("## Target Task Meta-Cognitive Analysis (Invariants & Affordances):")
        lines.append(
            f"- Shape Transition: {obs_report.in_shape} -> {obs_report.out_shape} "
            f"(Ratio: {obs_report.shape_ratio})"
        )
        lines.append(f"- Transformation Hypothesis: {obs_report.transformation_hint}")
        if obs_report.new_colors:
            lines.append(f"- Newly Introduced Colors: {list(obs_report.new_colors)}")
        if obs_report.removed_colors:
            lines.append(f"- Removed Colors: {list(obs_report.removed_colors)}")
        lines.append(f"- Salient Input Objects Detected: {len(obs_report.objects)}")
        lines.append("")

        # 3. Subgoal-Decomposer による階層分解計画
        lines.append("## Human-Inspired Subgoal Plan:")
        lines.append(f"Goal: {plan.task_hint}")
        for s in plan.subgoals:
            lines.append(f"  [Step {s.index}] {s.name}: {s.objective}")
            lines.append(f"          Reasoning: {s.reasoning}")
            lines.append(f"          Expected Op: {s.expected_operation}")
        lines.append("")

        # 4. 具体的な入出力グリッド
        lines.append("## Training Examples:")
        for i, pair in enumerate(train_pairs):
            lines.append(f"--- Example {i + 1} ---")
            lines.append("Input grid:")
            lines.append(self._format_grid(np.array(pair["input"], dtype=int)))
            lines.append("Output grid:")
            lines.append(self._format_grid(np.array(pair["output"], dtype=int)))
            lines.append("")

        # 5. 自己修復フィードバック (Failure Diagnoser)
        if feedback:
            lines.append("## PREVIOUS ATTEMPT FAILURE DIAGNOSIS:")
            lines.append(feedback)
            lines.append(
                "Modify your algorithm according to the diagnosed discrepancy. "
                "Return only the fixed `def transform(grid: np.ndarray) -> np.ndarray:`."
            )
        else:
            lines.append(
                "Synthesize a robust Python function "
                "`def transform(grid: np.ndarray) -> np.ndarray:` implementing "
                "the subgoal sequence. Provide only the Python code in ```python ``` block."
            )

        return "\n".join(lines)

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

    def diagnose_failure(
        self,
        code: str,
        verification: Dict[str, Any],
        train_pairs: List[Dict[str, Any]],
    ) -> str:
        """Failure-Diagnoser メタスキルによる詳細差分診断."""
        if verification.get("error"):
            return f"Runtime/Syntax Error: {verification['error']}"

        failures = verification.get("failures", [])
        if not failures:
            return "Unknown verification failure."

        tot = verification["total_count"]
        diag_lines = [f"Verification failed on {len(failures)}/{tot} pairs:"]
        for fail in failures[:2]:
            pair_idx = fail["pair_index"]
            expected = np.array(train_pairs[pair_idx]["output"], dtype=int)
            reason = fail["reason"]
            diag_lines.append(f"- Pair {pair_idx}: {reason}")

            # 形状が一致している場合は不一致セルの詳細を診断
            try:
                local_scope: Dict[str, Any] = {"np": np}
                exec(code, {"np": np, "__builtins__": __builtins__}, local_scope)
                inp_arr = np.array(train_pairs[pair_idx]["input"], dtype=int)
                predicted = local_scope["transform"](inp_arr)
                if predicted.shape == expected.shape:
                    diff_mask = (predicted != expected)
                    diff_coords = np.argwhere(diff_mask)[:3]
                    for r, c in diff_coords:
                        exp_c = expected[r, c]
                        got_c = predicted[r, c]
                        diag_lines.append(
                            f"    * Cell ({r}, {c}): Expected color {exp_c}, Got {got_c}"
                        )
            except Exception as e:
                diag_lines.append(f"    * Exception during diagnosis: {e}")

        return "\n".join(diag_lines)

    def solve(
        self,
        train_pairs: List[Dict[str, Any]],
        max_iterations: int = 3,
        task_id: str = "task",
    ) -> Dict[str, Any]:
        """メタスキル駆動型の自己改善ループを実行."""
        obs_report, plan = self.analyze_task(train_pairs)
        feedback: Optional[str] = None
        history: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            prompt = self.build_meta_prompt(train_pairs, obs_report, plan, feedback=feedback)
            session_id = f"meta_sess_{task_id}_{iteration}"

            response = asyncio.run(self._run_agent_turn(prompt, session_id=session_id))
            code = extract_python_code(response)
            verification = execute_and_verify_code(code, train_pairs)

            history.append({
                "iteration": iteration,
                "code": code,
                "verification": verification,
            })

            if verification["is_valid"]:
                return {
                    "is_solved": True,
                    "code": code,
                    "iterations": iteration,
                    "history": history,
                    "plan": plan,
                    "obs_report": obs_report,
                }

            feedback = self.diagnose_failure(code, verification, train_pairs)

        return {
            "is_solved": False,
            "code": history[-1]["code"] if history else None,
            "iterations": max_iterations,
            "history": history,
            "plan": plan,
            "obs_report": obs_report,
        }

    def synthesize_subgoal_skill(
        self,
        subgoal: Subgoal,
        train_pairs: List[Dict[str, Any]],
        task_id: str = "task",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """EDD ツールを用いてサブゴール特化スキルを自律開発・契約テスト."""
        skill_name = f"sub_{task_id}_{subgoal.index}_{subgoal.expected_operation}"
        edd_init_skill(skill_name)

        prompt = (
            f"Synthesize an EDD skill function for Subgoal {subgoal.index}: {subgoal.name}\n"
            f"Objective: {subgoal.objective}\n"
            f"Reasoning: {subgoal.reasoning}\n"
            f"Expected Operation: {subgoal.expected_operation}\n"
            "Write a Python function `def transform(grid: np.ndarray) -> np.ndarray:` "
            "implementing this specific transformation step. Provide Python code."
        )

        feedback = None
        code = ""
        for attempt in range(1, max_retries + 1):
            cur_prompt = prompt if not feedback else f"{prompt}\n\n[FEEDBACK]: {feedback}"
            resp = asyncio.run(
                self._run_agent_turn(cur_prompt, session_id=f"sess_{skill_name}_{attempt}")
            )
            code = extract_python_code(resp)
            edd_write_skill_code(skill_name, code)
            val_res = edd_validate_skill(skill_name)
            test_res = edd_run_contract_test(skill_name, train_pairs)

            if test_res["is_valid"]:
                return {
                    "success": True,
                    "skill_name": skill_name,
                    "attempt": attempt,
                    "code": code,
                    "validation": val_res,
                    "test_result": test_res,
                }
            feedback = self.diagnose_failure(code, test_res, train_pairs)

        return {
            "success": False,
            "skill_name": skill_name,
            "attempt": max_retries,
            "code": code,
            "feedback": feedback,
        }
