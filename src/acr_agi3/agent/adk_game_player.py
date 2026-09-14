"""Google ADK 2.0 準拠・自律ワークフロー駆動型ゲームプレイエージェント (ADKGamePlayer).

プロンプトへの禁則事項の肥大化を廃止し、Google ADK 2.0 の構造的ワークフロー
（思考・計画: Planner -> 検証・審査: Reviewer -> 実行: GameController）
によって質の高い自律的行動決定と自己改善ループを実現するエージェント。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.adk import Context
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from unittest.mock import MagicMock

from acr_agi3.agent.cognitive_workflow import CognitiveState, assess_cognitive_state
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.agent.workflow_schemas import PlanProposal, ReviewFeedback
from acr_agi3.harness.game_action_tools import ActionDecision, GameActionTools
from acr_agi3.harness.vision_observation import VisionObservationHarness, normalize_grid
from acr_agi3.meta.skill_harness import SkillHarness

logger = logging.getLogger(__name__)

# リアルタイム監視用ログファイル (logs/agent_live_execution.log) のセットアップ
LIVE_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "logs" / "agent_live_execution.log"
LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(LIVE_LOG_PATH, mode="a", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file_handler)
logger.setLevel(logging.INFO)

# game-controller メタスキルのインポート
GAME_CONTROLLER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "meta_skills" / "game-controller" / "scripts"
if str(GAME_CONTROLLER_DIR) not in sys.path and GAME_CONTROLLER_DIR.exists():
    sys.path.insert(0, str(GAME_CONTROLLER_DIR))

try:
    from game_controller import GameController
except ImportError:
    GameController = None

# episodic-memory メタスキルのインポート
EPISODIC_MEMORY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "meta_skills" / "episodic-memory" / "scripts"
if str(EPISODIC_MEMORY_DIR) not in sys.path and EPISODIC_MEMORY_DIR.exists():
    sys.path.insert(0, str(EPISODIC_MEMORY_DIR))

try:
    from episodic_memory import EpisodicMemoryManager
except ImportError:
    EpisodicMemoryManager = None


class ADKGamePlayer:
    """Google ADK 2.0 準拠・Plan-Review-Act ワークフロー自律ゲームプレイエージェント."""

    def __init__(
        self,
        model: Optional[Any] = None,
        name: str = "adk_game_player",
        app_name: str = "acr_agi3_adk_player",
        cell_size: int = 12,
        session_refresh_interval: int = 1,
    ) -> None:
        self.name = name
        self.app_name = app_name
        self.session_refresh_interval = session_refresh_interval
        self.model = model or LocalQwenVL(model_name_or_path="auto")

        # 1. 視覚認識ハーネス & 操作ツール
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()

        # 2. ADK 2.0 公式 SkillToolset (Progressive Disclosure)
        self.skill_harness = SkillHarness()
        self.skill_toolset = self.skill_harness.get_toolset()

        # 3. エピソード記憶マネージャー (Transcript からの因果蒸留 & Working Memory)
        self.memory = EpisodicMemoryManager(capacity=100) if EpisodicMemoryManager else None

        # 4. Google ADK 2.0 Planner Agent (思考・計画立案者)
        planner_instruction = (
            "You are the Chief Strategist and Planning Agent playing ARC-AGI-3 dynamic games.\n"
            "Your goal is to inspect the current visual observation and Working Memory, formulate a hypothesis about game mechanics, and propose a concrete action plan.\n\n"
            "=== 2-STEP COGNITIVE PLANNING PROTOCOL ===\n"
            "1. Analyze visual entities (player, targets, obstacles, colors) and past step outcomes.\n"
            "2. Output your plan as a structured JSON object:\n"
            "```json\n"
            "{\n"
            '  "hypothesis": "What the board state means (e.g. red square is player, gold star is goal)",\n'
            '  "goal": "Immediate objective (e.g. move toward gold star, click the blue tile)",\n'
            '  "action": "UP" | "DOWN" | "LEFT" | "RIGHT" | "click_at" | "RESET" | "ACTION1"..."ACTION7",\n'
            '  "coordinates": {"x": 5, "y": 8}, // REQUIRED if action is click_at or ACTION6! (x=col, y=row)\n'
            '  "reasoning": "Logical reason why this action advances your goal"\n'
            "}\n"
            "```\n"
            "If you need strategic guidance from an expert skill, include `\"load_skill\": \"visual-inspector\"` or `\"backward-planner\"`.\n"
        )
        self.planner_agent = Agent(
            name=f"{self.name}_planner",
            model=self.model,
            tools=[self.skill_toolset],
            instruction=planner_instruction,
        )

        # 5. Google ADK 2.0 Reviewer Agent (計画レビュー・審査者)
        reviewer_instruction = (
            "You are the Senior Game Reviewer and Quality Gatekeeper for ARC-AGI-3 gameplay.\n"
            "Your role is to rigorously review the Planner's proposed action plan before execution.\n\n"
            "=== 3-POINT OBJECTIVE QUALITY AUDIT ===\n"
            "1. SPECIFICITY: If action is `click_at` or `ACTION6`, are valid numeric coordinates {\"x\": col, \"y\": row} provided? If missing, mark REVISE or provide refined coordinates.\n"
            "2. REFLECTIVE: Does this action repeat an action that was just marked as INEFFECTIVE or barrier hit in Working Memory without justification?\n"
            "3. INTENTIONALITY: Does the reasoning clearly explain how this action advances the hypothesis or goal, avoiding mechanical repetitive cycling?\n\n"
            "Output your audit result as a structured JSON object:\n"
            "```json\n"
            "{\n"
            '  "status": "APPROVED" | "REVISE",\n'
            '  "critique": "Brief explanation of whether the plan is sound or flawed",\n'
            '  "suggested_fix": "If REVISE, explicit instructions on what to change (e.g. specify click coordinates, try a different direction)",\n'
            '  "refined_action": "Optional fallback action name if easily corrected",\n'
            '  "refined_coordinates": {"x": col, "y": row} // Optional fallback coordinates\n'
            "}\n"
            "```\n"
        )
        self.reviewer_agent = Agent(
            name=f"{self.name}_reviewer",
            model=self.model,
            tools=[],
            instruction=reviewer_instruction,
        )

        self.session_service = InMemorySessionService()
        self.planner_runner = Runner(
            agent=self.planner_agent,
            app_name=f"{self.app_name}_planner",
            session_service=self.session_service,
            auto_create_session=True,
        )
        self.reviewer_runner = Runner(
            agent=self.reviewer_agent,
            app_name=f"{self.app_name}_reviewer",
            session_service=self.session_service,
            auto_create_session=True,
        )

        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid: Optional[np.ndarray] = None
        self.last_action_info: Optional[Dict[str, Any]] = None
        self.planner_session_id: Optional[str] = None
        self.reviewer_session_id: Optional[str] = None

    def reset(self) -> None:
        """エージェントの状態とセッションを初期化."""
        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid = None
        self.last_action_info = None
        self.action_tools.pending_decision = None
        self.action_tools.history.clear()
        if self.memory:
            self.memory.clear()
        self.planner_session_id = None
        self.reviewer_session_id = None

    def decide_next_action(
        self,
        grid: Any,
        available_actions: Optional[List[int]] = None,
        state_str: str = "NOT_FINISHED",
    ) -> ActionDecision:
        """ADK 2.0 Plan-Review-Act ワークフローを経て次のアクションを決定."""
        self.step_index += 1
        arr = normalize_grid(grid)

        # 差分情報の算出
        pixels_changed = 0
        is_effective = False
        if self.last_grid is not None and self.last_grid.shape == arr.shape:
            diff_mask = (self.last_grid != arr)
            pixels_changed = int(np.sum(diff_mask))
            is_effective = (pixels_changed > 0)

        # 停滞カウントの更新 (無変化が連続した場合にインクリメント)
        if self.last_action_info is not None:
            if pixels_changed == 0:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        else:
            self.stagnation_count = 0

        # 直前ステップの結果をエピソード記憶に登録（自己反省・因果更新）
        if self.last_action_info is not None and self.memory is not None:
            self.memory.record_step(
                step_index=self.step_index - 1,
                action_name=self.last_action_info.get("action", "UNKNOWN"),
                action_id=self.last_action_info.get("action_id", 1),
                pixels_changed=pixels_changed,
                is_effective=is_effective,
                state_before=self.last_action_info.get("state_before", "NOT_FINISHED"),
                state_after=state_str,
                reflection=self.last_action_info.get("reasoning", ""),
            )

        # 利用可能アクションの更新
        avail_ids = available_actions or [1, 2, 3, 4]
        avail_names = [f"ACTION{i}" for i in avail_ids]
        self.action_tools.set_available_actions(avail_ids)

        # 視覚観測 Parts の生成 (Phase 1: OBSERVE)
        parts = self.vision_harness.create_observation_parts(
            grid_data=arr,
            step_index=self.step_index,
            available_actions=avail_names,
            last_action_info=self.last_action_info,
        )

        memory_summary = ""
        if self.memory is not None:
            memory_summary = self.memory.get_working_memory_summary(max_recent=4)
            parts.append(Part.from_text(text=f"\n=== [RECALL: Working Memory Context] ===\n{memory_summary}\n"))

        # ADK 2.0 認知グラフワークフローによる状況評価とメタスキルルーティング
        cog_state = CognitiveState(
            step=self.step_index,
            observation=arr,
            working_memory=memory_summary,
            stagnation_count=self.stagnation_count,
        )
        dummy_ctx = MagicMock(spec=Context)
        assess_cognitive_state(dummy_ctx, cog_state)
        active_skill = cog_state.selected_skill or "visual-inspector"

        skill_obj = self.skill_harness.get_skill(active_skill)
        skill_desc = getattr(skill_obj, "description", "") if skill_obj else ""
        parts.append(
            Part.from_text(
                text=(
                    f"\n=== [ADK 2.0 COGNITIVE GRAPH: ACTIVE META-SKILL '{active_skill}'] ===\n"
                    f"Cognitive Mode: {cog_state.cognitive_mode.upper()}\n"
                    f"Description: {skill_desc}\n"
                    f"Directive: Align your hypothesis, goal, and action plan with the '{active_skill}' protocol.\n"
                )
            )
        )

        # 非同期 Runner を同期実行
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        decision = loop.run_until_complete(
            self._run_plan_review_act_workflow(
                obs_parts=parts,
                available_action_ids=avail_ids,
                avail_names=avail_names,
                memory_summary=memory_summary,
                grid_shape=arr.shape[:2],
                active_skill=active_skill,
            )
        )
        self.last_grid = arr.copy()
        self.last_action_info = {
            "action": decision.action_name,
            "action_id": decision.action_id,
            "reasoning": decision.reasoning,
            "state_before": state_str,
        }
        return decision

    async def _run_plan_review_act_workflow(
        self,
        obs_parts: List[Part],
        available_action_ids: List[int],
        avail_names: List[str],
        memory_summary: str,
        grid_shape: Tuple[int, int],
        active_skill: Optional[str] = None,
    ) -> ActionDecision:
        """Planner -> Reviewer -> Act の 3 フェーズ自律協調ワークフロー."""
        user_id = "arc_workflow_user"

        # 1. セッションの定期的新設 (会話履歴の肥大化・自己模倣ループ・CUDA OOM を完全防止)
        need_new_session = (
            self.planner_session_id is None
            or self.reviewer_session_id is None
            or self.session_refresh_interval <= 1
            or (self.step_index % self.session_refresh_interval == 1)
        )

        if need_new_session:
            s_plan = await self.session_service.create_session(
                app_name=f"{self.app_name}_planner", user_id=user_id
            )
            self.planner_session_id = s_plan.id

            s_rev = await self.session_service.create_session(
                app_name=f"{self.app_name}_reviewer", user_id=user_id
            )
            self.reviewer_session_id = s_rev.id
        else:
            await self._archive_session_images(f"{self.app_name}_planner", self.planner_session_id, user_id)
            await self._archive_session_images(f"{self.app_name}_reviewer", self.reviewer_session_id, user_id)

        # -------------------------------------------------------------
        # Phase 1: Planning (思考・仮説・行動計画の立案)
        # -------------------------------------------------------------
        plan_content = Content(role="user", parts=obs_parts)
        planner_events = self.planner_runner.run_async(
            session_id=self.planner_session_id,
            user_id=user_id,
            new_message=plan_content,
        )

        plan_raw_text = ""
        async for ev in planner_events:
            if hasattr(ev, "content") and ev.content:
                for p in getattr(ev.content, "parts", []):
                    if hasattr(p, "text") and p.text:
                        plan_raw_text += p.text

        logger.info(f"Step {self.step_index} [Phase 1: Planner] raw text: {plan_raw_text!r}")
        proposal = PlanProposal.from_text(plan_raw_text)

        # Progressive Disclosure: メタスキルが要求された場合の展開
        triggered_skill = proposal.load_skill
        if triggered_skill:
            logger.info(f"Step {self.step_index} [Phase 1: Progressive Disclosure] Expanding skill: {triggered_skill}")
            try:
                skill_content = self.skill_harness.read_skill_content(triggered_skill)
                skill_prompt = (
                    f"Expert meta-skill `{triggered_skill}` instructions loaded:\n"
                    f"```markdown\n{skill_content[:1500]}\n```\n\n"
                    f"Now provide your finalized plan JSON according to the planning protocol:"
                )
                skill_events = self.planner_runner.run_async(
                    session_id=self.planner_session_id,
                    user_id=user_id,
                    new_message=Content(role="user", parts=[Part.from_text(text=skill_prompt)]),
                )
                plan_raw_text = ""
                async for ev in skill_events:
                    if hasattr(ev, "content") and ev.content:
                        for p in getattr(ev.content, "parts", []):
                            if hasattr(p, "text") and p.text:
                                plan_raw_text += p.text
                proposal = PlanProposal.from_text(plan_raw_text)
                proposal.load_skill = triggered_skill
            except Exception as e:
                logger.warning(f"Failed to expand skill {triggered_skill}: {e}")

        # -------------------------------------------------------------
        # Phase 2: Review (計画の客観的検証・レビュー)
        # -------------------------------------------------------------
        review_prompt = (
            f"=== [PLAN REVIEW REQUEST for Step {self.step_index}] ===\n"
            f"Available Actions: {avail_names} ({available_action_ids})\n"
            f"Grid Shape: {grid_shape[0]} rows x {grid_shape[1]} cols\n"
            f"Working Memory:\n{memory_summary or 'No previous actions.'}\n\n"
            f"Planner's Proposed Plan:\n"
            f"```json\n{json.dumps(proposal.to_dict(), indent=2)}\n```\n\n"
            f"Perform the 3-point audit. If action is click/ACTION6 without valid coordinates, "
            f"or repeats a proven ineffective action, mark REVISE and provide specific fixes."
        )
        review_content = Content(role="user", parts=[Part.from_text(text=review_prompt)])
        reviewer_events = self.reviewer_runner.run_async(
            session_id=self.reviewer_session_id,
            user_id=user_id,
            new_message=review_content,
        )

        review_raw_text = ""
        async for ev in reviewer_events:
            if hasattr(ev, "content") and ev.content:
                for p in getattr(ev.content, "parts", []):
                    if hasattr(p, "text") and p.text:
                        review_raw_text += p.text

        logger.info(f"Step {self.step_index} [Phase 2: Reviewer] raw text: {review_raw_text!r}")
        review = ReviewFeedback.from_text(review_raw_text)

        # -------------------------------------------------------------
        # Phase 3: Revision Loop (不合格時の修正)
        # -------------------------------------------------------------
        if not review.is_approved:
            logger.info(f"Step {self.step_index} [Phase 3: Revision Required] Critique: {review.critique}")
            revise_prompt = (
                f"Your proposed plan was REVISED by the Quality Reviewer:\n"
                f"Critique: {review.critique}\n"
                f"Suggested Fix: {review.suggested_fix or 'Provide valid coordinates or select an alternative productive action.'}\n\n"
                f"Please produce an updated, corrected plan JSON now:"
            )
            revise_events = self.planner_runner.run_async(
                session_id=self.planner_session_id,
                user_id=user_id,
                new_message=Content(role="user", parts=[Part.from_text(text=revise_prompt)]),
            )
            revised_text = ""
            async for ev in revise_events:
                if hasattr(ev, "content") and ev.content:
                    for p in getattr(ev.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            revised_text += p.text

            logger.info(f"Step {self.step_index} [Phase 3: Revised Plan] text: {revised_text!r}")
            proposal = PlanProposal.from_text(revised_text)

        # Reviewer からの直接オーバーライド補正（座標やアクション）があれば適用
        chosen_action = review.refined_action or proposal.action
        chosen_coords = review.refined_coordinates or proposal.coordinates

        # -------------------------------------------------------------
        # Phase 4: Act (GameController による確定行動の検証と実行)
        # -------------------------------------------------------------
        logger.info(
            f"Step {self.step_index} [Phase 4: Act] Final Action: {chosen_action}, "
            f"Coords: {chosen_coords}, Hypothesis: {proposal.hypothesis!r}"
        )

        # 認知ワークフローで選定されたメタスキルを反映 (明示指定された場合はそれを尊重)
        if proposal.load_skill_explicit:
            final_skill = proposal.load_skill
        else:
            final_skill = proposal.load_skill or active_skill

        decision = self._convert_to_decision(
            action_str=chosen_action,
            coordinates=chosen_coords,
            reasoning=f"[{proposal.goal}] {proposal.reasoning or proposal.hypothesis}",
            available_action_ids=available_action_ids,
            grid_shape=grid_shape,
            loaded_skill=final_skill,
        )

        # VRAM キャッシュ解放
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return decision

    def _convert_to_decision(
        self,
        action_str: str,
        coordinates: Optional[Dict[str, int]],
        reasoning: str,
        available_action_ids: List[int],
        grid_shape: Tuple[int, int],
        loaded_skill: Optional[str] = None,
    ) -> ActionDecision:
        """計画内容を GameController を用いて安全に ActionDecision へ変換."""
        h, w = grid_shape
        if GameController is not None:
            controller = GameController(available_actions=available_action_ids)
            # JSON 形式の文字列に変換して GameController の厳格バリデーションへ渡す
            payload: Dict[str, Any] = {"action": action_str, "reasoning": reasoning}
            if coordinates and "x" in coordinates and "y" in coordinates:
                payload["x"] = coordinates["x"]
                payload["y"] = coordinates["y"]

            validation = controller.parse_and_validate(
                json.dumps(payload),
                available_actions=available_action_ids,
                grid_shape=(h, w),
            )
            if validation["success"]:
                return ActionDecision(
                    action_type=validation["action_type"],
                    action_name=validation["action_name"],
                    action_id=validation["action_id"],
                    coordinates=validation["coordinates"],
                    reasoning=validation["reasoning"],
                    loaded_skill=loaded_skill,
                )

        # フォールバック安全処理
        act_upper = action_str.upper()
        name_map = {
            "RESET": 0,
            "ACTION1": 1, "UP": 1,
            "ACTION2": 2, "DOWN": 2,
            "ACTION3": 3, "LEFT": 3,
            "ACTION4": 4, "RIGHT": 4,
            "ACTION5": 5,
            "ACTION6": 6, "CLICK": 6, "CLICK_AT": 6,
            "ACTION7": 7,
        }
        for k, aid in name_map.items():
            if k in act_upper:
                if aid == 6:
                    coords = coordinates or {"x": w // 2, "y": h // 2}
                    return ActionDecision(
                        action_type="CLICK",
                        action_name="ACTION6",
                        action_id=6,
                        coordinates=coords,
                        reasoning=reasoning,
                        loaded_skill=loaded_skill,
                    )
                return ActionDecision(
                    action_type="STEP" if aid != 0 else "RESET",
                    action_name=f"ACTION{aid}" if aid != 0 else "RESET",
                    action_id=aid,
                    coordinates=None,
                    reasoning=reasoning,
                    loaded_skill=loaded_skill,
                )

        default_id = available_action_ids[0] if available_action_ids else 1
        return ActionDecision(
            action_type="STEP",
            action_name=f"ACTION{default_id}",
            action_id=default_id,
            coordinates=None,
            reasoning=f"Fallback default action: {reasoning}",
            loaded_skill=loaded_skill,
        )

    async def _archive_session_images(self, app_name: str, session_id: str, user_id: str) -> None:
        """セッション内の過去フレーム画像をテキストマーカーに置き換え VRAM 累積を防止."""
        session = await self.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        if session and getattr(session, "events", None):
            for ev in session.events:
                if hasattr(ev, "content") and ev.content:
                    for p in getattr(ev.content, "parts", []):
                        if hasattr(p, "inline_data") and p.inline_data:
                            p.inline_data = None
                            p.text = "[Previous Visual Frame archived]"
