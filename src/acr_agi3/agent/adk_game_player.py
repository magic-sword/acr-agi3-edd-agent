"""Google ADK 2.0 準拠・自律ゲームプレイエージェント (ADKGamePlayer).

過剰に複雑化したワークフローや多数のメタスキルを全廃し、
人間が画面を見てプレイするのと同様の、
「視覚入力 (visual-inspector) -> 画像認識思考 (Planner) -> 1手出力 (game-controller)」
の直線的かつ超高速な自律ゲームプレイを実現するエージェント。
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
from google.adk.agents import Agent
from google.adk.runners import Runner, RunConfig
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.agent.workflow_schemas import PlanProposal
from acr_agi3.harness.game_action_tools import ActionDecision, GameActionTools
from acr_agi3.harness.vision_observation import (
    VisionObservationHarness,
    normalize_grid,
)
from acr_agi3.meta.skill_harness import SkillHarness
from acr_agi3.tools import MemoryTools, VisionTools

logger = logging.getLogger(__name__)
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)

# game-controller メタスキルのインポート
GAME_CONTROLLER_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "game-controller" / "scripts"
if str(GAME_CONTROLLER_DIR) not in sys.path and GAME_CONTROLLER_DIR.exists():
    sys.path.insert(0, str(GAME_CONTROLLER_DIR))

try:
    from game_controller import GameController
except ImportError:
    GameController = None


class ADKGamePlayer:
    """Google ADK 2.0 準拠・自律ゲームプレイエージェント (3フェーズワークフロー版)."""

    def __init__(
        self,
        model: Optional[Any] = None,
        name: str = "adk_game_player",
        app_name: str = "acr_agi3_adk_player",
        cell_size: int = 12,
        session_refresh_interval: int = 1,
        autonomous_probing: Optional[bool] = None,
    ) -> None:
        self.name = name
        self.app_name = app_name
        self.session_refresh_interval = session_refresh_interval
        self.model = model or LocalQwenVL(model_name_or_path="auto")
        self.autonomous_probing = bool(autonomous_probing) if autonomous_probing is not None else False

        # 1. ハーネス初期化
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()
        self.vision_tools = VisionTools()
        self.memory_tools = MemoryTools()
        self.skill_harness = SkillHarness()

        # 2. 各ノード専用の最小限スキルセット（最小権限の原則 ＋ Level 3 実行ツールの動的解放）
        self.perceive_additional_tools = self.vision_tools.get_tools()
        self.perceive_toolset = self.skill_harness.get_scoped_toolset(
            ["visual-inspector"],
            additional_tools=self.perceive_additional_tools,
        )

        self.plan_additional_tools = self.memory_tools.get_tools()
        self.plan_toolset = self.skill_harness.get_scoped_toolset(
            ["memory-notebook"],
            additional_tools=self.plan_additional_tools,
        )

        self.act_additional_tools = self.action_tools.get_tools()
        self.act_toolset = self.skill_harness.get_scoped_toolset(
            ["game-controller"],
            additional_tools=self.act_additional_tools,
        )
        self.skill_toolset = self.skill_harness.get_toolset(
            additional_tools=self.perceive_additional_tools + self.plan_additional_tools + self.act_additional_tools
        )

        # 3. 動的操作力学マップ (ボタンと移動方向の同定結果: 例 {'UP': 3, 'DOWN': 4})
        self.dynamics_map: Dict[str, int] = {}

        # 4. Google ADK 2.0 3フェーズ Agent 定義
        # Node 1: Visual Inspection (Perceive Node) - 盤面・アフォーダンス目視点検
        perceive_instruction = (
            "You are the Visual Inspection Specialist for ARC-AGI-3 dynamic games.\n"
            "Your objective is to observe the visual console screen (game board and controller HUD) and extract objective spatial layout, active colors, player candidates, targets, and affordances.\n"
            "1. You have access to the skill: 'visual-inspector'. You may call `load_skill(skill_name='visual-inspector')` if you need detailed inspection guides or scripts.\n"
            "2. You may also call inspection tools directly if needed: `inspect_affordances(mode='deep')` or `inspect_board_summary()`.\n"
            "Output a concise visual observation summary:\n"
            "- Board layout geometry, active colors, and spatial symmetry\n"
            "- Discovered entities (player, goal, obstacles, movable blocks)\n"
            "- Visual displacement or state changes from the previous action"
        )
        self.perceive_agent = Agent(
            name=f"{self.name}_perceive",
            model=self.model,
            tools=[self.perceive_toolset] + self.perceive_additional_tools,
            instruction=perceive_instruction,
        )

        # Node 2: Cognitive Planning (Plan Node) - 逆算プランニング・記憶連携
        plan_instruction = (
            "You are the Cognitive Planner for ARC-AGI-3 dynamic games.\n"
            "Your objective is to review the visual observation summary from Node 1 and formulate a backward-chaining strategy and immediate subgoal.\n"
            "CRITICAL CONSTRAINT: You are ONLY a planner. You MUST NOT execute actions or call step_action or click_at.\n"
            "Node 3 will execute the action based on your plan.\n"
            "1. You have access to the skill: 'memory-notebook'. You may call `load_skill(skill_name='memory-notebook')` if you need memory recall or state-tracking guides.\n"
            "2. You may use memory tools to recall or store findings: `memory_write(section_id=..., content=...)`, `memory_read(section_id=...)`, `memory_toc()`, `memory_search(query=...)`.\n"
            "Output your planning strategy as plain text:\n"
            "- Immediate subgoal (e.g. advance towards target, stage piece in buffer, test unexplored button, avoid trap)\n"
            "- Keystone piece or dependency ordering (Backward Chaining)\n"
            "- Key hypothesis on causal dynamics"
        )
        self.plan_agent = Agent(
            name=f"{self.name}_plan",
            model=self.model,
            tools=[self.plan_toolset] + self.plan_additional_tools,
            instruction=plan_instruction,
        )

        # Node 3: 1-Step Execution (Act Node) - 1手決定・安全検証
        act_instruction = (
            "You are the Action Decision Specialist for ARC-AGI-3 dynamic games.\n"
            "Your objective is to execute the immediate subgoal from Node 2 using available tools.\n"
            "1. You have access to the skill: 'game-controller'. Call `load_skill(skill_name='game-controller')` if you need to review its operational instructions and rules.\n"
            "2. To execute your action, you MUST call one of the execution tools:\n"
            "   - `step_action(action='...', reasoning='...')`: Execute a directional move ('UP', 'DOWN', 'LEFT', 'RIGHT') or physical button ('ACTION1'-'ACTION7').\n"
            "   - `click_at(x=col, y=row, reasoning='...')`: Click at coordinate (ACTION6) or auto-snap to interactive element.\n"
            "   - `reset_game(reasoning='...')`: Reset level when deadlocked.\n"
            "DO NOT call load_skill with action names (e.g. do NOT call load_skill('ACTION1')). Always use `step_action` or `click_at` to execute actions."
        )
        self.act_agent = Agent(
            name=f"{self.name}_act",
            model=self.model,
            tools=[self.act_toolset] + self.act_additional_tools,
            instruction=act_instruction,
        )

        # 後方互換性エイリアス
        self.planner_agent = self.act_agent

        # 5. セッションとランナーの初期化
        self.session_service = InMemorySessionService()
        self.perceive_runner = Runner(
            agent=self.perceive_agent,
            app_name=f"{self.app_name}_perceive",
            session_service=self.session_service,
            auto_create_session=True,
        )
        self.plan_runner = Runner(
            agent=self.plan_agent,
            app_name=f"{self.app_name}_plan",
            session_service=self.session_service,
            auto_create_session=True,
        )
        self.act_runner = Runner(
            agent=self.act_agent,
            app_name=f"{self.app_name}_act",
            session_service=self.session_service,
            auto_create_session=True,
        )
        self.planner_runner = self.act_runner

        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid: Optional[np.ndarray] = None
        self.last_action_info: Optional[Dict[str, Any]] = None
        self.perceive_session_id: Optional[str] = None
        self.plan_session_id: Optional[str] = None
        self.act_session_id: Optional[str] = None
        self.planner_session_id: Optional[str] = None

    def reset(self) -> None:
        """エージェントの状態とセッションを初期化."""
        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid = None
        self.last_action_info = None
        self.action_tools.pending_decision = None
        self.action_tools.history.clear()
        self.perceive_session_id = None
        self.plan_session_id = None
        self.act_session_id = None
        self.planner_session_id = None

    def decide_next_action(
        self,
        grid: Any,
        available_actions: Optional[List[int]] = None,
        state_str: str = "NOT_FINISHED",
    ) -> ActionDecision:
        """最新観測から 3フェーズ (Perceive -> Plan -> Act) を経て game-controller で 1 手を実行."""
        self.step_index += 1
        arr = normalize_grid(grid)

        # 差分情報の算出
        pixels_changed = 0
        is_effective = False
        if self.last_grid is not None and self.last_grid.shape == arr.shape:
            diff_mask = (self.last_grid != arr)
            pixels_changed = int(np.sum(diff_mask))
            is_effective = (pixels_changed > 0)

        # 停滞カウントの更新
        if self.last_action_info is not None:
            if pixels_changed == 0:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        else:
            self.stagnation_count = 0

        # 利用可能アクションの更新
        avail_ids = available_actions or [1, 2, 3, 4]
        avail_names = [f"ACTION{i}" for i in avail_ids]
        self.action_tools.set_available_actions(avail_ids)
        self.action_tools.set_dynamics_map(self.dynamics_map)
        self.vision_tools.set_context(arr, step_index=self.step_index)
        self.memory_tools.set_step(self.step_index)

        # 視覚観測 Parts の生成 (統合コンソール画面 + 客観的事実)
        parts = self.vision_harness.create_observation_parts(
            grid_data=arr,
            step_index=self.step_index,
            available_actions=avail_names,
            last_action_info=self.last_action_info,
        )

        # 非同期 Runner を実行
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        decision = loop.run_until_complete(
            self._run_plan_act_workflow(
                obs_parts=parts,
                available_action_ids=avail_ids,
                available_action_names=avail_names,
                grid_shape=arr.shape[:2],
                grid=arr,
                state_str=state_str,
            )
        )

        self.last_grid = arr.copy()
        self.last_action_info = {
            "action": decision.action_name,
            "action_id": decision.action_id,
            "reasoning": decision.reasoning,
            "state_before": state_str,
            "pixels_changed": pixels_changed,
            "is_effective": is_effective,
        }
        return decision

    async def _run_plan_act_workflow(
        self,
        obs_parts: List[Part],
        available_action_ids: List[int],
        available_action_names: List[str],
        grid_shape: Tuple[int, int],
        grid: Optional[np.ndarray] = None,
        state_str: str = "NOT_FINISHED",
    ) -> ActionDecision:
        """3フェーズワークフロー: Node 1 (Perceive) -> Node 2 (Plan) -> Node 3 (Act)."""
        user_id = "arc_workflow_user"

        need_new_session = (
            self.act_session_id is None
            or self.session_refresh_interval <= 1
            or (self.step_index % self.session_refresh_interval == 1)
        )

        if need_new_session:
            s_perceive = await self.session_service.create_session(
                app_name=f"{self.app_name}_perceive", user_id=user_id
            )
            self.perceive_session_id = s_perceive.id

            s_plan = await self.session_service.create_session(
                app_name=f"{self.app_name}_plan", user_id=user_id
            )
            self.plan_session_id = s_plan.id

            s_act = await self.session_service.create_session(
                app_name=f"{self.app_name}_act", user_id=user_id
            )
            self.act_session_id = s_act.id
            self.planner_session_id = s_act.id
        else:
            await self._archive_session_images(f"{self.app_name}_perceive", self.perceive_session_id, user_id)
            await self._archive_session_images(f"{self.app_name}_act", self.act_session_id, user_id)

        # ---------------------------------------------------------------------
        # Phase 1: Visual Inspection (Perceive Node) - 最小限ツール: visual-inspector
        # ---------------------------------------------------------------------
        node_run_config = RunConfig(max_llm_calls=4)
        perceive_content = Content(role="user", parts=obs_parts)
        perceive_events = self.perceive_runner.run_async(
            session_id=self.perceive_session_id,
            user_id=user_id,
            new_message=perceive_content,
            run_config=node_run_config,
        )
        perceive_summary = ""
        try:
            async for ev in perceive_events:
                if hasattr(ev, "content") and ev.content:
                    for p in getattr(ev.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            perceive_summary += p.text
        except Exception as e:
            logger.warning("Step %d [Perceive Node notice]: %s", self.step_index, e)

        logger.info("Step %d [Phase 1: Perceive Complete] Summary: %s...", self.step_index, perceive_summary[:120].strip())

        # ---------------------------------------------------------------------
        # Phase 2: Cognitive Planning (Plan Node) - 最小限ツール: memory-notebook
        # ---------------------------------------------------------------------
        plan_prompt = (
            f"=== VISUAL PERCEPTION SUMMARY (Phase 1) ===\n"
            f"{perceive_summary}\n\n"
            f"Step: {self.step_index}, Game State: {state_str}, Available Action Buttons: {available_action_names}\n"
            f"Formulate your backward-chaining strategy and immediate subgoal. (Do not call any action tools like step_action)."
        )
        plan_content = Content(role="user", parts=[Part.from_text(text=plan_prompt)])
        plan_events = self.plan_runner.run_async(
            session_id=self.plan_session_id,
            user_id=user_id,
            new_message=plan_content,
            run_config=node_run_config,
        )
        plan_summary = ""
        try:
            async for ev in plan_events:
                if hasattr(ev, "content") and ev.content:
                    for p in getattr(ev.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            plan_summary += p.text
        except Exception as e:
            logger.warning("Step %d [Plan Node notice]: %s", self.step_index, e)

        logger.info("Step %d [Phase 2: Plan Complete] Strategy: %s...", self.step_index, plan_summary[:120].strip())

        # ---------------------------------------------------------------------
        # Phase 3: Action Execution (Act Node) - 最小限ツール: game-controller
        # ---------------------------------------------------------------------
        self.action_tools.pending_decision = None

        act_prompt = (
            f"=== IMMEDIATE SUBGOAL & STRATEGY (Phase 2) ===\n"
            f"{plan_summary}\n\n"
            f"Step: {self.step_index}, Game State: {state_str}, Available Action Buttons: {available_action_names}\n"
            f"Execute your 1-step action by calling `step_action` or `click_at` tool."
        )
        act_content = Content(role="user", parts=[Part.from_text(text=act_prompt)])
        act_events = self.act_runner.run_async(
            session_id=self.act_session_id,
            user_id=user_id,
            new_message=act_content,
            run_config=node_run_config,
        )
        act_raw_text = ""
        try:
            async for ev in act_events:
                if hasattr(ev, "content") and ev.content:
                    for p in getattr(ev.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            act_raw_text += p.text
                if self.action_tools.pending_decision is not None:
                    break
        except (Exception, GeneratorExit) as e:
            pass

        if self.action_tools.pending_decision is not None:
            decision = self.action_tools.pending_decision
            logger.info(
                "Step %d [Act Node: Tool Call Succeeded] Action: %s (ID: %d) Reasoning: %s",
                self.step_index, decision.action_name, decision.action_id, decision.reasoning
            )
        else:
            proposal = PlanProposal.from_text(act_raw_text)
            if not proposal.action and (perceive_summary or plan_summary):
                # モックモデル等のフォールバック抽出
                proposal = PlanProposal.from_text(f"{act_raw_text}\n{plan_summary}\n{perceive_summary}")

            decision = self._convert_to_decision(
                action_str=proposal.action,
                coordinates=proposal.coordinates,
                reasoning=proposal.reasoning or plan_summary or proposal.hypothesis,
                available_action_ids=available_action_ids,
                grid_shape=grid_shape,
                grid=grid,
                loaded_skill=proposal.load_skill,
            )

        # game-controller によるアクション拒絶時の自律的再検討 (Re-think) ループ
        retry_count = 0
        max_retries = 2
        while not decision.metadata.get("success", True) and retry_count < max_retries:
            retry_count += 1
            err_msg = decision.metadata.get("error", "Invalid action proposed")
            logger.warning(
                "Step %d [Action Rejected by game-controller] Attempt %d: %s. Requesting re-thinking...",
                self.step_index, retry_count, err_msg
            )
            feedback_prompt = (
                f"=== ACTION REJECTED BY GAME CONTROLLER ===\n"
                f"Error: {err_msg}\n"
                f"Currently available actions in this game are: {available_action_ids}\n"
                f"You CANNOT perform this action. You must re-think your hypothesis, respect available actions, and choose from the available actions.\n"
                f"Output your revised JSON block enclosed in ```json ... ``` with a valid action."
            )
            feedback_content = Content(role="user", parts=[Part.from_text(text=feedback_prompt)])
            retry_events = self.act_runner.run_async(
                session_id=self.act_session_id,
                user_id=user_id,
                new_message=feedback_content,
            )
            retry_raw_text = ""
            async for ev in retry_events:
                if hasattr(ev, "content") and ev.content:
                    for p in getattr(ev.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            retry_raw_text += p.text

            proposal = PlanProposal.from_text(retry_raw_text)
            decision = self._convert_to_decision(
                action_str=proposal.action,
                coordinates=proposal.coordinates,
                reasoning=proposal.reasoning or proposal.hypothesis,
                available_action_ids=available_action_ids,
                grid_shape=grid_shape,
                grid=grid,
                loaded_skill=proposal.load_skill,
                metadata={"rethink_attempts": retry_count, "rethink_feedback": err_msg},
            )
            if decision.metadata.get("success", True):
                logger.info(
                    "Step %d [Re-think Succeeded] Planner self-corrected action to %s (ID: %d) after %d attempt(s)",
                    self.step_index, decision.action_name, decision.action_id, retry_count
                )
                break

        # リトライ上限を超えても無効な場合の最外周フェイルセーフ
        if not decision.metadata.get("success", True):
            default_id = available_action_ids[0] if available_action_ids else 1
            logger.error(
                "Step %d [Re-think Exhausted] Action still invalid after %d retries. Applying safety fail-safe to ACTION%d.",
                self.step_index, retry_count, default_id
            )
            decision.action_id = default_id
            decision.action_name = f"ACTION{default_id}"
            decision.action_type = "STEP"
            decision.reasoning += f" [Safety fail-safe after {retry_count} re-thinks: {decision.metadata.get('error')}]"

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
        grid: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionDecision:
        """game-controller メタスキルを用いて安全・決定論的に ActionDecision へ変換."""
        h, w = grid_shape
        meta_dict = dict(metadata or {})

        if self.action_tools is not None and getattr(self.action_tools, "controller", None) is not None:
            controller = self.action_tools.controller
            controller.set_available_actions(available_action_ids)
            controller.set_dynamics_map(self.dynamics_map)
        elif GameController is not None:
            controller = GameController(
                available_actions=available_action_ids,
                dynamics_map=self.dynamics_map,
            )
        else:
            controller = None

        if controller is not None:
            payload: Dict[str, Any] = {
                "action": action_str,
                "reasoning": reasoning,
            }
            if coordinates and "x" in coordinates and "y" in coordinates:
                payload["x"] = coordinates["x"]
                payload["y"] = coordinates["y"]

            validation = controller.parse_and_validate(
                payload,
                available_actions=available_action_ids,
                grid_shape=(h, w),
                grid=grid,
            )
            meta_dict["success"] = validation.get("success", True)
            meta_dict["error"] = validation.get("error")
            return ActionDecision(
                action_type=validation.get("action_type", "STEP"),
                action_name=validation.get("action_name", "UNKNOWN"),
                action_id=validation.get("action_id", available_action_ids[0] if available_action_ids else 1),
                coordinates=validation.get("coordinates"),
                reasoning=validation.get("reasoning", reasoning),
                loaded_skill=loaded_skill,
                metadata=meta_dict,
            )

        # 最低限のフォールバック
        default_id = available_action_ids[0] if available_action_ids else 1
        meta_dict["success"] = True
        return ActionDecision(
            action_type="STEP",
            action_name=f"ACTION{default_id}",
            action_id=default_id,
            coordinates=None,
            reasoning=f"Fallback action: {reasoning}",
            loaded_skill=loaded_skill,
            metadata=meta_dict,
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
