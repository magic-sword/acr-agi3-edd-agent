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
from google.adk.runners import Runner
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
from acr_agi3.tools.observation_tools import ObservationTools

logger = logging.getLogger(__name__)

# game-controller メタスキルのインポート
GAME_CONTROLLER_DIR = Path(__file__).resolve().parents[3] / "meta_skills" / "game-controller" / "scripts"
if str(GAME_CONTROLLER_DIR) not in sys.path and GAME_CONTROLLER_DIR.exists():
    sys.path.insert(0, str(GAME_CONTROLLER_DIR))

try:
    from game_controller import GameController
except ImportError:
    GameController = None


class ADKGamePlayer:
    """Google ADK 2.0 準拠・自律ゲームプレイエージェント."""

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

        # 1. 視覚入力ハーネス (visual-inspector) & 1手出力ツール (game-controller)
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()
        self.skill_harness = SkillHarness()
        self.skill_toolset = self.skill_harness.get_toolset()
        self.observation_tools = ObservationTools()

        # 2. 動的操作力学マップ (ボタンと移動方向の同定結果: 例 {'UP': 3, 'DOWN': 4})
        self.dynamics_map: Dict[str, int] = {}

        # 3. Google ADK 2.0 Planner Agent
        planner_instruction = (
            "You are an expert autonomous game player playing ARC-AGI-3 dynamic interactive games.\n\n"
            "=== CORE OBJECTIVE & ARC-AGI-3 GAME PRINCIPLES ===\n"
            "1. MISSION: Play the game actively, adapt to unknown dynamics, and achieve victory (WIN / CLEAR).\n"
            "2. ADAPTIVE DISCOVERY: Game environments, physical mechanics, and keybindings change drastically across tasks. "
            "No rules are provided upfront. You must discover and verify the rules yourself by observing visual feedback "
            "(the highlighted glowing button on the controller HUD and the resulting pixel displacement Δ on the board).\n"
            "3. STEP MINIMIZATION: Aim to clear the game in the fewest possible steps. Avoid blind trial-and-error, wall bumping, "
            "or looping in traps. Every single action must be intentional and purposeful.\n\n"
            "=== PLANNING-FIRST PROTOCOL & REASONING (REACT LOOP) ===\n"
            "Before issuing any action, you must always construct an explicit action plan:\n"
            "  Step A. SITUATION ANALYSIS: Inspect the integrated console image. "
            "Identify candidate controllable objects, target patterns/goals, hazards, and layout geometry.\n"
            "  Step B. ON-DEMAND OBSERVATION (TOOLS): You have access to specialized observation and skill tools. "
            "If you need deeper clarity or want to confirm hypotheses, call tools before finalizing your move:\n"
            "    - `inspect_board(target_pattern=None)`: Global board geometry, colors, and estimated game style.\n"
            "    - `inspect_affordances(detail=True)`: Candidate player position, target goal positions, and obstacles.\n"
            "    - `inspect_action_effect()`: Last action's displacement, pixel delta, and keybinding verification.\n"
            "    - `inspect_roi(top, left, height, width)`: Close-up zoom into critical maze junctions or intricate patterns.\n"
            "    - Skill tools (`list_skills`, `load_skill`, `run_skill_script`): Load or run meta-skills on demand.\n"
            "  Step C. GOAL FORMATION: Formulate a clear immediate subgoal that brings you closer to victory.\n"
            "  Step D. ACTION SELECTION: Choose the most efficient 1-step action that advances your subgoal.\n\n"
            "=== STRUCTURED OUTPUT JSON FORMAT ===\n"
            "When you are ready to execute your 1-step action, respond with your final JSON block enclosed in ```json ... ```:\n"
            "```json\n"
            "{\n"
            '  "hypothesis": "What do you perceive? (e.g. entities, roles, observed causal mechanics from last action)",\n'
            '  "goal": "Immediate subgoal (e.g. move toward switch, align color block, test unexplored direction)",\n'
            '  "action": "UP" | "DOWN" | "LEFT" | "RIGHT" | "click_at" | "RESET" | "ACTION1"..."ACTION7",\n'
            '  "coordinates": {"x": col, "y": row}, // Specify if action is click_at or ACTION6 (optional)\n'
            '  "reasoning": "Why this specific action is the optimal step to achieve the goal with fewest moves"\n'
            "}\n"
            "```\n"
        )
        all_tools: List[Any] = []
        if self.skill_toolset:
            all_tools.append(self.skill_toolset)
        all_tools.extend(self.observation_tools.get_tools())

        self.planner_agent = Agent(
            name=f"{self.name}_planner",
            model=self.model,
            tools=all_tools,
            instruction=planner_instruction,
        )

        self.session_service = InMemorySessionService()
        self.planner_runner = Runner(
            agent=self.planner_agent,
            app_name=f"{self.app_name}_planner",
            session_service=self.session_service,
            auto_create_session=True,
        )

        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid: Optional[np.ndarray] = None
        self.last_action_info: Optional[Dict[str, Any]] = None
        self.planner_session_id: Optional[str] = None

    def reset(self) -> None:
        """エージェントの状態とセッションを初期化."""
        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid = None
        self.last_action_info = None
        self.action_tools.pending_decision = None
        self.action_tools.history.clear()
        self.planner_session_id = None

    def decide_next_action(
        self,
        grid: Any,
        available_actions: Optional[List[int]] = None,
        state_str: str = "NOT_FINISHED",
    ) -> ActionDecision:
        """最新観測から Planner の画像認識思考を経て game-controller で 1 手を実行."""
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

        # 観測ツールのコンテキスト更新 (ReAct ループ中にエージェントが参照可能)
        self.observation_tools.update_context(
            grid=arr,
            prev_grid=self.last_grid,
            step_index=self.step_index,
            last_action_info=self.last_action_info,
            dynamics_map=self.dynamics_map,
            available_actions=avail_names,
        )

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
                grid_shape=arr.shape[:2],
                grid=arr,
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
        grid_shape: Tuple[int, int],
        grid: Optional[np.ndarray] = None,
    ) -> ActionDecision:
        """Planner による ReAct 推論 (Tool-Calling) -> game-controller による決定論的 1 手確定."""
        user_id = "arc_workflow_user"

        need_new_session = (
            self.planner_session_id is None
            or self.session_refresh_interval <= 1
            or (self.step_index % self.session_refresh_interval == 1)
        )

        if need_new_session:
            s_plan = await self.session_service.create_session(
                app_name=f"{self.app_name}_planner", user_id=user_id
            )
            self.planner_session_id = s_plan.id
        else:
            await self._archive_session_images(f"{self.app_name}_planner", self.planner_session_id, user_id)

        # 思考・行動立案 (Planner)
        plan_content = Content(role="user", parts=obs_parts)
        planner_events = self.planner_runner.run_async(
            session_id=self.planner_session_id,
            user_id=user_id,
            new_message=plan_content,
        )

        plan_raw_text = ""
        rethink_count = 0
        async for ev in planner_events:
            if hasattr(ev, "content") and ev.content:
                for p in getattr(ev.content, "parts", []):
                    if hasattr(p, "function_call") and p.function_call:
                        rethink_count += 1
                        logger.info(
                            "Step %d [ReAct Loop #%d] Tool Call: %s(args=%s)",
                            self.step_index, rethink_count, p.function_call.name, p.function_call.args
                        )
                    elif hasattr(p, "function_response") and p.function_response:
                        resp_preview = str(p.function_response.response)[:150]
                        logger.info(
                            "Step %d [ReAct Loop #%d] Tool Response: %s -> %s...",
                            self.step_index, rethink_count, p.function_response.name, resp_preview
                        )
                    elif hasattr(p, "text") and p.text:
                        plan_raw_text += p.text

        logger.info(
            "Step %d [Planner Complete] (ReAct Tool Calls: %d) Raw: %r",
            self.step_index, rethink_count, plan_raw_text
        )
        proposal = PlanProposal.from_text(plan_raw_text)

        # game-controller スキルを通じた決定論的変換
        decision = self._convert_to_decision(
            action_str=proposal.action,
            coordinates=proposal.coordinates,
            reasoning=proposal.reasoning or proposal.hypothesis,
            available_action_ids=available_action_ids,
            grid_shape=grid_shape,
            grid=grid,
            loaded_skill=proposal.load_skill,
        )

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
        meta_dict = metadata or {}

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
            return ActionDecision(
                action_type=validation["action_type"],
                action_name=validation["action_name"],
                action_id=validation["action_id"],
                coordinates=validation.get("coordinates"),
                reasoning=validation.get("reasoning", reasoning),
                loaded_skill=loaded_skill,
                metadata=meta_dict,
            )

        # 最低限のフォールバック
        default_id = available_action_ids[0] if available_action_ids else 1
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
