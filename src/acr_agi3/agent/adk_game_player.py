"""Google ADK 2.0 準拠・自律ワークフロー駆動型ゲームプレイエージェント (ADKGamePlayer).

プロンプトへの禁則事項の肥大化を廃止し、Google ADK 2.0 の構造的ワークフロー
（思考・計画: Planner -> 実行: GameController）
によって高速かつ質の高い自律的行動決定と自己改善ループを実現するエージェント。
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
from acr_agi3.agent.online_skill_developer import OnlineSkillDeveloper
from acr_agi3.agent.workflow_schemas import PlanProposal
from acr_agi3.harness.game_action_tools import ActionDecision, GameActionTools
from acr_agi3.harness.vision_observation import (
    VisionObservationHarness,
    detect_interactable_objects,
    normalize_grid,
)
from acr_agi3.meta.skill_harness import SkillHarness

logger = logging.getLogger(__name__)

# リアルタイム監視用ログファイル (logs/agent_live_execution.log) のセットアップ
LIVE_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "logs" / "agent_live_execution.log"
LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# acr_agi3 パッケージ全体のログを統一的に LIVE_LOG_PATH へ集約
_pkg_logger = logging.getLogger("acr_agi3")
_pkg_logger.setLevel(logging.INFO)
_abs_log_path = str(LIVE_LOG_PATH.resolve())
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == _abs_log_path for h in _pkg_logger.handlers):
    _file_handler = logging.FileHandler(LIVE_LOG_PATH, mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
    _pkg_logger.addHandler(_file_handler)

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
    """Google ADK 2.0 準拠・Plan-Act ワークフロー自律ゲームプレイエージェント."""

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
        if autonomous_probing is not None:
            self.autonomous_probing = autonomous_probing
        else:
            # mock-model (結合テスト用モック) の場合は自動で自律プローブを有効化
            self.autonomous_probing = (self.model == "mock-model")

        # 1. 視覚認識ハーネス & 操作ツール
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()

        # 2. ADK 2.0 公式 SkillToolset (Progressive Disclosure)
        self.skill_harness = SkillHarness()
        self.skill_toolset = self.skill_harness.get_toolset()

        # 3. エピソード記憶マネージャー (Transcript からの因果蒸留 & Working Memory)
        self.memory = EpisodicMemoryManager(capacity=100) if EpisodicMemoryManager else None

        # 4. Google ADK 2.0 Planner Agent (階層的逆算プランナー)
        planner_instruction = (
            "You are the Chief Strategist and Backward Planning Agent playing ARC-AGI-3 dynamic games.\n"
            "Your core competence is HIERARCHICAL SUBGOAL DECOMPOSITION (Backward Chaining):\n"
            "1. Inspect the terminal goal/target pattern and working memory.\n"
            "2. Break down the objective into ordered intermediate subgoals (e.g. key before door, isolate pieces into staging buffer).\n"
            "3. Honestly declare whether the method to achieve the immediate subgoal is KNOWN or UNKNOWN:\n"
            "   - If UNKNOWN (actuator laws or physics unverified): set `\"need_probe\": true` and `\"cognitive_intent\": \"PROBE\"` to request experimental validation.\n"
            "   - If KNOWN (rules verified): select the purposeful action to advance the immediate milestone.\n\n"
            "=== STRUCTURED OUTPUT JSON FORMAT ===\n"
            "```json\n"
            "{\n"
            '  "hypothesis": "Mental model of board entities, colors, and actuator physics",\n'
            '  "goal": "Ultimate game objective (e.g. arrange blocks horizontally R-O-B-G)",\n'
            '  "subgoal": "Immediate milestone (e.g. move blue obstacle to bottom staging buffer)",\n'
            '  "cognitive_intent": "PLAN" | "PROBE" | "DIRECT",\n'
            '  "need_probe": false, // Set true if actuator dynamics are unverified or you need to experiment\n'
            '  "action": "UP" | "DOWN" | "LEFT" | "RIGHT" | "click_at" | "ACTION1"..."ACTION7",\n'
            '  "coordinates": {"x": col, "y": row}, // REQUIRED if action is click_at or ACTION6!\n'
            '  "reasoning": "Logical explanation of how this action advances the immediate milestone"\n'
            "}\n"
            "```\n"
            "Consult expert meta-skills when needed: `\"load_skill\": \"backward-planner\"` or `\"epistemic-prober\"`.\n"
        )
        self.planner_agent = Agent(
            name=f"{self.name}_planner",
            model=self.model,
            tools=[self.skill_toolset],
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

        # オンライン自己改善・スキル開発エンジン (仮説検証プローブ・ルール同定・EDD契約テスト・動的スキル生成)
        self.skill_developer = OnlineSkillDeveloper(game_id=self.name)

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
        if hasattr(self, "skill_developer"):
            self.skill_developer.reset()

    def decide_next_action(
        self,
        grid: Any,
        available_actions: Optional[List[int]] = None,
        state_str: str = "NOT_FINISHED",
    ) -> ActionDecision:
        """ADK 2.0 Plan-Act ワークフローと自己改善エンジンを経て次のアクションを決定."""
        self.step_index += 1
        arr = normalize_grid(grid)

        # 差分情報の算出
        pixels_changed = 0
        is_effective = False
        if self.last_grid is not None and self.last_grid.shape == arr.shape:
            diff_mask = (self.last_grid != arr)
            pixels_changed = int(np.sum(diff_mask))
            is_effective = (pixels_changed > 0)

        if self.last_action_info is not None:
            act_n = self.last_action_info.get("action", "UNKNOWN")
            logger.info(
                "⚡ [Direct Action Outcome] Action %s produced ΔPixels: %d (Effective: %s, Stagnation: %d)",
                act_n, pixels_changed, is_effective, self.stagnation_count
            )

        # オンライン自己改善エンジンによるプローブ・遷移の因果差分解析
        if self.last_action_info is not None and self.last_grid is not None:
            last_act_name = self.last_action_info.get("action", "UNKNOWN")
            self.skill_developer.analyze_probe_transition(
                prev_grid=self.last_grid,
                curr_grid=arr,
                action_id=last_act_name,
            )

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
        self.action_tools.set_dynamics_map(self.skill_developer.action_semantics)

        # 1. 停滞検出時のマクロポリシー解除（手詰まり・壁衝突時は即座に解除し LLM / Taboo Reset へ委託）
        if self.stagnation_count >= 2 and self.skill_developer.active_policy is not None:
            logger.warning(
                "🛡️ [OnlineSkillDeveloper] Stagnation detected (%d steps) - Deactivating macro policy '%s'",
                self.stagnation_count, self.skill_developer.active_skill_name
            )
            self.skill_developer.reset_policy()

        # 2. ルール同定完了時: EDD防壁ゲート（正例3+負例3）付き動的スキル生成
        if (
            self.skill_developer.is_rule_identified
            and self.skill_developer.active_policy is None
            and not getattr(self.skill_developer, "policy_suppressed", False)
            and self.stagnation_count == 0
        ):
            synth_ok = self.skill_developer.develop_and_register_skill(
                grid=arr,
                available_actions=avail_ids,
            )
            if synth_ok:
                logger.info(
                    "🎉 [OnlineSkillDeveloper] Successfully synthesized & EDD-verified skill '%s'!",
                    self.skill_developer.active_skill_name,
                )

        # 3. 高速マクロポリシー実行（承認済みスキルが存在し停滞していない場合、LLM推論を完全バイパス）
        if self.stagnation_count < 2 and self.skill_developer.active_policy is not None:
            macro_decision = self.skill_developer.execute_active_policy(arr, avail_ids)
            if macro_decision is not None:
                self.last_grid = arr.copy()
                self.last_action_info = {
                    "action": macro_decision.action_name,
                    "action_id": macro_decision.action_id,
                    "reasoning": macro_decision.reasoning,
                    "state_before": state_str,
                }
                logger.info(
                    "⚡ [Fast Macro Exec] Step %d: %s (skill=%s) -> LLM bypassed (1ms)",
                    self.step_index, macro_decision.action_name, macro_decision.loaded_skill
                )
                return macro_decision

        # 4. 能動的プローブ行動（ルール未確定かつ自律プローブが有効な場合）
        if self.autonomous_probing and self.skill_developer.is_probing:
            probe_decision = self.skill_developer.get_next_probe_action(avail_ids)
            if probe_decision is not None:
                self.last_grid = arr.copy()
                self.last_action_info = {
                    "action": probe_decision.action_name,
                    "action_id": probe_decision.action_id,
                    "reasoning": probe_decision.reasoning,
                    "state_before": state_str,
                }
                logger.info(
                    "🧪 [Epistemic Probe Exec] Step %d: %s (skill=%s) -> %s",
                    self.step_index, probe_decision.action_name, probe_decision.loaded_skill,
                    probe_decision.reasoning
                )
                return probe_decision

        # 5. 通常ワークフロー: 視覚観測 Parts の生成 (Phase 1: OBSERVE)
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
            current_subgoal=getattr(self, "last_subgoal", ""),
            cognitive_intent=getattr(self, "last_cognitive_intent", "PLAN"),
            method_known=getattr(self, "last_method_known", True),
            is_probing=self.skill_developer.is_probing,
            active_macro_skill=self.skill_developer.active_skill_name,
            is_macro_mode=(self.skill_developer.active_policy is not None),
        )
        dummy_ctx = MagicMock(spec=Context)
        assess_cognitive_state(dummy_ctx, cog_state)
        active_skill = cog_state.selected_skill or "backward-planner"

        logger.info(
            "🧠 [Cognitive Routing] Step %d: Active Skill='%s' (Mode='%s', Stagnation=%d)\n"
            "   Reason: %s\n"
            "   Working Memory: %s",
            self.step_index, active_skill, cog_state.cognitive_mode.upper(), self.stagnation_count,
            cog_state.routing_reason,
            memory_summary.replace("\n", " | ") if memory_summary else "(empty)"
        )

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
            self._run_plan_act_workflow(
                obs_parts=parts,
                available_action_ids=avail_ids,
                avail_names=avail_names,
                memory_summary=memory_summary,
                grid_shape=arr.shape[:2],
                active_skill=active_skill,
                grid=arr,
                cog_state=cog_state,
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

    async def _run_plan_act_workflow(
        self,
        obs_parts: List[Part],
        available_action_ids: List[int],
        avail_names: List[str],
        memory_summary: str,
        grid_shape: Tuple[int, int],
        active_skill: Optional[str] = None,
        grid: Optional[np.ndarray] = None,
        cog_state: Optional[CognitiveState] = None,
    ) -> ActionDecision:
        """Planner -> Act の高速単一推論ワークフロー."""
        user_id = "arc_workflow_user"

        # 1. セッションの定期的新設 (会話履歴の肥大化・自己模倣ループ・CUDA OOM を完全防止)
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

        logger.info(f"Step {self.step_index} [Phase 1: Planner Raw] {plan_raw_text!r}")
        proposal = PlanProposal.from_text(plan_raw_text)
        original_proposed_action = proposal.action
        original_proposed_coords = proposal.coordinates
        logger.info(
            "Step %d [Phase 1: Proposed Plan] Hypothesis: %r | Goal: %r | Proposed Action: %s (Coords: %s) | Requested Skill: %s",
            self.step_index, proposal.hypothesis, proposal.goal, original_proposed_action, original_proposed_coords, proposal.load_skill
        )

        # Progressive Disclosure: メタスキルが要求された場合の展開
        triggered_skill = proposal.load_skill
        if triggered_skill:
            logger.info(f"Step {self.step_index} [Phase 1: Progressive Disclosure] Expanding requested skill: {triggered_skill}")
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
                original_proposed_action = proposal.action
                original_proposed_coords = proposal.coordinates
            except Exception as e:
                logger.warning(f"Failed to expand skill {triggered_skill}: {e}")
        else:
            logger.info(
                "Step %d [Phase 1: Progressive Disclosure] No on-demand skill requested by Planner. Active meta-skill remains '%s'",
                self.step_index, active_skill
            )

        chosen_action = proposal.action
        chosen_coords = proposal.coordinates

        # 高速ルールサニタイズ（LLM再推論不要の即時幾何補正: 1ms）
        # クリック行動で座標が欠落している場合は、画面内のアフォーダンス中心に即時オートスナップ
        if (chosen_action == "ACTION6" or chosen_action == "click_at" or "6" in str(chosen_action)) and not chosen_coords:
            if grid is not None:
                detected_objs = detect_interactable_objects(grid)
                if detected_objs:
                    best = detected_objs[0]
                    chosen_coords = {"x": best["center"]["x"], "y": best["center"]["y"]}
                    logger.info(
                        "⚡ [Fast Python Sanitize] Auto-snapped missing click coordinates to object center: %s",
                        chosen_coords
                    )

        # -------------------------------------------------------------
        # Phase 4: Act (確定行動の検証と ActionDecision 生成)
        # -------------------------------------------------------------
        # 認知ワークフローで選定されたメタスキルを反映 (明示指定された場合はそれを尊重)
        if proposal.load_skill_explicit:
            final_skill = proposal.load_skill
        else:
            final_skill = proposal.load_skill or active_skill

        self.last_subgoal = proposal.subgoal
        self.last_cognitive_intent = proposal.cognitive_intent
        self.last_method_known = proposal.method_known

        meta = {
            "original_action": original_proposed_action,
            "final_action": chosen_action,
            "subgoal": proposal.subgoal,
            "cognitive_intent": proposal.cognitive_intent,
            "method_known": proposal.method_known,
            "routing_reason": cog_state.routing_reason if cog_state else "",
            "cognitive_mode": cog_state.cognitive_mode if cog_state else "",
            "working_memory_snippet": memory_summary[:100] if memory_summary else "",
        }

        logger.info(
            "Step %d [Phase 4: Act Finalized] Action: %s, Coords: %s, Skill: %s, Hypothesis: %r",
            self.step_index, chosen_action, chosen_coords, final_skill, proposal.hypothesis
        )

        decision = self._convert_to_decision(
            action_str=chosen_action,
            coordinates=chosen_coords,
            reasoning=f"[{proposal.goal}] {proposal.reasoning or proposal.hypothesis}",
            available_action_ids=available_action_ids,
            grid_shape=grid_shape,
            loaded_skill=final_skill,
            grid=grid,
            metadata=meta,
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
        grid: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionDecision:
        """計画内容をメタスキル game-controller (GameController) を通じて安全・決定論的に ActionDecision へ変換."""
        h, w = grid_shape
        meta_dict = metadata or {}
        dynamics = getattr(self.skill_developer, "action_semantics", {})

        # GameController インスタンスの用意 (同定済み力学マップを注入)
        if self.action_tools is not None and getattr(self.action_tools, "controller", None) is not None:
            controller = self.action_tools.controller
            controller.set_available_actions(available_action_ids)
            controller.set_dynamics_map(dynamics)
        elif GameController is not None:
            controller = GameController(
                available_actions=available_action_ids,
                dynamics_map=dynamics,
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
            reasoning=f"Emergency fallback: {reasoning}",
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
