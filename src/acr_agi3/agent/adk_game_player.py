"""Google ADK 2.0 準拠・自律ゲームプレイエージェント (ADKGamePlayer).

過剰に複雑化したワークフローや多数のメタスキルを全廃し、
人間が画面を見てプレイするのと同様の、
「視覚入力 (visual-inspector) -> 画像認識思考 (Planner) -> 1手出力 (game-controller)」
の直線的かつ超高速な自律ゲームプレイを実現するエージェント。
"""

from __future__ import annotations

import asyncio
from enum import Enum
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
from acr_agi3.tools import MemoryTools, PlanningTools, SpatialTools, VisionTools

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


class CognitiveState(str, Enum):
    """Google ADK 2.0 認知的ステートマシン (Cognitive State Machine) の主要状態."""
    PROBING = "PROBING"          # [探索期] 操作力学・オブジェクト因果の能動解明
    PLANNING = "PLANNING"        # [計画期] 逆算A*・サブゴール策定・アクションキュー構築
    EXECUTING = "EXECUTING"      # [実行期] 計画追従・LLMバイパスの超高速サクサク実行
    RECOVERY = "RECOVERY"        # [回復期] 計画逸脱（壁衝突・0変化・罠）の診断・再計画


class CognitiveMode(str, Enum):
    """人間プレイスタイル (VCGT) に基づく 5段階認知的モード."""
    PROBING_SCIENTIST = "PROBING_SCIENTIST"      # [探索期] 操作力学の仮説検証・プロービング
    CAUSAL_PROGRAMMER = "CAUSAL_PROGRAMMER"      # [因果同定期] 鍵・扉・スイッチ等の前提条件規則発見
    BACKWARD_ARCHITECT = "BACKWARD_ARCHITECT"    # [逆算期] ゴールからの A* 最短経路・トポロジカル計画
    RISK_NAVIGATOR = "RISK_NAVIGATOR"            # [実行期] 局所障害物・トラップ回避ナビゲーション
    TABOO_RECOVERY = "TABOO_RECOVERY"            # [例外対処] 壁衝突・振動・停滞からの脱出・リセット


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

        # 1. ハーネスおよび Level 3 実行ツールの初期化
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()
        self.vision_tools = VisionTools()
        self.spatial_tools = SpatialTools()
        self.planning_tools = PlanningTools()
        self.memory_tools = MemoryTools()
        self.skill_harness = SkillHarness()

        # 2. 各ノード専用の最小限スキルセット（共有黒板 memory-notebook を全ノードへ解放）
        self.perceive_additional_tools = self.vision_tools.get_tools() + self.spatial_tools.get_tools() + self.memory_tools.get_tools()
        self.perceive_toolset = self.skill_harness.get_scoped_toolset(
            ["visual-inspector", "spatial-grounder", "memory-notebook"],
            additional_tools=self.perceive_additional_tools,
        )

        self.plan_additional_tools = self.memory_tools.get_tools() + self.planning_tools.get_tools() + self.spatial_tools.get_tools()
        self.plan_toolset = self.skill_harness.get_scoped_toolset(
            ["memory-notebook", "backward-planner", "spatial-grounder"],
            additional_tools=self.plan_additional_tools,
        )

        self.act_additional_tools = (
            self.action_tools.get_tools()
            + self.spatial_tools.get_tools()
            + self.planning_tools.get_tools()
            + self.memory_tools.get_tools()
        )
        self.act_toolset = self.skill_harness.get_scoped_toolset(
            ["game-controller", "spatial-grounder", "taboo-reset-guard", "epistemic-prober", "memory-notebook"],
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
            "1. You have access to the skills: 'visual-inspector', 'spatial-grounder'. You may call `load_skill(skill_name='visual-inspector')` if you need detailed inspection guides or scripts.\n"
            "2. You may also call inspection tools directly if needed: `inspect_board_summary()`, `inspect_affordances(mode='deep')`, `inspect_detected_objects()`, or `inspect_clickable_anchors()`.\n"
            "Output a concise visual observation summary:\n"
            "- Board layout geometry, active colors, and spatial symmetry\n"
            "- Discovered entities (player, goal, obstacles, movable blocks, clickable buttons)\n"
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
            "1. You have access to the skills: 'memory-notebook', 'backward-planner', 'spatial-grounder'. You may call `load_skill(skill_name=...)` if needed.\n"
            "2. You may use memory tools: `memory_write(section_id=..., content=...)`, `memory_read(section_id=...)`, `memory_toc()`, `memory_search(query=...)`.\n"
            "3. If planning an interaction or click, you can query candidate targets and coordinates using `inspect_clickable_anchors()` or `inspect_detected_objects()`.\n"
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
            "1. You have access to the skills: 'game-controller', 'spatial-grounder'. Call `load_skill(skill_name='game-controller')` if needed.\n"
            "2. To inspect click targets on-demand, call `inspect_clickable_anchors()`, `inspect_detected_objects()`, or `inspect_cursor()`.\n"
            "3. Mouse Cursor & Clicking (Two-Stage Safe Aiming):\n"
            "   - Current cursor position is shown on the game board image with reticle `[ + ]` and in the HUD as CURSOR: (col, row).\n"
            "   - `move_cursor(x=col, y=row, reasoning='...')`: Move cursor to aim at target without consuming environment turns.\n"
            "   - `click_at_cursor(reasoning='...')`: Fire click at current cursor position (consumes turn).\n"
            "   - `click_at(x=col, y=row, reasoning='...')`: Direct click (moves cursor and fires click).\n"
            "4. Directional Steps & Reset:\n"
            "   - `step_action(action='...', reasoning='...')`: Execute move using D-Pad ('UP', 'DOWN', 'LEFT', 'RIGHT') or button ('ACTION1'-'ACTION7').\n"
            "   - `reset_game(reasoning='...')`: Reset level when deadlocked.\n"
            "DO NOT call load_skill with action names (e.g. do NOT call load_skill('ACTION1')). Always use execution tools."
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
        self.current_cognitive_mode: CognitiveMode = CognitiveMode.PROBING_SCIENTIST
        self.cognitive_state: CognitiveState = CognitiveState.PROBING
        self.plan_queue: List[Dict[str, Any]] = []
        self.last_expected_action: Optional[str] = None
        self.current_game_id: str = "default"
        self.game_dynamics: Dict[str, Dict[str, int]] = {}
        self.taboo_click_coords: List[Tuple[int, int]] = []
        self.cursor_pos: Tuple[int, int] = (32, 32)

    def switch_game(self, game_id: str) -> None:
        """指定されたゲーム環境 (game_id) に切り替え、環境ごとのノートと力学を復元."""
        if not game_id or game_id == self.current_game_id:
            return
        logger.info("Switching game environment from %s to %s", self.current_game_id, game_id)
        self.current_game_id = game_id
        self.memory_tools.switch_game(game_id)
        self.dynamics_map = self.game_dynamics.setdefault(game_id, {})
        self.action_tools.set_dynamics_map(self.dynamics_map)
        if self.planning_tools.prober is not None:
            self.planning_tools.prober.dynamics_map = dict(self.dynamics_map)
        self.cursor_pos = (32, 32)
        if self.action_tools.controller is not None:
            self.action_tools.controller.set_cursor(32, 32)
        self.reset(full_wipe=False)

    def reset(self, full_wipe: bool = False) -> None:
        """エージェントの状態とセッションを初期化.

        Args:
            full_wipe: True の場合、ノートブックと操作力学マップを全消去します。
                       False（デフォルト）の場合、同一ゲームのリトライとして
                       causality.*, rules.*, taboo.* などの永続知識を保持し、
                       破綻した一時的計画 (plan.active, plan_queue) のみをリフレッシュします。
        """
        self.step_index = 0
        self.stagnation_count = 0
        self.last_grid = None
        self.last_action_info = None
        self.action_tools.pending_decision = None
        self.action_tools.history.clear()
        self.cursor_pos = (32, 32)
        if self.action_tools.controller is not None:
            self.action_tools.controller.set_cursor(32, 32)
        self.perceive_session_id = None
        self.plan_session_id = None
        self.act_session_id = None
        self.planner_session_id = None
        self.current_cognitive_mode = CognitiveMode.PROBING_SCIENTIST
        self.cognitive_state = CognitiveState.PROBING
        self.plan_queue.clear()
        self.last_expected_action = None

        if full_wipe:
            self.memory_tools.clear()
            self.dynamics_map.clear()
            self.action_tools.set_dynamics_map({})
            self.taboo_click_coords.clear()
            if self.planning_tools.prober is not None:
                self.planning_tools.prober.dynamics_map.clear()
                self.planning_tools.prober.tested_actions.clear()
        else:
            self.memory_tools.reset_episode()

    def determine_cognitive_mode(
        self,
        step_index: int,
        stagnation_count: int,
        available_action_ids: List[int],
        has_probe_rec: bool,
        has_nav_path: bool,
        has_anchors: bool = False,
        has_preconditions: bool = False,
    ) -> CognitiveMode:
        """決定論的状態判定による認知的モード (Cognitive Mode) の決定.

        人間プレイスタイル (VCGT: 探針科学者 -> 因果プログラマー -> 逆算建築家 -> 動的ナビゲーター -> 禁忌ガード)
        の思考遷移をワークフロー層で自律制御する。

        優先順位:
        1. TABOO_RECOVERY: 停滞・壁衝突・デッドロック検知時は最優先で回避行動をとる
        2. PROBING_SCIENTIST: 初動 (step_index <= 4) で未検証アクションが存在する場合
        3. CAUSAL_PROGRAMMER: 鍵・扉・スイッチ等の前提条件が存在、またはクリックアンカーが存在する場合
        4. BACKWARD_ARCHITECT: ゴールまでの A* 最短幾何経路が同定できている場合
        5. RISK_NAVIGATOR: 通常の障害物・トラップ回避ナビゲーション
        """
        if stagnation_count >= 1:
            return CognitiveMode.TABOO_RECOVERY
        if step_index <= 4 and has_probe_rec:
            return CognitiveMode.PROBING_SCIENTIST
        if has_preconditions:
            return CognitiveMode.CAUSAL_PROGRAMMER
        if has_nav_path:
            return CognitiveMode.BACKWARD_ARCHITECT
        if has_anchors and 6 in available_action_ids:
            return CognitiveMode.CAUSAL_PROGRAMMER
        return CognitiveMode.RISK_NAVIGATOR

    def decide_next_action(
        self,
        grid: Any,
        available_actions: Optional[List[int]] = None,
        state_str: str = "NOT_FINISHED",
        game_id: Optional[str] = None,
    ) -> ActionDecision:
        """最新観測から 3フェーズ (Perceive -> Plan -> Act) を経て game-controller で 1 手を実行."""
        if game_id and game_id != self.current_game_id:
            self.switch_game(game_id)
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
        self.action_tools.set_available_actions(avail_ids)
        self.action_tools.set_dynamics_map(self.dynamics_map)
        self.action_tools.set_grid(arr)
        self.vision_tools.set_context(arr, step_index=self.step_index)
        self.spatial_tools.set_context(arr, step_index=self.step_index, last_grid=self.last_grid)
        self.planning_tools.set_context(arr, step_index=self.step_index, available_actions=avail_ids)
        self.memory_tools.set_step(self.step_index)

        # 操作力学同定 (Epistemic Prober) のオンライン更新
        if self.last_grid is not None and self.last_action_info is not None:
            self.planning_tools.prober.analyze_displacement(
                grid_before=self.last_grid,
                grid_after=arr,
                action_id=self.last_action_info.get("action_id", 1),
            )
            self.dynamics_map.update(self.planning_tools.prober.get_dynamics_map())
            self.action_tools.set_dynamics_map(self.dynamics_map)
            # 共有黒板 (memory-notebook) に同定済み力学を自動同期
            if self.dynamics_map:
                self.memory_tools.memory_write(
                    section_id="causality.dynamics",
                    title="Controller Mechanics Map",
                    content=json.dumps(self.dynamics_map, ensure_ascii=False),
                    summary=f"Mapped {len(self.dynamics_map)} controller actions",
                    tags="causality,dynamics",
                )

        # 十字キー (D-Pad: UP/DOWN/LEFT/RIGHT) およびアクションボタン名付きの直感的ラベル生成
        avail_names = self._format_available_action_labels(avail_ids)

        # ---------------------------------------------------------------------
        # 認知的ステートマシン: 失敗・計画逸脱（壁衝突・空振り・0ピクセル変化）検知と黒板記録
        # ---------------------------------------------------------------------
        if pixels_changed == 0 and self.last_action_info is not None:
            last_act_name = self.last_action_info.get("action_name") or self.last_expected_action or "UNKNOWN"
            last_act_id = self.last_action_info.get("action_id", 0)
            last_coords = self.last_action_info.get("coordinates") or {}

            logger.warning(
                "Step %d [Failure Detected] Action %s (ID: %s) coords=%s caused 0 pixel changes. Transitioning to RECOVERY.",
                self.step_index, last_act_name, last_act_id, last_coords
            )
            self.cognitive_state = CognitiveState.RECOVERY
            if self.plan_queue:
                self.plan_queue.clear()
            self.memory_tools.memory_delete("plan.active")

            # クリック失敗の場合: 禁忌座標として記録し、SpatialTools にも伝播
            if last_act_id == 6 or last_act_name == "ACTION6" or "x" in last_coords:
                cx = last_coords.get("x", 0)
                cy = last_coords.get("y", 0)
                if (cx, cy) not in self.taboo_click_coords:
                    self.taboo_click_coords.append((cx, cy))
                self.spatial_tools.set_taboo_coords(self.taboo_click_coords)
                self.memory_tools.memory_write(
                    section_id=f"taboo.click_{cx}_{cy}",
                    title=f"Failed Click at ({cx}, {cy})",
                    content=f"Click at coordinate (col={cx}, row={cy}) caused 0 pixel changes (inactive or missed object). Do not repeat this coordinate.",
                    summary=f"Inactive click coordinate ({cx}, {cy})",
                    tags="taboo,click,constraint",
                )
            else:
                # 移動・ボタン失敗の場合: 禁忌アクションとして記録 (互換性のために taboo.step_{step} も同期)
                self.memory_tools.memory_write(
                    section_id=f"taboo.step_{self.step_index}",
                    title=f"Taboo Barrier at Step {self.step_index}",
                    content=f"Action '{last_act_name}' (ID: {last_act_id}) caused 0 pixel change.",
                    summary="Wall bump or obstacle collision",
                    tags="taboo,constraint",
                )
                self.memory_tools.memory_write(
                    section_id=f"taboo.action_{last_act_id}",
                    title=f"Taboo Action {last_act_name} at Step {self.step_index}",
                    content=f"Action '{last_act_name}' (ID: {last_act_id}) caused 0 pixel change (wall bump or deadlocked position). Avoid repeating without state change.",
                    summary=f"Wall bump with {last_act_name}",
                    tags="taboo,movement,constraint",
                )

        # ---------------------------------------------------------------------
        # 認知的ステートマシン: 計画追従・高速サクサク実行 (Fast Path: LLMバイパス)
        # ---------------------------------------------------------------------
        if self.cognitive_state == CognitiveState.EXECUTING and len(self.plan_queue) > 0:
            self.current_cognitive_mode = CognitiveMode.BACKWARD_ARCHITECT
            next_plan = self.plan_queue.pop(0)
            action_str = next_plan.get("action", "")
            coords = next_plan.get("coordinates")
            reasoning = next_plan.get("reasoning", f"Fast execution along planned route ({len(self.plan_queue)} remaining in queue)")

            decision = self._convert_to_decision(
                action_str=action_str,
                coordinates=coords,
                reasoning=reasoning,
                available_action_ids=avail_ids,
                grid_shape=arr.shape[:2],
                grid=arr,
                metadata={"fast_path": True, "remaining_plan_steps": len(self.plan_queue)},
            )

            # Taboo Guard チェック (予期せぬ壁衝突の即時安全刈り込み)
            if self.planning_tools.guard is not None:
                last_aid = self.last_action_info.get("action_id") if self.last_action_info else None
                is_allowed, sanitized, reason = self.planning_tools.guard.filter_taboo_actions(
                    proposed_action=decision.action_id,
                    last_action_effective=True,
                    last_action_id=last_aid,
                    available_actions=avail_ids,
                    stagnation_count=self.stagnation_count,
                )
                if not is_allowed:
                    self.cognitive_state = CognitiveState.RECOVERY
                    self.plan_queue.clear()
                    decision.action_id = sanitized
                    decision.action_name = f"ACTION{sanitized}"
                    decision.reasoning += f" [TabooGuard: {reason}]"

            if decision.coordinates and "x" in decision.coordinates and "y" in decision.coordinates:
                self.cursor_pos = (int(decision.coordinates["x"]), int(decision.coordinates["y"]))
                if self.action_tools.controller is not None:
                    self.action_tools.controller.set_cursor(self.cursor_pos[0], self.cursor_pos[1])

            self.last_grid = arr.copy()
            self.last_expected_action = decision.action_name
            self.last_action_info = {
                "action": decision.action_name,
                "action_id": decision.action_id,
                "coordinates": decision.coordinates,
                "reasoning": decision.reasoning,
                "state_before": state_str,
                "pixels_changed": pixels_changed,
                "is_effective": is_effective,
            }
            logger.info(
                "Step %d [Cognitive State: EXECUTING (Fast Path)] %s (ID: %d), Queue remaining: %d",
                self.step_index, decision.action_name, decision.action_id, len(self.plan_queue)
            )
            if len(self.plan_queue) == 0:
                self.cognitive_state = CognitiveState.PLANNING
                self.memory_tools.memory_delete("plan.active")
            return decision

        # コントローラーのカーソル位置と同期
        if self.action_tools.controller is not None:
            self.cursor_pos = self.action_tools.controller.cursor

        # 視覚観測 Parts の生成 (統合コンソール画面 + 客観的事実 + マウスカーソル照準)
        parts = self.vision_harness.create_observation_parts(
            grid_data=arr,
            step_index=self.step_index,
            available_actions=avail_names,
            last_action_info=self.last_action_info,
            cursor_pos=self.cursor_pos,
        )

        # 非同期 Runner を実行 (Slow Path: 計画策定または能動プロービング)
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

        # アクション決定後のカーソル位置同期
        if decision.coordinates and "x" in decision.coordinates and "y" in decision.coordinates:
            self.cursor_pos = (int(decision.coordinates["x"]), int(decision.coordinates["y"]))
            if self.action_tools.controller is not None:
                self.action_tools.controller.set_cursor(self.cursor_pos[0], self.cursor_pos[1])
        elif self.action_tools.controller is not None:
            self.cursor_pos = self.action_tools.controller.cursor

        self.last_grid = arr.copy()
        self.last_expected_action = decision.action_name
        self.last_action_info = {
            "action": decision.action_name,
            "action_id": decision.action_id,
            "coordinates": decision.coordinates,
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
        # Python ワークフロー層: 認知的コンテキストの自動解析 (Cognitive Context Analysis)
        # ---------------------------------------------------------------------
        probe_rec = None
        if self.planning_tools.prober is not None and self.planning_tools.prober.is_probing_needed(self.step_index, available_action_ids):
            probe_rec = self.planning_tools.prober.recommend_probe_action(available_action_ids)

        nav_rec_dir = None
        nav_path_len = 0
        if grid is not None and self.vision_tools.inspector is not None and self.planning_tools.planner is not None:
            try:
                report = self.vision_tools.inspector.analyze_frame(grid, step_index=self.step_index)
                if report.agent_pos and report.goal_pos:
                    sc, sr = report.agent_pos[1], report.agent_pos[0]
                    gc, gr = report.goal_pos[1], report.goal_pos[0]
                    actions = self.planning_tools.plan_action_sequence(
                        start_col=sc,
                        start_row=sr,
                        goal_col=gc,
                        goal_row=gr,
                        impassable_colors=[1],
                    )
                    if actions:
                        nav_rec_dir = actions[0]
                        nav_path_len = len(actions)
                        if len(actions) > 1 and self.stagnation_count == 0 and self.cognitive_state != CognitiveState.RECOVERY:
                            self.plan_queue = [{"action": a} for a in actions[1:]]
                            self.cognitive_state = CognitiveState.EXECUTING
                            self.memory_tools.memory_write(
                                section_id="plan.active",
                                title="Active Shortest Path Plan",
                                content=json.dumps({"planned_actions": actions, "next_steps": [a["action"] for a in self.plan_queue]}, ensure_ascii=False),
                                summary=f"A* route with {len(actions)} steps",
                                tags="plan,active",
                            )
                            logger.info(
                                "Step %d [Plan Formulated] Enqueued %d future actions: %s. State -> EXECUTING.",
                                self.step_index, len(self.plan_queue), [a["action"] for a in self.plan_queue]
                            )
            except Exception as e:
                logger.debug("Automatic pathfinding notice: %s", e)

        anchors_hint = ""
        if 6 in available_action_ids and self.spatial_tools.cached_anchors:
            total_anchors = len(self.spatial_tools.cached_anchors)
            anchors_hint = (
                f"Interactive Elements Detected: {total_anchors} clickable candidate objects/anchors found on board.\n"
                "Query candidate coordinates, colors, and object IDs using `inspect_clickable_anchors()` or `inspect_detected_objects()`."
            )

        taboo_warning = ""
        if self.stagnation_count >= 1 and self.last_action_info:
            taboo_act = self.last_action_info.get("action")
            taboo_warning = f"NOTE: Last action {taboo_act} produced 0 pixel change (wall bump/stuck). Avoid repeating {taboo_act}."

        # 認知的モードの決定 (Cognitive Mode Selection)
        has_probe = (probe_rec is not None)
        has_nav = (nav_rec_dir is not None)
        has_anchors = bool(self.spatial_tools.cached_anchors)
        has_preconditions = False

        self.current_cognitive_mode = self.determine_cognitive_mode(
            step_index=self.step_index,
            stagnation_count=self.stagnation_count,
            available_action_ids=available_action_ids,
            has_probe_rec=has_probe,
            has_nav_path=has_nav,
            has_anchors=has_anchors,
            has_preconditions=has_preconditions,
        )
        if self.stagnation_count >= 1:
            self.cognitive_state = CognitiveState.RECOVERY
        elif len(self.plan_queue) > 0:
            self.cognitive_state = CognitiveState.EXECUTING
        elif probe_rec is not None:
            self.cognitive_state = CognitiveState.PROBING
        else:
            self.cognitive_state = CognitiveState.PLANNING

        logger.info(
            "Step %d [Cognitive Mode: %s] Stagnation: %d, PathLen: %d, Probe: %s",
            self.step_index,
            self.current_cognitive_mode.value,
            self.stagnation_count,
            nav_path_len,
            probe_rec,
        )

        mode_header = f"=== CURRENT COGNITIVE MODE: {self.current_cognitive_mode.value} ==="
        mode_instructions = []

        if self.current_cognitive_mode == CognitiveMode.TABOO_RECOVERY:
            if 6 in available_action_ids and len(available_action_ids) == 1:
                taboo_str = f"Recorded Taboo Clicks: {self.taboo_click_coords}" if self.taboo_click_coords else ""
                mode_instructions.append(
                    "🚨 [Mode: TABOO RECOVERY / INTERACTION RE-PLANNING]\n"
                    "The previous click action caused 0 pixel changes (target was inactive or missed).\n"
                    f"{taboo_str}\n"
                    "Goal: Re-plan your target! Choose a DIFFERENT target from taboo coordinates.\n"
                    "Aim safely: Use `move_cursor(x=col, y=row)` to align reticle, then `click_at_cursor()` to execute.\n"
                    "CRITICAL: Only ACTION6 (click) is available. Do NOT output movement directions (UP/DOWN/LEFT/RIGHT) or repeat taboo coordinates."
                )
            elif 6 in available_action_ids and any(a in [1, 2, 3, 4] for a in available_action_ids):
                mode_instructions.append(
                    "🚨 [Mode: TABOO RECOVERY / CAUSAL RE-PLANNING]\n"
                    "The previous action caused 0 pixel changes.\n"
                    f"{taboo_warning}\n"
                    f"Goal: Re-evaluate causal constraints and re-plan. Choose an orthogonal movement direction or test an alternative interactive switch/anchor from available actions: {available_action_names}. Avoid repeating the failed action."
                )
            else:
                mode_instructions.append(
                    "🚨 [Mode: TABOO RECOVERY / DEADLOCK ESCAPE]\n"
                    "The previous action caused 0 pixel changes (wall bump or deadlocked position).\n"
                    f"{taboo_warning}\n"
                    f"Goal: Re-plan movement route! Select an alternative valid direction or unblock action from available actions: {available_action_names}. Avoid repeating the failed action."
                )
        elif self.current_cognitive_mode == CognitiveMode.PROBING_SCIENTIST:
            mode_instructions.append(
                "🎯 [Mode: PROBING SCIENTIST / ONSET EXPLORATION]\n"
                "Controller physics/mappings are unmapped in this initial phase.\n"
                f"Recommended probe: Test ACTION{probe_rec} to observe board response and identify its movement vector."
            )
        elif self.current_cognitive_mode == CognitiveMode.BACKWARD_ARCHITECT:
            mode_instructions.append(
                "🧭 [Mode: BACKWARD ARCHITECT / GEOMETRIC SHORTEST PATH]\n"
                f"A* shortest path to goal identified ({nav_path_len} steps).\n"
                f"Recommended next movement: `{nav_rec_dir}`. Execute `{nav_rec_dir}` directly to advance along optimal path."
            )
        elif self.current_cognitive_mode == CognitiveMode.CAUSAL_PROGRAMMER:
            mode_instructions.append(
                "🖱️ [Mode: CAUSAL PROGRAMMER / DISCRETE INTERACTION]\n"
                f"{anchors_hint}\n"
                "Call `inspect_clickable_anchors()` or `inspect_detected_objects()` to inspect candidate targets, then call `click_at(x=col, y=row)` or `click_at(object_id=...)`."
            )
        else:  # RISK_NAVIGATOR
            mode_instructions.append(
                "🛡️ [Mode: RISK NAVIGATOR / ADAPTIVE NAVIGATION]\n"
                "Observe obstacles and boundaries carefully. Advance towards unexplored or promising regions while avoiding dead ends."
            )

        workflow_guidance = f"\n\n{mode_header}\n" + "\n\n".join(mode_instructions)

        # ---------------------------------------------------------------------
        # Phase 2: Cognitive Planning (Plan Node) - 最小限ツール: memory-notebook
        # ---------------------------------------------------------------------
        blackboard_toc = self.memory_tools.memory_toc(as_markdown=True)
        blackboard_section = ""
        if blackboard_toc and "*(Notebook is currently empty)*" not in blackboard_toc:
            blackboard_section = f"\n\n=== SHARED BLACKBOARD (Memory Notebook TOC) ===\n{blackboard_toc}\n(Use `memory_read(section_id=...)` to inspect details or `memory_write` to update knowledge.)"

        plan_prompt = (
            f"=== VISUAL PERCEPTION SUMMARY (Phase 1) ===\n"
            f"{perceive_summary}\n\n"
            f"Step: {self.step_index}, Game State: {state_str}, Available Action Buttons: {available_action_names}\n"
            f"{workflow_guidance}"
            f"{blackboard_section}\n\n"
            f"According to Cognitive Mode {self.current_cognitive_mode.value}, formulate your immediate subgoal and strategy. (Do not call any action tools like step_action)."
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

        act_guidance = "calling `step_action` or `click_at` tool."
        if 6 in available_action_ids and len(available_action_ids) == 1:
            act_guidance = (
                "using the 2-step mouse cursor workflow:\n"
                "  1. Call `move_cursor(x=col, y=row, reasoning='...')` to aim your reticle at the target element.\n"
                "  2. Check the reticle feedback (cell color & coords), then call `click_at_cursor(reasoning='...')` to fire.\n"
                "  (You may also call `click_at(x=col, y=row)` directly). CRITICAL: Only ACTION6 is available; do NOT call step_action."
            )
        elif 6 not in available_action_ids:
            act_guidance = "calling `step_action(direction=..., reasoning='...')`. (CRITICAL: Only directional buttons are available; do NOT call click_at or move_cursor)."

        act_prompt = (
            f"=== IMMEDIATE SUBGOAL & STRATEGY (Phase 2) ===\n"
            f"{plan_summary}\n\n"
            f"Step: {self.step_index}, Game State: {state_str}, Available Action Buttons: {available_action_names}\n"
            f"{workflow_guidance}\n\n"
            f"Execute your 1-step action aligned with Cognitive Mode {self.current_cognitive_mode.value} by {act_guidance}"
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
            proposal = PlanProposal.from_text(act_raw_text, available_action_ids=available_action_ids)
            if not proposal.action and (perceive_summary or plan_summary):
                # モックモデル等のフォールバック抽出
                proposal = PlanProposal.from_text(f"{act_raw_text}\n{plan_summary}\n{perceive_summary}", available_action_ids=available_action_ids)

            # ワークフローによる認知的推論補完 (決定論的パスまたはプローブ手の適用)
            if not proposal.action:
                if self.current_cognitive_mode == CognitiveMode.BACKWARD_ARCHITECT and nav_rec_dir is not None:
                    proposal.action = nav_rec_dir
                elif self.current_cognitive_mode == CognitiveMode.PROBING_SCIENTIST and probe_rec is not None:
                    proposal.action = f"ACTION{probe_rec}"
                elif nav_rec_dir is not None:
                    proposal.action = nav_rec_dir
                elif probe_rec is not None:
                    proposal.action = f"ACTION{probe_rec}"

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

            proposal = PlanProposal.from_text(retry_raw_text, available_action_ids=available_action_ids)
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
            decision.action_type = "CLICK" if default_id == 6 else "STEP"
            decision.reasoning += f" [Safety fail-safe after {retry_count} re-thinks: {decision.metadata.get('error')}]"

        # クリック座標の幾何接地 (Spatial Grounding: 座標未指定時の自動吸着 & 禁忌除外)
        if decision.action_id == 6 or decision.action_name == "ACTION6":
            coords = decision.coordinates or {}
            # エージェントが明示的に (x, y) を指定している場合はその生座標を尊重
            # 座標が未指定の場合のみ、禁忌でないアンカーへ自動フォールバック吸着
            if not coords or ("x" not in coords and "y" not in coords):
                valid_anchors = [
                    a for a in self.spatial_tools.cached_anchors
                    if not any(np.hypot(a["x"] - tx, a["y"] - ty) <= 3.0 for tx, ty in self.taboo_click_coords)
                ]
                if valid_anchors:
                    best = valid_anchors[0]
                    decision.coordinates = {"x": best["x"], "y": best["y"]}
                    decision.reasoning += f" [SpatialGrounder: Auto-snapped click to valid Anchor {best['id']} at ({best['x']}, {best['y']})]"
                elif self.spatial_tools.cached_anchors:
                    best = self.spatial_tools.cached_anchors[0]
                    decision.coordinates = {"x": best["x"], "y": best["y"]}
                    decision.reasoning += f" [SpatialGrounder: Fallback click to Anchor {best['id']} at ({best['x']}, {best['y']})]"

        # 禁忌アクション刈り込み＆能動的リセット (Taboo Reset Guard)
        if self.planning_tools.guard is not None:
            last_eff = self.last_action_info.get("is_effective", True) if self.last_action_info else True
            last_aid = self.last_action_info.get("action_id") if self.last_action_info else None
            is_allowed, sanitized, reason = self.planning_tools.guard.filter_taboo_actions(
                proposed_action=decision.action_id,
                last_action_effective=last_eff,
                last_action_id=last_aid,
                available_actions=available_action_ids,
                stagnation_count=self.stagnation_count,
            )
            if not is_allowed:
                decision.action_id = sanitized
                decision.action_name = f"ACTION{sanitized}"
                decision.reasoning += f" [TabooGuard: {reason}]"

            if self.planning_tools.guard.should_active_reset(self.stagnation_count) and 0 in available_action_ids:
                decision.action_id = 0
                decision.action_name = "RESET"
                decision.action_type = "RESET"
                decision.reasoning += f" [TabooGuard: Active reset triggered after {self.stagnation_count} stagnant steps]"

            self.planning_tools.guard.record_step(decision.action_id)

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

    def _format_available_action_labels(self, avail_ids: List[int]) -> List[str]:
        """利用可能アクションIDを十字キー方向名・ボタン名付きの直感的な表現へ整形."""
        rev_map = {v: k for k, v in self.dynamics_map.items() if k in ("UP", "DOWN", "LEFT", "RIGHT")}
        default_dir_map = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
        labels: List[str] = []
        for aid in avail_ids:
            if aid in rev_map:
                labels.append(f"{rev_map[aid]} (ACTION{aid})")
            elif aid in default_dir_map:
                labels.append(f"{default_dir_map[aid]} (ACTION{aid})")
            elif aid == 6:
                labels.append("CLICK (ACTION6)")
            elif aid == 0:
                labels.append("RESET (ACTION0)")
            else:
                labels.append(f"ACTION{aid}")
        return labels
