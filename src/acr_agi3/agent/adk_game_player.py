"""Google ADK 2.0 準拠・ローカルLLM/VLM思考駆動型自律ゲームプレイエージェント (ADKGamePlayer).

決定論的なプログラム実行ではなく、
1. 視覚認識ハーネス (VisionObservationHarness) による公式10色カラー画像変換
2. Google ADK 2.0 ネイティブ Agent / Runner による推論
3. SkillToolset (Progressive Disclosure) によるメタスキル SKILL.md のオンデマンド読み込み
4. GameActionTools (Function Calling) による自律的アクション・クリック決定
を統合した本質的ゲームプレイエージェント。
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from pathlib import Path

from acr_agi3.agent.llm.local_vlm import LocalQwenVL
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
    """Google ADK 2.0 ネイティブ・マルチモーダル自律ゲームプレイエージェント."""

    def __init__(
        self,
        model: Optional[Any] = None,
        name: str = "adk_game_player",
        app_name: str = "acr_agi3_adk_player",
        cell_size: int = 12,
    ) -> None:
        self.name = name
        self.app_name = app_name
        self.model = model or LocalQwenVL(model_name_or_path="auto")

        # 1. 視覚認識ハーネス & ゲーム操作ツール
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()

        # 2. ADK 2.0 公式 SkillToolset (meta_skills/ の Progressive Disclosure)
        self.skill_harness = SkillHarness()
        self.skill_toolset = self.skill_harness.get_toolset()

        # 3. エピソード記憶マネージャー (Transcript からの因果蒸留 & Working Memory 保持)
        self.memory = EpisodicMemoryManager(capacity=100) if EpisodicMemoryManager else None

        # 4. エージェントのツール群 (操作ツール + メタスキルツールセット)
        self.tools = [
            *self.action_tools.get_tools(),
            self.skill_toolset,
        ]

        # 5. ADK 2.0 Agent 指示書 (5-Step 自律ワークフロー規約)
        instruction = (
            "You are an elite autonomous agent playing ARC-AGI-3 dynamic games.\n"
            "Your goal is to inspect the visual board, reason about rules and affordances, and choose the optimal action.\n\n"
            "=== 5-STEP AUTONOMOUS WORKFLOW ===\n"
            "1. [OBSERVE]: Inspect the latest color grid image and identify key entities (player, goal, obstacles).\n"
            "2. [RECALL]: Review Working Memory (provided in user prompt) to check past action outcomes and avoid repeated mistakes.\n"
            "3. [PLAN]: Formulate an immediate subgoal. If complex, load an expert meta-skill (e.g. `visual-inspector`, `backward-planner`).\n"
            "4. [ACT]: Execute your chosen action following the `game-controller` JSON protocol.\n"
            "5. [REFLECT]: The environment will measure pixel diffs to update your episodic memory automatically.\n\n"
            "=== GAME OPERATION (game-controller protocol) ===\n"
            "You operate the game by outputting a single valid JSON block:\n"
            "  1. Movement (UP, DOWN, LEFT, RIGHT, ACTION1-7):\n"
            '     ```json\n     {"action": "UP", "reasoning": "Move toward active target"}\n     ```\n'
            "  2. Coordinate Click (for clicking on grid):\n"
            '     ```json\n     {"action": "click_at", "x": 10, "y": 5, "reasoning": "Click interactive object"}\n     ```\n'
            "  3. Reset (if deadlocked or trapped):\n"
            '     ```json\n     {"action": "RESET", "reasoning": "No valid paths remaining"}\n     ```\n\n'
            "=== CONSULTING META-SKILLS ON DEMAND ===\n"
            "If you need strategic analysis, you can load a meta-skill first:\n"
            '     ```json\n     {"action": "load_skill", "skill_name": "visual-inspector"}\n     ```\n'
            "Available meta-skills: `visual-inspector`, `episodic-memory`, `epistemic-prober`, `backward-planner`, `taboo-reset-guard`, `macro-skill-compiler`.\n"
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

        self.step_index = 0
        self.last_grid: Optional[np.ndarray] = None
        self.last_action_info: Optional[Dict[str, Any]] = None
        self.session_id: Optional[str] = None

    def reset(self) -> None:
        """エージェントの状態とセッションを初期化."""
        self.step_index = 0
        self.last_grid = None
        self.last_action_info = None
        self.action_tools.pending_decision = None
        self.action_tools.history.clear()
        if self.memory:
            self.memory.clear()
        self.session_id = None

    def decide_next_action(
        self,
        grid: Any,
        available_actions: Optional[List[int]] = None,
        state_str: str = "NOT_FINISHED",
    ) -> ActionDecision:
        """現在の観測フレームから、LLM の思考・ツール呼び出しを経て次のアクションを決定."""
        self.step_index += 1
        arr = normalize_grid(grid)

        # 差分情報の算出
        pixels_changed = 0
        is_effective = False
        if self.last_grid is not None and self.last_grid.shape == arr.shape:
            diff_mask = (self.last_grid != arr)
            pixels_changed = int(np.sum(diff_mask))
            is_effective = (pixels_changed > 0)

        # Phase 5: 直前ステップの結果をエピソード記憶に登録（自己反省・因果更新）
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

        # Phase 2: Working Memory 要約の注入 (Phase 2: RECALL)
        if self.memory is not None:
            memory_summary = self.memory.get_working_memory_summary(max_recent=4)
            parts.append(Part.from_text(text=f"\n=== [RECALL: Working Memory Context] ===\n{memory_summary}\n"))

        # 非同期 Runner を同期実行
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # 既存のイベントループ内での安全な実行
            import nest_asyncio
            nest_asyncio.apply()

        decision = loop.run_until_complete(self._run_adk_cycle(parts, avail_ids))
        self.last_grid = arr.copy()
        self.last_action_info = {
            "action": decision.action_name,
            "action_id": decision.action_id,
            "reasoning": decision.reasoning,
            "state_before": state_str,
        }
        return decision

    async def _run_adk_cycle(self, parts: List[Part], available_action_ids: List[int]) -> ActionDecision:
        """ADK Runner を呼び出し、LLM による推論と Function Calling を実行."""
        user_id = "arc_player_user"
        if not self.session_id:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
            )
            self.session_id = session.id
        else:
            # 過去イベントから画像バイナリをアーカイブし、VRAM の累積肥大化 (OOM) を防止
            # テキスト履歴（推論ログ・アクション）はすべて保持
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=self.session_id,
            )
            if session and getattr(session, "events", None):
                for ev in session.events:
                    if hasattr(ev, "content") and ev.content:
                        for p in getattr(ev.content, "parts", []):
                            if hasattr(p, "inline_data") and p.inline_data:
                                p.inline_data = None
                                p.text = "[Previous Visual Frame archived]"

        content = Content(role="user", parts=parts)
        events = self.runner.run_async(
            session_id=self.session_id,
            user_id=user_id,
            new_message=content,
        )

        response_text = ""
        async for event in events:
            if hasattr(event, "content") and event.content:
                for p in getattr(event.content, "parts", []):
                    if hasattr(p, "text") and p.text:
                        response_text += p.text
                    if hasattr(p, "function_call") and p.function_call:
                        logger.info(f"Detected function_call part: {p.function_call.name}({p.function_call.args})")

        logger.info(f"Step {self.step_index} accumulated response_text: {response_text!r}")
        logger.info(f"Step {self.step_index} pending_decision: {self.action_tools.pending_decision}")

        # 推論終了後の VRAM クリーンアップ
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 1. ツール呼び出し（Function Call）によって pending_decision がセットされた場合
        if self.action_tools.pending_decision is not None:
            decision = self.action_tools.pending_decision
            self.action_tools.pending_decision = None
            return decision

        # 2. モデルが load_skill を要求した場合、Progressive Disclosure (Level 2 Instructions 展開)
        import json
        skill_to_load = None
        json_match = re.search(r"\{[^{}]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if data.get("action") == "load_skill" and "skill_name" in data:
                    skill_to_load = data["skill_name"]
            except Exception:
                pass

        if skill_to_load:
            logger.info(f"[Progressive Disclosure] Agent requested meta-skill: {skill_to_load}")
            try:
                skill_content = self.skill_harness.read_skill_content(skill_to_load)
                # スキルの Level 2 Instructions を提供し、ゲーム操作の決定を即座に促す
                followup_prompt = (
                    f"Successfully loaded expert meta-skill `{skill_to_load}` instructions:\n"
                    f"```markdown\n{skill_content[:1500]}\n```\n\n"
                    f"Now, based on the above skill and available actions {available_action_ids}, "
                    f"execute your immediate game action following game-controller protocol:\n"
                    f'```json\n{{"action": "UP" (or DOWN, LEFT, RIGHT, click_at, RESET), "reasoning": "..."}}\n```'
                )
                followup_content = Content(role="user", parts=[Part.from_text(text=followup_prompt)])
                followup_events = self.runner.run_async(
                    session_id=self.session_id,
                    user_id=user_id,
                    new_message=followup_content,
                )
                response_text = ""
                async for event in followup_events:
                    if hasattr(event, "content") and event.content:
                        for p in getattr(event.content, "parts", []):
                            if hasattr(p, "text") and p.text:
                                response_text += p.text

                logger.info(f"Step {self.step_index} post-skill response_text: {response_text!r}")
            except Exception as e:
                logger.warning(f"Failed to expand skill {skill_to_load}: {e}")

        # 3. ツールが直接呼ばれなかった場合：LLM の応答テキストからアクションを安全にパース
        return self._fallback_parse_decision(response_text, available_action_ids)

    def _fallback_parse_decision(self, text: str, available_action_ids: List[int]) -> ActionDecision:
        logger.info(f"[_fallback_parse_decision] text length: {len(text)}, repr: {text!r}")
        if GameController is not None:
            controller = GameController(available_actions=available_action_ids)
            h, w = (self.last_grid.shape[0], self.last_grid.shape[1]) if self.last_grid is not None else (30, 30)
            res = controller.parse_and_validate(text, available_actions=available_action_ids, grid_shape=(h, w))
            logger.info(f"[_fallback_parse_decision] GameController result: {res}")
            if res["success"]:
                return ActionDecision(
                    action_type=res["action_type"],
                    action_name=res["action_name"],
                    action_id=res["action_id"],
                    coordinates=res["coordinates"],
                    reasoning=res["reasoning"],
                )
        else:
            logger.warning("[_fallback_parse_decision] GameController is None!")

        # GameController 未初期化または解析失敗時のセーフティフォールバック
        text_upper = text.upper()
        action_candidates = [
            ("RESET", 0),
            ("ACTION1", 1), ("UP", 1),
            ("ACTION2", 2), ("DOWN", 2),
            ("ACTION3", 3), ("LEFT", 3),
            ("ACTION4", 4), ("RIGHT", 4),
            ("ACTION5", 5),
            ("ACTION6", 6),
            ("ACTION7", 7),
        ]
        for name, aid in action_candidates:
            if re.search(rf"\b{name}\b", text_upper):
                if aid in available_action_ids or aid == 0:
                    return ActionDecision(
                        action_type="STEP" if aid != 0 else "RESET",
                        action_name=name,
                        action_id=aid,
                        reasoning=f"Extracted from LLM text: {name}",
                    )

        # 最終デフォルト
        default_id = available_action_ids[0] if available_action_ids else 1
        return ActionDecision(
            action_type="STEP",
            action_name=f"ACTION{default_id}",
            action_id=default_id,
            reasoning="Default action when unparseable",
        )
