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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.llm.local_model import LocalTransformersLlm
from acr_agi3.harness.game_action_tools import ActionDecision, GameActionTools
from acr_agi3.harness.vision_observation import VisionObservationHarness, normalize_grid
from acr_agi3.meta.skill_harness import SkillHarness

logger = logging.getLogger(__name__)


class ADKGamePlayer:
    """Google ADK 2.0 ネイティブ・マルチモーダル自律ゲームプレイエージェント."""

    def __init__(
        self,
        model: Optional[Any] = None,
        name: str = "adk_game_player",
        app_name: str = "acr_agi3_adk_player",
        cell_size: int = 24,
    ) -> None:
        self.name = name
        self.app_name = app_name
        self.model = model or LocalTransformersLlm(model_name_or_path="mock")

        # 1. 視覚認識ハーネス & ゲーム操作ツール
        self.vision_harness = VisionObservationHarness(cell_size=cell_size)
        self.action_tools = GameActionTools()

        # 2. ADK 2.0 公式 SkillToolset (meta_skills/ の Progressive Disclosure)
        self.skill_harness = SkillHarness()
        self.skill_toolset = self.skill_harness.get_toolset()

        # 3. エージェントのツール群 (操作ツール + メタスキルツールセット)
        self.tools = [
            *self.action_tools.get_tools(),
            self.skill_toolset,
        ]

        # 4. ADK 2.0 Agent 指示書
        instruction = (
            "You are an elite autonomous agent playing ARC-AGI-3 dynamic games.\n"
            "Your goal is to inspect the visual game board, reason about rules and affordances, and choose the optimal action.\n\n"
            "=== HOW TO OPERATE ===\n"
            "1. Visual Gestalt: Inspect the rendered game image. Identify agent position, target goals, walls, and interactables.\n"
            "2. Meta-Skills: You have access to expert problem-solving meta-skills via `load_skill`:\n"
            "   - `visual-inspector`: Gestalt difference and style identification.\n"
            "   - `epistemic-prober`: Active exploratory testing for unknown mechanics.\n"
            "   - `backward-planner`: Backward chaining from goal and buffer staging.\n"
            "   - `taboo-reset-guard`: No-go collision avoidance and active reset.\n"
            "   - `macro-skill-compiler`: Predefined strategic movement macros.\n"
            "3. Action Execution: You MUST call one of the following tools every turn:\n"
            "   - `step_action(action='UP'|'DOWN'|'LEFT'|'RIGHT'|'ACTION1'..)` for movement/actions.\n"
            "   - `click_at(x=col, y=row)` for interactive coordinate clicking.\n"
            "   - `reset_game()` for active reset when deadlocked or in an infinite loop.\n"
            "Always state your strategic reasoning before calling the action tool."
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

        if self.action_tools.history:
            last_dec = self.action_tools.history[-1]
            self.last_action_info = {
                "action": last_dec.action_name,
                "pixels_changed": pixels_changed,
                "is_effective": is_effective,
            }

        # 利用可能アクションの更新
        avail_ids = available_actions or [1, 2, 3, 4]
        avail_names = [f"ACTION{i}" for i in avail_ids]
        self.action_tools.set_available_actions(avail_ids)

        # 視覚観測 Parts の生成
        parts = self.vision_harness.create_observation_parts(
            grid_data=arr,
            step_index=self.step_index,
            available_actions=avail_names,
            last_action_info=self.last_action_info,
        )

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

        # 1. ツール呼び出し（Function Call）によって pending_decision がセットされた場合
        if self.action_tools.pending_decision is not None:
            decision = self.action_tools.pending_decision
            self.action_tools.pending_decision = None
            return decision

        # 2. ツールが直接呼ばれなかった場合：LLM の応答テキストからアクションを安全にパース
        return self._fallback_parse_decision(response_text, available_action_ids)

    def _fallback_parse_decision(self, text: str, available_action_ids: List[int]) -> ActionDecision:
        """テキストからアクション名やクリック座標を抽出する安全なフォールバック."""
        text_upper = text.upper()

        # クリック座標の検出: click(x, y) or [x, y]
        click_match = re.search(r"CLICK.*?(\d+)\s*[,xX\s]\s*(\d+)", text, re.IGNORECASE)
        if click_match and 6 in available_action_ids:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            return ActionDecision(
                action_type="CLICK",
                action_name="ACTION6",
                action_id=6,
                coordinates={"x": x, "y": y},
                reasoning=f"Extracted from LLM text: click at ({x}, {y})",
            )

        # アクション名の検出
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
