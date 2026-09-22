"""ADK game player driven by missing knowledge and on-demand visual inspection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from functools import wraps
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.runners import RunConfig, Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from acr_agi3.agent.deliberation import ThoughtMode, ThoughtState
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.harness.game_action_tools import ActionDecision, GameActionTools
from acr_agi3.meta.skill_harness import SkillHarness
from acr_agi3.tools.screen_tools import ScreenTools


class DeliberationBudgetExceededError(RuntimeError):
    """No justified action was selected within the inference budget."""


def tool_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, KeyError, TypeError) as error:
            return {"error": str(error)}

    return wrapped


class DeliberativeGamePlayer:
    """One reasoning session per observation; no mandatory perception phase.

    Questions and causal knowledge survive inference sessions. External actions
    end deliberation; their consequences are reviewed on the next gateway frame.
    """

    def __init__(self, model=None, name: str = "deliberative_player", max_llm_calls: int = 24):
        self.model = model or LocalQwenVL(model_name_or_path="auto")
        self.name = name
        self.max_llm_calls = max_llm_calls
        self.state = ThoughtState()
        self.screen = ScreenTools()
        self.actions = GameActionTools()
        self.skill_harness = SkillHarness()
        self.current_game_id = "default"
        self.games: dict[str, ThoughtState] = {}
        self.levels_completed = 0
        self.step_index = 0
        self.available_actions: list[int] = []
        self.reset_states: set[bytes] = set()
        self._reset_pending = False
        self.decision: ActionDecision | None = None
        self.session_service = InMemorySessionService()
        bindings = [
            self.set_goal,
            self.need_causal_knowledge,
            self.need_experiment,
            self.assess_result,
            self.resolve_question,
            self.plan_actions,
            self.continue_plan,
            self.answer_visible_question,
            self.use_known_rules,
            self.observe_screen,
            self.step_action,
            self.click_at,
            self.reset_game,
        ]
        scoped_skills = ["causal-deliberation", "visual-inspector", "game-controller"]
        toolset = self.skill_harness.get_scoped_toolset(
            scoped_skills,
            additional_tools=bindings,
            tool_filter=self._tool_is_relevant,
        )
        catalog = self.skill_harness.get_level1_catalog(scoped_skills)
        instruction = (
            "You are an autonomous cognitive game-playing agent for ARC-AGI-3.\n"
            "You solve games through a rigorous 5-Mode Cognitive State Machine.\n"
            "NEVER take blind actions without grounded evidence or hypothesis.\n\n"
            f"{catalog}\n\n"
            "=== COGNITIVE STATE MACHINE ARCHITECTURE ===\n"
            "1. [PLAN Mode] (Current Goal & Strategy)\n"
            "   - Purpose: Observe the screen, identify the goal, and formulate a plan.\n"
            "   - If screen unobserved: Load 'visual-inspector' and call observe_screen(view='current').\n"
            "   - If causal rules are UNKNOWN (how objects move or what buttons do):\n"
            "     -> Call need_causal_knowledge(question='...') to advance to CAUSAL mode.\n"
            "   - If plan is ready: Call plan_actions(subgoal=..., steps=[...], rule_ids=[...]) to advance to EXECUTE mode.\n"
            "   - Constraint: Environment actions (step_action/click_at) are NOT available in PLAN mode.\n\n"
            "2. [CAUSAL Mode] (Knowledge Gap Resolution)\n"
            "   - Purpose: Resolve open questions through hypothesis testing.\n"
            "   - If visible by inspection: Call answer_visible_question(answer=..., visual_evidence=...).\n"
            "   - If test is needed: Call need_experiment(hypothesis=..., prediction=..., alternative=...) to advance to EXPERIMENT mode.\n"
            "   - Constraint: Environment actions are NOT available in CAUSAL mode.\n\n"
            "3. [EXPERIMENT Mode] (Controlled Hypothesis Testing)\n"
            "   - Purpose: Execute ONE minimal intervention to test the active hypothesis.\n"
            "   - Action: Load 'game-controller' and call step_action(action_id=..., reasoning=...) or click_at(x=..., y=..., reasoning=...).\n\n"
            "4. [EXECUTE Mode] (Plan Step Dispatch)\n"
            "   - Purpose: Execute the planned step.\n"
            "   - Action: Load 'game-controller' and call step_action or click_at matching the plan.\n\n"
            "5. [REVIEW Mode] (Outcome Assessment & Rule Learning)\n"
            "   - Purpose: Compare before and after frames to assess hypothesis validity.\n"
            "   - Steps: Call observe_screen(view='both'), then assess_result(...).\n"
            "   - If supported: Call resolve_question(...) to record a learned rule.\n\n"
            "=== OUTPUT RULES ===\n"
            "- Keep reasoning extremely concise (1-2 sentences maximum).\n"
            "- In every turn, you MUST call exactly one tool from the available tools matching your current mode.\n"
            "- Do NOT hallucinate that an action already took place until you actually execute it and enter REVIEW mode."
        )
        self.agent = Agent(
            name=name,
            model=self.model,
            tools=[toolset],
            instruction=instruction,
        )
        self.runner = Runner(agent=self.agent, app_name=name, session_service=self.session_service)

    def _wrap_snapshot(self, res: dict) -> dict:
        """ツールの実行結果に最新の Cognitive Guidance を埋め込み、モデルが次の行動を自律理解できるようにする."""
        if isinstance(res, dict):
            res["mode_guidance"] = self._build_mode_guidance()
        return res

    def _build_mode_guidance(self) -> str:
        """現在地の ThoughtMode に応じた目的、達成条件、遷移ルールを明確に生成."""
        observed = self.screen.current_id in self.screen.viewed
        mode = self.state.mode

        if mode == ThoughtMode.PLAN:
            if not observed:
                return (
                    "📍 Current State: [PLAN] (Screen Unobserved)\n"
                    "🎯 Goal: Observe the board to inspect layout, entities, and colors.\n"
                    "👉 Next Action: Call load_skill('visual-inspector') and observe_screen(view='current')."
                )
            if not self.state.rules:
                return (
                    "📍 Current State: [PLAN] (Screen Observed, Zero Causal Rules Known)\n"
                    "🎯 Goal: Since how actions affect the game is unknown, you cannot plan yet. You must investigate causality.\n"
                    "👉 Next Action: Call need_causal_knowledge(question='Which action moves the piece or interacts with targets?') to enter CAUSAL mode.\n"
                    "⚠️ Notice: You have 0 learned rules. 'plan_actions' and 'step_action' are NOT available yet."
                )
            return (
                "📍 Current State: [PLAN] (Rules Available)\n"
                f"📚 Learned Rules: {list(self.state.rules.keys())}\n"
                "🎯 Goal: Formulate an execution plan using your learned rules.\n"
                "👉 Next Action: Call plan_actions(subgoal=..., steps=[...], rule_ids=[...]) to advance to EXECUTE mode.\n"
                "⚠️ Notice: Environment actions (step_action/click_at) are NOT available directly in PLAN mode."
            )
        elif mode == ThoughtMode.CAUSAL:
            active_q = self.state.questions[-1].question if self.state.questions else "unknown causal relation"
            if not observed:
                return (
                    f"📍 Current State: [CAUSAL] (Screen Unobserved)\n"
                    f"❓ Open Question: {active_q}\n"
                    "🎯 Goal: Observe the screen to check for visible answers.\n"
                    "👉 Next Action: Call observe_screen(view='current')."
                )
            return (
                f"📍 Current State: [CAUSAL] (Screen Observed)\n"
                f"❓ Open Question: {active_q}\n"
                "🎯 Goal: Design a minimal experiment to test how an action works.\n"
                "👉 Next Action: Call need_experiment(hypothesis='Action 1 moves the piece', prediction='Piece moves one cell', alternative='Piece does not move') to enter EXPERIMENT mode.\n"
                "⚠️ Notice: Direct actions and plan_actions are NOT available in CAUSAL mode."
            )
        elif mode == ThoughtMode.EXPERIMENT:
            exp = self.state.experiment or {}
            hyp = exp.get("hypothesis", "active hypothesis")
            return (
                f"📍 Current State: [EXPERIMENT]\n"
                f"🧪 Active Hypothesis: {hyp}\n"
                "🎯 Goal: Execute the single test action for this experiment.\n"
                "👉 Next Action: Call load_skill('game-controller') and step_action(action_id=..., reasoning='...') or click_at(x=..., y=..., reasoning='...').\n"
                "   (If this hypothesis is no longer viable, call need_causal_knowledge to propose a different question)."
            )
        elif mode == ThoughtMode.EXECUTE:
            next_step = self.state.plan[0] if self.state.plan else {}
            return (
                f"📍 Current State: [EXECUTE]\n"
                f"📋 Next Planned Step: {next_step}\n"
                "🎯 Goal: Execute the planned step.\n"
                "👉 Next Action: Call load_skill('game-controller') and step_action(action_id=..., reasoning='...') or click_at(x=..., y=..., reasoning='...').\n"
                "   (If an unexpected obstacle or rule violation appears, call need_causal_knowledge to pause and investigate)."
            )
        elif mode == ThoughtMode.REVIEW:
            if not observed:
                return (
                    "📍 Current State: [REVIEW] (Screen Unobserved)\n"
                    "🎯 Goal: Observe the consequence of the last action.\n"
                    "👉 Next Action: Call observe_screen(view='both') to inspect changes before and after."
                )
            before_id = (self.state.pending or {}).get("frame_id", max(1, self.screen.current_id - 1))
            after_id = self.screen.current_id
            return (
                "📍 Current State: [REVIEW] (Screen Observed)\n"
                "🎯 Goal: Assess whether the observed outcome supports or refutes the hypothesis.\n"
                f"👉 Next Action: Call assess_result(outcome='supported', evidence='Observed movement/change', before_frame_id={before_id}, after_frame_id={after_id}).\n"
                "⚠️ Notice: 'resolve_question' is NOT available in REVIEW mode. You MUST call assess_result first!"
            )
        return ""

    @tool_errors
    def observe_screen(self, view: str = "current") -> dict:
        """Inspect the current game frame on demand or compare before/after frames."""
        res = self.screen.observe_screen(view=view)
        return self._wrap_snapshot(res)

    def _tool_is_relevant(self, tool, context) -> bool:
        """Filter ADK-activated tools by thought purpose; ADK still owns loading.

        Observation and skill discovery remain available in every mode. Execution
        checks remain authoritative even if a stale call reaches the dispatcher.
        """
        common = {"list_skills", "load_skill", "load_skill_resource", "observe_screen"}
        by_mode = {
            ThoughtMode.PLAN: {
                "set_goal",
                "need_causal_knowledge",
                "plan_actions",
                "continue_plan",
                "reset_game",
            },
            ThoughtMode.CAUSAL: {
                "need_causal_knowledge",
                "need_experiment",
                "resolve_question",
                "answer_visible_question",
                "use_known_rules",
                "reset_game",
            },
            ThoughtMode.EXPERIMENT: {"need_causal_knowledge", "step_action", "click_at"},
            ThoughtMode.EXECUTE: {"need_causal_knowledge", "step_action", "click_at"},
            ThoughtMode.REVIEW: {"assess_result"},
        }
        return tool.name in common | by_mode[self.state.mode]

    def _ensure_observed(self) -> None:
        if self.screen.current_id not in self.screen.viewed:
            raise ValueError(
                "Call observe_screen on the current frame before grounding this decision."
            )
        if self.decision is not None:
            raise ValueError("One action is already scheduled; await the next gateway frame.")

    @tool_errors
    def set_goal(self, goal: str, visual_evidence: str) -> dict:
        """Record the visually grounded winning condition without taking an action."""
        self._ensure_observed()
        self.state.require_mode(ThoughtMode.PLAN)
        if not goal.strip() or not visual_evidence.strip():
            raise ValueError("Supply a goal and its visible evidence.")
        self.state.goal = goal
        return self._wrap_snapshot(self.state.snapshot())

    @tool_errors
    def need_causal_knowledge(self, question: str) -> dict:
        """Suspend reasoning or an unissued action because a necessary causal fact is missing."""
        return self._wrap_snapshot(self.state.ask(question))

    @tool_errors
    def need_experiment(self, hypothesis: str, prediction: str, alternative: str) -> dict:
        """Evidence is insufficient: design a minimal intervention to distinguish explanations."""
        self._ensure_observed()
        return self._wrap_snapshot(self.state.propose_experiment(hypothesis, prediction, alternative))

    @tool_errors
    def assess_result(
        self,
        outcome: str,
        evidence: str,
        before_frame_id: int,
        after_frame_id: int,
    ) -> dict:
        """Review observed consequences as supported/refuted/inconclusive, never by pixel count."""
        self._ensure_observed()
        if before_frame_id not in self.screen.viewed or after_frame_id != self.screen.current_id:
            raise ValueError("Observe both the actual before and latest after frames first.")
        return self._wrap_snapshot(self.state.review(outcome, evidence, before_frame_id, after_frame_id))

    @tool_errors
    def resolve_question(
        self,
        answer: str,
        precondition: str,
        effect: str,
        result_ids: list[str],
    ) -> dict:
        """Record an evidence-backed conditional rule and resume the suspended thought."""
        return self._wrap_snapshot(self.state.resolve(answer, precondition, effect, result_ids))

    @tool_errors
    def answer_visible_question(self, answer: str, visual_evidence: str) -> dict:
        """Resolve a visible fact (goal, object, layout), not an untested causal relation."""
        self._ensure_observed()
        self.state.require_mode(ThoughtMode.CAUSAL)
        if not self.state.questions or not answer.strip() or not visual_evidence.strip():
            raise ValueError("An open visible-fact question and its observation are required.")
        question = self.state.questions.pop()
        self.state.facts.append(
            {
                "question": question.question,
                "answer": answer,
                "evidence": visual_evidence,
                "frame_id": self.screen.current_id,
            }
        )
        self.state.facts = self.state.facts[-16:]
        self.state.transition(question.return_mode, "Visible fact resolved without an intervention")
        return self._wrap_snapshot(self.state.snapshot())

    @tool_errors
    def use_known_rules(
        self, answer: str, rule_ids: list[str], observed_preconditions: str
    ) -> dict:
        """Answer a suspended question using existing rules whose conditions still hold."""
        self._ensure_observed()
        self.state.require_mode(ThoughtMode.CAUSAL)
        if not self.state.questions or not answer.strip() or not observed_preconditions.strip():
            raise ValueError("Supply an answer and checked preconditions for the open question.")
        if not rule_ids or any(key not in self.state.rules for key in rule_ids):
            raise ValueError("Cite known rules; otherwise infer or gather evidence first.")
        question = self.state.questions.pop()
        self.state.transition(
            question.return_mode, f"Known rules answer {question.question}: {answer}"
        )
        return self._wrap_snapshot(self.state.snapshot())

    @tool_errors
    def plan_actions(self, subgoal: str, steps: list[dict[str, Any]], rule_ids: list[str]) -> dict:
        """Plan backwards: ordered steps need action_id, precondition and expected_result.

        Click steps additionally need x and y in original game coordinates.
        Cite rules supporting the dependency order. Temporary detours are valid.
        """
        self._ensure_observed()
        for step in steps:
            self._validate_action(step.get("action_id"), step.get("x"), step.get("y"))
        return self._wrap_snapshot(self.state.make_plan(subgoal, steps, rule_ids))

    @tool_errors
    def continue_plan(self, precondition_evidence: str) -> dict:
        """After reviewing success, recheck the remaining plan's next precondition on screen."""
        self._ensure_observed()
        self.state.require_mode(ThoughtMode.PLAN)
        if not self.state.plan or not precondition_evidence.strip():
            raise ValueError("There is no remaining plan, or its precondition was not checked.")
        self.state.transition(ThoughtMode.EXECUTE, precondition_evidence)
        return self._wrap_snapshot(self.state.snapshot())

    def _validate_action(self, action_id, x=None, y=None) -> None:
        if type(action_id) is not int or action_id not in self.available_actions or action_id == 0:
            raise ValueError(f"Choose an available non-reset action: {self.available_actions}")
        if action_id == 6:
            h, w = self.screen.frames[self.screen.current_id].shape
            if type(x) is not int or type(y) is not int or not (0 <= x < w and 0 <= y < h):
                raise ValueError(
                    "Click coordinates must be integers inside the current game frame."
                )

    def _schedule(self, action_id: int, x, y, reasoning: str) -> dict:
        self._ensure_observed()
        self.state.require_mode(ThoughtMode.EXPERIMENT, ThoughtMode.EXECUTE)
        self._validate_action(action_id, x, y)
        if not reasoning.strip():
            raise ValueError("Explain the intervention or the checked plan precondition.")
        if self.state.mode == ThoughtMode.EXECUTE:
            first = self.state.plan[0]
            if (action_id, x, y) != (first["action_id"], first.get("x"), first.get("y")):
                raise ValueError("Action differs from the current plan; revise the plan first.")
            expected = first["expected_result"]
            intent = "plan"
        else:
            expected = self.state.experiment["prediction"]
            intent = "experiment"
        self.actions.pending_decision = None
        if action_id == 6:
            self.actions.click_at(x=x, y=y, reasoning=reasoning)
        else:
            self.actions.step_action(action=f"ACTION{action_id}", reasoning=reasoning)
        if self.actions.pending_decision is None:
            raise ValueError("Execution tool rejected the action.")
        self.decision = self.actions.pending_decision
        self.state.pending = {
            "intent": intent,
            "frame_id": self.screen.current_id,
            "action_id": action_id,
            "coordinates": {"x": x, "y": y} if action_id == 6 else None,
            "expected_result": expected,
            "reasoning": reasoning,
            "question": self.state.questions[-1].question if self.state.questions else None,
        }
        if intent == "plan":
            self.state.pending["subgoal"] = first["subgoal"]
            self.state.pending["rule_ids"] = first["rule_ids"]
            self.state.plan.pop(0)
        self.state.transition(ThoughtMode.REVIEW, "Action scheduled; await its actual outcome")
        return {"scheduled": True, "action_id": action_id, "expected_result": expected}

    @tool_errors
    def step_action(self, action_id: int, reasoning: str) -> dict:
        """Execute one physical button by verified ID; do not guess directional aliases."""
        if action_id == 6:
            raise ValueError("Use click_at for ACTION6.")
        return self._schedule(action_id, None, None, reasoning)

    @tool_errors
    def click_at(self, x: int, y: int, reasoning: str) -> dict:
        """Execute one planned click or causal experiment in original frame coordinates."""
        return self._schedule(6, x, y, reasoning)

    @tool_errors
    def reset_game(self, diagnosis: str, revised_approach: str) -> dict:
        """Reset after a causal diagnosis with a different approach, never from stagnation alone."""
        self._ensure_observed()
        self.state.require_mode(ThoughtMode.CAUSAL, ThoughtMode.PLAN)
        if not diagnosis.strip() or not revised_approach.strip():
            raise ValueError("A strategic reset needs a diagnosis and a changed approach.")
        grid = self.screen.frames[self.screen.current_id]
        key = hashlib.sha256(grid.tobytes()).digest()
        if key in self.reset_states:
            raise ValueError("Already reset this board; revise the hypothesis or explore instead.")
        self.reset_states.add(key)
        self.actions.reset_game(reasoning=f"{diagnosis}; next: {revised_approach}")
        self.decision = self.actions.pending_decision
        self._reset_pending = True
        self.state.plan.clear()
        self.state.transition(ThoughtMode.PLAN, f"Restart strategy: {revised_approach}")
        return {"scheduled": True, "action_id": 0}

    def reset(self, full_wipe: bool = False) -> None:
        """Clear episode execution; retain causal knowledge unless explicitly wiped."""
        if full_wipe:
            self.state = ThoughtState()
        else:
            self.state.plan.clear()
            self.state.pending = None
            self.state.experiment = None
            self.state.questions.clear()
            self.state.facts.clear()
            self.state.transition(ThoughtMode.PLAN, "Episode boundary; retain learned causal rules")
        # SkillToolset holds bound methods: retain the screen instance across
        # episodes so on-demand tools always see the current gateway frame.
        self.screen.frames.clear()
        self.screen.viewed.clear()
        self.decision = None
        self._reset_pending = False
        self.reset_states.clear()

    def switch_game(self, game_id: str) -> None:
        if game_id == self.current_game_id:
            return
        self.games[self.current_game_id] = self.state
        self.current_game_id = game_id
        self.state = self.games.setdefault(game_id, ThoughtState())
        self.reset()
        self.levels_completed = 0

    def decide_next_action(
        self,
        grid,
        available_actions=None,
        state_str="NOT_FINISHED",
        game_id=None,
        levels_completed=0,
    ) -> ActionDecision:
        if game_id:
            self.switch_game(game_id)
        if self._reset_pending:
            keys = self.reset_states.copy()
            self.reset()
            self.reset_states = keys
        self.step_index += 1
        frame_id = self.screen.publish(grid)
        self.available_actions = [a for a in (available_actions or []) if a != 0]
        self.actions.set_available_actions(self.available_actions)
        self.actions.set_grid(self.screen.frames[frame_id])
        self.decision = None
        if levels_completed != self.levels_completed:
            self.state.plan.clear()
            self.state.pending = None
            self.state.experiment = None
            self.state.questions.clear()
            self.state.goal = "Identify and reach the new level's displayed winning condition"
            self.state.facts.clear()
            self.state.transition(
                ThoughtMode.PLAN, "Level changed; inspect new goal and preconditions"
            )
        self.levels_completed = levels_completed
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
        return loop.run_until_complete(self._think(state_str))

    async def _think(self, state_str: str) -> ActionDecision:
        start = time.perf_counter()
        session = await self.session_service.create_session(app_name=self.name, user_id="player")
        context = {
            "game_id": self.current_game_id,
            "frame_id": self.screen.current_id,
            "game_state": state_str,
            "available_actions": self.available_actions,
            "thought": self.state.snapshot(),
        }
        guidance = self._build_mode_guidance()
        prompt_text = (
            f"Observation:\n{json.dumps(context, indent=2)}\n\n"
            f"=== COGNITIVE GUIDANCE ===\n{guidance}\n\n"
            "Task: Advance the cognitive state machine or execute an action using the appropriate tool."
        )
        message = Content(role="user", parts=[Part.from_text(text=prompt_text)])
        remaining = self.max_llm_calls
        try:
            while remaining > 0 and self.decision is None:
                calls = 0
                events = self.runner.run_async(
                    user_id="player",
                    session_id=session.id,
                    new_message=message,
                    run_config=RunConfig(max_llm_calls=remaining),
                )
                try:
                    async for event in events:
                        if event.content and event.content.role == "model":
                            calls += 1
                        if self.decision is not None:
                            break
                except LlmCallsLimitExceededError:
                    remaining = 0
                finally:
                    await events.aclose()
                remaining -= max(1, calls)
                guidance = self._build_mode_guidance()
                followup_text = (
                    f"State update:\n{json.dumps(self.state.snapshot(), indent=2)}\n\n"
                    f"=== COGNITIVE GUIDANCE ===\n{guidance}\n\n"
                    "Task: Advance the state machine or execute an action using the opened tools."
                )
                message = Content(
                    role="user",
                    parts=[Part.from_text(text=followup_text)],
                )
        finally:
            await self.session_service.delete_session(
                app_name=self.name,
                user_id="player",
                session_id=session.id,
            )
        if self.decision is None:
            raise DeliberationBudgetExceededError(
                "No justified tool action was selected; no fallback move issued"
            )
        self.decision.metadata.update(
            {
                "thought_mode": self.state.mode.value,
                "transitions": list(self.state.transitions),
                "pending_question": self.state.questions[-1].question
                if self.state.questions
                else None,
                "expected_result": (self.state.pending or {}).get("expected_result"),
                "observed_frame_ids": sorted(self.screen.viewed),
                "deliberation_ms": round((time.perf_counter() - start) * 1000, 3),
            }
        )
        return self.decision
