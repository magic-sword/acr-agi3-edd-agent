"""ADK game player driven by missing knowledge and on-demand visual inspection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from functools import wraps
from typing import Any

import numpy as np
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
            return {"error": str(error), "mode_guidance": args[0]._build_mode_guidance()}

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
        self.last_trace: list[dict] = []
        self._loaded_skills: set[str] = set()
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
            self.move_cursor,
            self.click_at_cursor,
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
            "Before any cognitive transition, load_skill(skill_name='causal-deliberation'). "
            "To observe, load 'visual-inspector'; to aim or act, load 'game-controller'. "
            "Tools absent from the current schemas cannot be called. Loading a skill never changes mode.\n\n"
            "=== COGNITIVE STATE MACHINE ARCHITECTURE ===\n"
            "1. [PLAN Mode] (Current Goal & Strategy)\n"
            "   - Purpose: Observe the screen, identify the goal, and formulate a plan.\n"
            "   - If screen unobserved: Load 'visual-inspector' and call observe_screen(view='current').\n"
            "   - If causal rules are UNKNOWN (how objects move or what buttons do):\n"
            "     -> Call need_causal_knowledge(question='...') to advance to CAUSAL mode.\n"
            "   - If plan is ready: Call plan_actions(subgoal=..., steps=[...], rule_ids=[...]) to advance to EXECUTE mode.\n"
            "   - Constraint: Environment actions (step_action/click_at_cursor) are NOT available in PLAN mode.\n\n"
            "2. [CAUSAL Mode] (Knowledge Gap Resolution)\n"
            "   - Purpose: Resolve open questions through hypothesis testing.\n"
            "   - If visible by inspection: Call answer_visible_question(answer=..., visual_evidence=...).\n"
            "   - If test is needed: Call need_experiment(hypothesis=..., prediction=..., alternative=..., expected_visual_change=...) to advance to EXPERIMENT mode.\n"
            "   - Constraint: Environment actions are NOT available in CAUSAL mode.\n\n"
            "3. [EXPERIMENT Mode] (Controlled Hypothesis Testing)\n"
            "   - Purpose: Execute ONE minimal intervention to test the active hypothesis.\n"
            "   - Action: Load 'game-controller' and call step_action(action_id=..., reasoning=...) or move_cursor, observe_screen, then click_at_cursor.\n\n"
            "4. [EXECUTE Mode] (Plan Step Dispatch)\n"
            "   - Purpose: Execute the planned step.\n"
            "   - Action: Load 'game-controller' and call step_action or click_at_cursor matching the plan.\n\n"
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
            before_tool_callback=self._before_tool,
            after_tool_callback=self._after_tool,
        )
        self.runner = Runner(agent=self.agent, app_name=name, session_service=self.session_service)

    def _before_tool(self, tool, args, tool_context):
        if tool.name == "load_skill" and args.get("skill_name") in self._loaded_skills:
            return self._wrap_snapshot(
                {
                    "skill_name": args["skill_name"],
                    "already_loaded": True,
                    "message": "Skill is active in this session. Use its exposed tools; do not reload it.",
                }
            )
        return None

    def _after_tool(self, tool, args, tool_context, tool_response):
        if not isinstance(tool_response, dict):
            return None
        if tool.name == "load_skill" and "instructions" in tool_response:
            self._loaded_skills.add(tool_response["skill_name"])
        return self._wrap_snapshot(dict(tool_response))

    def _wrap_snapshot(self, res: dict) -> dict:
        """ツールの実行結果に最新の Cognitive Guidance を埋め込み、モデルが次の行動を自律理解できるようにする."""
        if isinstance(res, dict):
            res["mode_guidance"] = self._build_mode_guidance()
            res["active_skills"] = sorted(self._loaded_skills)
        return res

    def _build_mode_guidance(self) -> str:
        """現在地の ThoughtMode に応じた目的、達成条件、遷移ルールを明確に生成."""
        observed = self.screen.current_id in self.screen.viewed
        mode = self.state.mode

        def load_guidance(skill_name: str) -> str:
            call = json.dumps({"name": "load_skill", "arguments": {"skill_name": skill_name}})
            return (
                f"{mode.value}: Required skill is not loaded. Next tool call: {call}. "
                f"'{skill_name}' is a skill name, not a callable tool. "
                "Use the load_skill tool; do not call the skill name as a function."
            )

        if not observed:
            if "visual-inspector" not in self._loaded_skills:
                return load_guidance("visual-inspector")
            view = "both" if mode == ThoughtMode.REVIEW else "current"
            return f"{mode.value}: Visual inspector is active. Call observe_screen(view='{view}')."
        if mode in (ThoughtMode.PLAN, ThoughtMode.CAUSAL, ThoughtMode.REVIEW):
            if "causal-deliberation" not in self._loaded_skills:
                return load_guidance("causal-deliberation")
        if mode in (ThoughtMode.EXPERIMENT, ThoughtMode.EXECUTE):
            if "game-controller" not in self._loaded_skills:
                return load_guidance("game-controller")
            if self.available_actions == [6] or self.screen.cursor is not None:
                if self.screen.cursor is None:
                    return f"{mode.value}: Call move_cursor to aim at the observed target; no click yet."
                if self.screen.cursor_observed != (
                    self.screen.current_id,
                    self.screen.cursor_revision,
                ):
                    return f"{mode.value}: Cursor moved. Call observe_screen to inspect the reticle before clicking."
                return (
                    f"{mode.value}: Reticle has been observed. If aligned, call click_at_cursor "
                    "with visual_evidence and reasoning. If misplaced, move_cursor and observe again."
                )

        if mode == ThoughtMode.PLAN:
            if self.state.goal_evidence is None:
                return (
                    "PLAN: Call set_goal with the displayed goal and visual evidence. "
                    "If the goal is unknown, state that explicitly and describe visible candidates."
                )
            if not self.state.rules:
                return (
                    "PLAN: Goal recorded; causal rules unknown. Call need_causal_knowledge "
                    "with the specific missing relation blocking this goal."
                )
            return (
                f"PLAN: Goal {self.state.goal}. Rules: {list(self.state.rules)}. "
                "Plan backwards from the winning condition and its prerequisites; allow staging "
                "and temporary detours. Call plan_actions with conditional steps and cited rule_ids, "
                "continue_plan for a verified remaining plan, or need_causal_knowledge for a gap."
            )
        if mode == ThoughtMode.CAUSAL:
            active_q = (
                self.state.questions[-1].question if self.state.questions else "unknown relation"
            )
            relevant = [
                (key, value)
                for key, value in self.state.results.items()
                if value.get("action", {}).get("question") == active_q
            ]
            if relevant:
                result_id, result = relevant[-1]
                return (
                    f"CAUSAL: Question: {active_q}. Latest evidence: {result_id}: "
                    f"{result['outcome']}: {result['evidence']}. "
                    "If this answers the question, call resolve_question with cited result_ids, "
                    "a conditional precondition and effect to resume the suspended plan. "
                    "Refutation establishes a conditional limitation, not a universal prohibition. "
                    "If inconclusive or insufficient, revise the hypothesis or observation and "
                    "call need_experiment. Repeating an intervention requires retry_reason "
                    "explaining changed conditions or new information."
                )
            return (
                f"CAUSAL: Question: {active_q}. Use answer_visible_question for visible facts, "
                "use_known_rules for existing knowledge, or need_experiment with distinguishable "
                "conditional predictions and explicit expected_visual_change (boolean) for a missing relation. Choose the intervention from "
                "the observed board and available_actions. Do not ask the same question again."
            )
        if mode in (ThoughtMode.EXPERIMENT, ThoughtMode.EXECUTE):
            return (
                f"{mode.value}: Execute the prepared button using step_action, or aim at a "
                "click target with move_cursor, observe_screen, then click_at_cursor. "
                "Call need_causal_knowledge if a new gap prevents the prepared action."
            )
        if mode == ThoughtMode.REVIEW:
            before_id = (self.state.pending or {}).get("frame_id", self.screen.current_id - 1)
            return (
                f"REVIEW: Compare the predicted target effect in frames {before_id} and "
                f"{self.screen.current_id}. Observe both if needed, then call assess_result "
                "with supported, refuted or inconclusive and concrete evidence. Pixel or HUD "
                "changes alone do not support the prediction. Resolve the question only after assessment."
            )
        return ""

    @tool_errors
    def observe_screen(
        self,
        view: str = "current",
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> dict:
        """Inspect the current game frame on demand or compare before/after frames."""
        res = self.screen.observe_screen(view=view, x=x, y=y, width=width, height=height)
        return self._wrap_snapshot(res)

    def _tool_is_relevant(self, tool, context) -> bool:
        """Filter ADK-activated tools by thought purpose; ADK still owns loading.

        Observation and skill discovery remain available in every mode. Execution
        checks remain authoritative even if a stale call reaches the dispatcher.
        """
        common = {
            "list_skills",
            "load_skill",
            "load_skill_resource",
            "observe_screen",
            "move_cursor",
        }
        if tool.name == "move_cursor" and 6 not in self.available_actions:
            return False
        if tool.name == "step_action" and not any(a != 6 for a in self.available_actions):
            return False
        if tool.name == "click_at_cursor" and (
            6 not in self.available_actions
            or self.screen.cursor is None
            or self.screen.cursor_observed != (self.screen.current_id, self.screen.cursor_revision)
        ):
            return False
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
            ThoughtMode.EXPERIMENT: {"need_causal_knowledge", "step_action", "click_at_cursor"},
            ThoughtMode.EXECUTE: {"need_causal_knowledge", "step_action", "click_at_cursor"},
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
        self.state.goal_evidence = {"frame_id": self.screen.current_id, "evidence": visual_evidence}
        return self._wrap_snapshot(self.state.snapshot())

    @tool_errors
    def need_causal_knowledge(self, question: str) -> dict:
        """Suspend reasoning or an unissued action because a necessary causal fact is missing."""
        return self._wrap_snapshot(self.state.ask(question))

    @tool_errors
    def need_experiment(
        self,
        hypothesis: str,
        prediction: str,
        alternative: str,
        expected_visual_change: bool,
        retry_reason: str = "",
    ) -> dict:
        """Evidence is insufficient: design a minimal intervention to distinguish explanations."""
        self._ensure_observed()
        if type(expected_visual_change) is not bool:
            raise ValueError("expected_visual_change must explicitly be true or false.")
        self.state.propose_experiment(hypothesis, prediction, alternative, retry_reason)
        self.state.experiment["expected_visual_change"] = expected_visual_change
        return self._wrap_snapshot(self.state.snapshot())

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
        self.state.require_mode(ThoughtMode.REVIEW)
        pending = self.state.pending
        if not pending or before_frame_id != pending["frame_id"]:
            raise ValueError("Review the actual pending action's before frame.")
        changed = not np.array_equal(
            self.screen.frames[before_frame_id], self.screen.frames[after_frame_id]
        )
        expected = (pending.get("experiment") or {}).get("expected_visual_change")
        if outcome == "supported" and isinstance(expected, bool) and changed != expected:
            raise ValueError(
                "The predicted visual change contradicts the actual frames. "
                "Reinspect the evidence and assess as refuted or inconclusive. "
                "Pixel change alone never proves the predicted effect."
            )
        result = self.state.review(outcome, evidence, before_frame_id, after_frame_id)
        self.state.results[result["result_id"]]["observed_visual_change"] = changed
        return self._wrap_snapshot({"result_id": result["result_id"], **self.state.snapshot()})

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
        frame = self.screen.frames[self.screen.current_id]
        scope = f"{self.current_game_id}:{self.levels_completed}:{frame.shape}".encode()
        state_hash = hashlib.sha256(scope + frame.tobytes()).hexdigest()
        repeated = any(
            result["action"].get("state_hash") == state_hash
            and result["action"].get("action_id") == action_id
            and result["action"].get("coordinates")
            == ({"x": x, "y": y} if action_id == 6 else None)
            for result in self.state.results.values()
            if "action" in result
        )
        if (
            intent == "experiment"
            and repeated
            and not self.state.experiment.get("retry_reason", "").strip()
        ):
            raise ValueError(
                "This intervention was already tested on this board. Return to CAUSAL, "
                "resolve the evidence or specify a changed condition/information gain in retry_reason."
            )
        self.actions.pending_decision = None
        if action_id == 6:
            if (x, y) != self.screen.require_observed_cursor():
                raise ValueError("Click must use the observed cursor coordinates.")
            self.actions.click_at(x=x, y=y, reasoning=reasoning)
        else:
            self.actions.step_action(action=f"ACTION{action_id}", reasoning=reasoning)
        if self.actions.pending_decision is None:
            raise ValueError("Execution tool rejected the action.")
        self.decision = self.actions.pending_decision
        if action_id == 6 and self.decision.coordinates != {"x": x, "y": y}:
            self.decision = None
            self.actions.pending_decision = None
            raise ValueError("Controller changed the verified click coordinates.")
        self.state.pending = {
            "intent": intent,
            "frame_id": self.screen.current_id,
            "action_id": action_id,
            "coordinates": {"x": x, "y": y} if action_id == 6 else None,
            "expected_result": expected,
            "reasoning": reasoning,
            "question": self.state.questions[-1].question if self.state.questions else None,
            "state_hash": state_hash,
            "experiment": dict(self.state.experiment or {}) if intent == "experiment" else None,
            "repeated_intervention": repeated,
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
            raise ValueError("Use move_cursor, observe_screen, then click_at_cursor for ACTION6.")
        return self._schedule(action_id, None, None, reasoning)

    @tool_errors
    def move_cursor(self, x: int, y: int, reasoning: str) -> dict:
        """Position an internal reticle without a game action, in any thought mode.

        Load visual-inspector and observe the reticle after every movement.
        """
        if self.decision is not None:
            raise ValueError("Await the gateway frame before moving the cursor.")
        if 6 not in self.available_actions or not reasoning.strip():
            raise ValueError("Cursor aiming needs ACTION6 availability and a target rationale.")
        return self._wrap_snapshot(self.screen.move_cursor(x, y))

    @tool_errors
    def click_at_cursor(self, visual_evidence: str, reasoning: str) -> dict:
        """Click only the freshly observed reticle; coordinates cannot be changed here."""
        if not visual_evidence.strip():
            raise ValueError("Describe how the observed reticle aligns with the intended target.")
        x, y = self.screen.require_observed_cursor()
        result = self._schedule(6, x, y, reasoning)
        self.state.pending["cursor_confirmation"] = {
            "frame_id": self.screen.current_id,
            "cursor_revision": self.screen.cursor_revision,
            "coordinates": {"x": x, "y": y},
            "visual_evidence": visual_evidence,
        }
        self.screen.cursor_observed = None
        return result

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
            self.state.goal_evidence = None
            self.state.transition(ThoughtMode.PLAN, "Episode boundary; retain learned causal rules")
        # SkillToolset holds bound methods: retain the screen instance across
        # episodes so on-demand tools always see the current gateway frame.
        self.screen.clear_cursor()
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
            self.state.goal_evidence = None
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
        self.last_trace = []
        self._loaded_skills.clear()
        if hasattr(self.model, "_inference_trace"):
            self.model._inference_trace.clear()
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
                        if event.content:
                            for part in event.content.parts or []:
                                if part.function_call:
                                    self.last_trace.append(
                                        {
                                            "kind": "call",
                                            "name": part.function_call.name,
                                            "args": dict(part.function_call.args or {}),
                                        }
                                    )
                                elif part.function_response:
                                    response = dict(part.function_response.response or {})
                                    if "screen_observation" in response:
                                        obs = dict(response["screen_observation"])
                                        obs["images"] = [
                                            {k: v for k, v in item.items() if k != "data"}
                                            for item in obs["images"]
                                        ]
                                        response["screen_observation"] = obs
                                    self.last_trace.append(
                                        {
                                            "kind": "result",
                                            "name": part.function_response.name,
                                            "response": response,
                                        }
                                    )
                                elif part.text:
                                    self.last_trace.append({"kind": "text", "text": part.text})
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
                    f"Tool feedback: {getattr(self.model, '_last_output_error', '')}\n"
                    f"State update:\n{json.dumps(self.state.snapshot(), indent=2)}\n\n"
                    f"=== COGNITIVE GUIDANCE ===\n{guidance}\n\n"
                    "Task: Advance the state machine or execute an action using the opened tools."
                )
                message = Content(
                    role="user",
                    parts=[Part.from_text(text=followup_text)],
                )
        finally:
            self.last_trace.extend(
                {"kind": "inference", **entry}
                for entry in getattr(self.model, "_inference_trace", [])
            )
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
                "thought_snapshot": self.state.snapshot(),
                "tool_trace": list(self.last_trace),
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
