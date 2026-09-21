"""Goal-directed thought transitions, independent of screen-change heuristics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ThoughtMode(str, Enum):
    PLAN = "PLAN"
    CAUSAL = "CAUSAL"
    EXPERIMENT = "EXPERIMENT"
    EXECUTE = "EXECUTE"
    REVIEW = "REVIEW"


@dataclass
class Question:
    question: str
    return_mode: ThoughtMode
    suspended_goal: str


@dataclass
class ThoughtState:
    mode: ThoughtMode = ThoughtMode.PLAN
    goal: str = "Identify and reach the game's displayed winning condition"
    questions: list[Question] = field(default_factory=list)
    rules: dict[str, dict] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    facts: list[dict] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    experiment: dict | None = None
    pending: dict | None = None
    transitions: list[dict] = field(default_factory=list)
    serial: int = 0

    def transition(self, mode: ThoughtMode, reason: str) -> None:
        self.transitions.append({"from": self.mode.value, "to": mode.value, "reason": reason})
        self.transitions = self.transitions[-64:]
        self.mode = mode

    def snapshot(self) -> dict:
        return {
            "mode": self.mode.value,
            "goal": self.goal,
            "questions": [asdict(q) for q in self.questions],
            "rules": self.rules,
            "recent_results": dict(list(self.results.items())[-8:]),
            "visible_facts": self.facts,
            "remaining_plan": self.plan,
            "experiment": self.experiment,
            "pending": self.pending,
        }

    def require_mode(self, *modes: ThoughtMode) -> None:
        if self.mode not in modes:
            raise ValueError(
                f"This operation needs {[m.value for m in modes]}; current={self.mode.value}"
            )

    def ask(self, question: str) -> dict:
        self.require_mode(
            ThoughtMode.PLAN, ThoughtMode.CAUSAL, ThoughtMode.EXECUTE, ThoughtMode.EXPERIMENT
        )
        if not question.strip() or len(self.questions) >= 8:
            raise ValueError("Supply a specific missing fact; at most eight suspended questions.")
        # A newly discovered gap can interrupt an unissued action. Resume reasoning,
        # never the obsolete execution intent, after answering that question.
        parent_mode = self.mode
        if self.mode == ThoughtMode.EXECUTE:
            self.plan.clear()
            parent_mode = ThoughtMode.PLAN
        elif self.mode == ThoughtMode.EXPERIMENT:
            self.experiment = None
            parent_mode = ThoughtMode.CAUSAL
        self.questions.append(Question(question, parent_mode, self.goal))
        self.transition(ThoughtMode.CAUSAL, f"Knowledge missing: {question}")
        return self.snapshot()

    def propose_experiment(self, hypothesis: str, prediction: str, alternative: str) -> dict:
        self.require_mode(ThoughtMode.CAUSAL)
        if not self.questions or not all(s.strip() for s in (hypothesis, prediction, alternative)):
            raise ValueError(
                "An experiment needs an open question and distinguishable predictions."
            )
        self.experiment = {
            "hypothesis": hypothesis,
            "prediction": prediction,
            "alternative": alternative,
            "question": self.questions[-1].question,
        }
        self.transition(ThoughtMode.EXPERIMENT, "Evidence is insufficient; test the hypothesis")
        return self.snapshot()

    def review(self, outcome: str, evidence: str, before: int, after: int) -> dict:
        self.require_mode(ThoughtMode.REVIEW)
        if not self.pending or outcome not in ("supported", "refuted", "inconclusive"):
            raise ValueError("Review the pending action as supported, refuted or inconclusive.")
        if not evidence.strip() or before != self.pending["frame_id"] or after <= before:
            raise ValueError("Evidence must describe the actual before/after observations.")
        self.serial += 1
        result_id = f"result_{self.serial}"
        self.results[result_id] = {
            "outcome": outcome,
            "evidence": evidence,
            "before": before,
            "after": after,
            "action": self.pending.copy(),
        }
        intent = self.pending["intent"]
        self.pending = None
        if intent == "experiment":
            self.transition(ThoughtMode.CAUSAL, f"Experiment assessed: {result_id}")
        elif outcome == "supported":
            self.transition(ThoughtMode.PLAN, "Reassess remaining plan against the new observation")
        else:
            self.plan.clear()
            self.questions.append(
                Question("Why did the planned effect fail?", ThoughtMode.PLAN, self.goal)
            )
            self.transition(
                ThoughtMode.CAUSAL, "Diagnose the failed prediction without erasing known rules"
            )
        return {"result_id": result_id, **self.snapshot()}

    def resolve(self, answer: str, precondition: str, effect: str, result_ids: list[str]) -> dict:
        self.require_mode(ThoughtMode.CAUSAL)
        if not self.questions or not all(s.strip() for s in (answer, precondition, effect)):
            raise ValueError("Resolve an open question with a conditional causal statement.")
        if not result_ids or any(
            key not in self.results or self.results[key]["outcome"] == "inconclusive"
            for key in result_ids
        ):
            raise ValueError(
                "Cite assessed evidence; uncertainty alone does not resolve a question."
            )
        self.serial += 1
        rule_id = f"rule_{self.serial}"
        self.rules[rule_id] = {
            "answer": answer,
            "precondition": precondition,
            "effect": effect,
            "evidence": result_ids,
            "status": "provisional",
        }
        question = self.questions.pop()
        self.goal = question.suspended_goal
        self.experiment = None
        self.transition(question.return_mode, f"Resume suspended reasoning: {question.question}")
        return {"rule_id": rule_id, **self.snapshot()}

    def make_plan(self, subgoal: str, steps: list[dict[str, Any]], rule_ids: list[str]) -> dict:
        self.require_mode(ThoughtMode.PLAN)
        if not subgoal.strip() or not 1 <= len(steps) <= 8:
            raise ValueError("Give a subgoal and one to eight conditional action steps.")
        if not rule_ids or any(key not in self.rules for key in rule_ids):
            raise ValueError("Cite causal rules; ask a knowledge question when rules are missing.")
        for step in steps:
            if not isinstance(step.get("action_id"), int) or not step.get("expected_result"):
                raise ValueError("Each step needs an action_id and an expected_result.")
            if not step.get("precondition"):
                raise ValueError("Each step needs a precondition to check on screen.")
        self.plan = [{**step, "subgoal": subgoal, "rule_ids": list(rule_ids)} for step in steps]
        self.transition(ThoughtMode.EXECUTE, f"Conditional plan ready for subgoal: {subgoal}")
        return self.snapshot()
