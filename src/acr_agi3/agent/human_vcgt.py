"""人間プレイ思考アノテーション (VCGT: Visual Concept Guided Thinking) データローダー."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acr_agi3.meta.skill_harness import SkillHarness

_harness = SkillHarness()
_dec_mod = _harness.get_skill_module("subgoal-decomposer")
DecompositionPlan = _dec_mod.DecompositionPlan
Subgoal = _dec_mod.Subgoal


@dataclass
class VCGTExplanation:
    """人間の思考解説構造 (Goal -> Reasoning -> Steps -> Reflection)."""

    goal: str
    reasoning: str
    steps: list[str] = field(default_factory=list)
    reflection: str = ""


@dataclass
class VCGTRecord:
    """単一の VCGT アノテーションレコード."""

    task_id: str
    step_index: int
    environment: str
    observation: dict[str, Any]
    human_vcgt: VCGTExplanation
    invariants_identified: list[str] = field(default_factory=list)

    def to_subgoal_plan(self) -> DecompositionPlan:
        """人間の VCGT 思考ログをエージェントの DecompositionPlan に変換."""
        subgoals: list[Subgoal] = []
        for i, step_desc in enumerate(self.human_vcgt.steps):
            subgoals.append(
                Subgoal(
                    index=i + 1,
                    name=f"Subgoal_{i + 1}",
                    objective=step_desc,
                    reasoning=self.human_vcgt.reasoning if i == 0 else "Sequential progression",
                    expected_operation="execute_step",
                    parameters={"step_detail": step_desc},
                )
            )

        return DecompositionPlan(
            task_hint=self.human_vcgt.goal,
            subgoals=subgoals,
            total_steps=len(subgoals),
            constraints=self.invariants_identified,
            reasoning_trace=(
                f"Reasoning: {self.human_vcgt.reasoning}\nReflection: {self.human_vcgt.reflection}"
            ),
        )

    def format_prompt_demonstration(self) -> str:
        """LLM/VLM プロンプト用の思考デモンストレーションテキストに変換."""
        steps_text = "\n".join(
            f"  - Step {i + 1}: {s}" for i, s in enumerate(self.human_vcgt.steps)
        )
        invariants_text = "\n".join(f"  * {inv}" for inv in self.invariants_identified)

        return (
            f"### Example Task [{self.task_id}]:\n"
            f"- Environment: {self.environment}\n"
            f"- Goal: {self.human_vcgt.goal}\n"
            f"- Invariants:\n{invariants_text}\n"
            f"- Reasoning: {self.human_vcgt.reasoning}\n"
            f"- Action Plan:\n{steps_text}\n"
            f"- Reflection: {self.human_vcgt.reflection}\n"
        )


class VCGTDataset:
    """VCGT データセット管理クラス."""

    def __init__(self, records: list[VCGTRecord] | None = None) -> None:
        self.records: list[VCGTRecord] = records or []

    @classmethod
    def load_from_json(cls, file_path: str | Path) -> VCGTDataset:
        """JSON ファイルから VCGT レコードを読み込む."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"VCGT file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        records: list[VCGTRecord] = []
        for item in data:
            expl_data = item.get("human_vcgt", {})
            explanation = VCGTExplanation(
                goal=expl_data.get("goal", ""),
                reasoning=expl_data.get("reasoning", ""),
                steps=expl_data.get("steps", []),
                reflection=expl_data.get("reflection", ""),
            )
            record = VCGTRecord(
                task_id=item.get("task_id", ""),
                step_index=item.get("step_index", 0),
                environment=item.get("environment", ""),
                observation=item.get("observation", {}),
                human_vcgt=explanation,
                invariants_identified=item.get("invariants_identified", []),
            )
            records.append(record)

        return cls(records=records)

    def get_task(self, task_id: str) -> list[VCGTRecord]:
        """指定したタスク ID のレコード一覧を取得."""
        return [r for r in self.records if r.task_id == task_id]

    def build_few_shot_prompt(self, max_examples: int = 2) -> str:
        """Few-shot 用のプロンプトブロックを構築."""
        examples = self.records[:max_examples]
        prompt_parts = ["## Human Visual Concept-Guided Thinking (VCGT) Reference Examples:"]
        for ex in examples:
            prompt_parts.append(ex.format_prompt_demonstration())
        return "\n\n".join(prompt_parts)

    def __len__(self) -> int:
        return len(self.records)
