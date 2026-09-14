#!/usr/bin/env python3
"""Epistemic Prober - Hypothesis Probing & Epistemic Action Selection (ACR-AGI-3).

Implements the Human Epistemic Action Principle (Kirsh & Maglio 1994, Lake et al. 2017):
1. Distinguishes Epistemic Actions (probing rules/mechanics) from Pragmatic Actions (exploiting goals).
2. Generates single-step minimal interventions (Isolation of Variables) to learn actuator reach,
   linked movement, or collision boundaries.
3. Quantifies and tracks Epistemic Action Ratio (EAR).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class EpistemicProber:
    """仮説検証型アクティブ探索と認識論的行動（Epistemic Actions）の策定エンジン."""

    PROBE_ACTUATOR_REACH = "PROBE_ACTUATOR_REACH"
    PROBE_LINKAGE_DYNAMICS = "PROBE_LINKAGE_DYNAMICS"
    PROBE_BOUNDARY_COLLISION = "PROBE_BOUNDARY_COLLISION"
    PROBE_OBJECT_AFFORDANCE = "PROBE_OBJECT_AFFORDANCE"
    PRAGMATIC_EXPLOIT = "PRAGMATIC_EXPLOIT"

    def __init__(self) -> None:
        pass

    def evaluate_and_propose_probe(
        self,
        step_index: int,
        available_actions: List[int],
        unexplored_affordances: Optional[Dict[str, Any]] = None,
        transition_history: Optional[List[Dict[str, Any]]] = None,
        force_epistemic: bool = False,
    ) -> Dict[str, Any]:
        """未知の環境力学を解明するための認識論的行動（プローブ）を提案."""
        if not available_actions:
            return {
                "success": False,
                "error": "No available actions provided",
                "is_epistemic": False,
                "recommended_action_id": None,
                "probe_type": None,
            }

        unexplored = unexplored_affordances or {}
        history = transition_history or []

        # 1. 探索初期（ステップ 1〜5）または強制フラグ時は認識論的行動を優先
        is_early_exploration = (step_index <= 5)
        has_untested_mechanics = bool(
            unexplored.get("has_actuator_rail")
            or unexplored.get("has_linked_objects")
            or unexplored.get("has_untested_switches")
        )

        should_probe = force_epistemic or is_early_exploration or has_untested_mechanics

        if not should_probe:
            # 既知の力学に対する実利行動（Pragmatic）へ移行
            return {
                "success": True,
                "is_epistemic": False,
                "probe_type": self.PRAGMATIC_EXPLOIT,
                "recommended_action_id": available_actions[0],
                "hypothesis": "Mechanics established; proceeding with pragmatic goal execution.",
                "target_variable": "GOAL_PROGRESSION",
                "epistemic_ratio_recommendation": 0.1,
            }

        # 2. 未知力学に応じたプローブ種別と最小介入手の選択
        probe_type = self.PROBE_OBJECT_AFFORDANCE
        hypothesis = "Testing baseline environmental response to single-step action."
        target_var = "GENERAL_MOVEMENT"

        if unexplored.get("has_actuator_rail"):
            probe_type = self.PROBE_ACTUATOR_REACH
            hypothesis = "Testing piston/carriage extension range and holding contact."
            target_var = "ACTUATOR_DISPLACEMENT"
        elif unexplored.get("has_linked_objects"):
            probe_type = self.PROBE_LINKAGE_DYNAMICS
            hypothesis = "Testing mirrored or indirect linkage displacement on secondary blocks."
            target_var = "LINKAGE_CORRELATION"
        elif unexplored.get("has_untested_switches"):
            probe_type = self.PROBE_OBJECT_AFFORDANCE
            hypothesis = "Testing state toggle effect upon interacting with switch."
            target_var = "SWITCH_STATE_CHANGE"

        # 最小介入の原則: 過去にまだ試していない、または低リスクな方向を選択
        tested_actions = {t.get("action") for t in history if "action" in t}
        untested_actions = [a for a in available_actions if a not in tested_actions]

        chosen_action = untested_actions[0] if untested_actions else available_actions[0]

        return {
            "success": True,
            "is_epistemic": True,
            "probe_type": probe_type,
            "recommended_action_id": chosen_action,
            "hypothesis": hypothesis,
            "target_variable": target_var,
            "epistemic_ratio_recommendation": 0.8 if step_index <= 3 else 0.4,
            "minimal_intervention_rule": "Execute exactly 1 primitive step, then observe state diff.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epistemic Prober - Epistemic Action Selection CLI"
    )
    parser.add_argument("--step", type=int, default=1, help="Current step index")
    parser.add_argument(
        "--actions", type=str, default="[0, 1, 2, 3]", help="JSON list of available action IDs"
    )
    parser.add_argument(
        "--affordances", type=str, default="{}", help="JSON dict of unexplored affordances"
    )
    parser.add_argument("--file", type=str, help="Path to input JSON file")
    args = parser.parse_args()

    prober = EpistemicProber()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            step = data.get("step", 1)
            actions = data.get("actions", [0, 1, 2, 3])
            affordances = data.get("affordances", {})
    else:
        step = args.step
        actions = json.loads(args.actions)
        affordances = json.loads(args.affordances)

    res = prober.evaluate_and_propose_probe(step, actions, unexplored_affordances=affordances)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
