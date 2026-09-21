#!/usr/bin/env python3
"""Rule Inducer - Core deterministic logic for empirical causal rule induction.

Analyzes environmental state transitions, pixel changes, and level advance events
to induce deterministic causal rules and win-condition invariants.
"""

from typing import Any, Dict, List, Optional, Tuple


class RuleInducerCore:
    """Manages empirical rule induction and win-condition discovery."""

    def __init__(self):
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.win_condition_hypotheses: List[Dict[str, Any]] = []

    def induce_rule_from_transition(
        self,
        step_index: int,
        action_name: str,
        action_id: int,
        pixels_changed: int,
        coords: Optional[Dict[str, int]] = None,
        level_before: int = 0,
        level_after: int = 0,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Induces a causal rule or win-condition from a single transition.

        Returns:
            {"status": "ok", "is_new": bool, "rule": dict} or error.
        """
        if not action_name:
            return {"status": "error", "message": "Action name cannot be empty."}

        is_new = False
        rule_record: Optional[Dict[str, Any]] = None

        # 1. Level Advance Event -> Win Condition Induction
        if level_after > level_before:
            win_id = f"win_condition_level_{level_before}_to_{level_after}"
            rule_record = {
                "rule_id": win_id,
                "rule_type": "win_condition",
                "trigger_step": step_index,
                "trigger_action": action_name,
                "trigger_coords": coords or {},
                "statement": f"Triggering {action_name} at {coords or 'current position'} advances level from {level_before} to {level_after}.",
                "confidence": 1.0,
            }
            self.win_condition_hypotheses.append(rule_record)
            self.rules[win_id] = rule_record
            return {"status": "ok", "is_new": True, "rule": rule_record}

        # 2. Effective State Change -> Interaction / Affordance Rule
        if pixels_changed > 0:
            coord_suffix = f"_{coords.get('x')}_{coords.get('y')}" if (coords and "x" in coords and "y" in coords) else ""
            rule_id = f"rule_effective_{action_name}{coord_suffix}"
            if rule_id not in self.rules:
                is_new = True
                rule_record = {
                    "rule_id": rule_id,
                    "rule_type": "causal_interaction",
                    "action_name": action_name,
                    "action_id": action_id,
                    "coords": coords or {},
                    "effect": f"Causes environmental state change ({pixels_changed} pixels changed).",
                    "observation_count": 1,
                    "confidence": 0.8,
                }
                self.rules[rule_id] = rule_record
            else:
                self.rules[rule_id]["observation_count"] += 1
                self.rules[rule_id]["confidence"] = min(1.0, self.rules[rule_id]["confidence"] + 0.1)
                rule_record = self.rules[rule_id]
            return {"status": "ok", "is_new": is_new, "rule": rule_record}

        # 3. Ineffective (0 pixels changed) -> Invariant Barrier / No-op Rule
        if pixels_changed == 0:
            coord_suffix = f"_{coords.get('x')}_{coords.get('y')}" if (coords and "x" in coords and "y" in coords) else ""
            rule_id = f"rule_barrier_{action_name}{coord_suffix}"
            if rule_id not in self.rules:
                is_new = True
                rule_record = {
                    "rule_id": rule_id,
                    "rule_type": "invariant_barrier",
                    "action_name": action_name,
                    "action_id": action_id,
                    "coords": coords or {},
                    "effect": "Produces 0 pixel changes (barrier or inactive affordance).",
                    "observation_count": 1,
                    "confidence": 0.9,
                }
                self.rules[rule_id] = rule_record
            else:
                self.rules[rule_id]["observation_count"] += 1
                rule_record = self.rules[rule_id]
            return {"status": "ok", "is_new": is_new, "rule": rule_record}

        return {"status": "error", "message": "Unknown transition outcome"}

    def get_known_rules(self, rule_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of induced rules, optionally filtered by type."""
        if rule_type:
            return [r for r in self.rules.values() if r.get("rule_type") == rule_type]
        return list(self.rules.values())

    def get_win_condition_hypotheses(self) -> List[str]:
        """Returns readable statements of win condition hypotheses."""
        return [w.get("statement", "") for w in self.win_condition_hypotheses]

    def reset_episode(self) -> None:
        """Clears transient rules while preserving verified win conditions."""
        pass


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Rule Inducer CLI Tool.")
    parser.add_argument("--action-name", type=str, default="ACTION1")
    parser.add_argument("--action-id", type=int, default=1)
    parser.add_argument("--pixels-changed", type=int, default=1)
    parser.add_argument("--level-before", type=int, default=0)
    parser.add_argument("--level-after", type=int, default=0)
    args = parser.parse_args()

    inducer = RuleInducerCore()
    res = inducer.induce_rule_from_transition(
        step_index=1,
        action_name=args.action_name,
        action_id=args.action_id,
        pixels_changed=args.pixels_changed,
        level_before=args.level_before,
        level_after=args.level_after,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
