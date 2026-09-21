#!/usr/bin/env python3
"""Hypothesis Engine - Core deterministic logic for hypothesis lifecycle management.

Handles scientific formulation, outcome verification, falsification archiving,
and taboo-free alternative proposal for ARC-AGI-3 gameplay.
"""

from typing import Any, Dict, List, Optional, Tuple


class HypothesisEngineCore:
    """Manages the lifecycle of hypotheses during game exploration."""

    def __init__(self):
        self.active_hypothesis: Optional[Dict[str, Any]] = None
        self.refuted_hypotheses: Dict[str, Dict[str, Any]] = {}
        self.verified_hypotheses: List[Dict[str, Any]] = []

    def formulate_hypothesis(
        self,
        step_index: int,
        claim: str,
        action_name: str,
        action_id: int,
        coords: Optional[Dict[str, int]] = None,
        expected_effect: str = "",
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """Formulates a new causal hypothesis if not already refuted."""
        if not claim or not action_name:
            return {"status": "error", "message": "Claim and action_name must be non-empty."}

        # Check if identical action/coord is already refuted
        if self.is_refuted(action_id, coords):
            return {
                "status": "rejected",
                "message": f"Action {action_name} at coords {coords} has already been refuted.",
            }

        hypo = {
            "step_index": step_index,
            "claim": claim,
            "action_name": action_name,
            "action_id": action_id,
            "coords": coords or {},
            "expected_effect": expected_effect or "Cause meaningful state change",
            "reasoning": reasoning,
        }
        self.active_hypothesis = hypo
        return {"status": "ok", "hypothesis": hypo}

    def evaluate_outcome(
        self,
        step_index: int,
        pixels_changed: int,
        is_effective: bool = True,
        actual_notes: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Evaluates whether the active hypothesis was verified or refuted.

        Returns:
            (status, record) where status is 'VERIFIED', 'REFUTED', or 'NO_ACTIVE_HYPOTHESIS'
        """
        if not self.active_hypothesis:
            return "NO_ACTIVE_HYPOTHESIS", {"message": "No active hypothesis to evaluate"}

        hypo = self.active_hypothesis
        act_name = hypo.get("action_name", "UNKNOWN")
        act_id = hypo.get("action_id", 0)
        coords = hypo.get("coords", {})
        coord_str = f"_{coords.get('x')}_{coords.get('y')}" if ("x" in coords and "y" in coords) else ""

        if pixels_changed == 0 or not is_effective:
            # Refuted
            refuted_id = f"hypothesis.refuted.s{step_index}_{act_name}{coord_str}"
            record = {
                "refuted_id": refuted_id,
                "step_index": step_index,
                "action_name": act_name,
                "action_id": act_id,
                "coords": coords,
                "original_claim": hypo.get("claim", ""),
                "pixels_changed": pixels_changed,
                "lesson": f"Action {act_name}{coord_str} produced 0 pixel change; assumption refuted.",
                "tags": ["hypothesis", "refuted", "falsified", "constraint"],
            }
            self.refuted_hypotheses[refuted_id] = record
            self.active_hypothesis = None
            return "REFUTED", record
        else:
            # Verified
            verified_id = f"hypothesis.verified.s{step_index}_{act_name}{coord_str}"
            record = {
                "verified_id": verified_id,
                "step_index": step_index,
                "action_name": act_name,
                "action_id": act_id,
                "coords": coords,
                "claim": hypo.get("claim", ""),
                "pixels_changed": pixels_changed,
                "notes": actual_notes,
                "tags": ["hypothesis", "verified", "causality"],
            }
            self.verified_hypotheses.append(record)
            self.active_hypothesis = None
            return "VERIFIED", record

    def is_refuted(self, action_id: int, coords: Optional[Dict[str, int]] = None) -> bool:
        """Checks if an action ID and coordinate pair has been previously refuted."""
        for r in self.refuted_hypotheses.values():
            if r.get("action_id") == action_id:
                if coords and "x" in coords and "y" in coords:
                    r_coords = r.get("coords", {})
                    if r_coords.get("x") == coords.get("x") and r_coords.get("y") == coords.get("y"):
                        return True
                elif not coords:
                    # Generic action with no coordinates
                    return True
        return False

    def get_refuted_bookmarks(self) -> List[str]:
        """Returns lightweight Level 1 bookmark strings (TOC entries) for all refuted hypotheses."""
        bookmarks = []
        for rid, record in self.refuted_hypotheses.items():
            act = record.get("action_name", "")
            coords = record.get("coords", {})
            coord_info = f" pos=({coords.get('x')},{coords.get('y')})" if coords else ""
            bookmarks.append(f"[{rid}] Refuted: {act}{coord_info} (0 pixel change) (tags: refuted, falsified)")
        return bookmarks

    def propose_alternative_hypothesis(
        self,
        step_index: int,
        available_actions: List[int],
        candidate_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Proposes the first unrefuted action or coordinate pair."""
        # 1. Coordinate-based actions (e.g. click at ACTION6)
        if (not available_actions or 6 in available_actions) and candidate_coords:
            for cx, cy in candidate_coords:
                coords = {"x": cx, "y": cy}
                if not self.is_refuted(6, coords):
                    return {
                        "action_name": "ACTION6",
                        "action_id": 6,
                        "coords": coords,
                        "claim": f"Interact at coordinate ({cx}, {cy}) to discover affordance",
                    }


        # 2. Discrete actions
        for aid in available_actions:
            if aid != 0 and not self.is_refuted(aid, None):
                return {
                    "action_name": f"ACTION{aid}",
                    "action_id": aid,
                    "coords": {},
                    "claim": f"Probe action ACTION{aid} to measure directional dynamics",
                }

        # If everything is refuted, fall back to RESET (ACTION0) if available
        if 0 in available_actions:
            return {
                "action_name": "RESET",
                "action_id": 0,
                "coords": {},
                "claim": "Trigger active reset to escape fully-refuted deadlock",
            }
        return None

    def reset_episode(self) -> None:
        """Clears active hypotheses while preserving refuted/verified causal knowledge."""
        self.active_hypothesis = None


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Hypothesis Engine CLI tool.")
    parser.add_argument("--action", choices=["formulate", "evaluate", "bookmarks"], default="bookmarks")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--claim", type=str, default="")
    parser.add_argument("--action-name", type=str, default="")
    parser.add_argument("--action-id", type=int, default=1)
    parser.add_argument("--pixels-changed", type=int, default=0)

    args = parser.parse_args()
    engine = HypothesisEngineCore()

    if args.action == "formulate":
        res = engine.formulate_hypothesis(args.step, args.claim, args.action_name, args.action_id)
        print(json.dumps(res))
    elif args.action == "evaluate":
        status, record = engine.evaluate_outcome(args.step, args.pixels_changed)
        print(json.dumps({"status": status, "record": record}))
    else:
        print(json.dumps({"bookmarks": engine.get_refuted_bookmarks()}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
