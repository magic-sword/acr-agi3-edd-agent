"""Taboo Reset Guard: Failure attribution, taboo constraint learning, and active reset engine."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


class TabooResetGuard:
    """Monitors execution state to prune ineffective actions and trigger active resets."""

    def __init__(self, max_stagnation: int = 6, oscillation_threshold: int = 2) -> None:
        self.max_stagnation = max_stagnation
        self.oscillation_threshold = oscillation_threshold
        self.action_history: List[int] = []

    def record_step(self, action_id: int) -> None:
        self.action_history.append(action_id)
        if len(self.action_history) > 20:
            self.action_history.pop(0)

    def is_oscillating(self) -> bool:
        """Detects 2-step oscillation like [A, B, A, B]."""
        if len(self.action_history) < 4:
            return False
        h = self.action_history
        return h[-1] == h[-3] and h[-2] == h[-4] and h[-1] != h[-2]

    def filter_taboo_actions(
        self,
        proposed_action: int,
        last_action_effective: bool,
        last_action_id: Optional[int],
        available_actions: List[int],
        stagnation_count: int,
    ) -> Tuple[bool, int, str]:
        """Filters proposed action against taboo constraints.

        Returns:
            (is_allowed, sanitized_action, reason)
        """
        valid_actions = [a for a in available_actions if a != 0]
        if not valid_actions:
            return True, proposed_action, "No alternative actions available."

        # Case 1: Only 1 action available (e.g. click-only) -> cannot veto
        if len(valid_actions) == 1:
            return True, proposed_action, "Single action available; bypass taboo."

        # Case 2: Wall bump veto (last action caused 0 pixel change, model proposes same action)
        if not last_action_effective and last_action_id == proposed_action and stagnation_count >= 1:
            # Pick an alternative action from available
            alternatives = [a for a in valid_actions if a != proposed_action]
            alt = alternatives[0] if alternatives else proposed_action
            return False, alt, f"Vetoed ACTION{proposed_action} due to wall bump/zero pixel change; selected alternative ACTION{alt}."

        # Case 3: 2-step oscillation veto
        if self.is_oscillating():
            vetoed = self.action_history[-2]  # Veto the cyclic return
            alternatives = [a for a in valid_actions if a != proposed_action and a != vetoed]
            alt = alternatives[0] if alternatives else (valid_actions[0] if valid_actions[0] != proposed_action else valid_actions[-1])
            return False, alt, f"Vetoed ACTION{proposed_action} due to 2-step oscillation; break loop with ACTION{alt}."

        return True, proposed_action, "Action allowed."

    def should_active_reset(self, stagnation_count: int) -> bool:
        """Determines if active level reset is warranted due to unrecoverable stagnation."""
        return stagnation_count >= self.max_stagnation


def main() -> None:
    parser = argparse.ArgumentParser(description="Taboo Reset Guard CLI for loop and stagnation prevention.")
    parser.add_argument("--proposed", type=int, default=1, help="Proposed action ID")
    parser.add_argument("--last-effective", action="store_true", help="Was last action effective")
    parser.add_argument("--last-action", type=int, default=1, help="Last action ID executed")
    parser.add_argument("--stagnation", type=int, default=0, help="Consecutive stagnation steps")
    parser.add_argument("--available", type=str, default="[1, 2, 3, 4]", help="Available action IDs JSON")
    parser.add_argument("--history", type=str, default="[]", help="Action history IDs JSON")
    args = parser.parse_args()

    guard = TabooResetGuard()
    history = json.loads(args.history)
    for h in history:
        guard.record_step(h)

    available = json.loads(args.available)
    is_allowed, sanitized, reason = guard.filter_taboo_actions(
        proposed_action=args.proposed,
        last_action_effective=args.last_effective,
        last_action_id=args.last_action,
        available_actions=available,
        stagnation_count=args.stagnation,
    )

    should_reset = guard.should_active_reset(args.stagnation)

    result = {
        "is_allowed": is_allowed,
        "sanitized_action": sanitized,
        "should_reset": should_reset,
        "reason": reason,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
