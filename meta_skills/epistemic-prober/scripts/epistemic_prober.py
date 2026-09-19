"""Epistemic Prober: Hypothesis-driven active exploration and action dynamics probing engine."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _normalize_2d(grid: Any) -> np.ndarray:
    if isinstance(grid, np.ndarray):
        arr = grid
    else:
        arr = np.array(grid, dtype=int)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[-1]
    return arr


class EpistemicProber:
    """Manages empirical probing of controller actions to infer physical dynamics (UP, DOWN, LEFT, RIGHT, CLICK)."""

    def __init__(self) -> None:
        self.dynamics_map: Dict[str, int] = {}
        self.tested_actions: List[int] = []

    def is_probing_needed(self, step_index: int, available_actions: List[int], max_probe_steps: int = 4) -> bool:
        """Determines if epistemic probing is recommended for the current episode onset."""
        if step_index > max_probe_steps:
            return False
        # If click is the only action, no directional probing needed
        if available_actions == [6]:
            self.dynamics_map["CLICK"] = 6
            return False
        # If all 4 cardinal directions are mapped, no probing needed
        cardinals = {"UP", "DOWN", "LEFT", "RIGHT"}
        if cardinals.issubset(set(self.dynamics_map.keys())):
            return False
        # If untested actions remain
        untested = [a for a in available_actions if a not in self.tested_actions and a != 0]
        return len(untested) > 0

    def recommend_probe_action(self, available_actions: List[int]) -> Optional[int]:
        """Selects the next action to test in isolation."""
        untested = [a for a in available_actions if a not in self.tested_actions and a != 0]
        if untested:
            return untested[0]
        return None

    def analyze_displacement(
        self,
        grid_before: Any,
        grid_after: Any,
        action_id: int,
    ) -> Optional[str]:
        """Infers semantic action role from frame difference before and after action execution."""
        arr_b = _normalize_2d(grid_before)
        arr_a = _normalize_2d(grid_after)

        if arr_b.shape != arr_a.shape or arr_b.size == 0:
            return None

        diff = (arr_b != arr_a)
        diff_count = int(np.sum(diff))
        if diff_count == 0:
            # Action was ineffective or hit boundary
            return None

        self.tested_actions.append(action_id)

        # Find rows and cols where values changed
        rows_b, cols_b = np.where(diff)
        # Check if there is a moving object: pixels that vanished vs pixels that appeared
        vanished_mask = diff & (arr_a == 0)
        appeared_mask = diff & (arr_b == 0)

        if np.sum(vanished_mask) > 0 and np.sum(appeared_mask) > 0:
            mean_vr, mean_vc = np.mean(np.where(vanished_mask)[0]), np.mean(np.where(vanished_mask)[1])
            mean_ar, mean_ac = np.mean(np.where(appeared_mask)[0]), np.mean(np.where(appeared_mask)[1])
            dr = mean_ar - mean_vr
            dc = mean_ac - mean_vc
        else:
            # Simple centroid shift of all changed pixels
            dr = np.mean(rows_b) - (arr_b.shape[0] / 2.0)
            dc = np.mean(cols_b) - (arr_b.shape[1] / 2.0)

        role = None
        if abs(dr) > abs(dc):
            if dr < -0.1:
                role = "UP"
            elif dr > 0.1:
                role = "DOWN"
        elif abs(dc) > abs(dr):
            if dc < -0.1:
                role = "LEFT"
            elif dc > 0.1:
                role = "RIGHT"
        elif diff_count <= 4 and action_id == 6:
            role = "CLICK"

        if role:
            self.dynamics_map[role] = action_id
        return role

    def get_dynamics_map(self) -> Dict[str, int]:
        return dict(self.dynamics_map)

    def set_dynamics_map(self, d_map: Dict[str, int]) -> None:
        self.dynamics_map.update(d_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Epistemic Prober CLI for action dynamics inference.")
    parser.add_argument("--before", type=str, help="JSON string of grid before action")
    parser.add_argument("--after", type=str, help="JSON string of grid after action")
    parser.add_argument("--action", type=int, default=1, help="Action ID that was executed")
    parser.add_argument("--file", type=str, help="JSON file containing before, after, and action")
    args = parser.parse_args()

    prober = EpistemicProber()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            grid_b = data.get("before", [])
            grid_a = data.get("after", [])
            act_id = data.get("action", 1)
    elif args.before and args.after:
        grid_b = json.loads(args.before)
        grid_a = json.loads(args.after)
        act_id = args.action
    else:
        grid_b = [[0, 2, 0], [0, 0, 0]]
        grid_a = [[0, 0, 0], [0, 2, 0]]
        act_id = 2

    inferred_role = prober.analyze_displacement(grid_b, grid_a, act_id)
    result = {
        "action_id": act_id,
        "inferred_role": inferred_role,
        "dynamics_map": prober.get_dynamics_map(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
