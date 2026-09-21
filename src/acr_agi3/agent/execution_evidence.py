"""Game-frame evidence for bounded execution and state-scoped experiments.

Pixel changes are observations, not proof of progress or controller semantics.
Only repeated, unambiguous rigid translations establish a movement mapping.
"""
from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Motion:
    color: int
    dr: int
    dc: int

    @property
    def direction(self) -> str:
        return ("DOWN" if self.dr > 0 else "UP") if self.dr else (
            "RIGHT" if self.dc > 0 else "LEFT"
        )


class ExecutionEvidence:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.expected: tuple[int, np.ndarray] | None = None
        self.samples: dict[int, tuple[Motion, int]] = {}
        self.attempts: Counter = Counter()
        self.trials: OrderedDict[bytes, dict[tuple[int, int, int], str]] = OrderedDict()
        self.visits: Counter = Counter()
        self.state_key = b""
        self.levels = 0
        self.prediction_match: bool | None = None
        self.meaningful_change = False
        self.level_progress = False
        self.pixels_changed = 0
        self.cycle = False

    def reset_episode(self) -> None:
        self.previous = None
        self.expected = None
        self.visits.clear()
        self.prediction_match = None
        self.state_key = b""

    @staticmethod
    def key(grid: np.ndarray, levels: int) -> bytes:
        arr = np.asarray(grid, dtype=np.int64)
        return hashlib.sha256(str((arr.shape, levels)).encode() + arr.tobytes()).digest()

    @staticmethod
    def translation(before: np.ndarray, after: np.ndarray) -> Motion | None:
        if before.shape != after.shape or not before.size:
            return None
        colors, counts = np.unique(before, return_counts=True)
        background = int(colors[np.argmax(counts)])
        motions = []
        for color in colors:
            if color == background:
                continue
            a, b = np.argwhere(before == color), np.argwhere(after == color)
            if len(a) == 0 or len(a) != len(b):
                continue
            delta = b[0] - a[0]
            dr, dc = map(int, delta)
            if (dr == 0) == (dc == 0):
                continue
            if np.array_equal(a + delta, b):
                motions.append(Motion(int(color), dr, dc))
        return motions[0] if len(motions) == 1 else None

    def confirmed(self, action_id: int) -> Motion | None:
        sample = self.samples.get(action_id)
        return sample[0] if sample and sample[1] >= 2 else None

    def dynamics(self) -> dict[str, int]:
        candidates: dict[str, list[int]] = {}
        for aid in self.samples:
            motion = self.confirmed(aid)
            if motion:
                candidates.setdefault(motion.direction, []).append(aid)
        return {direction: aids[0] for direction, aids in candidates.items() if len(aids) == 1}

    def observe(self, grid: np.ndarray, action: dict | None, levels: int = 0) -> None:
        arr = np.asarray(grid)
        old_key = self.state_key
        self.level_progress = levels > self.levels
        self.levels = levels
        comparable = self.previous is not None and self.previous.shape == arr.shape
        self.pixels_changed = int(np.count_nonzero(self.previous != arr)) if comparable else 0
        motion = self.translation(self.previous, arr) if comparable else None
        self.prediction_match = None
        if self.expected is not None:
            color, mask = self.expected
            self.prediction_match = mask.shape == arr.shape and np.array_equal(mask, arr == color)
        self.expected = None
        aid = action.get("action_id") if action else None
        if aid not in (None, 0):
            self.attempts[aid] += 1
        # Clicks, reset, and level transitions must never be learned as movement.
        if aid not in (None, 0, 6) and comparable and not self.level_progress:
            if motion:
                old = self.samples.get(aid)
                count = old[1] + 1 if old and old[0] == motion else 1
                self.samples[aid] = (motion, count)
            elif self.prediction_match is False:
                self.samples.pop(aid, None)
        self.meaningful_change = bool(self.level_progress or motion or self.pixels_changed > 2)
        # A tiny unexplained delta is uncertain (possibly HUD), not new evidence
        # that a previously tested click should be tried again.
        if not old_key or not comparable or self.meaningful_change:
            self.state_key = self.key(arr, levels)
        if aid == 6 and action.get("coordinates") and old_key:
            coords = action["coordinates"]
            trial = (6, int(coords["x"]), int(coords["y"]))
            outcome = (
                "level_progress" if self.level_progress else
                "changed" if self.meaningful_change else
                "uncertain" if self.pixels_changed else "no_effect"
            )
            self.trials.setdefault(old_key, {})[trial] = outcome
            self.trials.move_to_end(old_key)
            if len(self.trials) > 256:
                self.trials.popitem(last=False)
        self.visits[self.state_key] += 1
        self.cycle = self.visits[self.state_key] > 2
        if len(self.visits) > 256:
            self.visits = Counter({self.state_key: self.visits[self.state_key]})
        self.previous = arr.copy()

    def tried_clicks(self) -> set[tuple[int, int]]:
        return {
            (x, y) for (_, x, y), outcome in self.trials.get(self.state_key, {}).items()
            if outcome in ("no_effect", "uncertain") or self.cycle
        }

    def arm(self, grid: np.ndarray, action_id: int) -> bool:
        self.expected = None
        motion = self.confirmed(action_id)
        if motion is None:
            return False
        pts = np.argwhere(grid == motion.color)
        if not len(pts):
            return False
        dest = pts + (motion.dr, motion.dc)
        if (np.any(dest < 0) or np.any(dest[:, 0] >= grid.shape[0])
                or np.any(dest[:, 1] >= grid.shape[1])):
            return False
        mask = np.zeros(grid.shape, dtype=bool)
        mask[dest[:, 0], dest[:, 1]] = True
        self.expected = (motion.color, mask)
        return True

    def metrics(self) -> dict:
        return {
            "pixels_changed": self.pixels_changed,
            "meaningful_change": self.meaningful_change,
            "prediction_match": self.prediction_match,
            "level_progress": self.level_progress,
            "state_revisited": self.cycle,
            "confirmed_dynamics": self.dynamics(),
        }
