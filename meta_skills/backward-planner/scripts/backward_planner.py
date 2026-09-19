"""Backward Planner: Geometric pathfinding (A*) and precondition subgoal decomposition engine."""

from __future__ import annotations

import argparse
import heapq
import json
from typing import Any, Dict, List, Optional, Set, Tuple
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


class BackwardPlanner:
    """Computes deterministic grid paths and orders hierarchical causal subgoals."""

    @staticmethod
    def plan_path(
        grid: Any,
        start_pos: Tuple[int, int],  # (col, row)
        goal_pos: Tuple[int, int],   # (col, row)
        impassable_colors: Optional[List[int]] = None,
        max_nodes: int = 4000,
    ) -> Tuple[List[Tuple[int, int]], Optional[str]]:
        """A* shortest path on 2D grid avoiding impassable colors.
        
        Returns:
            path: List of (col, row) coordinates from start to goal.
            next_direction: One of 'UP', 'DOWN', 'LEFT', 'RIGHT' or None.
        """
        arr = _normalize_2d(grid)
        if arr.size == 0:
            return [], None

        h, w = arr.shape
        sc, sr = start_pos
        gc, gr = goal_pos

        # Bounds validation
        if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
            return [], None

        if (sc, sr) == (gc, gr):
            return [(sc, sr)], None

        impassable_set = set(impassable_colors if impassable_colors is not None else [1])

        # Priority queue for A*: (f_score, cost, (c, r), path)
        def h_cost(c: int, r: int) -> int:
            return abs(c - gc) + abs(r - gr)

        start_h = h_cost(sc, sr)
        pq: List[Tuple[int, int, Tuple[int, int], List[Tuple[int, int]]]] = [
            (start_h, 0, (sc, sr), [(sc, sr)])
        ]
        visited: Set[Tuple[int, int]] = {(sc, sr)}
        nodes_explored = 0

        while pq and nodes_explored < max_nodes:
            nodes_explored += 1
            f, cost, (curr_c, curr_r), path = heapq.heappop(pq)

            if (curr_c, curr_r) == (gc, gr):
                # Path found
                next_dir = None
                if len(path) >= 2:
                    next_c, next_r = path[1]
                    dc = next_c - curr_c if len(path) == 1 else next_c - sc
                    dr = next_r - curr_r if len(path) == 1 else next_r - sr
                    if dr < 0:
                        next_dir = "UP"
                    elif dr > 0:
                        next_dir = "DOWN"
                    elif dc < 0:
                        next_dir = "LEFT"
                    elif dc > 0:
                        next_dir = "RIGHT"
                return path, next_dir

            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nc, nr = curr_c + dc, curr_r + dr
                if 0 <= nr < h and 0 <= nc < w:
                    if (nc, nr) not in visited:
                        cell_color = int(arr[nr, nc])
                        # If cell is goal, ignore impassable check
                        if (nc, nr) == (gc, gr) or cell_color not in impassable_set:
                            visited.add((nc, nr))
                            new_cost = cost + 1
                            new_f = new_cost + h_cost(nc, nr)
                            heapq.heappush(pq, (new_f, new_cost, (nc, nr), path + [(nc, nr)]))

        # No path reachable
        return [], None

    @staticmethod
    def sequence_preconditions(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Topologically orders items: keys first, switches/doors second, goals third."""
        priority_order = {"key": 0, "item": 1, "switch": 2, "door": 3, "barrier": 4, "goal": 5}
        return sorted(items, key=lambda x: priority_order.get(x.get("type", "goal").lower(), 99))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backward Planner CLI for A* pathfinding and subgoal decomposition.")
    parser.add_argument("--grid", type=str, help="JSON string representing 2D grid array")
    parser.add_argument("--file", type=str, help="Path to JSON file containing grid array")
    parser.add_argument("--start", type=str, default="[0, 0]", help="Start coordinate [col, row]")
    parser.add_argument("--goal", type=str, default="[1, 1]", help="Goal coordinate [col, row]")
    parser.add_argument("--impassable", type=str, default="[1]", help="Impassable colors list JSON")
    args = parser.parse_args()

    grid = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            grid = data.get("grid", data)
    elif args.grid:
        grid = json.loads(args.grid)
    else:
        grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]

    start = json.loads(args.start)
    goal = json.loads(args.goal)
    impassable = json.loads(args.impassable)

    path, next_dir = BackwardPlanner.plan_path(
        grid,
        start_pos=(start[0], start[1]),
        goal_pos=(goal[0], goal[1]),
        impassable_colors=impassable,
    )

    result = {
        "start": start,
        "goal": goal,
        "path_length": len(path),
        "next_direction": next_dir,
        "path": path,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
