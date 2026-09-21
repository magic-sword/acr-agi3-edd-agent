#!/usr/bin/env python3
"""Subgoal Decomposer - Core deterministic logic for hierarchical backward planning.

Decomposes complex ARC-AGI-3 task targets into topologically ordered subgoals,
resolving precondition dependencies (keys before doors) and allocating temporary
staging buffers for disordered objects.
"""

from typing import Any, Dict, List, Optional, Tuple


class SubgoalDecomposerCore:
    """Manages hierarchical backward subgoal decomposition and buffer allocation."""

    def __init__(self):
        self.subgoals: List[Dict[str, Any]] = []
        self.active_index: int = 0
        self.staging_buffers: List[Dict[str, int]] = []

    def decompose_hierarchical_subgoals(
        self,
        entities: List[Dict[str, Any]],
        target_pattern: Optional[List[Dict[str, Any]]] = None,
        grid_dims: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Decomposes target objectives into ordered subgoals based on backward chaining.

        Priority Ordering (Topological Dependency):
        0: Displace/Stage blocking objects into buffer (Detour/Staging)
        1: Collect keys / interact with switches (Preconditions)
        2: Pass through or open doors/barriers (Access)
        3: Assemble keystone / anchor items (Keystone Anchor)
        4: Reach final target or align goal objects (Terminal Goal)
        """
        if not entities:
            return {"status": "error", "message": "Entities list cannot be empty."}

        generated_subgoals: List[Dict[str, Any]] = []
        subgoal_id = 1

        # 1. Check for keys and switches (Preconditions)
        keys_and_switches = [e for e in entities if e.get("type") in ("key", "switch", "button")]
        doors_and_barriers = [e for e in entities if e.get("type") in ("door", "barrier", "laser")]
        movable_items = [e for e in entities if e.get("type") in ("box", "block", "item", "piece")]
        targets = [e for e in entities if e.get("type") in ("goal", "target", "exit")]

        # Circular dependency check
        for e in entities:
            if e.get("requires") == e.get("id") and e.get("id"):
                return {"status": "error", "message": f"Circular dependency detected in entity {e.get('id')}."}

        # Step A: Identify need for temporary buffer staging if multiple movable blocks block each other
        if len(movable_items) > 1 and target_pattern:
            for item in movable_items:
                target_pos = next((t for t in target_pattern if t.get("color") == item.get("color")), None)
                if target_pos and (item.get("x") != target_pos.get("x") or item.get("y") != target_pos.get("y")):
                    # Needs staging
                    buf_coords = {"x": item.get("x", 0) + 1, "y": item.get("y", 0)}
                    generated_subgoals.append({
                        "id": f"subgoal_{subgoal_id}",
                        "type": "stage_buffer",
                        "title": f"Stage piece {item.get('id', item.get('color'))} into temporary buffer",
                        "target_entity": item.get("id"),
                        "staging_coords": buf_coords,
                        "priority": 0,
                        "completed": False,
                    })
                    subgoal_id += 1
                    break  # Single buffer staging per plan

        # Step B: Keys and Switches (Preconditions)
        for ks in keys_and_switches:
            generated_subgoals.append({
                "id": f"subgoal_{subgoal_id}",
                "type": "precondition_interaction",
                "title": f"Activate/Acquire {ks.get('type')} at pos=({ks.get('x')},{ks.get('y')})",
                "target_entity": ks.get("id", ks.get("type")),
                "coords": {"x": ks.get("x", 0), "y": ks.get("y", 0)},
                "priority": 1,
                "completed": False,
            })
            subgoal_id += 1

        # Step C: Doors and Barriers (Access)
        for db in doors_and_barriers:
            generated_subgoals.append({
                "id": f"subgoal_{subgoal_id}",
                "type": "barrier_clearance",
                "title": f"Unlock/Traverse {db.get('type')} at pos=({db.get('x')},{db.get('y')})",
                "target_entity": db.get("id", db.get("type")),
                "coords": {"x": db.get("x", 0), "y": db.get("y", 0)},
                "priority": 2,
                "completed": False,
            })
            subgoal_id += 1

        # Step D: Movable Pieces to Target Slots
        for item in movable_items:
            generated_subgoals.append({
                "id": f"subgoal_{subgoal_id}",
                "type": "item_positioning",
                "title": f"Position movable item {item.get('id', 'item')} to target location",
                "target_entity": item.get("id", "item"),
                "coords": {"x": item.get("x", 0), "y": item.get("y", 0)},
                "priority": 3,
                "completed": False,
            })
            subgoal_id += 1

        # Step E: Terminal Target
        for t in targets:
            generated_subgoals.append({
                "id": f"subgoal_{subgoal_id}",
                "type": "reach_goal",
                "title": f"Reach terminal goal at pos=({t.get('x')},{t.get('y')})",
                "target_entity": t.get("id", "goal"),
                "coords": {"x": t.get("x", 0), "y": t.get("y", 0)},
                "priority": 4,
                "completed": False,
            })
            subgoal_id += 1

        # Default fallback if no specific targets
        if not generated_subgoals:
            generated_subgoals.append({
                "id": f"subgoal_{subgoal_id}",
                "type": "general_exploration",
                "title": "Explore available affordances and boundaries",
                "target_entity": "board",
                "priority": 5,
                "completed": False,
            })

        self.subgoals = generated_subgoals
        self.active_index = 0
        return {
            "status": "ok",
            "subgoal_count": len(self.subgoals),
            "subgoals": self.subgoals,
            "active_subgoal": self.get_active_subgoal(),
        }

    def get_active_subgoal(self) -> Optional[Dict[str, Any]]:
        """Returns the current incomplete subgoal."""
        if 0 <= self.active_index < len(self.subgoals):
            return self.subgoals[self.active_index]
        return None

    def advance_subgoal(self, subgoal_id: Optional[str] = None) -> Dict[str, Any]:
        """Marks the current active subgoal as completed and advances."""
        if self.active_index >= len(self.subgoals):
            return {"status": "exhausted", "message": "All subgoals already completed"}

        current = self.subgoals[self.active_index]
        if subgoal_id and current["id"] != subgoal_id:
            return {"status": "error", "message": f"Expected active subgoal {current['id']}, got {subgoal_id}"}

        current["completed"] = True
        self.active_index += 1
        next_subgoal = self.get_active_subgoal()
        return {
            "status": "ok",
            "completed_subgoal": current,
            "next_subgoal": next_subgoal,
            "is_all_completed": (next_subgoal is None),
        }

    def reset(self) -> None:
        """Resets all subgoals."""
        self.subgoals.clear()
        self.active_index = 0
        self.staging_buffers.clear()


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Subgoal Decomposer CLI Tool.")
    parser.add_argument("--entities", type=str, default="[]", help="JSON string of detected entities")
    parser.add_argument("--advance", action="store_true", help="Advance active subgoal")
    args = parser.parse_args()

    core = SubgoalDecomposerCore()
    try:
        entities = json.loads(args.entities)
        if entities:
            res = core.decompose_hierarchical_subgoals(entities)
            print(json.dumps(res))
        else:
            print(json.dumps({"subgoals": [], "active_subgoal": None}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
