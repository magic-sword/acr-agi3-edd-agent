#!/usr/bin/env python3
"""Macro Skill Compiler - Core deterministic logic for procedural macro synthesis.

Compiles reusable sequences of actions (e.g. Aim -> Inspect -> Click, Cardinal Probing)
to enable fast-path execution without invoking LLM inference every turn.
"""

from typing import Any, Dict, List, Optional, Tuple


class MacroSkillCompilerCore:
    """Manages compilation, instantiation, and step-by-step execution of macro skills."""

    def __init__(self):
        self.macro_registry: Dict[str, Dict[str, Any]] = {}
        self.active_macro: Optional[Dict[str, Any]] = None
        self.active_queue: List[Dict[str, Any]] = []

        # Register standard built-in human play macros
        self._register_builtin_macros()

    def _register_builtin_macros(self) -> None:
        """Registers common procedural macros derived from human gameplay analysis."""
        self.register_macro(
            name="AIM_AND_CLICK",
            description="Aim reticle at target coords, inspect object, and execute click.",
            steps=[
                {"action_type": "MOVE_CURSOR", "action_id": 6, "requires_coords": True},
                {"action_type": "INSPECT", "action_id": 0, "requires_coords": False},
                {"action_type": "CLICK", "action_id": 6, "requires_coords": False},
            ],
        )
        self.register_macro(
            name="CARDINAL_PROBE",
            description="Probe all 4 cardinal directions (UP, DOWN, LEFT, RIGHT) sequentially.",
            steps=[
                {"action_type": "STEP", "action_id": 1, "action_name": "ACTION1"},
                {"action_type": "STEP", "action_id": 2, "action_name": "ACTION2"},
                {"action_type": "STEP", "action_id": 3, "action_name": "ACTION3"},
                {"action_type": "STEP", "action_id": 4, "action_name": "ACTION4"},
            ],
        )

    def register_macro(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Registers a new procedural macro skill."""
        if not name or not steps:
            return {"status": "error", "message": "Name and non-empty steps required."}

        self.macro_registry[name] = {
            "name": name,
            "description": description,
            "steps": steps,
            "step_count": len(steps),
        }
        return {"status": "ok", "name": name, "step_count": len(steps)}

    def instantiate_macro(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Instantiates a registered macro into the active execution queue."""
        if name not in self.macro_registry:
            return {"status": "error", "message": f"Macro '{name}' not found in registry."}

        macro_def = self.macro_registry[name]
        queue: List[Dict[str, Any]] = []
        params = parameters or {}

        for s in macro_def["steps"]:
            step_copy = dict(s)
            if step_copy.get("requires_coords") and "coords" in params:
                step_copy["coords"] = params["coords"]
            queue.append(step_copy)

        self.active_macro = {"name": name, "params": params, "total_steps": len(queue)}
        self.active_queue = queue
        return {
            "status": "ok",
            "active_macro": name,
            "queued_steps": len(queue),
        }

    def pop_next_step(self) -> Optional[Dict[str, Any]]:
        """Pops and returns the next executable step from the active macro queue."""
        if not self.active_queue:
            self.active_macro = None
            return None
        step = self.active_queue.pop(0)
        if not self.active_queue:
            self.active_macro = None
        return step

    def abort_macro(self, reason: str = "Interrupted") -> Dict[str, Any]:
        """Aborts the currently executing macro."""
        aborted_name = self.active_macro.get("name") if self.active_macro else "None"
        rem = len(self.active_queue)
        self.active_macro = None
        self.active_queue.clear()
        return {
            "status": "aborted",
            "aborted_macro": aborted_name,
            "remaining_steps_discarded": rem,
            "reason": reason,
        }

    def list_available_macros(self) -> List[Dict[str, Any]]:
        """Lists all registered macros."""
        return list(self.macro_registry.values())


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Macro Skill Compiler CLI Tool.")
    parser.add_argument("--list", action="store_true", help="List registered macros")
    parser.add_argument("--instantiate", type=str, default="", help="Instantiate macro by name")
    args = parser.parse_args()

    compiler = MacroSkillCompilerCore()
    if args.list:
        print(json.dumps(compiler.list_available_macros(), indent=2))
    elif args.instantiate:
        res = compiler.instantiate_macro(args.instantiate)
        print(json.dumps(res, indent=2))
    else:
        print(json.dumps({"macros": [m["name"] for m in compiler.list_available_macros()]}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
