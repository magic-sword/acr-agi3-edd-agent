"""Contract tests for macro-skill-compiler meta-skill (3 positive + 3 negative cases)."""

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from macro_skill_compiler import MacroSkillCompilerCore


# =============================================================================
# Positive Test Cases (正例 3 件)
# =============================================================================

def test_positive_case_1_register_and_instantiate_custom_macro():
    """Case 1: Registers a custom multi-step macro and instantiates with params."""
    core = MacroSkillCompilerCore()
    steps = [
        {"action_type": "STEP", "action_id": 1, "action_name": "UP"},
        {"action_type": "CLICK", "action_id": 6, "requires_coords": True},
    ]
    res = core.register_macro("STEP_AND_CLICK", "Move then click", steps)
    assert res["status"] == "ok"
    assert res["step_count"] == 2

    inst = core.instantiate_macro("STEP_AND_CLICK", {"coords": {"x": 4, "y": 8}})
    assert inst["status"] == "ok"
    assert inst["queued_steps"] == 2

    # Verify execution of 1st and 2nd step
    s1 = core.pop_next_step()
    assert s1["action_name"] == "UP"
    s2 = core.pop_next_step()
    assert s2["action_id"] == 6
    assert s2["coords"] == {"x": 4, "y": 8}


def test_positive_case_2_builtin_aim_and_click_macro():
    """Case 2: Instantiates and executes built-in AIM_AND_CLICK macro."""
    core = MacroSkillCompilerCore()
    inst = core.instantiate_macro("AIM_AND_CLICK", {"coords": {"x": 10, "y": 12}})
    assert inst["status"] == "ok"
    assert inst["queued_steps"] == 3

    step1 = core.pop_next_step()
    assert step1["action_type"] == "MOVE_CURSOR"
    assert step1["coords"] == {"x": 10, "y": 12}

    step2 = core.pop_next_step()
    assert step2["action_type"] == "INSPECT"

    step3 = core.pop_next_step()
    assert step3["action_type"] == "CLICK"

    # Queue should be empty now
    assert core.pop_next_step() is None


def test_positive_case_3_list_available_macros():
    """Case 3: Correctly lists all built-in and registered macros."""
    core = MacroSkillCompilerCore()
    macros = core.list_available_macros()
    names = [m["name"] for m in macros]
    assert "AIM_AND_CLICK" in names
    assert "CARDINAL_PROBE" in names
    assert len(names) >= 2


# =============================================================================
# Negative Test Cases (負例 3 件)
# =============================================================================

def test_negative_case_1_abort_active_macro():
    """Case 4: Aborts active macro and discards all remaining steps."""
    core = MacroSkillCompilerCore()
    core.instantiate_macro("CARDINAL_PROBE")
    assert len(core.active_queue) == 4

    step1 = core.pop_next_step()
    assert step1["action_name"] == "ACTION1"

    # Unexpected collision happens -> abort
    abort_res = core.abort_macro(reason="Hit unexpected wall")
    assert abort_res["status"] == "aborted"
    assert abort_res["remaining_steps_discarded"] == 3
    assert len(core.active_queue) == 0
    assert core.pop_next_step() is None


def test_negative_case_2_instantiate_nonexistent_macro():
    """Case 5: Returns error when attempting to instantiate an unknown macro."""
    core = MacroSkillCompilerCore()
    res = core.instantiate_macro("NONEXISTENT_MACRO_XYZ")
    assert res["status"] == "error"
    assert "not found" in res["message"]


def test_negative_case_3_register_empty_steps_error():
    """Case 6: Returns error when registering a macro with empty steps."""
    core = MacroSkillCompilerCore()
    res = core.register_macro("EMPTY", "No steps", [])
    assert res["status"] == "error"
    assert "required" in res["message"]
