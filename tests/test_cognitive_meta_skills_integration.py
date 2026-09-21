import json
import numpy as np
from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.llm.local_vlm import LocalQwenVL
from acr_agi3.harness.game_action_tools import ActionDecision


def test_adk_player_subgoal_and_rule_tools_integration():
    mock_vlm = LocalQwenVL(model_name_or_path="mock")
    player = ADKGamePlayer(model=mock_vlm, name="test_meta_player")
    player.switch_game(game_id="test_game_meta_integration")

    # 1. Subgoal Tools
    decompose_str = player.subgoal_tools.decompose_hierarchical_subgoals(
        entities=[
            {"id": "goal_1", "type": "goal", "x": 9, "y": 9},
            {"id": "door_1", "type": "door", "x": 5, "y": 5},
            {"id": "key_1", "type": "key", "x": 1, "y": 2},
        ]
    )
    decompose_res = json.loads(decompose_str)
    assert decompose_res["status"] == "ok"
    assert len(decompose_res["subgoals"]) == 3
    assert player.subgoal_tools.core.get_active_subgoal()["id"] == "subgoal_1"

    advance_str = player.subgoal_tools.advance_subgoal("subgoal_1")
    advance_res = json.loads(advance_str)
    assert advance_res["status"] == "ok"
    assert advance_res["next_subgoal"]["id"] == "subgoal_2"

    # 2. Rule Tools
    rule_str = player.rule_tools.induce_rule_from_transition(
        action_name="ACTION6",
        action_id=6,
        pixels_changed=5,
        coords={"x": 2, "y": 3},
        level_before=0,
        level_after=1,
        notes="Level advanced after clicking key"
    )
    rule_res = json.loads(rule_str)
    assert rule_res["status"] == "ok"
    assert rule_res["rule"]["rule_type"] == "win_condition"

    query_str = player.rule_tools.get_known_rules(rule_type="win_condition")
    query_res = json.loads(query_str)
    assert len(query_res) == 1
    assert query_res[0]["rule_id"] == "win_condition_level_0_to_1"


def test_adk_player_macro_fast_path_and_zero_pixel_abort():
    mock_vlm = LocalQwenVL(model_name_or_path="mock")
    player = ADKGamePlayer(model=mock_vlm, name="test_macro_player")
    player.switch_game(game_id="test_macro_fastpath")

    # Instantiate CARDINAL_PROBE macro: steps are ACTION1, ACTION2, ACTION3, ACTION4
    inst_str = player.macro_tools.instantiate_macro("CARDINAL_PROBE", {})
    inst_res = json.loads(inst_str)
    assert inst_res["status"] == "ok"
    assert player.macro_tools.has_active_macro() is True

    # Frame 1: Agent executes step 1 of macro (ACTION1)
    grid = np.zeros((10, 10), dtype=np.int32)
    available_actions = [1, 2, 3, 4, 5, 6]

    # Call decide_next_action with last_action_info and last_grid simulating pixel change
    player.last_action_info = {"name": "ACTION1", "action_id": 1}
    player.last_grid = np.zeros((10, 10), dtype=np.int32)
    # A macro needs a verified object-motion prediction, not arbitrary pixels.
    from acr_agi3.agent.execution_evidence import Motion
    player.last_grid[5, 3] = 2
    grid[5, 4] = 2
    player.execution_evidence.samples[1] = (Motion(2, 0, 1), 2)
    player.execution_evidence.observe(player.last_grid, None)
    player.execution_evidence.arm(player.last_grid, 1)
    decision = player.decide_next_action(grid, available_actions=available_actions)
    assert decision.action_name == "ACTION1"  # First step is ACTION1
    assert player.macro_tools.has_active_macro() is True

    # Frame 2: Suppose step resulted in 0 pixel change (collision/abort)
    # last_grid equals grid -> pixels_changed == 0
    player.last_grid = grid.copy()
    decision = player.decide_next_action(grid, available_actions=available_actions)
    # Stagnation / zero pixel change should abort the macro
    assert player.macro_tools.has_active_macro() is False


