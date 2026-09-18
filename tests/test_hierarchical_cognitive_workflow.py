"""階層的逆算プランニング & 自律仮説検証スキル合成ワークフロー結合テスト.

人間行動解析レポートに基づく：
1. 大目標逆算 & サブゴール分解 (Backward Chaining)
2. 不確実性・知識欠如時のオンデマンド仮説検証 (Epistemic Probe)
3. 実証法則の定石スキル合成 & EDD防壁ゲート (Contract Tester)
4. 超高速マクロ実行 (Macro Execution) & 停滞時禁忌学習 (Taboo Reset)
のライフサイクルを検証する。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.agent.cognitive_workflow import CognitiveState, assess_cognitive_state
from acr_agi3.agent.workflow_schemas import PlanProposal
from acr_agi3.harness.game_action_tools import ActionDecision


class TestHierarchicalCognitiveWorkflow(unittest.TestCase):
    def setUp(self):
        self.player = ADKGamePlayer(model="mock-model", name="test_hierarchical_player")
        # LLM ワークフローのモック
        self.mock_proposal = PlanProposal(
            hypothesis="Red block is keystone, blue block is obstacle",
            goal="Align blocks R-O-B-G horizontally",
            subgoal="isolate_blue_obstacle_to_bottom_buffer",
            cognitive_intent="PROBE",
            method_known=False,
            action="ACTION1",
            reasoning="Testing if ACTION1 can displace blue obstacle downwards",
            load_skill="backward-planner",
        )
        fallback_decision = ActionDecision(
            action_type="STEP",
            action_name="ACTION1",
            action_id=1,
            coordinates=None,
            reasoning="Mocked backward plan execution",
            loaded_skill="backward-planner",
        )
        self.player._run_plan_act_workflow = AsyncMock(return_value=fallback_decision)

    def test_plan_proposal_parsing_with_subgoal_and_intent(self):
        """Planner の JSON 出力から subgoal と cognitive_intent が正しく抽出されること."""
        llm_json = """
        ```json
        {
          "hypothesis": "Orange key opens green door. Player is red square.",
          "goal": "Reach the green exit door",
          "subgoal": "collect_orange_key_before_door",
          "cognitive_intent": "PROBE",
          "need_probe": true,
          "action": "ACTION1",
          "coordinates": null,
          "reasoning": "Need to test which action moves player toward the orange key"
        }
        ```
        """
        proposal = PlanProposal.from_text(llm_json)
        self.assertEqual(proposal.subgoal, "collect_orange_key_before_door")
        self.assertEqual(proposal.cognitive_intent, "PROBE")
        self.assertFalse(proposal.method_known)
        self.assertEqual(proposal.action, "ACTION1")

    def test_cognitive_state_routing_with_subgoal_uncertainty(self):
        """サブゴール達成方法が未知の際、route_epistemic_probe が選ばれること."""
        state = CognitiveState(
            step=2,
            stagnation_count=0,
            current_subgoal="move_obstacle_to_buffer",
            cognitive_intent="PROBE",
            method_known=False,
        )
        ctx = MagicMock()
        assess_cognitive_state(ctx, state)
        self.assertEqual(ctx.route, "route_epistemic_probe")
        self.assertEqual(state.selected_skill, "epistemic-prober")

    def test_full_hierarchical_cycle_probe_synthesis_macro_taboo(self):
        """未知サブゴール ➔ プローブ実験 ➔ EDD契約テスト合格 ➔ マクロ実行 ➔ 停滞時解除 の全結合サイクル."""
        # 1. 初期盤面 (自機: 1, ターゲット/サブゴール対象: 2)
        grid1 = np.zeros((10, 10), dtype=np.uint8)
        grid1[5, 5] = 1
        grid1[2, 5] = 2

        # Step 1: プローブ実行
        d1 = self.player.decide_next_action(grid1, available_actions=[1, 2, 3, 4])
        self.assertEqual(d1.loaded_skill, "epistemic-prober")

        # Step 2: 上移動が観測されたとする (ACTION1 -> UP)
        grid2 = np.zeros((10, 10), dtype=np.uint8)
        grid2[4, 5] = 1
        grid2[2, 5] = 2
        self.player.last_action_info["action"] = "ACTION1"
        self.player.last_action_info["action_id"] = 1
        d2 = self.player.decide_next_action(grid2, available_actions=[1, 2, 3, 4])

        developer = self.player.skill_developer
        self.assertEqual(developer.agent_color, 1)
        self.assertIn("UP", developer.action_semantics)

        # Step 3: 右移動が観測されたとする (ACTION4 -> RIGHT)
        if developer.is_probing:
            grid3 = np.zeros((10, 10), dtype=np.uint8)
            grid3[4, 6] = 1
            grid3[2, 5] = 2
            self.player.last_action_info["action"] = "ACTION4"
            self.player.last_action_info["action_id"] = 4
            d3 = self.player.decide_next_action(grid3, available_actions=[1, 2, 3, 4])

        # Step 4: 対向移動が同定され、EDD防壁ゲート付きでスキルが自動生成されること
        self.assertTrue(developer.is_rule_identified)
        self.assertIsNotNone(developer.active_policy)
        self.assertTrue(developer.active_skill_name.startswith("skill-"))

        # Step 5: マクロ実行 (LLMバイパス)
        grid_macro = np.zeros((10, 10), dtype=np.uint8)
        grid_macro[4, 5] = 1
        grid_macro[2, 5] = 2  # 上にターゲット
        d_macro = self.player.decide_next_action(grid_macro, available_actions=[1, 2, 3, 4])
        self.assertEqual(d_macro.action_id, 1)  # UP
        self.assertTrue(d_macro.metadata.get("is_macro_execution") or d_macro.metadata.get("macro_bypass"))

        # Step 6: 停滞シミュレーション (連続無変化)
        d_stag1 = self.player.decide_next_action(grid_macro, available_actions=[1, 2, 3, 4])
        self.assertEqual(self.player.stagnation_count, 1)

        d_stag2 = self.player.decide_next_action(grid_macro, available_actions=[1, 2, 3, 4])
        self.assertGreaterEqual(self.player.stagnation_count, 2)
        # 停滞によりマクロが解除され、再合成が抑制されていること
        self.assertIsNone(developer.active_policy)
        self.assertTrue(developer.policy_suppressed)


if __name__ == "__main__":
    unittest.main()
