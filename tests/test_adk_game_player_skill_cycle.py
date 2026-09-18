"""ADKGamePlayer と OnlineSkillDeveloper の自己改善サイクル結合テスト.

プローブ実行 -> 差分解析 -> EDD契約テスト付きスキル合成 -> 高速マクロ実行 -> 停滞時解除
の一連のライフサイクルを検証する。
"""

import unittest
import numpy as np

from acr_agi3.agent.adk_game_player import ADKGamePlayer
from acr_agi3.harness.game_action_tools import ActionDecision


from unittest.mock import AsyncMock

class TestADKGamePlayerSkillCycle(unittest.TestCase):
    def setUp(self):
        # Google ADK Agent に適合するモックモデル名
        self.player = ADKGamePlayer(model="mock-model", name="test_agent")
        # LLMワークフローのフォールバック呼び出しをモック化
        fallback_decision = ActionDecision(
            action_type="STEP",
            action_name="UP",
            action_id=1,
            coordinates=None,
            reasoning="Fallback mock reasoning",
            loaded_skill="taboo-reset-guard",
        )
        self.player._run_plan_act_workflow = AsyncMock(return_value=fallback_decision)

    def test_epistemic_probe_then_synthesize_and_macro_execute(self):
        """プローブから同定、スキル合成、マクロ実行、停滞時解除までの全フロー検証."""
        # 1. 初期盤面 (自機: 1 (青), ゴール: 2 (赤), 背景: 0)
        grid1 = np.zeros((10, 10), dtype=np.uint8)
        grid1[5, 5] = 1
        grid1[2, 5] = 2

        # Step 1: 初期プローブアクションが発行されること
        d1 = self.player.decide_next_action(grid1, available_actions=[1, 2, 3, 4])
        self.assertTrue(d1.metadata.get("is_epistemic_probe") or d1.metadata.get("probe"))
        self.assertEqual(d1.loaded_skill, "epistemic-prober")

        # Step 2: プローブによって自機が上に移動したとする (5, 5 -> 4, 5)
        # （ACTION1 が UP に対応）
        grid2 = np.zeros((10, 10), dtype=np.uint8)
        grid2[4, 5] = 1
        grid2[2, 5] = 2

        # 直前アクションの結果を反映
        self.player.last_action_info["action"] = "ACTION1"
        self.player.last_action_info["action_id"] = 1

        d2 = self.player.decide_next_action(grid2, available_actions=[1, 2, 3, 4])
        developer = self.player.skill_developer

        self.assertEqual(developer.agent_color, 1)
        self.assertIn("UP", developer.action_semantics)

        # Step 3: さらに右プローブをシミュレート（ACTION4 が RIGHT に対応）
        if developer.is_probing:
            self.player.last_action_info["action"] = "ACTION4"
            self.player.last_action_info["action_id"] = 4
            grid3 = np.zeros((10, 10), dtype=np.uint8)
            grid3[4, 6] = 1
            grid3[2, 5] = 2
            d3 = self.player.decide_next_action(grid3, available_actions=[1, 2, 3, 4])

        # Step 4: さらに下プローブ（ACTION2 が DOWN に対応）
        if developer.is_probing:
            self.player.last_action_info["action"] = "ACTION2"
            self.player.last_action_info["action_id"] = 2
            grid4 = np.zeros((10, 10), dtype=np.uint8)
            grid4[5, 6] = 1
            grid4[2, 5] = 2
            d4 = self.player.decide_next_action(grid4, available_actions=[1, 2, 3, 4])

        # ルール同定完了後、スキルが合成されEDDゲートを通過していること
        self.assertTrue(developer.is_rule_identified)
        self.assertIsNotNone(developer.active_policy)
        self.assertIsNotNone(developer.active_skill_name)

        # Step 5: マクロポリシーによる高速実行 (LLMバイパス)
        grid_macro = np.zeros((10, 10), dtype=np.uint8)
        grid_macro[5, 5] = 1
        grid_macro[2, 5] = 2  # 上にターゲット
        d_macro = self.player.decide_next_action(grid_macro, available_actions=[1, 2, 3, 4])

        self.assertTrue(d_macro.metadata.get("is_macro_execution") or d_macro.metadata.get("macro_bypass"))
        # ACTION1 (UP) が選ばれること
        self.assertEqual(d_macro.action_id, 1)

        # Step 6: 停滞シミュレーション (壁に激突して2ステップ無変化)
        d_stag1 = self.player.decide_next_action(grid_macro, available_actions=[1, 2, 3, 4])
        self.assertEqual(self.player.stagnation_count, 1)

        # 2回目の無変化 -> 停滞カウント2でマクロポリシーが解除されること
        d_stag2 = self.player.decide_next_action(grid_macro, available_actions=[1, 2, 3, 4])
        self.assertGreaterEqual(self.player.stagnation_count, 2)
        self.assertIsNone(developer.active_policy)


if __name__ == "__main__":
    unittest.main()
