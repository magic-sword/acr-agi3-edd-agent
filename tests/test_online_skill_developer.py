"""OnlineSkillDeveloper の単体テスト (認識論的プローブ・コード合成・EDD契約テスト・マクロ実行)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from acr_agi3.agent.online_skill_developer import OnlineSkillDeveloper


class TestOnlineSkillDeveloper(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dev = OnlineSkillDeveloper(
            game_id="test_game_42",
            generated_skills_dir=self.temp_dir,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_epistemic_probing_cycle(self) -> None:
        # 1. プローブが必要と判定される
        self.assertTrue(self.dev.should_probe(step_index=1, available_actions=[1, 2, 3, 4]))

        # 2. プローブアクションの提案
        decision = self.dev.get_next_probe_action(available_actions=[1, 2, 3, 4])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.loaded_skill, "epistemic-prober")
        self.assertTrue(decision.metadata.get("is_epistemic_probe"))

        # 3. プローブ前後のグリッド差分解析 (ACTION1 で自機(2)が (3, 3) -> (2, 3) へ上移動したシミュレーション)
        g_before = np.zeros((8, 8), dtype=int)
        g_before[3, 3] = 2
        g_before[6, 6] = 3  # ターゲット

        g_after = np.zeros((8, 8), dtype=int)
        g_after[2, 3] = 2  # 上に1マス移動
        g_after[6, 6] = 3

        res = self.dev.analyze_probe_transition(g_before, g_after, action_id=1)
        self.assertTrue(res["identified"])
        self.assertEqual(res["direction"], "UP")
        self.assertEqual(self.dev.agent_color, 2)
        self.assertEqual(self.dev.action_semantics.get("UP"), 1)

    def test_edd_contract_gate_and_synthesis(self) -> None:
        # ナビゲーションコードの合成
        code = self.dev.synthesize_navigation_skill_code(
            agent_color=2,
            target_color=3,
            obstacle_colors=[1],
            action_map={"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4},
        )
        self.assertIn("choose_action", code)

        # EDD 防壁ゲート（正例3件＋負例3件）の検証
        passed, diag = self.dev.run_edd_contract_gate(code)
        self.assertTrue(passed)
        self.assertEqual(diag["positive_passed"], 3)
        self.assertEqual(diag["negative_passed"], 3)
        self.assertEqual(diag["total_passed"], 6)
        self.assertEqual(len(diag["failures"]), 0)

    def test_skill_development_and_macro_execution(self) -> None:
        grid = np.zeros((10, 10), dtype=int)
        grid[1, 1] = 2  # agent
        grid[5, 5] = 3  # target

        self.dev.agent_color = 2
        self.dev.action_semantics = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}

        # スキル自動開発と登録
        success = self.dev.develop_and_register_skill(grid, available_actions=[1, 2, 3, 4])
        self.assertTrue(success)
        self.assertIsNotNone(self.dev.active_policy_instance)
        self.assertTrue(self.dev.active_policy_dir.exists())
        self.assertTrue((self.dev.active_policy_dir / "SKILL.md").exists())
        self.assertTrue((self.dev.active_policy_dir / "scripts" / "policy.py").exists())

        # 高速マクロ実行テスト (LLMバイパス)
        decision = self.dev.execute_active_policy(grid, available_actions=[1, 2, 3, 4])
        self.assertIsNotNone(decision)
        self.assertIn(decision.action_id, [1, 2, 3, 4])
        self.assertTrue(decision.metadata.get("is_macro_execution"))
        self.assertEqual(decision.metadata.get("macro_step"), 1)


if __name__ == "__main__":
    unittest.main()
