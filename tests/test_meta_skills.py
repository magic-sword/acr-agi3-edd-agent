"""メタスキル (Meta-Observer & Subgoal-Decomposer) の単体・結合テスト."""

from pathlib import Path

import numpy as np

from acr_agi3.agent.human_vcgt import VCGTDataset
from acr_agi3.meta.skill_harness import SkillHarness

_harness = SkillHarness()
VisualInspector = _harness.get_skill_module("visual-inspector").VisualInspector
BackwardPlanner = _harness.get_skill_module("backward-planner").BackwardPlanner


def test_meta_skills_spec_files_exist():
    """meta_skills/ 配下の全メタスキル仕様書の存在確認 (6スキル直交体系)."""
    repo_root = Path(__file__).resolve().parent.parent
    meta_skills_dir = repo_root / "meta_skills"

    expected_meta_skills = [
        "visual-inspector",
        "epistemic-prober",
        "backward-planner",
        "taboo-reset-guard",
        "macro-skill-compiler",
        "contract-tester",
    ]

    for ms_name in expected_meta_skills:
        skill_md = meta_skills_dir / ms_name / "SKILL.md"
        assert skill_md.exists(), f"Missing SKILL.md for {ms_name}"
        content = skill_md.read_text(encoding="utf-8")
        assert content.startswith("---"), f"YAML frontmatter missing in {ms_name}"
        assert f"name: {ms_name}" in content


def test_human_cognitive_meta_skills_via_harness():
    """新設された5大人間問題解決メタスキルがSkillHarness経由で正しく動的解決できること."""
    harness = SkillHarness()

    # 1. visual-inspector
    vi_mod = harness.get_skill_module("visual-inspector")
    assert hasattr(vi_mod, "VisualInspector")
    vi = vi_mod.VisualInspector()
    res_vi = vi.inspect_board([[0, 1], [2, 0]], [1, 2], step_index=0)
    assert res_vi["success"] is True
    assert res_vi["pause_required"] is True

    # 2. epistemic-prober
    ep_mod = harness.get_skill_module("epistemic-prober")
    assert hasattr(ep_mod, "EpistemicProber")
    ep = ep_mod.EpistemicProber()
    res_ep = ep.evaluate_and_propose_probe(step_index=1, available_actions=[1, 2], unexplored_affordances={"has_actuator_rail": True})
    assert res_ep["success"] is True
    assert res_ep["is_epistemic"] is True

    # 3. backward-planner
    bp_mod = harness.get_skill_module("backward-planner")
    assert hasattr(bp_mod, "BackwardPlanner")
    bp = bp_mod.BackwardPlanner()
    res_bp = bp.plan_backward_subgoals([3, 2, 1], [1, 2, 3])
    assert res_bp["success"] is True
    assert res_bp["backward_chaining_applied"] is True

    # 4. taboo-reset-guard
    tr_mod = harness.get_skill_module("taboo-reset-guard")
    assert hasattr(tr_mod, "TabooResetGuard")
    tr = tr_mod.TabooResetGuard()
    res_tr = tr.evaluate_state_and_failure([], consecutive_ineffective_actions=5)
    assert res_tr["success"] is True
    assert res_tr["is_reset_recommended"] is True

    # 5. macro-skill-compiler
    mc_mod = harness.get_skill_module("macro-skill-compiler")
    assert hasattr(mc_mod, "MacroSkillCompiler")
    mc = mc_mod.MacroSkillCompiler()
    res_mc = mc.compile_macro_skill({"phase": "ISOLATE_AND_STAGE"})
    assert res_mc["success"] is True



def test_human_vcgt_loader_and_plan_conversion():
    """人間プレイ思考データ (VCGT) の読み込みと DecompositionPlan 変換テスト."""
    repo_root = Path(__file__).resolve().parent.parent
    sample_path = repo_root / "data" / "human_vcgt" / "sample_vcgt.json"
    assert sample_path.exists(), "Sample VCGT JSON dataset must exist"

    dataset = VCGTDataset.load_from_json(sample_path)
    assert len(dataset) >= 2

    # レコード検証
    rec = dataset.records[0]
    assert rec.task_id == "vcgt_game_001"
    assert len(rec.human_vcgt.steps) == 3
    assert len(rec.invariants_identified) == 3

    # DecompositionPlan への変換テスト
    plan = rec.to_subgoal_plan()
    assert plan.task_hint == rec.human_vcgt.goal
    assert plan.total_steps == 3
    assert plan.subgoals[0].objective == "Move right along row 1 until column 6"
    assert len(plan.constraints) == 3

    # Few-shot プロンプト構築テスト
    few_shot_prompt = dataset.build_few_shot_prompt(max_examples=1)
    assert "Example Task [vcgt_game_001]" in few_shot_prompt
    assert "Human Visual Concept-Guided Thinking (VCGT)" in few_shot_prompt


def test_meta_skill_driven_agent_solve_game():
    """MetaSkillDrivenAgent によるゲーム環境のメタ認知分解と行動ポリシー合成・解決テスト."""
    from acr_agi3.agent.llm.local_model import LocalTransformersLlm
    from acr_agi3.agent.meta_agent import MetaSkillDrivenAgent
    from acr_agi3.game.vcgt_game import GridWorldGameEnv

    def mock_policy_fn(prompt: str) -> str:
        return (
            "```python\n"
            "from acr_agi3.game.env import Action\n"
            "def choose_action(obs, info=None):\n"
            "    return Action.RIGHT\n"
            "```"
        )

    mock_llm = LocalTransformersLlm(model_name_or_path="mock", generation_fn=mock_policy_fn)
    agent = MetaSkillDrivenAgent(model=mock_llm)
    env = GridWorldGameEnv(grid_shape=(3, 3), initial_player_pos=(1, 0), goal_pos=(1, 1))
    res = agent.solve_game(env)

    assert res["is_solved"] is True
    assert res["policy_code"] is not None
    assert res["steps_taken"] > 0


def test_visual_inspector_game_frame_and_transition():
    """ACR-AGI-3 ゲーム環境におけるフレーム観測と状態遷移の因果抽出テスト."""
    from acr_agi3.game.env import Action

    inspector = VisualInspector()

    # 5x5 ゲーム環境グリッド (0: 背景, 1: 壁, 2: プレイヤー, 3: ゴール, 4: 鍵)
    grid = np.zeros((5, 5), dtype=int)
    grid[0, :] = 1  # 上壁
    grid[1, 1] = 2  # プレイヤー (1, 1)
    grid[4, 4] = 3  # ゴール (4, 4)
    grid[1, 3] = 4  # 鍵 (1, 3)

    report = inspector.analyze_frame(grid, known_roles={"agent": 2, "goal": 3})
    assert report.grid_shape == (5, 5)
    assert report.background_color == 0
    assert report.player_pos == (1, 1)
    assert report.goal_pos == (4, 4)
    assert (0, 1) in report.obstacles
    assert "item_color_4" in report.interactables
    assert report.interactables["item_color_4"] == (1, 3)

    # 状態遷移: RIGHT に移動した場合 (1, 1) -> (1, 2)
    next_grid = np.copy(grid)
    next_grid[1, 1] = 0
    next_grid[1, 2] = 2

    trans_success = inspector.analyze_transition(
        obs_before=grid,
        action=Action.RIGHT,
        obs_after=next_grid,
        reward=0.0,
        done=False,
    )
    assert trans_success["moved"] is True
    assert trans_success["displacement"] == (0, 1)
    assert trans_success["hit_obstacle"] is False

    # 状態遷移: UP に移動して壁に衝突した場合 (位置不変)
    trans_hit = inspector.analyze_transition(
        obs_before=grid,
        action=Action.UP,
        obs_after=grid,
        reward=-0.1,
        done=False,
    )
    assert trans_hit["moved"] is False
    assert trans_hit["hit_obstacle"] is True


def test_backward_planner_game_milestones():
    """ゲーム環境に対する BackwardPlanner の自律中間マイルストーン策定テスト."""
    planner = BackwardPlanner()

    # 障害物壁と鍵が存在するマップ
    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1] = 2  # プレイヤー
    grid[8, 8] = 3  # ゴール
    grid[1, 7] = 4  # 鍵
    grid[3:7, 4] = 1  # 中央の縦壁

    plan = planner.decompose_game(grid, known_roles={"agent": 2, "goal": 3})

    assert plan.total_steps >= 3
    step_names = [s.name for s in plan.subgoals]
    assert "Acquire_item_color_4" in step_names
    assert "BypassCentralObstacle" in step_names
    assert "ReachGoalAndClearStage" in step_names

    # 制約に壁回避が含まれているか
    assert any("Avoid impassable obstacle walls" in c for c in plan.constraints)


def test_vcgt_game_002_record_loading():
    """更新された sample_vcgt.json の vcgt_game_002 の読み込みとプラン整合性テスト."""
    repo_root = Path(__file__).resolve().parent.parent
    sample_path = repo_root / "data" / "human_vcgt" / "sample_vcgt.json"

    dataset = VCGTDataset.load_from_json(sample_path)
    game_002_recs = dataset.get_task("vcgt_game_002")
    assert len(game_002_recs) == 1

    rec = game_002_recs[0]
    assert rec.environment == "interactive_switch_and_key_navigation"
    assert "Collect key" in rec.human_vcgt.goal
    assert len(rec.human_vcgt.steps) == 3
    assert any(
        "Door (5) is impassable until player touches Key (4)" in inv
        for inv in rec.invariants_identified
    )

    plan = rec.to_subgoal_plan()
    assert plan.total_steps == 3
    assert "Move east" in plan.subgoals[0].objective
