"""メタスキル (visual-inspector & game-controller) の単体・結合テスト."""

from pathlib import Path
import numpy as np

from acr_agi3.meta.skill_harness import SkillHarness

_harness = SkillHarness()
VisualInspector = _harness.get_skill_module("visual-inspector").VisualInspector
GameController = _harness.get_skill_module("game-controller").GameController


def test_meta_skills_spec_files_exist():
    """meta_skills/ 配下の全メタスキル仕様書 (visual-inspector, game-controller) の存在確認."""
    repo_root = Path(__file__).resolve().parent.parent
    meta_skills_dir = repo_root / "meta_skills"

    expected_meta_skills = [
        "visual-inspector",
        "game-controller",
    ]

    for ms_name in expected_meta_skills:
        skill_md = meta_skills_dir / ms_name / "SKILL.md"
        assert skill_md.exists(), f"Missing SKILL.md for {ms_name}"
        content = skill_md.read_text(encoding="utf-8")
        assert content.startswith("---"), f"YAML frontmatter missing in {ms_name}"
        assert f"name: {ms_name}" in content


def test_meta_skills_via_harness():
    """残された2大メタスキルが SkillHarness 経由で正しく動的解決できること."""
    harness = SkillHarness()

    # 1. visual-inspector
    vi_mod = harness.get_skill_module("visual-inspector")
    assert hasattr(vi_mod, "VisualInspector")
    vi = vi_mod.VisualInspector()
    res_vi = vi.inspect_board([[0, 1], [2, 0]], [1, 2], step_index=0)
    assert res_vi["success"] is True
    assert res_vi["grid_dimensions"] == [2, 2]

    # 2. game-controller
    gc_mod = harness.get_skill_module("game-controller")
    assert hasattr(gc_mod, "GameController")
    gc = gc_mod.GameController(available_actions=[1, 2, 3, 4], dynamics_map={"UP": 3})
    res_gc = gc.step_action("UP", reasoning="Testing harness resolution")
    assert res_gc["success"] is True
    assert res_gc["action_id"] == 3


def test_visual_inspector_game_frame_and_transition():
    """visual-inspector におけるフレーム観測と状態遷移の因果抽出テスト."""
    inspector = VisualInspector()

    grid = np.zeros((5, 5), dtype=int)
    grid[0, :] = 1
    grid[1, 1] = 2
    grid[4, 4] = 3
    grid[1, 3] = 4

    report = inspector.analyze_frame(grid, known_roles={"agent": 2, "goal": 3})
    assert report.grid_shape == (5, 5)
    assert report.background_color == 0
    assert report.player_pos == (1, 1)
    assert report.goal_pos == (4, 4)
