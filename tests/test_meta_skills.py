"""メタスキル (Meta-Observer & Subgoal-Decomposer) の単体・結合テスト."""

from pathlib import Path

import numpy as np

from acr_agi3.meta.decomposer import SubgoalDecomposer
from acr_agi3.meta.human_vcgt import VCGTDataset
from acr_agi3.meta.observer import MetaObserver


def test_meta_observer_pair_analysis():
    """MetaObserver による不変量とアフォーダンス分析テスト."""
    observer = MetaObserver()

    # 入力: 3x3 (背景: 0, 物体: 1)
    inp = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ])
    # 出力: 6x6 (2倍拡大, 新色 2 が登場)
    out = np.zeros((6, 6), dtype=int)
    out[:3, :3] = inp
    out[3:, 3:] = 2

    report = observer.analyze_pair(inp, out)

    assert report.in_shape == (3, 3)
    assert report.out_shape == (6, 6)
    assert report.shape_ratio == (2.0, 2.0)
    assert report.background_color == 0
    assert 2 in report.new_colors
    assert report.transformation_hint == "scaling_or_tiling"
    assert len(report.objects) == 1
    assert report.objects[0].color == 1
    assert report.objects[0].size == 3


def test_meta_observer_task_summary():
    """タスク全体の全 Train ペア集約テスト."""
    observer = MetaObserver()
    train_pairs = [
        {"input": np.zeros((3, 3)), "output": np.zeros((6, 6))},
        {"input": np.zeros((4, 4)), "output": np.zeros((8, 8))},
    ]

    summary = observer.analyze_task(train_pairs)
    assert summary["consistent_shape_ratio"] == (2.0, 2.0)
    assert summary["consistent_background"] == 0
    assert summary["num_train_pairs"] == 2


def test_subgoal_decomposer_vcgt_structure():
    """SubgoalDecomposer による VCGT 構造の中間マイルストーン分解テスト."""
    decomposer = SubgoalDecomposer()

    # 形状が拡大し、新色が出るタスク
    inp = np.array([[1, 0], [0, 1]])
    out = np.array([[1, 0, 2], [0, 1, 2], [2, 2, 2]])

    plan = decomposer.decompose(inp, out)

    assert plan.total_steps >= 2
    step_names = [s.name for s in plan.subgoals]
    assert "AdjustGridDimensions" in step_names
    assert "IntroduceNewColors" in step_names

    # 各サブゴールに VCGT 思考理由 (reasoning) が含まれていることを確認
    for subgoal in plan.subgoals:
        assert len(subgoal.objective) > 0
        assert len(subgoal.reasoning) > 0
        assert len(subgoal.expected_operation) > 0

    plan_dict = plan.to_dict()
    assert "subgoals" in plan_dict
    assert len(plan_dict["subgoals"]) == plan.total_steps


def test_meta_skills_spec_files_exist():
    """meta_skills/ 配下の全メタスキル仕様書の存在確認."""
    repo_root = Path(__file__).resolve().parent.parent
    meta_skills_dir = repo_root / "meta_skills"

    expected_meta_skills = [
        "env-observer",
        "skill-synthesizer",
        "contract-tester",
        "failure-diagnoser",
        "subgoal-decomposer",
        "constraint-learner",
    ]

    for ms_name in expected_meta_skills:
        skill_md = meta_skills_dir / ms_name / "SKILL.md"
        assert skill_md.exists(), f"Missing SKILL.md for {ms_name}"
        content = skill_md.read_text(encoding="utf-8")
        assert content.startswith("---"), f"YAML frontmatter missing in {ms_name}"
        assert f"name: {ms_name}" in content


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

