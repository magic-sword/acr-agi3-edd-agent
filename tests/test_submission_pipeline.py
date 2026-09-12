"""Kaggle リーダーボード提出パイプラインの結合テスト."""

import json
from pathlib import Path

import numpy as np
import pytest

from acr_agi3.submission.bundle import create_submission_bundle
from acr_agi3.submission.entrypoint import (
    KaggleSubmissionPipeline,
    task_dict_to_env,
)
from acr_agi3.submission.path_resolver import ModelPathResolver


def test_model_path_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelPathResolver の探索・解決ロジックをテスト."""
    # 1. 指定パスが存在する場合
    fake_model_dir = tmp_path / "custom_model"
    fake_model_dir.mkdir()
    resolved = ModelPathResolver.resolve_model_path(fake_model_dir)
    assert resolved == fake_model_dir.resolve()

    # 2. 環境変数が設定されている場合
    env_model_dir = tmp_path / "env_model"
    env_model_dir.mkdir()
    monkeypatch.setenv("ARC_MODEL_PATH", str(env_model_dir))
    resolved_env = ModelPathResolver.resolve_model_path()
    assert resolved_env == env_model_dir.resolve()


def test_task_dict_to_env_explicit() -> None:
    """明示的パラメータを含むタスク辞書から環境が正しく復元されることをテスト."""
    task_data = {
        "grid_shape": [12, 12],
        "initial_player_pos": [2, 2],
        "goal_pos": [10, 10],
        "walls": [[5, 5], [5, 6]],
        "hazards": [[4, 4]],
    }
    env = task_dict_to_env(task_data)
    assert env.grid_shape == (12, 12)
    assert env.initial_player_pos == (2, 2)
    assert env.goal_pos == (10, 10)
    assert (5, 5) in env.walls
    assert (4, 4) in env.hazards


def test_task_dict_to_env_grid_inference() -> None:
    """グリッド行列（色情報）から環境アフォーダンスが自動抽出されることをテスト."""
    grid = np.zeros((8, 8), dtype=int)
    grid[1, 1] = 2  # Player
    grid[6, 6] = 3  # Goal
    grid[3, :] = 1  # Wall row
    grid[4, 4] = 4  # Hazard

    task_data = {
        "test": [{"input": grid.tolist()}],
    }
    env = task_dict_to_env(task_data)
    assert env.player_pos == (1, 1)
    assert env.goal_pos == (6, 6)
    assert (3, 2) in env.walls
    assert (4, 4) in env.hazards


def test_kaggle_submission_pipeline_execution(tmp_path: Path) -> None:
    """パイプラインがゲームをプレイし、有効な submission.json を出力することをテスト."""
    output_file = tmp_path / "output_submission.json"

    mock_challenges = {
        "task_open_01": {
            "grid_shape": [8, 8],
            "initial_player_pos": [1, 1],
            "goal_pos": [3, 1],
            "walls": [],
        },
        "task_simple_02": {
            "grid_shape": [8, 8],
            "initial_player_pos": [2, 2],
            "goal_pos": [2, 4],
            "walls": [],
        },
    }

    pipeline = KaggleSubmissionPipeline(max_steps_per_task=20)
    results = pipeline.run_on_challenges(
        challenges_source=mock_challenges,
        output_submission_path=output_file,
    )

    assert output_file.exists()
    assert len(results) == 2
    assert "task_open_01" in results
    assert "task_simple_02" in results

    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    for tid, entry in saved_data.items():
        assert "actions" in entry
        assert "status" in entry
        assert "steps" in entry
        assert isinstance(entry["actions"], list)
        for a in entry["actions"]:
            assert isinstance(a, int)
            assert 0 <= a <= 5


def test_create_submission_bundle(tmp_path: Path) -> None:
    """提出スクリプトのバンドル生成テスト."""
    bundle_file = tmp_path / "run_submission_bundle.py"
    res_path = create_submission_bundle(bundle_file)
    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "ARC-AGI-3 Kaggle Submission Runner" in content
    assert "run_submission" in content
