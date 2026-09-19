# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

# src ディレクトリを sys.path に追加 (ローカル・CI双方の互換性確保)
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

root_dir = Path(__file__).resolve().parent.parent
for scripts_dir in (root_dir / "meta_skills").glob("*/scripts"):
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

from acr_agi3.game.vcgt_game import GridWorldGameEnv


@pytest.fixture
def sample_game_env() -> GridWorldGameEnv:
    """ACR-AGI-3 簡易グリッドゲーム環境フィクスチャ."""
    return GridWorldGameEnv(
        grid_shape=(5, 5),
        initial_player_pos=(1, 1),
        goal_pos=(1, 3),
        walls=set(),
    )
