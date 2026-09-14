"""Contract tests for game-controller meta-skill (3 Positive + 3 Negative)."""

import sys
from pathlib import Path
import pytest

skill_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(skill_root / "scripts"))

from game_controller import GameController


@pytest.fixture
def controller():
    return GameController(available_actions=[1, 2, 3, 4, 6])


# ==========================================
# Positive Contract Tests (3件)
# ==========================================

def test_positive_directional_step_action(controller):
    """Positive 1: 有効な移動アクション (RIGHT) の正常解析と ActionID 検証."""
    res = controller.parse_and_validate("RIGHT", available_actions=[1, 2, 3, 4])
    assert res["success"] is True
    assert res["action_name"] == "ACTION4"
    assert res["action_id"] == 4
    assert res["action_type"] == "STEP"
    assert res["error"] is None


def test_positive_valid_click_coordinates(controller):
    """Positive 2: 盤面範囲内の有効な座標クリック (x=5, y=3) の解析."""
    res = controller.parse_and_validate(
        {"action": "click_at", "x": 5, "y": 3, "reasoning": "Click switch"},
        available_actions=[1, 2, 3, 4, 6],
        grid_shape=(10, 10),
    )
    assert res["success"] is True
    assert res["action_type"] == "CLICK"
    assert res["action_id"] == 6
    assert res["coordinates"] == {"x": 5, "y": 3}
    assert res["error"] is None


def test_positive_structured_json_parsing(controller):
    """Positive 3: JSON テキスト形式からのアクション抽出と検証."""
    json_text = '{"action": "step_action", "action_name": "UP", "reasoning": "Climb ladder"}'
    res = controller.parse_and_validate(json_text, available_actions=[1, 2, 3, 4])
    assert res["success"] is True
    assert res["action_name"] == "ACTION1"
    assert res["action_id"] == 1
    assert res["action_type"] == "STEP"


# ==========================================
# Negative Contract Tests (3件)
# ==========================================

def test_negative_out_of_bounds_click_rejected(controller):
    """Negative 1: 盤面サイズ (10x10) を超える座標 (x=15, y=20) のクリック拒絶."""
    res = controller.parse_and_validate(
        {"action": "click", "x": 15, "y": 20},
        available_actions=[6],
        grid_shape=(10, 10),
    )
    assert res["success"] is False
    assert res["error"] is not None
    assert "out of grid bounds" in res["error"]


def test_negative_unavailable_action_rejected(controller):
    """Negative 2: 現在の状態で許可されていないアクション (ACTION5) の拒絶."""
    res = controller.parse_and_validate("ACTION5", available_actions=[1, 2, 3, 4])
    assert res["success"] is False
    assert res["error"] is not None
    assert "not in available actions" in res["error"]


def test_negative_unparseable_garbage_input_rejected(controller):
    """Negative 3: 無効な文字列や未知のコマンドの拒絶."""
    res = controller.parse_and_validate("some random nonsense without action words", available_actions=[1, 2, 3, 4])
    assert res["success"] is False
    assert res["error"] is not None
    assert res["action_id"] == -1
