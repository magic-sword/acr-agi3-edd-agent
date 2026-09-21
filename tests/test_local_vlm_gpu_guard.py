import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from acr_agi3.agent.llm.local_vlm import LocalQwenVL


def test_cuda_not_available_raises_runtime_error():
    """torch.cuda が False のとき allow_cpu_fallback=False で RuntimeError を送出すること."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError) as excinfo:
            LocalQwenVL(
                model_name_or_path="models/Qwen2.5-VL-3B-Instruct",
                device="cuda",
                allow_cpu_fallback=False,
            )
        assert "GPU (CUDA) が指定されていますが" in str(excinfo.value)
        assert "CPU 推論は 30〜40 倍遅延し" in str(excinfo.value)


def test_mock_mode_bypasses_cuda_check():
    """model_name_or_path='mock' の場合は GPU チェックを行わずモック動作すること."""
    vlm = LocalQwenVL(model_name_or_path="mock")
    assert vlm.model == "mock"


def test_leaderboard_preflight_fails_when_cuda_missing():
    """CUDA が使用できない場合、--allow-cpu なしでは即時 sys.exit(1) すること."""
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_local_leaderboard

    with patch("torch.cuda.is_available", return_value=False):
        with patch.object(sys, "argv", ["run_local_leaderboard.py"]):
            with pytest.raises(SystemExit) as excinfo:
                run_local_leaderboard.main()
            assert excinfo.value.code == 1

