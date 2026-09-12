"""オフライン提出用の単一スクリプト・バンドル生成ツール."""

from pathlib import Path


def create_submission_bundle(output_path: Path) -> Path:
    """Kaggle リーダーボード提出用の自律ゲームプレイ推論スクリプトを生成する."""
    bundle_code = '''# ARC-AGI-3 Kaggle Submission Runner (Auto-generated)
import os
import sys
import json
from pathlib import Path

# /kaggle/working またはローカルパスを探索
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from acr_agi3.submission.entrypoint import run_submission
    from acr_agi3.submission.path_resolver import ModelPathResolver
except ImportError:
    print("⚠️ acr_agi3 not found in PYTHONPATH, searching /kaggle/working/src...")
    sys.path.insert(0, "/kaggle/working/src")
    from acr_agi3.submission.entrypoint import run_submission
    from acr_agi3.submission.path_resolver import ModelPathResolver

def main():
    print("🚀 Initializing ACR-AGI-3 Kaggle Submission Runner...")
    data_dir = ModelPathResolver.resolve_data_dir()
    challenges_file = data_dir / "arc-agi_test_challenges.json"

    if not challenges_file.exists():
        # 代替パス候補
        candidates = [
            Path("/kaggle/input/arc-prize-2026-arc-agi-3/arc-agi_test_challenges.json"),
            Path("data/arc-agi_test_challenges.json"),
            Path("data/test_challenges.json"),
        ]
        for c in candidates:
            if c.exists():
                challenges_file = c
                break

    output_sub = Path("submission.json")
    print(f"📂 Reading challenges from: {challenges_file}")
    print(f"📝 Output target: {output_sub.resolve()}")

    if challenges_file.exists():
        run_submission(
            test_challenges_path=challenges_file,
            output_submission_path=output_sub,
            max_steps_per_task=50,
        )
    else:
        print("⚠️ Warning: No challenge file found! Emitting empty submission.")
        with open(output_sub, "w", encoding="utf-8") as f:
            json.dump({}, f)

    print("🎉 Submission process finished successfully!")

if __name__ == "__main__":
    main()
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bundle_code)
    return output_path
