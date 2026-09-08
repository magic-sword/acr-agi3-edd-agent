"""オフライン提出用の単一スクリプト・バンドル生成ツール."""

from pathlib import Path


def create_submission_bundle(output_path: Path) -> Path:
    """スタンドアロンで動作する提出用スクリプトを生成する."""
    bundle_code = '''# ARC-AGI-3 Kaggle Submission Bundle (Auto-generated)
import sys
import json
import numpy as np

def solve_task(task):
    """単一タスクに対するベースライン推論."""
    predictions = []
    for test_case in task.get("test", []):
        inp = np.array(test_case["input"])
        # フォールバック: 恒等変換
        predictions.append([inp.tolist(), inp.tolist()])
    return predictions

def main():
    challenges_path = "arc-agi_test_challenges.json"
    with open(challenges_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sub = {}
    for task_id, task in data.items():
        preds = solve_task(task)
        sub[task_id] = [{"attempt_1": p[0], "attempt_2": p[1]} for p in preds]

    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump(sub, f)

if __name__ == "__main__":
    main()
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bundle_code)
    return output_path
