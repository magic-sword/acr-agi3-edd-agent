#!/usr/bin/env python3
"""ARC-AGI 公式タスクデータセット自動取得スクリプト.

GitHub の fchollet/ARC-AGI から公式の training / evaluation タスクを取得し、
data/raw/ ディレクトリに配置・結合します。
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARC_GITHUB_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
SAMPLE_TASK_IDS = [
    "007bbfb7",  # 基本的なタイリング/反復パターン
    "00d62c1b",  # 閉領域塗りつぶし (Flood Fill)
    "017c7c7b",  # 幾何形状拡大 (Scaling)
    "025d127b",  # 移動・整列 (Movement)
    "045e512c",  # 色置換・条件抽出
]


def download_sample_tasks(raw_dir: Path) -> None:
    """代表的なサンプルタスクを GitHub Raw から高速にダウンロードする."""
    print(f"📥 代表サンプルタスク ({len(SAMPLE_TASK_IDS)}件) をダウンロード中...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    challenges = {}
    solutions = {}

    for task_id in SAMPLE_TASK_IDS:
        url = f"https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/{task_id}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                task_data = json.loads(resp.read().decode("utf-8"))
                challenges[task_id] = {
                    "train": task_data.get("train", []),
                    "test": [{"input": tc["input"]} for tc in task_data.get("test", [])],
                }
                solutions[task_id] = [tc["output"] for tc in task_data.get("test", [])]
                # 個別JSONも保存
                task_file = raw_dir / f"{task_id}.json"
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=2)
                print(f"  ✓ {task_id}.json 保存完了")
        except Exception as e:
            print(f"  ✗ {task_id} ダウンロード失敗: {e}", file=sys.stderr)

    # Kaggle形式のまとめJSONを生成
    ch_path = raw_dir / "arc-agi_training_challenges.json"
    sol_path = raw_dir / "arc-agi_training_solutions.json"
    with open(ch_path, "w", encoding="utf-8") as f:
        json.dump(challenges, f, indent=2)
    with open(sol_path, "w", encoding="utf-8") as f:
        json.dump(solutions, f, indent=2)
    print(f"✅ まとめデータ作成完了: {ch_path.name}, {sol_path.name}")


def download_full_arc_dataset(raw_dir: Path) -> None:
    """公式リポジトリ全体の ZIP をダウンロードして全タスクを展開・生成する."""
    print(f"📥 ARC-AGI 全データセット ZIP をダウンロード中 ({ARC_GITHUB_ZIP_URL})...")
    raw_dir.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(ARC_GITHUB_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        zip_bytes = resp.read()

    print("📦 ZIP アーカイブを展開中...")
    challenges = {}
    solutions = {}
    count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for filename in z.namelist():
            if "/data/training/" in filename and filename.endswith(".json"):
                task_id = Path(filename).stem
                with z.open(filename) as f:
                    task_data = json.loads(f.read().decode("utf-8"))
                    challenges[task_id] = {
                        "train": task_data.get("train", []),
                        "test": [{"input": tc["input"]} for tc in task_data.get("test", [])],
                    }
                    solutions[task_id] = [tc["output"] for tc in task_data.get("test", [])]
                    # 個別ファイルも配置
                    with open(raw_dir / f"{task_id}.json", "w", encoding="utf-8") as out_f:
                        json.dump(task_data, out_f, indent=2)
                    count += 1

    ch_path = raw_dir / "arc-agi_training_challenges.json"
    sol_path = raw_dir / "arc-agi_training_solutions.json"
    with open(ch_path, "w", encoding="utf-8") as f:
        json.dump(challenges, f, indent=2)
    with open(sol_path, "w", encoding="utf-8") as f:
        json.dump(solutions, f, indent=2)

    print(f"✅ 全 {count} 件のトレーニングタスクを {raw_dir} に展開しました。")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI データセットダウンロードスクリプト")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="代表的なサンプルタスク (5件) のみをダウンロードする (高速)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="保存先ディレクトリ (デフォルト: data/raw)",
    )
    args = parser.parse_args()

    if args.sample:
        download_sample_tasks(args.out_dir)
    else:
        download_full_arc_dataset(args.out_dir)


if __name__ == "__main__":
    main()
