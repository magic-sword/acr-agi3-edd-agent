"""Kaggle から Human VCGT データセットを取得・サンプリングするスクリプト."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def download_dataset(target_dir: Path) -> Path:
    """Kaggle API を使用してデータセットを取得."""
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_file = target_dir / "results.csv"
    if csv_file.exists():
        print(f"[Info] Found existing {csv_file}")
        return csv_file

    print("[Info] Downloading results.csv from magicsword001/acr-agi-3-human-vcgt...")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "magicsword001/acr-agi-3-human-vcgt",
        "-f",
        "results.csv",
        "-p",
        str(target_dir),
        "--unzip",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Error] Kaggle download failed: {res.stderr}", file=sys.stderr)
        return csv_file

    print("[Info] Download complete.")
    return csv_file


def parse_sample_rows(csv_path: Path, output_json: Path, limit: int = 50) -> None:
    """results.csv の先頭 N 行からテキスト思考データを抽出して JSON に保存."""
    if not csv_path.exists():
        print(f"[Warning] CSV file not found: {csv_path}. Skipping parse.")
        return

    records = []
    print(f"[Info] Parsing {limit} records from {csv_path}...")
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            # CSV カラム構造に応じた安全な抽出
            task_id = row.get("task_id", row.get("id", f"task_{i}"))
            step_idx = int(row.get("step", row.get("step_index", i)))
            goal = row.get("goal", row.get("objective", ""))
            reasoning = row.get("reasoning", row.get("explanation", ""))
            steps_raw = row.get("steps", row.get("actions", ""))
            steps = [s.strip() for s in steps_raw.split(";") if s.strip()] if steps_raw else []

            records.append(
                {
                    "task_id": task_id,
                    "step_index": step_idx,
                    "environment": row.get("game", row.get("env", "arc_game")),
                    "observation": {"raw_row_idx": i},
                    "human_vcgt": {
                        "goal": goal,
                        "reasoning": reasoning,
                        "steps": steps,
                        "reflection": row.get("reflection", ""),
                    },
                    "invariants_identified": [],
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[Info] Saved {len(records)} parsed VCGT samples to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human VCGT dataset loader")
    parser.add_argument("--data-dir", type=str, default="data/human_vcgt")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_file = download_dataset(data_dir)
    sample_json = data_dir / "parsed_vcgt.json"
    parse_sample_rows(csv_file, sample_json, limit=args.limit)


if __name__ == "__main__":
    main()
