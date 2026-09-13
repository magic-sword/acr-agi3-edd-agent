#!/usr/bin/env python3
"""Kaggle Dataset 準備 & 同期スクリプト (ACR-AGI-3).

リポジトリの `src/` および `meta_skills/` を Kaggle Dataset 用ディレクトリ (`build/kaggle_dataset/`)
に構造化・同期し、Kaggle CLI によるワンクリック更新を可能にします。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build" / "kaggle_dataset"


def sync_dataset_files(output_dir: Path) -> int:
    """src/ と meta_skills/ を output_dir にクリーンコピー."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ignore_patterns = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", ".DS_Store", "*.swp"
    )

    # 1. src/ コピー
    src_dest = output_dir / "src"
    shutil.copytree(REPO_ROOT / "src", src_dest, ignore=ignore_patterns)

    # 2. meta_skills/ コピー
    meta_dest = output_dir / "meta_skills"
    shutil.copytree(REPO_ROOT / "meta_skills", meta_dest, ignore=ignore_patterns)

    # 3. dataset-metadata.json 生成
    meta_json = {
        "title": "ACR-AGI-3 Agent Framework",
        "id": "magic-sword/acr-agi3-agent",
        "licenses": [{"name": "mit"}],
    }
    with (output_dir / "dataset-metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    total_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
    print(f"📦 Successfully synced {total_files} files to {output_dir}")
    return total_files


def main():
    parser = argparse.ArgumentParser(description="Prepare and sync Kaggle Dataset.")
    parser.add_argument("--upload", "-u", action="store_true", help="Upload to Kaggle via kaggle CLI")
    parser.add_argument("--message", "-m", type=str, default="Update agent and meta skills", help="Version message")
    args = parser.parse_args()

    total_files = sync_dataset_files(BUILD_DIR)

    if args.upload:
        print("🚀 Uploading dataset to Kaggle...")
        try:
            cmd = ["kaggle", "datasets", "version", "-p", str(BUILD_DIR), "-m", args.message, "--dir-mode", "tar"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Upload failed: {res.stderr}")
                print("Tip: If dataset does not exist yet, run: kaggle datasets create -p build/kaggle_dataset")
            else:
                print(res.stdout)
        except Exception as e:
            print(f"Error running kaggle CLI: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
