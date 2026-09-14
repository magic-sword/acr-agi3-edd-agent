#!/usr/bin/env python3
"""Qwen2.5-VL-3B-Instruct モデル重みダウンロードスクリプト.

Kaggle ARC-AGI-3 オフライン実行環境およびローカル実験用に、
Qwen/Qwen2.5-VL-3B-Instruct の重みファイルをローカルディレクトリ (models/) にダウンロードします。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = REPO_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def download_model(
    model_id: str = DEFAULT_MODEL_ID,
    target_dir: Path = DEFAULT_TARGET_DIR,
) -> Path:
    """Hugging Face Hub からモデルを snapshot_download."""
    print(f"📦 Downloading {model_id} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub is not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"✅ Successfully downloaded {model_id} to {local_dir}")
    # 必須ファイルの存在確認
    required_files = ["config.json", "processor_config.json"]
    for rf in required_files:
        p = target_dir / rf
        if not p.exists():
            print(f"⚠️ Warning: Expected file {rf} not found in {target_dir}")
        else:
            print(f"  • Found: {rf}")

    return Path(local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen2.5-VL model weights")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (default: Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Target local directory",
    )
    args = parser.parse_args()
    download_model(model_id=args.model_id, target_dir=args.target_dir)
