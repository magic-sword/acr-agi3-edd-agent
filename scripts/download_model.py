#!/usr/bin/env python3
"""オフライン推論用 LLM モデルダウンロードスクリプト.

HuggingFace から指定モデルの全ウェイト・設定ファイルを
ローカルの models/ ディレクトリにダウンロードします。
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(model_id: str, output_dir: Path) -> Path:
    """HuggingFace からモデルをオフライン保存."""
    print(f"📥 モデル '{model_id}' のダウンロードを開始します...")
    print(f"📁 保存先: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dest_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"✅ ダウンロード完了: {dest_path}")
    return Path(dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LLM model for offline usage")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/Qwen2.5-Coder-1.5B-Instruct"),
        help="Local output directory",
    )
    args = parser.parse_args()

    download_model(args.model_id, args.output_dir)
