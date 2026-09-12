"""Kaggle ノートブック & ソースコード CLI 自動デプロイスクリプト.

コマンド一発で自己完結型提出ノートブックを Kaggle へデプロイ（push & 実行）し、
実行状態の確認およびリーダーボード提出を行う。

使用例:
    # 1. ノートブックをビルドして Kaggle にプッシュ（実行開始）
    docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --push

    # 2. 実行ステータスを確認
    docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --status

    # 3. ソースコードを Dataset としてプッシュ（任意）
    docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --push-dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_auth_env() -> None:
    """新トークン (~/.kaggle/access_token) または環境変数から認証を確立."""
    import os

    if "KAGGLE_API_TOKEN" not in os.environ:
        candidates = [
            Path.home() / ".kaggle/access_token",
            Path("/root/.kaggle/access_token"),
            REPO_ROOT / ".kaggle/access_token",
        ]
        for p in candidates:
            if p.exists():
                token = p.read_text().strip()
                if token:
                    os.environ["KAGGLE_API_TOKEN"] = token
                    break


def get_kaggle_username() -> str:
    """ユーザー名を取得 (デフォルト: magicsword001)."""
    ensure_auth_env()
    kaggle_json_candidates = [
        Path.home() / ".kaggle/kaggle.json",
        Path("/root/.kaggle/kaggle.json"),
    ]
    for p in kaggle_json_candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if "username" in data:
                    return str(data["username"])
            except Exception:
                pass
    return "magicsword001"


def prepare_deploy_dir(notebook_slug: str = "acr-agi3-agent-submission") -> Path:
    """デプロイ用ディレクトリとメタデータ JSON を生成."""
    deploy_dir = REPO_ROOT / "deploy" / "kaggle_kernel"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # 1. まず自己完結型ノートブックを最新ビルド
    print("🔨 Building latest self-contained submission notebook...")
    build_script = REPO_ROOT / "scripts" / "build_self_contained_notebook.py"
    if build_script.exists():
        subprocess.run([sys.executable, str(build_script)], check=True)

    # 2. 最新ノートブックをコピー
    src_nb = REPO_ROOT / "notebooks" / "submission_template.ipynb"
    dst_nb = deploy_dir / "submission_template.ipynb"
    shutil.copy2(src_nb, dst_nb)
    print(f"📄 Copied notebook to {dst_nb}")

    # 3. kernel-metadata.json の作成
    username = get_kaggle_username()
    kernel_id = f"{username}/{notebook_slug}"
    metadata = {
        "id": kernel_id,
        "title": notebook_slug,
        "code_file": "submission_template.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    meta_file = deploy_dir / "kernel-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Generated kernel metadata: {kernel_id}")
    return deploy_dir


def push_kernel(deploy_dir: Path) -> None:
    """Kaggle へノートブックを push してバックグラウンド実行を開始."""
    print(f"🚀 Pushing kernel to Kaggle from {deploy_dir}...")
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(deploy_dir)],
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    if res.returncode == 0:
        print("✅ Kernel pushed successfully! Kaggle is now running 'Save & Run All'.")
    else:
        print(f"⚠️ Kernel push exited with code {res.returncode}")


def check_status(notebook_slug: str = "acr-agi3-agent-submission") -> None:
    """Kaggle 上でのノートブック実行ステータスを確認."""
    username = get_kaggle_username()
    kernel_id = f"{username}/{notebook_slug}"
    print(f"🔍 Checking kernel status for {kernel_id}...")
    res = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)


def push_dataset(dataset_slug: str = "acr-agi3-source") -> None:
    """ソースコードアーカイブを Kaggle Dataset として push."""
    dist_tar = REPO_ROOT / "dist" / "acr_agi3_source.tar.gz"
    if not dist_tar.exists():
        print(f"❌ Archive not found at {dist_tar}. Building archive first...")
        import tarfile
        dist_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dist_tar, "w:gz") as tar:
            tar.add(REPO_ROOT / "src", arcname="src")

    deploy_ds_dir = REPO_ROOT / "deploy" / "kaggle_dataset"
    deploy_ds_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dist_tar, deploy_ds_dir / dist_tar.name)

    username = get_kaggle_username()
    metadata = {
        "title": "ACR-AGI-3 Source Package",
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "mit"}],
    }
    with open(deploy_ds_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"🚀 Creating / Updating Kaggle Dataset {username}/{dataset_slug}...")
    # まず更新を試み、存在しなければ新規作成
    cmd_version = [
        "kaggle", "datasets", "version",
        "-p", str(deploy_ds_dir),
        "-m", "Auto-update source package",
        "-r", "zip",
    ]
    res = subprocess.run(cmd_version, capture_output=True, text=True)
    if res.returncode != 0:
        res = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(deploy_ds_dir), "-r", "zip"],
            capture_output=True,
            text=True,
        )
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle CLI Deployment Tool")
    parser.add_argument("--push", action="store_true", help="Build and push notebook to Kaggle")
    parser.add_argument("--status", action="store_true", help="Check notebook status on Kaggle")
    parser.add_argument(
        "--push-dataset",
        action="store_true",
        help="Upload source archive as Kaggle Dataset",
    )
    parser.add_argument("--slug", default="acr-agi3-agent-submission", help="Notebook slug")

    args = parser.parse_args()

    if args.push:
        d = prepare_deploy_dir(notebook_slug=args.slug)
        push_kernel(d)
    elif args.status:
        check_status(notebook_slug=args.slug)
    elif args.push_dataset:
        push_dataset()
    else:
        # デフォルトは push
        d = prepare_deploy_dir(notebook_slug=args.slug)
        push_kernel(d)


if __name__ == "__main__":
    main()
