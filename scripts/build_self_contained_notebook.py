"""自己完結型 Kaggle 提出ノートブック生成スクリプト.

src/ のコードを Base64 アーカイブとしてノートブック内に安全に埋め込み、
外部データセットなしでも完全オフラインで 100% 動作するノートブックをビルドする。
"""

import base64
import json
import tarfile
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_notebook() -> None:
    # 1. src ディレクトリをメモリ上で tar.gz 化
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REPO_ROOT / "src", arcname="src")
    tar_bytes = buf.getvalue()
    b64_str = base64.b64encode(tar_bytes).decode("utf-8")
    print(f"📦 Encoded src package: {len(tar_bytes)} bytes -> {len(b64_str)} base64 chars")

    nb_path = REPO_ROOT / "notebooks" / "submission_template.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 2. セル 2（インポート & 自動展開セル）のコードを構築
    bootstrap_code = f"""# === acr_agi3 ライブラリの自動検出 & 自己解凍セットアップ ===
import sys
import os
import tarfile
import base64
from io import BytesIO
from pathlib import Path

def setup_acr_agi3():
    # 1. 既存の sys.path やカレントディレクトリから検出
    candidates = [
        Path("src"),
        Path("../src"),
        Path("/kaggle/working/src"),
        Path("/kaggle/working/acr-agi3-edd-agent/src"),
    ]
    for p in candidates:
        if (p / "acr_agi3").exists():
            resolved = str(p.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            print(f"✅ Found and added local package: {{resolved}}")
            return

    # 2. /kaggle/input 配下の再帰走査（データセットとして追加されている場合）
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("acr_agi3"):
            if candidate.is_dir():
                parent_dir = str(candidate.parent.resolve())
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                print(f"✅ Discovered acr_agi3 in Kaggle Input: {{parent_dir}}")
                return

        # .tar.gz や .zip アーカイブの解凍
        for archive in kaggle_input.rglob("*acr*source*.tar.gz"):
            print(f"📦 Extracting archive from {{archive}}...")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path="/kaggle/working")
            if Path("/kaggle/working/src").exists():
                sys.path.insert(0, "/kaggle/working/src")
                print("✅ Extracted and added /kaggle/working/src to sys.path")
                return

    # 3. 自己完結埋め込みコードからの自動自己解凍 (ゼロ外部依存保証)
    print("🚀 Auto-extracting embedded acr_agi3 package into /kaggle/working/src...")
    EMBEDDED_SRC_B64 = "{b64_str}"
    tar_bytes = base64.b64decode(EMBEDDED_SRC_B64)
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
        target_dir = Path("/kaggle/working") if Path("/kaggle").exists() else Path(".")
        tar.extractall(path=target_dir)
    extracted_src = (target_dir / "src").resolve()
    if str(extracted_src) not in sys.path:
        sys.path.insert(0, str(extracted_src))
    print(f"✅ Extracted embedded package to {{extracted_src}} and added to sys.path")

setup_acr_agi3()

# インポート確認
from acr_agi3.submission.path_resolver import ModelPathResolver
from acr_agi3.submission.entrypoint import KaggleSubmissionPipeline, run_submission
from acr_agi3.agent.orchestrator import ARCOrchestrator
from acr_agi3.game.env import Action

print("🎉 Successfully loaded acr_agi3 package!")
"""

    # セルの更新 (cell index 2: 3番目のセル)
    # nb["cells"][2] がライブラリパス解決セル
    nb["cells"][2]["source"] = [line + "\n" for line in bootstrap_code.strip().split("\n")]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"✅ Successfully updated {nb_path} with self-contained bootstrap!")


if __name__ == "__main__":
    build_notebook()
