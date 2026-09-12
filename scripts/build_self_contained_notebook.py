"""クリーンな Kaggle 提出ノートブック生成スクリプト.

Base64 などの巨大文字列を一切埋め込まず、
Kaggle Dataset (acr-agi3-source) またはローカル src から
美しくクリーンにインポートするノートブックをビルドする。
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_notebook() -> None:
    nb_path = REPO_ROOT / "notebooks" / "submission_template.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    clean_import_code = """# === acr_agi3 ライブラリの読み込み ===
import os
import sys
import tarfile
from pathlib import Path

def init_library_path():
    # 1. ローカル開発環境または作業ディレクトリの検出
    for p in [Path("src"), Path("../src"), Path("/kaggle/working/src")]:
        if (p / "acr_agi3").exists():
            resolved = str(p.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            print(f"✅ Loaded acr_agi3 from local: {resolved}")
            return

    # 2. Kaggle Dataset (/kaggle/input) からの読み込み
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # アーカイブ（src.tar 等）があれば自動展開
        for arc in kaggle_input.rglob("*.tar*"):
            try:
                with tarfile.open(arc, "r:*") as tar:
                    tar.extractall(path="/kaggle/working")
                if Path("/kaggle/working/src").exists():
                    sys.path.insert(0, "/kaggle/working/src")
                    print("✅ Extracted and added /kaggle/working/src to sys.path")
                    return
            except Exception:
                pass

        # 展開済みフォルダがある場合は直接追加
        for candidate in kaggle_input.rglob("acr_agi3"):
            if candidate.is_dir():
                p_dir = str(candidate.parent.resolve())
                if p_dir not in sys.path:
                    sys.path.insert(0, p_dir)
                print(f"✅ Found acr_agi3 in Kaggle Input: {p_dir}")
                return

init_library_path()

# インポート確認
from acr_agi3.submission.path_resolver import ModelPathResolver
from acr_agi3.submission.entrypoint import KaggleSubmissionPipeline, run_submission
from acr_agi3.agent.orchestrator import ARCOrchestrator
from acr_agi3.game.env import Action

print("🎉 Successfully loaded acr_agi3 package!")
"""

    # セル 2 を置換
    nb["cells"][2]["source"] = [line + "\n" for line in clean_import_code.strip().split("\n")]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"✅ Successfully updated {nb_path} with clean, elegant imports (no Base64)!")


if __name__ == "__main__":
    build_notebook()
