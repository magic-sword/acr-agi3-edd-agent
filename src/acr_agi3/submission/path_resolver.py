"""Kaggle 環境とローカル環境でモデルパスとデータパスを自動解決するリゾルバ."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class ModelPathResolver:
    """Kaggle Dataset パスおよびローカルキャッシュパスを自動解決するクラス."""

    # 探索対象の既知候補パス
    DEFAULT_CANDIDATE_PATHS = [
        # Kaggle input datasets
        Path("/kaggle/input/qwen2-5-coder-1-5b-instruct"),
        Path("/kaggle/input/qwen2-5-coder-7b-instruct"),
        Path("/kaggle/input/qwen2-5-coder-1.5b-instruct"),
        Path("/kaggle/input/qwen-2.5-coder-1.5b"),
        # ローカルプロジェクト配下
        Path("models/qwen2.5-coder-1.5b-instruct"),
        Path("../models/qwen2.5-coder-1.5b-instruct"),
        # HuggingFace ローカルキャッシュスナップショット
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots",
    ]

    @classmethod
    def resolve_model_path(cls, custom_path: Optional[str | Path] = None) -> Optional[Path]:
        """利用可能なローカルモデルディレクトリの絶対パスを返す.

        Args:
            custom_path: 明示的に指定されたモデルパスまたはモデル識別子

        Returns:
            存在するモデルディレクトリの Path、見つからない場合は None
        """
        # 1. 環境変数からの取得
        env_path = os.environ.get("ARC_MODEL_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path).resolve()

        # 2. 明示的指定の確認
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                return p.resolve()

        # 3. 既知候補の走査
        for candidate in cls.DEFAULT_CANDIDATE_PATHS:
            if candidate.exists():
                if candidate.is_dir():
                    # snapshots 内部のハッシュディレクトリ走査
                    if "snapshots" in candidate.parts:
                        snapshots = list(candidate.iterdir())
                        if snapshots:
                            return snapshots[0].resolve()
                    return candidate.resolve()

        return None

    @classmethod
    def resolve_data_dir(cls, custom_path: Optional[str | Path] = None) -> Path:
        """テストデータやゲーム定義が格納されたディレクトリを解決する."""
        if custom_path and Path(custom_path).exists():
            return Path(custom_path).resolve()

        kaggle_input = Path("/kaggle/input/arc-prize-2026-arc-agi-3")
        if kaggle_input.exists():
            return kaggle_input.resolve()

        local_data = Path("data")
        if local_data.exists():
            return local_data.resolve()

        return Path(".")
