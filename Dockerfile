# =============================================================================
# ARC-AGI-3 EDD Agent — Kaggle互換 統合開発環境
# =============================================================================
# Kaggle公式GPU Dockerイメージをベースに、EDD Agent 開発に必要な
# 追加依存とJupyterLabを統合した開発環境。
#
# Kaggle提出環境と同一のランタイム (Python 3.12, PyTorch, CUDA) で
# エージェント開発・テスト・ノートブック実験が可能。
# =============================================================================

FROM gcr.io/kaggle-gpu-images/python:latest

LABEL maintainer="magic-sword"
LABEL description="ARC-AGI-3 EDD Agent: Kaggle-compatible dev environment with JupyterLab"

# --- 作業ディレクトリ ---
WORKDIR /workspace

# --- プロジェクト依存のインストール ---
# pyproject.toml を先にコピーしてレイヤーキャッシュを活用
COPY pyproject.toml /workspace/pyproject.toml

# EDD Agent 追加依存のみインストール (Kaggle環境に既にあるものは skip される)
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
        "ruff>=0.4.0" \
        "pytest>=8.0.0" \
        "pytest-xdist>=3.5.0" \
        "matplotlib>=3.8.0" \
        "ipykernel>=6.29.0"

# JupyterLab を最新版にアップグレード (Kaggle環境は 3.x の場合がある)
RUN pip install --no-cache-dir --upgrade "jupyterlab>=4.0.0"

# --- JupyterLab 設定 ---
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.terminado_settings = {'shell_command': ['/bin/bash']}" >> /root/.jupyter/jupyter_lab_config.py

# --- プロジェクトソースのコピー ---
COPY . /workspace

# プロジェクトを editable モードでインストール
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || true

# --- ポート公開 ---
EXPOSE 8888

# --- エントリポイント ---
# デフォルトは JupyterLab 起動。docker exec で別プロセスも実行可能。
COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

CMD ["/usr/local/bin/start.sh"]
