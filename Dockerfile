# =============================================================================
# ARC-AGI-3 EDD Agent — Kaggle互換 統合開発環境
# =============================================================================
# Kaggle公式GPU Dockerイメージをベースにした開発環境。
# Kaggleイメージには以下がプリインストール済み:
#   - Python 3.12, PyTorch 2.10, CUDA 12.8
#   - NumPy, SciPy, Pydantic, Matplotlib, ipykernel
#   - JupyterLab, pytest, ruff, kaggle CLI
#
# この Dockerfile では不要な再ビルドを避け、
# JupyterLab 設定とプロジェクトパスの連携のみを行います。
# =============================================================================

FROM gcr.io/kaggle-gpu-images/python:latest

LABEL maintainer="magic-sword"
LABEL description="ARC-AGI-3 EDD Agent: Kaggle-compatible dev environment with JupyterLab"

# --- 作業ディレクトリ ---
WORKDIR /workspace

# --- Python パス設定 (バインドマウントされた src/ を直接インポート可能にする) ---
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

# --- JupyterLab 設定 ---
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.terminado_settings = {'shell_command': ['/bin/bash']}" >> /root/.jupyter/jupyter_lab_config.py

# --- エントリポイント ---
COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 8888

CMD ["/usr/local/bin/start.sh"]
