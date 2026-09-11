#!/bin/bash
# =============================================================================
# ARC-AGI-3 EDD Agent — コンテナ起動スクリプト
# =============================================================================
# JupyterLab を起動し、コンテナをアクティブに保つ。
# docker exec で EDD Agent やテストコマンドを別途実行可能。
# =============================================================================

set -e

echo "============================================="
echo "  ARC-AGI-3 EDD Agent — Development Environment"
echo "============================================="
echo ""

# --- 環境情報の表示 ---
echo "[INFO] Python:  $(python3 --version 2>&1)"
echo "[INFO] PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not available')"
echo "[INFO] CUDA:    $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
echo "[INFO] GPU:     $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'not available')"
echo ""

# --- JupyterLab 起動 ---
echo "[INFO] Starting JupyterLab on port 8888..."
echo "[INFO] Access: http://localhost:8888"
echo "[INFO] (SSH port forwarding: ssh -L 8888:localhost:8888 user@server)"
echo ""

exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --notebook-dir=/workspace \
    --ServerApp.token='' \
    --ServerApp.password=''
