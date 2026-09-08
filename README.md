# ACR-AGI-3 Self-Evolving EDD Agent
**ARC Prize 2026 (ARC-AGI-3) に向けた、評価駆動開発（EDD）型自己進化エージェント基盤**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Google ADK 2.0](https://img.shields.io/badge/Google%20ADK-2.0-green.svg)](https://github.com/google/adk)

本プロジェクトは、Google ADK 2.0 および [`skill-edd-agent`](https://github.com/magic-sword/skill-edd-agent) の自己進化アーキテクチャを活用し、**ARC-AGI-3 (ARC Prize 2026) コンテストに出場する推論エージェントと問題解決スキルを自律的に開発・自己改善・評価する**ためのプロジェクトです。

---

## 🏛 アーキテクチャ概要

```mermaid
graph TD
    A[agents-adk-devcontainer<br/>Docker Image / GHCR] -->|ベース環境提供| B[acr-agi3-edd-agent<br/>本プロジェクト]
    C[skill-edd-agent<br/>EDD フレームワーク] -->|依存/連携 (edd link)| B
    B --> D[skills/<br/>ARC ドメイン特化スキル群]
    B --> E[src/acr_agi3/<br/>エージェント・DSL・評価パイプライン]
    D -->|Evaluation Gating / 契約テスト| F[ARC-AGI-3 Benchmark & Submission]
    E -->|Pass@k 算出| F
```

1. **環境層 ([`agents-adk-devcontainer`](https://github.com/magic-sword/agents-adk-devcontainer))**:
   - Dev Container 上で統一された Python, uv, Google Agents CLI, Google Cloud CLI 環境を提供。
2. **フレームワーク層 ([`skill-edd-agent`](https://github.com/magic-sword/skill-edd-agent))**:
   - Evaluation Gating（テスト全勝を必須とする防壁）、自己修復ループ、`edd` CLI を提供。
3. **ドメイン層 (`acr-agi3-edd-agent`)**:
   - ARC-AGI-3 のタスクデータ、推論 DSL、幾何変換・物体抽出・探索スキル群、提出用バンドラー。

---

## 📂 プロジェクト構成

```text
acr-agi3-edd-agent/
├── .devcontainer/
│   └── devcontainer.json          # Dev Container 定義 (GHCR イメージ利用)
├── .github/
│   └── workflows/
│       └── test.yml               # CI: Ruff Lint & Pytest
├── data/                          # ARC-AGI-3 データセット (Git 管理外)
│   ├── raw/                       # 公式タスク (arc-agi_training_challenges.json 等)
│   ├── generated/                 # エージェント自己生成の合成タスク
│   └── solutions/                 # 解答データ
├── skills/                        # ARC-AGI-3 特化の自己改善スキル群
│   └── grid-analyzer/             # グリッド形状・色・対称性静的解析スキル
│       ├── SKILL.md               # スキル仕様書 (Markdown-First)
│       ├── scripts/               # 決定論的スクリプト
│       └── tests/                 # EDD 契約テスト (正例3 + 負例3)
├── src/
│   └── acr_agi3/
│       ├── agent/                 # 推論エージェント (Orchestrator, Hypothesis, Verifier, Evolver)
│       ├── dsl/                   # ARC ドメイン固有言語 (Primitives, Interpreter)
│       ├── eval/                  # 評価ハーネス & メトリクス (Pass@k, Exact Match)
│       └── submission/            # Kaggle / コンテスト提出用バンドラー
├── tests/                         # 全体統合・疎通テスト
├── pyproject.toml                 # uv / pip パッケージ定義
└── README.md
```

---

## ⚙️ エージェント自律開発のためのセットアップ

### 1. 動作環境の起動と依存関係のインストール
Dev Container を利用すると、GPU パススルーおよび必要なツールが自動構成されます。
```bash
# 基本依存 + 開発・評価ツールのインストール
uv pip install -e ".[dev,edd]"
```

### 2. 環境変数の設定 (LLM API)
エージェントの仮説生成および自己進化ループ（`Evolver`）で利用する API キーを設定します。
```bash
cp .env.example .env
# .env を編集し、GEMINI_API_KEY を設定
```

### 3. Kaggle & GitHub の認証
公式タスクデータのダウンロードや提出、Git 操作を行うための認証を行います。
```bash
# GitHub CLI 認証
gh auth login

# Kaggle API (ホストの ~/.kaggle/kaggle.json を配置するか環境変数を設定)
# export KAGGLE_USERNAME="..." && export KAGGLE_KEY="..."
kaggle competitions download -c arc-prize-2026-arc-agi-3 -p data/raw/
```

---

## 🚀 自律開発・評価ワークフロー

```bash
# 1. 静的解析とコード検証
ruff check .

# 2. 契約テスト・統合テストの実行 (防壁ゲート)
pytest -v

# 3. スキル仕様・Frontmatter の静的バリデーション (EDD)
edd validate skills/grid-analyzer

# 4. スキルの評価とカバレッジ計測
edd eval grid-analyzer --coverage

# 5. 失敗時の構造化診断
edd diagnose grid-analyzer
```