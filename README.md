# ACR-AGI-3 Meta-Skill Driven Self-Evolving Agent
**ARC Prize 2026 (ARC-AGI-3) に向けた、メタスキル駆動型自己進化エージェント基盤**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Google ADK 2.0](https://img.shields.io/badge/Google%20ADK-2.0-green.svg)](https://github.com/google/adk)

> **"Intelligence is the efficiency of skill acquisition on novel, unseen tasks."** — François Chollet

本プロジェクトは、Google ADK 2.0 および [`skill-edd-agent`](https://github.com/magic-sword/skill-edd-agent) の評価駆動開発（EDD）アーキテクチャを活用し、**不定・未知の ARC-AGI-3 ゲーム環境に適応するための「思考とスキル量産のメタスキル（Meta-Skills）」を自律的に開発・自己改善・評価する**ためのプロジェクトです。

詳細なアーキテクチャ設計書は [ARCHITECTURE.md](file:///home/prog/work/kaggle/acr-agi3-edd-agent/ARCHITECTURE.md)、AI エージェントの行動指針は [AGENTS.md](file:///home/prog/work/kaggle/acr-agi3-edd-agent/AGENTS.md)、学術論文（Paper Track）向けの設計経緯・認知科学的根拠は [docs/paper/RESEARCH_REPORT_META_SKILLS.md](file:///home/prog/work/kaggle/acr-agi3-edd-agent/docs/paper/RESEARCH_REPORT_META_SKILLS.md) を参照してください。

---

## 🏛 アーキテクチャ概要：メタスキル vs 具象スキルの分離

ARC-AGI-3 ではタスクごとに全く新しいゲーム・ルールが現れます。
個別のゲームを解く「具象コード」を集めるのではなく、**「未知のゲーム環境を観察し、契約テスト付きの行動スキルを即座に量産・自己修復するメタスキル」** こそがオープンソース化すべき永続的コア資産です。

```mermaid
graph TD
    A[gcr.io/kaggle-gpu-images/python<br/>Kaggle 公式 GPU イメージ] -->|ベース環境| B[acr-agi3-edd-agent<br/>本プロジェクト]
    C[skill-edd-agent<br/>EDD フレームワーク] -->|依存/連携| B
    B --> M[meta_skills/<br/>★OSS コア資産: 思考・スキル量産のメタスキル群]
    B --> E[src/acr_agi3/<br/>メタオーケストレーター・VLM/LLM推論基盤]
    M -->|オンデマンド自律量産| G[generated_skills/<br/>実行時生成の一時スキル群 ※Git管理外]
    G -->|Evaluation Gating / 契約テスト全勝| F[ARC-AGI-3 Benchmark & Submission]
    E -->|Pass@k 算出| F
```

### レイヤー構成

1. **環境層 (Kaggle 公式 GPU イメージ)**:
   - `gcr.io/kaggle-gpu-images/python:latest` をベースに Python 3.12, PyTorch, CUDA (RTX A2000 / RTX Pro 6000) 環境を提供。
2. **メタスキル層 (`meta_skills/`)【★OSS コア資産】**:
   - `env-observer`: 未知環境の不変量・対称性・因果関係の抽出
   - `skill-synthesizer`: 観察結果から `SKILL.md`（契約テスト付き）と Python 実装を自動執筆
   - `contract-tester`: EDD 防壁ゲート（正例3＋負例3の全勝検証）
   - `failure-diagnoser`: 失敗時の構造化診断と自己修復（Evolver）
3. **具象インスタンススキル層 (`generated_skills/`)【一時キャッシュ】**:
   - メタスキルが未知のゲームごとにオンデマンド生成する個別タスク用スキル（Git 除外）。
4. **コアエンジン層 (`src/acr_agi3/`)**:
   - Google ADK 2.0 統合エージェント、ローカル LLM / VLM (Qwen2.5-VL) 推論アダプター、幾何レンダラー。

---

## 📂 プロジェクト構成

```text
acr-agi3-edd-agent/
├── meta_skills/                       # ★【OSS コア資産】思考・スキル量産のメタスキル群
│   ├── env-observer/                  # 環境不変量・対称性・因果関係抽出メタスキル
│   ├── skill-synthesizer/             # SKILL.md ＋ 契約テスト自動生成メタスキル
│   ├── contract-tester/               # EDD 評価防壁ゲートメタスキル
│   └── failure-diagnoser/             # テスト失敗診断・自己修復メタスキル
├── generated_skills/                  # ★【実行時生成】オンデマンド量産されたタスク特化スキル群
│   └── .gitignore                     # Git 管理外（一時キャッシュ）
├── skills/seeds/                      # 検証・初期ブートストラップ用のシードスキル群
│   └── grid-analyzer/                 # グリッド形状・色・対称性解析シード
├── src/acr_agi3/
│   ├── agent/                         # ローカルLLM/VLM 推論アダプター (Qwen2.5-Coder/VL)
│   ├── dsl/                           # 幾何変換 DSL & 公式カラーパレットレンダラー
│   ├── eval/                          # 評価ハーネス & メトリクス (Pass@k, Exact Match)
│   └── submission/                    # Kaggle / コンテスト提出用オフラインバンドラー
├── data/
│   ├── human_vcgt/                    # 人間のプレイ思考ログ (magicsword001/acr-agi-3-human-vcgt)
│   ├── raw/                           # ARC 公式タスク (arc-agi_training_challenges.json 等)
│   └── solutions/                     # 解答・成功スクリプト
├── notebooks/                         # JupyterLab 実験ノートブック
├── scripts/
│   ├── run_evolution_loop.py          # ローカルLLM 自己改善ループランナー
│   ├── run_vlm_evolution_loop.py      # Qwen2.5-VL 画像認識自己改善ループランナー
│   └── download_arc_data.py           # ARC データダウンロードスクリプト
├── ARCHITECTURE.md                    # アーキテクチャ完全仕様書
├── AGENTS.md                          # AI エージェント運用ガイドライン
├── pyproject.toml                     # uv / pip パッケージ定義
└── README.md
```


---

## ⚙️ セットアップ

### 前提条件

- **Docker** (20.10+) & **Docker Compose** (v2+)
- **NVIDIA Container Toolkit** (GPU利用時)
- **Git**

### 1. リポジトリのクローン

```bash
git clone https://github.com/magic-sword/acr-agi3-edd-agent.git
cd acr-agi3-edd-agent
```

### 2. 環境変数の設定

```bash
# テンプレートをコピー
cp .env.example .env

# エディタで開き、各項目を設定
vi .env
# 設定が必要な項目:
#   GEMINI_API_KEY    - LLM API キー (https://aistudio.google.com/)
#   KAGGLE_USERNAME   - Kaggle ユーザー名
#   KAGGLE_KEY        - Kaggle API キー (https://www.kaggle.com/settings)
```

> **⚠️ セキュリティ:** `.env` は `.gitignore` に含まれており、Git にコミットされません。

### 3. Docker 環境の起動

```bash
# ビルド & 起動 (初回はKaggle公式イメージのpullに時間がかかります)
docker compose up -d

# ログの確認
docker compose logs -f
```

### 4. JupyterLab へのアクセス

```bash
# SSH ポートフォワーディング (ローカル PC から)
ssh -L 8888:localhost:8888 user@server

# ブラウザで開く
# http://localhost:8888
```

### 5. 環境の確認

JupyterLab で `notebooks/00_environment_check.ipynb` を開いて実行し、環境が正しく構築されていることを確認してください。

### 6. コンテナ内での作業

```bash
# コンテナ内シェルに入る
docker compose exec kaggle-dev bash

# テスト実行
pytest -v

# Ruff による静的解析
ruff check .
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

---

## 🏆 Kaggle 提出

`notebooks/submission_template.ipynb` をベースに提出用ノートブックを作成します。

**重要な制約事項:**
- Kaggle 評価環境では **インターネット接続なし**
- 外部 API (Gemini, Claude 等) は **利用不可**
- 必要な依存はすべて **オフライン** で提供する必要あり
- 実行時間: **最大 9 時間**
- GPU: **RTX Pro 6000 (96GB VRAM)**

---

## 📝 ライセンス

[MIT License](LICENSE)