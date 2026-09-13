# AGENTS.md — AI エージェント運用ガイドライン

本ファイルは、本リポジトリで作業するすべての AI エージェント（Pair Programming Assistant, Coding Agent, Autonomous Solver）に対する**行動原則・設計ルール**を定めたものです。作業前に必ず一読し、遵守してください。

---

## 🚨 最重要方針：ゲームプレイ（Game Play）とメタスキルの本質

本プロジェクト（`acr-agi3-edd-agent`）は、**ARC-1 / 2 のような静的な入出力パズル変換（`def transform(grid)`）を解くプロジェクトではありません。**
ACR-AGI-3 は **「未知の動的ゲーム環境において、状態観測からアフォーダンスを認識し、行動（Action: UP, DOWN, LEFT, RIGHT 等）によってステージクリアを目指すインタラクティブなゲームプレイ」** です。

目的は、**「未知のゲーム環境に直面したとき、自律的に思考し、高品質な行動スキルを即座に量産・検証・修復できるメタスキル（Meta-Skills）基盤を構築すること」** です。

### エージェントの遵守事項

1. **ARC-1/2 の静的パズル（一括グリッド変換）と混同しないこと**
   * 入力グリッド全体を一発で出力グリッドに変換するコードを書くのではなく、ゲーム環境（`GameEnvironment`）の状態遷移と行動ポリシー（`choose_action(obs) -> Action`）を設計すること。
   * 静的パズルの JSON データ（`arc-agi_training_challenges.json` 等）を本プロジェクトの解法対象と誤認しないこと。

2. **個別タスクの解法コードを `meta_skills/` にコミットしないこと**
   * `meta_skills/` は「スキルの作り方・検証の仕方」を規定する思考エンジン専用ディレクトリです。
   * 個別タスクに対する解法や生成されたスキルは、必ず `generated_skills/` またはインメモリで扱い、Git コミット対象外（`.gitignore`）とすること。

3. **EDD (Evaluation-Driven Development) 防壁ゲートの徹底**
   * スキルを生成・改良する際は、必ず **「正例 3 件 ＋ 負例 3 件」の契約テスト** を同時に設計・実行すること。
   * 契約テストで 1 件でもエラー（例外、形状不一致、境界値エラー、トラップ衝突）があるコードは、絶対に採用・マージしてはならない。

4. **EDD 汎用フレームワークと本リポジトリの厳格な分離 (Separation of Concerns)**
   * **禁止事項**: 汎用 EDD フレームワーク（ADK 2.0 評価器、テレメトリ収集器、根本原因診断アナライザー、CLI 等）のコアロジックを本リポジトリ（`acr-agi3-edd-agent`）に直接コミット・拡張してはならない。
   * **正当な手順**: EDD フレームワーク本体の機能拡張・修正は、必ず上流専用リポジトリ **`skill-edd-agent`**（ローカルパス: `/home/prog/work/skill-edd-agent`）にて作業・テスト・コミット・プッシュすること。
   * **本リポジトリの責務**: 本リポジトリ内の `src/acr_agi3/edd/` は、ARC-AGI-3 のゲーム固有 API（`GameAction`, `arcengine`）と上流 EDD を繋ぐ **アダプター層（Adapter Layer）のみ** とし、汎用ロジックの二重管理を行わないこと。

5. **完全オフライン互換の死守**
   * Kaggle 本番環境はインターネット接続がありません。
   * 外部 API（クラウド LLM API 等）に依存した提出用コードを書かないこと。
   * ローカル推論アダプター（`LocalTransformersLlm`, `LocalQwenVL`）は必ず `local_files_only=True` で動作させること。

6. **マルチモーダル認識の優先活用**
   * ARC-AGI のタスクを解く際は、数値テキスト（`[[0, 1], ...]`）だけでなく、`src/acr_agi3/dsl/renderer.py` を用いて公式 10 色カラー画像にレンダリングし、視覚的ゲシュタルト（空間的対称性、閉領域、境界）を活用すること。

7. **Google ADK 2.0 準拠「3段階 Progressive Disclosure（段階的開示）」の徹底**
   * 単なる Python スクリプトの直接呼び出しや、プロンプトへの全量指示一括注入を行わないこと。
   * **車輪の再発明の禁止**: 独自パーサーや独自 Progressive Disclosure ツールを自作せず、Google ADK 2.0 公式の `google.adk.skills.load_skills_from_dir` および `google.adk.tools.skill_toolset.SkillToolset` を直接活用すること。
   * **Level 1 (Metadata)**: `SKILL.md` の YAML Frontmatter カタログのみをコンテキストに常駐させ、極小トークンでスキルを俯瞰すること（ADK `list_skills`）。
   * **Level 2 (Instructions)**: エージェントが必要に応じてトリガーした時のみ、該当スキルの `SKILL.md` 本文（ワークフロー・思考プロトコル）をオンデマンド展開すること（ADK `load_skill`）。
   * **Level 3 (Execution)**: スキル配下の `scripts/` または許可されたツール（`allowed-tools`）をオンデマンド実行すること（ADK `run_skill_script`, `load_skill_resource`）。
   * スキルの管理・ロード・実行は必ず [`src/acr_agi3/meta/skill_harness.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/skill_harness.py) の `SkillHarness`（ADK 公式ラッパー）を経由すること。

8. **EDD MCP ツール（`edd-agent`）によるスキル作成・静的検証の必須化**
   * 新規スキルを作成・初期化する際は、必ず MCP ツール **`edd_init_skill`**（または `SkillScaffolder`）を用いて標準ディレクトリ構造（`SKILL.md`, `scripts/`, `tests/`）を生成すること。
   * スキルを作成・編集した後は、必ず MCP ツール **`edd_validate_skill`** を実行し、Markdown-First / Progressive Disclosure 規約に対する **エラー 0 件・警告 0 件** を確認してからコミットすること。

9. **Kaggle Dataset 直参照（Read-Only）アーキテクチャの徹底**
   * メタスキル（`meta_skills/`）およびソースコード（`src/`）は、ノートブック内に Base64/辞書として埋め込んで物理再展開してはならない（コード肥大化・I/Oオーバーヘッドの禁止）。
   * Kaggle 本番環境では、Kaggle Dataset（`/kaggle/input/acr-agi3-agent/`）のフォルダ構造をそのまま直接インポート・参照すること。
   * 提出ノートブック（`notebooks/submission_template.ipynb`）は、エージェントをロードして Gateway を叩くだけの極小・クリーンな構成（数十行）を維持すること。
   * 実行時動的生成スキル（`generated_skills/`）のみを書き込み可能領域（`/kaggle/working/generated_skills`）で扱うこと。

---

## 🧩 Google ADK 準拠 3段階 Progressive Disclosure 設計思想

```
[Level 1: Metadata Catalog] (常駐: 低コンテキスト消費)
  │  - name, description, inputs, outputs, allowed-tools を把握
  ▼ 自律判定により必要なスキルをトリガー (Trigger)
[Level 2: Instructions] (オンデマンド展開: SKILL.md 本文)
  │  - ワークフロー、思考プロトコル、制約事項、入出力例の理解
  ▼ ワークフローの各ステップを実行 (Execution)
[Level 3: Tools & Scripts] (オンデマンド実行: scripts/ & Tools)
     - env_observer.py, game_style_intuitor.py, contract_tester.py 等
```

### スキル作成・検証クイックリファレンス (MCP Tools)

```bash
# 1. 新規スキルの雛形作成 (MCP: edd_init_skill)
call_mcp_tool(ServerName="edd-agent", ToolName="edd_init_skill", Arguments={"name": "new-skill-name", "path": "generated_skills"})

# 2. スキルの規約静的検証 (MCP: edd_validate_skill)
call_mcp_tool(ServerName="edd-agent", ToolName="edd_validate_skill", Arguments={"skill_dir": "meta_skills/env-observer"})
```

---

## 📂 ディレクトリの役割と書き込み権限

* **`meta_skills/` [編集対象・永続]**:
  * `env-observer/`: 環境不変量・対称性・因果関係の抽出メタスキル
  * `game-style-intuitor/`: ゲームスタイル分類・探索方針策定メタスキル
  * `subgoal-decomposer/`: サブゴール階層分解メタスキル
  * `skill-synthesizer/`: `SKILL.md` ＋ 契約テスト自動生成メタスキル
  * `contract-tester/`: EDD 評価防壁ゲート（正例3+負例3）メタスキル
  * `failure-diagnoser/`: テスト失敗診断・自己修復メタスキル
  * `constraint-learner/`: 環境制約・禁忌状態学習メタスキル
* **`generated_skills/` [実行時生成・Git除外]**:
  * メタスキルがタスク解決のために一時生成する具象スキル置き場
* **`src/acr_agi3/` [コアエンジン]**:
  * `meta/`: メタ認知オーケストレーター & `skill_harness.py`
  * `agent/`: ローカル推論エージェント & VLM アダプター (`meta_agent.py`)
  * `dsl/`: 幾何プリミティブ & レンダラー
  * `submission/`: Kaggle 提出用バンドラー
* **`data/human_vcgt/` [学習源]**:
  * 人間のプレイ記録 ＋ 思考解説ログ（VCGT: Visual Concept Guided Thinking）
