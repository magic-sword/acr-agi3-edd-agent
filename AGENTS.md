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

---

## 📂 ディレクトリの役割と書き込み権限

* **`meta_skills/` [編集対象・永続]**:
  * `env-observer/`: 環境不変量・対称性・因果関係の抽出メタスキル
  * `skill-synthesizer/`: `SKILL.md` ＋ 契約テスト自動生成メタスキル
  * `contract-tester/`: EDD 評価防壁ゲートメタスキル
  * `failure-diagnoser/`: テスト失敗診断・自己修復メタスキル
* **`generated_skills/` [実行時生成・Git除外]**:
  * メタスキルがタスク解決のために一時生成する具象スキル置き場
* **`src/acr_agi3/` [コアエンジン]**:
  * `meta/`: メタ認知オーケストレーター
  * `agent/`: ローカル推論エージェント & VLM アダプター
  * `dsl/`: 幾何プリミティブ & レンダラー
  * `submission/`: Kaggle 提出用バンドラー
* **`data/human_vcgt/` [学習源]**:
  * 人間のプレイ記録 ＋ 思考解説ログ（VCGT: Visual Concept Guided Thinking）
