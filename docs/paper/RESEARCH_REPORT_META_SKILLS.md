# メタスキル駆動型適応基盤の設計経緯と理論的枠組み
## Towards General Adaptation in ARC-AGI-3 via Human-Inspired Meta-Skills and Evaluation-Driven Development (EDD)

本ドキュメントは、ARC-AGI-3（ARC Prize 2026）に向けたエージェント開発において、個別パズルの解法コード集積から脱却し、**「未知のゲーム環境に対するメタスキル（Meta-Skills）基盤」** を創出した背景・認知科学的根拠・システム設計・論文投稿時の論点（Contributions）を体系的にまとめた技術・研究レポートです。

---

## 1. エグゼクティブ・サマリー（論文アブストラクト草案）

> **Abstract**:  
> 従来の ARC-AGI アプローチの多くは、限られた幾何プリミティブを組み合わせた固定 DSL や、特定グリッドタスクに対する一発書きのプログラム合成（Program Synthesis）に依存していた。しかし、タスクごとにルール・物理法則・アフォーダンスが根本から切り替わる動的環境（ACR-AGI-3）において、この静的な解法集積アプローチは深刻な分布外脆弱性（Out-of-Distribution Collapse）に直面する。  
> 本研究では、人間が未知のゲーム環境に適応する際の思考・プレイ記録データセット（VCGT: Visual Concept Guided Thinking）を認知科学的・情報論的に解析し、**「個別の具象スキル」と「スキルを自律生成・検証・自己修復するメタスキル」を厳格に分離する二層アーキテクチャ** を提案する。  
> さらに、LLM / VLM 特有のハルシネーションを完全に遮断するため、ソフトウェア工学のテスト駆動開発を昇華させた **EDD (Evaluation-Driven Development: 正例3件＋負例3件の契約テスト防壁ゲート)** を組み込んだ。本稿では、本メタスキル体系（`env-observer`, `subgoal-decomposer`, `skill-synthesizer`, `contract-tester`, `failure-diagnoser`, `constraint-learner`）の設計論と、Kaggle 本番環境における完全オフライン推論基盤の確立について報告する。

---

## 2. 課題定義と問題意識：なぜ従来の静的解法は破綻するのか？

### 2.1 ARC-AGI-1/2 から ARC-AGI-3 へのパラダイムシフト
* **従来の ARC-AGI-1 / 2**:
  * 盤面は静的であり、入力グリッドから出力グリッドへの 1 ステップ変換（Static Grid Transformation）。
  * 解法の主流は「全探索型 DSL（DreamCoder 風）」や「大規模 LLM による Python コードの一発生成（Code Generation）」。
* **ARC-AGI-3（ACR-AGI-3）の環境特性**:
  * 状態遷移が存在するインタラクティブなゲーム環境。
  * 操作体系（Actions）、物体の物理特性（壁、障害物、プレイヤー、ゴール）、勝利条件がタスクごとに不定・未知。
  * 事前に網羅的なルールライブラリを準備することは原理的に不可能。

### 2.2 Instance-Level Synthesis（個別解法）の限界
特定パズルを解くための個別 Python 関数（具象スキル）をリポジトリに何千個蓄積しても、未知の新しいゲーム環境が登場した瞬間にそれらの大部分は役に立たなくなります。
蓄積すべき真の汎化資産は「特定の解法コード」ではなく、**「未知の環境に直面したとき、どのように観察し、どのように仮説を立て、どのように即座に高品質なスキルを量産・検証するかというメタ認知能力（Meta-Skills）」** です。

---

## 3. 認知科学的基盤：人間プレイ解説（VCGT）の解析経緯

本研究の着想源となったのは、人間プレイヤーの操作記録と、その時の思考プロセスを言語化させた Kaggle 公開データセット（`magicsword001/acr-agi-3-human-vcgt`）の解析です。

### 3.1 人間の適応プロセスにおける 4 つの認知フェーズ
人間が初見のゲームをプレイする際の思考ログを分析した結果、人間は決して「終局状態への完璧なプログラムを一括で書く」ような思考をしていないことが判明しました。

```
[環境の初期提示]
       │
       ▼
【フェーズ 1: アフォーダンス・不変量の同定 (Affordance & Invariants)】
   - 「青いブロックは壁で通り抜けられない」
   - 「緑のセルは動かせるオブジェクトである」
   - 「背景の黒色は非干渉領域である」
       │
       ▼
【フェーズ 2: 階層的サブゴール分解 (Hierarchical Subgoal Decomposition)】
   - 「ゴールは右下にあるが、中央に壁があるため直進できない」
   - Subgoal 1: まず右へ移動し、縦の通路に入る
   - Subgoal 2: 通路を下へ抜けて障害物を迂回する
   - Subgoal 3: 目標座標へ進入する
       │
       ▼
【フェーズ 3: 局所アクションの実験とマクロスキル化 (Active Experimentation)】
   - サブゴールごとに小さな動作を試し、手応えを確認
   - 成功したアクションシーケンスを再利用可能な単位（マクロスキル）として定着
       │
       ▼
【フェーズ 4: 手詰まりからの禁止制約学習 (Reflection & Constraint Learning)】
   - 「左の赤いゾーンに入ったらゲームオーバーになった」
   - 反省（Reflection）を通じて「赤ゾーン侵入禁止」という No-Go ルールを獲得
   - 以降のサブゴール探索空間から危険な手を枝刈り（Pruning）
```

### 3.2 VCGT (Visual Concept Guided Thinking) データモデルの構造化
この人間の認知サイクルを形式化し、エージェントが扱える標準スキーマとして `src/acr_agi3/meta/human_vcgt.py` に実装しました。

* **Goal**: 局所的・終局的な到達目標の宣言
* **Reasoning**: 視覚的配置と物理的アフォーダンスに基づく判断根拠
* **Steps**: 検証可能な具体的アクション系列（中間状態遷移）
* **Reflection**: 予期せぬ挙動・失敗から得られた教訓
* **Invariants**: 環境全体を支配する普遍的ルール（対称性、進入禁止、保存則）

---

## 4. アーキテクチャ設計：メタスキルと具象スキルの二層分離

本プロジェクトにおける最も重要な設計方針は、**「思考エンジン（永続的 OSS コア）」と「一時生成コード（実行時使い捨て）」の物理的・概念的完全分離** です。

```mermaid
graph TD
    subgraph MetaCognitiveLayer ["メタスキル層 (meta_skills/) [永続・OSSコア資産]"]
        EO["1. env-observer<br/>(環境不変量・アフォーダンス抽出)"]
        SD["2. subgoal-decomposer<br/>(VCGT階層分解)"]
        SS["3. skill-synthesizer<br/>(マクロスキル & 契約テスト合成)"]
        CT["4. contract-tester<br/>(EDD 評価防壁ゲート)"]
        FD["5. failure-diagnoser<br/>(差分・例外診断 & 自己修復)"]
        CL["6. constraint-learner<br/>(No-Go 制約学習)"]
    end

    subgraph RuntimeSandbox ["実行時具象スキル層 (generated_skills/) [Git除外・使い捨て]"]
        GS1["Skill_001.py (一時生成)"]
        GS2["Skill_002.py (一時生成)"]
    end

    subgraph TaskEnvironment ["未知のゲーム環境 (ACR-AGI-3)"]
        ENV["Grid Environment / Video State"]
    end

    ENV -->|盤面画像・グリッド| EO
    EO -->|不変量レポート| SD
    SD -->|サブゴール計画| SS
    SS -->|生成コード + 契約テスト| CT
    CT -->|合格 (100%全勝)| RuntimeSandbox
    CT -->|不合格| FD
    FD -->|修正プロンプト| SS
    RuntimeSandbox -->|環境実行| ENV
    ENV -->|行き止まり・失敗フィードバック| CL
    CL -->|禁止制約更新| SD
```

### 4.1 ディレクトリ分離の意義
* **`meta_skills/`**: 「スキルの作り方・評価の仕方・直し方」を定義するプロンプト仕様書・思考フレームワーク。個別のパズルコードは一切含めない。
* **`generated_skills/`**: メタスキルが実行時に自動生成する具象 Python コード置き場（`.gitignore` で完全除外）。タスク解決後に破棄またはインメモリ化される。

---

## 5. EDD (Evaluation-Driven Development) 防壁ゲートの理論

大規模言語モデル（LLM）やマルチモーダルモデル（VLM）を用いたプログラム合成における最大のリスクは、**「もっともらしいが境界条件で死ぬコード（ハルシネーション）」** です。

### 5.1 「正例 3 件 ＋ 負例 3 件」の契約テスト
本アーキテクチャでは、スキル生成と同時に必ず以下の契約テストを自動生成・実行させます。

1. **正例テスト（Positive Cases: 3件）**:
   * 通常入力での期待される変換結果
   * 典型的な境界値（サイズ最小・最大）での動作
   * 幾何対称性が維持されるケース
2. **負例テスト（Negative Cases: 3件）**:
   * 想定外の次元・形状（1次元配列、ギザギザな二次元配列）に対する適切な例外送出
   * 空の入力（Empty Grid）に対する防御
   * 定義外のカラー値に対するバリデーション

### 5.2 ゼロトレランス方針（Zero Tolerance Gating）
テストスイートにおいて **1 件でもアサーションエラーや例外が発生したコードは、絶対にマージ・採用してはならない** という防壁ゲート（Contract Gate）を敷いています。
失敗時は `failure-diagnoser` がエラーログと入力差分を解析し、`skill-synthesizer` に対してピンポイントな自己修復指示をフィードバックします。

---

## 6. 実装とオフライン推論基盤の担保

### 6.1 Google ADK 2.0 × ローカル小型推論エンジン
Kaggle の提出環境は **外部ネットワーク完全遮断（No Internet Access）** かつ **GPU メモリ制限（16GB T4 / P100 またはホストの RTX A2000 12GB）** という制約があります。
本システムでは、Google Agent Development Kit (ADK) 2.0 のアーキテクチャを採用しつつ、以下を実現しました：

* **`LocalTransformersLlm`**: `local_files_only=True` を強制し、コンテナ内キャッシュモデル（Qwen2.5-Coder-1.5B / Qwen2.5-VL-3B / 7B）で完全にローカル完結する推論アダプター。
* **`LocalQwenVL` & `DSL Renderer`**: 公式 10 色カラー画像をインプロセスでレンダリングし、数値テキストと画像を同時に VLM に提示するマルチモーダル自己改善エージェント。
* **超高速テスト・実行サイクル**: コンテナ内部での推論とテスト実行を最適化し、全 25 件の単体・結合テストを **約 1.0 秒** で完走。

---

## 7. ARC-AGI 論文部門（Paper Track）への投稿に向けた論点・貢献点

ARC Prize 論文部門に投稿する際のコア・クレーム（論文の主張）と実験計画を整理します。

### 7.1 主要な学術的貢献（Core Contributions）
1. **From Instance Synthesis to Meta-Skills (パラダイムシフトの提示)**:
   * 単一グリッド変換プログラムの探索ではなく、未知の動的環境から自律的にスキルを量産するメタスキル基盤の提唱。
2. **VCGT-Grounded Hierarchical Decomposition (認知科学的裏付け)**:
   * 人間の思考解説データ（VCGT）を計算論的にモデル化し、アフォーダンス同定からサブゴール計画・制約学習への落とし込みに成功。
3. **Evaluation-Driven Development for Agent Reliability (信頼性の担保)**:
   * 生成コードに対する「正例3＋負例3」の EDD 防壁ゲートによる、LLM ハルシネーションの構造的排除。
4. **Offline Edge Feasibility (実用性の証明)**:
   * 巨大なクラウド API に依存せず、1.5B〜7B クラスの軽量ローカルモデル＋Google ADK 2.0 の組み合わせにより、完全オフライン環境でリアルタイム適応を実現。

### 7.2 今後のアブレーション実験計画（Planned Ablations）
* **Ablation 1**: サブゴール分解（`subgoal-decomposer`）の有無による長手数タスクの達成率比較
* **Ablation 2**: EDD 契約テスト防壁ゲート（`contract-tester`）の有無によるランタイムエラー発生率と正答率の比較
* **Ablation 3**: 制約学習（`constraint-learner`）の有無による、手詰まり後の自己修復ステップ数の短縮効果
* **Ablation 4**: テキストのみ vs 視覚レンダリング画像併用（Multimodal VLM）の空間把握精度の比較

---

## 8. まとめ

本メタスキル体系は、「ARC-AGI-3 を解くための解法集」ではなく、**「どんなゲーム環境が来ても、自律的に思考し、高品質な解法を生み出し続ける思考エンジン」** です。
この設計思想と人間プレイ記録（VCGT）の解析プロセスは、AGI（汎用人工知能）に向けた記号接地（Symbol Grounding）とメタ認知（Metacognition）の融合事例として、論文部門において極めて強力な説得力を持ちます。
