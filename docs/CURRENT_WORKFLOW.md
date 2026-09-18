# ARC-AGI-3 一新された処理フロー仕様書 (Current Workflow)

本ドキュメントは、複雑な多分岐ルーティングや旧世代コードを全廃し、**「観測スキル・ツール（visual-inspector / ObservationTools）」＋「Google ADK 2.0 ReAct 反復思考（Planner Agent）」＋「出力スキル（game-controller）」** に一新された最新の自律ゲームプレイ処理フローをまとめたものです。

---

## 1. 全体構造マップ (ASCII ＆ フロー)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 視覚知覚・観測基盤: meta_skills/visual-inspector & ObservationTools       │
│    ├─ [統合画面] 公式10色カラー盤面 ＋ コントローラーHUD (前回押下ボタン💡発光)│
│    ├─ [客観事実] 盤面サイズ (H x W)、色一覧、Δ変化ピクセル数、有効キー一覧    │
│    └─ [観測ツール] Google ADK 2.0 FunctionTool (思考中にオンデマンド呼出):   │
│         • inspect_board: 大域幾何、色分布、ゲシュタルト分類、ゲームスタイル    │
│         • inspect_affordances: プレイヤー候補、ゴール、障害物、インタラクティブ│
│         • inspect_action_effect: 直前手による変位、効果判定、動的操作力学    │
│         • inspect_roi: 注目領域 (ROI) の拡大・局所色パレット検査             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. 思考: Google ADK 2.0 ReAct 反復思考ループ (Self-Iterative Tool-Use Loop)  │
│                                                                             │
│    [ Node 1: perceive_node ]                                                │
│      └─ 画像キャンバスと客観事実を CognitiveState に読み込み                  │
│                        │                                                    │
│                        ▼                                                    │
│    [ Node 2: plan_node (Planner Agent: ReAct 思考 ＆ ツール反復ループ) ]      │
│      ├─ 思考中の自律的ツール呼び出し (ReAct Loop):                           │
│      │   ① 状況分析 (Hypothesis)                                            │
│      │   ② 疑問・仮説検証 ➔ 【Tool Call】: inspect_affordances / ROI 等     │
│      │   ③ 観測結果受領 ➔ 【Tool Response】を思考履歴に反映して再考 (Rethink) │
│      │   ④ 目標策定 (Goal) ➔ 最小手決定 (Action)                            │
│      ├─ 将来の拡張性 (Google ADK 2.0 公式 SkillToolset):                     │
│      │   Level 1 目録 ➔ Level 2 SKILL.md ➔ Level 3 スクリプトをオンデマンド展開│
│      └─ 確定出力: PlanProposal { hypothesis, goal, action, reasoning, coords }│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. 出力スキル: meta_skills/game-controller (決定論的 1手発行)                │
│                                                                             │
│    [ Node 3: act_node (GameController) ]                                    │
│      ├─ 動的操作力学の解決 (resolve_action_id)                               │
│      │   └─ LLMの「UP」指示を、同定済みの物理キーIDへ自動変換 (例: ACTION3)    │
│      ├─ 幾何アフォーダンス吸着 (click_at)                                    │
│      │   └─ クリック座標が省略・空セルの場合、有色オブジェクトの重心へ自動補正│
│      └─ 出力: 100% 確実に環境へ発行可能な ActionDecision {action_id, coords}│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. 環境実行 ＆ フィードバックループ (Environment Execution)                  │
│    ├─ env.step(action) を実行                                               │
│    ├─ 完了判定:                                                             │
│    │   ├─ WIN (クリア) / GAME_OVER ──► [ 終了 ]                             │
│    │   └─ 継続 ──────────────────────► [ Step + 1 として 1. 観測基盤へ戻る ]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mermaid フロー図

```mermaid
flowchart TD
    subgraph S1["1. 視覚知覚・観測基盤: meta_skills/visual-inspector & ObservationTools"]
        START([ゲーム開始 / 各ステップ]) --> CONSOLE["統合コンソール画像の生成\n・上部: 公式10色カラー盤面\n・下部: コントローラーHUD (前回ボタンが発光💡)"]
        START --> FACTS["純粋な客観的事実の抽出\n(H x W, パレット, Δ変化ピクセル数, 有効キー)"]
        START --> TOOLS["Google ADK 2.0 観測ツール群\n・inspect_board\n・inspect_affordances\n・inspect_action_effect\n・inspect_roi"]
    end

    subgraph S2["2. 思考: Google ADK 2.0 ReAct 反復思考ループ (plan_node)"]
        CONSOLE & FACTS --> N1["Node 1: perceive_node\n(画像とコンソール状態の読み込み)"]
        N1 --> N2["Node 2: plan_node (Planner Agent)"]
        
        subgraph REACT["Planner Agent 自律 ReAct ループ"]
            THOUGHT["思考 (Hypothesis & Goal 検討)"]
            THOUGHT -- "盤面やアフォーダンスを再確認したい" --> TC["Tool Call 発行\n(inspect_affordances / ROI)"]
            TC --> TOOLS
            TOOLS -- "観測データ返却" --> TR["Tool Response 統合"]
            TR --> THOUGHT
            THOUGHT -- "確信が得られた" --> DECIDE["1手計画の確定\nPlanProposal {action, coords, reasoning}"]
        end
        
        N2 --> THOUGHT
    end

    subgraph S3["3. 出力スキル: meta_skills/game-controller"]
        DECIDE --> N3["Node 3: act_node (GameController)\n・動的操作力学の解決 (UP -> ACTION3 等)\n・幾何アフォーダンス吸着 (クリック座標オートスナップ)\n出力: ActionDecision {action_id, coordinates}"]
    end

    subgraph S4["4. 環境実行 ＆ フィードバック"]
        N3 --> STEP["env.step(action)\n(ゲーム環境へ1手発行)"]
        STEP --> CHECK{"クリア or 終了?"}
        CHECK -- "WIN" --> FINISH([クリア完了])
        CHECK -- "継続" --> START
    end

    classDef input fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    classDef cog fill:#fdf4ff,stroke:#c026d3,stroke-width:2px;
    classDef react fill:#faf5ff,stroke:#9333ea,stroke-width:1.5px,stroke-dasharray: 5 5;
    classDef act fill:#f0fdf4,stroke:#16a34a,stroke-width:2px;
    classDef env fill:#fffbeb,stroke:#d97706,stroke-width:2px;

    class S1 input;
    class S2 cog;
    class REACT react;
    class S3 act;
    class S4 env;
```

---

## 3. なぜこの設計が強力なのか？（以前の複雑な設計との違い）

| 比較項目 | 以前の設計（多分岐・レビュー・仮説進化） | 一新された新設計（本構成: ReAct + ADK 2.0） |
| :--- | :--- | :--- |
| **画面認識** | プログラムが勝手に自機やゴールを決め打ち推測（誤認が多発） | **プログラムによる決めつけを全廃**。公式10色カラー画像＋HUDをそのままLLMに見せて判断 |
| **思考フロー** | 5分岐ルーティング ＋ Reviewer（堂々巡りのリジェクトループでタイムアウト） | **Google ADK 2.0 ReAct 反復思考ループ**。エージェントが必要に応じて観測ツールを自発的に呼び出し、観測を取り込んで1手を確定 |
| **観測ツール** | 一括プロンプト注入（無関係な情報でコンテキストが肥大化） | **オンデマンドな `ObservationTools`**（`inspect_board`, `inspect_affordances`, `inspect_action_effect`, `inspect_roi`）をツール呼出でピンポイント取得 |
| **スキルの拡張性** | 新スキルが増えるたびにプロンプトがパンク | **Google ADK 2.0 公式 `SkillToolset` 準拠**。Level 1（目録）$\rightarrow$ Level 2（`SKILL.md`）$\rightarrow$ Level 3（スクリプト）の段階的開示でスケーラブル |
| **操作出力** | キーの対応関係がゲームごとに異なるARCで空振り | `game-controller` が**動的操作力学とクリック座標吸着を解決し、100% 確実に1手を発行** |
| **推論速度** | 1手あたり 40〜90秒（Kaggleで時間切れ） | **1手あたり 10〜15秒** で高速かつ柔軟に思考ループを回転 |

---

> **画像ファイル**: 上記フローをグラフィカルに可視化した高解像度 PNG 画像が `docs/architecture_workflow.png` に生成されています。
