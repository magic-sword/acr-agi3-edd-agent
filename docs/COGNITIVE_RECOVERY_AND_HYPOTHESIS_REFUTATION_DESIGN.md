# 認知リカバリー・反証仮説しおり管理・GPUガード設計書

本ドキュメントは、ARC-AGI-3 自律プレイエージェント（`acr-agi3-edd-agent`）における以下の設計課題と、それに対するアーキテクチャ修正の技術的理由（Rationale）および詳細仕様をまとめたものです。

1. **GPU Guard によるサイレント CPU フォールバックの防止 (Fail-Fast)**
2. **クリック専用環境における幾何 A*（`BACKWARD_ARCHITECT`）の誤作動防止**
3. **Progressive Disclosure 準拠の反証仮説（Refuted Hypotheses）しおり管理**
4. **エピソードリセット時の選択的記憶永続化 (Selective Memory Persistence)**

---

## 1. 背景と発見された課題 (Background & Issues)

### 1.1 長時間ハングとサイレント CPU フォールバック（14時間遅延）
* **現象**: ローカルオフラインリーダーボード実行時、推論が極端に遅延（1環境で14時間以上停止）。
* **原因分析**:
  - 長時間稼働する Docker コンテナ（`arc-agi3-dev`）内でホスト側のドライバ状態変化により NVML 接続が切断（`Failed to initialize NVML: Unknown Error`）。
  - この結果、`torch.cuda.is_available() == False` となり、ローカル VLM（`LocalQwenVL`）がサイレントに 16-core CPU 推論へとフォールバック。
  - GPU 推論時（1〜2秒/ステップ）に対し、CPU 推論時（35秒以上/ステップ）へと激甚な遅延が発生していた。
* **設計課題**: 競技環境およびローカル検証において、**GPU 未検知時に暗黙的に CPU で動き続けること（Silent Failure）を許容してはならない**。

### 1.2 クリック専用ゲーム（`s5i5`）における幾何 A* 誤作動と反復振動
* **現象**: 環境 `s5i5` において、エージェントが「`(0, 1)` へカーソル移動してクリック」→「無変化」→「`(5, 5)` へカーソル移動してクリック」という無意味な 2 点間を延々と往復し、パズル進行度 0 のままスタック。
* **原因分析**:
  - `s5i5` は利用可能アクションが **`ACTION6`（クリック）のみ** の環境であり、移動コマンド（`ACTION1`〜`ACTION4`: UP/DOWN/LEFT/RIGHT）は提供されていない。
  - しかし、汎用視覚解析器（`visual-inspector`）の色クラスタリングが盤面上の特定ブロックを幾何学的「スタート」および「ゴール」と誤認。
  - 幾何プランナーが「UP 方向への A* 経路」を出力したため、認知ステートマシンが誤って `BACKWARD_ARCHITECT` モードを選択。
  - 移動アクションが使えないにもかかわらず移動を前提としたプランニングが強制され、カーソルを交互に動かしてクリックするだけの振動ループに陥っていた。

### 1.3 失敗仮説（Refuted Hypotheses）の常時プロンプト注入によるローカル LLM の注意散漫
* **現象/懸念**:
  - 0 ピクセル変化で失敗した操作（無効なクリック座標や衝突）を記憶・学習して再発を防ぐ必要がある。
  - しかし、過去のすべての失敗・反証済み仮説・禁止事項テキストをプロンプトに常時注入すると、プロンプトのトークン数が膨れ上がり、ローカル小型モデル（Qwen2.5-VL-7B/3B）のアテンション（注意）が散漫化する。
  - その結果、本来最優先で認識すべき「現在の盤面状態（Observation）」や「ゲームコントローラーの利用可能アクション」の認識精度が劣化してしまう。

---

## 2. 設計修正方針と技術的理由 (Design Modifications & Rationales)

### 2.1 Fail-Fast GPU Guard
* **設計方針**:
  - `LocalQwenVL` および `run_local_leaderboard.py` において、`device="cuda"` が指定されている場合に `torch.cuda.is_available() == False` であれば、サイレントな CPU フォールバックを禁止し、直ちに明示的な `RuntimeError` を送出する。
  - 実行開始前に Preflight Check を行い、GPU が利用不可の場合は 0.01 秒以内に即時終了（Fail-Fast）させる（明示的な `--allow-cpu` オプション指定時のみ CPU 実行を許容）。
* **採用理由**:
  - 競技制約下において、何時間も無駄に CPU で待機させられる損失をゼロにし、コンテナ再起動（`docker restart`）によるドライバ再バインドを促すため。

### 2.2 アクションアフォーダンスに基づく幾何 A* ガード
* **設計方針**:
  - 移動アクション（`ACTION1`, `ACTION2`, `ACTION3`, `ACTION4`）が 1 つも含まれない環境（`is_click_only = True`）では、幾何 A* パスファインダーが経路を検出しても `BACKWARD_ARCHITECT` への遷移をブロック。
  - 強制的に `CAUSAL_PROGRAMMER`（因果プログラム探索）または `TABOO_RECOVERY`（禁忌回避）モードへ移行させる。
* **採用理由**:
  - 存在しない移動アクションを仮定したプランニングを構造的に排除し、クリック対象のアフォーダンス探索に集中させるため。

### 2.3 Google ADK 2.0 Progressive Disclosure 準拠の「しおり（TOC）」管理
* **設計方針**:
  - 0 ピクセル変化の行動が発生した際、その失敗仮説を Memory Notebook 内の `hypothesis.refuted.s{step}_{action}_{coords}` セクションへ退避・保存。
  - **プロンプトへの全量テキスト注入は厳禁**:
    - **Level 1 (しおり・目次)**: プロンプトには `memory_toc` の 1 行目次のみを表示。
      ```text
      - [hypothesis.refuted.s0_ACTION6_0_1] Refuted: ACTION6_0_1 (0 pixels changed) (tags: hypothesis, refuted, falsified)
      ```
    - **Level 2 / 3 (オンデマンドツール取得)**: エージェントが過去の失敗の詳細（なぜ無効だったのか）を知りたいときのみ、`memory_read(section_id=...)` ツールを能動的に呼び出して参照。
* **採用理由**:
  - プロンプトのトークン消費を最小化し、ローカル LLM の注意散漫（Negative Constraint による幻覚・萎縮）を防止しつつ、必要なときにツール経由で失敗知識にアクセスできるようにするため。

### 2.4 エピソードリセット時の選択的記憶永続化 (Selective Memory Persistence)
* **設計方針**:
  - ゲームオーバーやリセット（`ACTION0`）時、`MemoryTools.reset_episode()` を呼び出す。
  - この際、一時的な実行計画（`plan.active`）や作業中の仮説（`hypothesis.active`）はクリアするが、**`refuted`, `falsified`, `constraint`, `causality`, `rules` などの恒久知識タグを持つセクションは保持** する。
* **採用理由**:
  - リトライ時に前回の試行で反証された座標や操作を再実行する愚行を回避し、試行錯誤（Trial-and-Error）の学習効果を同一エピソード内で引き継ぐため。

---

## 3. アーキテクチャ図 (Cognitive Architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Observation (Visual Frame + Available Actions: [ACTION6])               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Node 1: Perceive Agent (Visual Inspection)                              │
│  - 画面構成、前景・背景色、カーソル座標の特定                           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Cognitive Mode Determination (Rule Engine)                              │
│  - Directional Actions (1..4) Missing? ───> BLOCKS BACKWARD_ARCHITECT   │
│  - Selects Mode: CAUSAL_PROGRAMMER / TABOO_RECOVERY                    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Node 2: Plan Agent                                                      │
│  - Prompt: Observation + Minimal TOC Bookmarks (Level 1)                │
│    * [hypothesis.refuted.s0_ACTION6_0_1] (tags: refuted, falsified)    │
│  - Optional Tool: memory_read() (Level 3 - On-demand only)             │
│  - Writes: hypothesis.active                                            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Node 3: Act Agent                                                       │
│  - Executes ACTION6 via game-controller                                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │ ΔPixels > 0?              │
                       ├─────────────┬─────────────┤
                      YES            NO (Failure)
                       │             │
                       │             ▼
                       │  ┌──────────────────────────────────────────────┐
                       │  │ Archive to hypothesis.refuted.s{step}_{act}  │
                       │  │ Delete hypothesis.active                     │
                       │  │ Register Taboo Coordinate & Transition to    │
                       │  │ TABOO_RECOVERY                               │
                       │  └──────────────────────────────────────────────┘
                       ▼
          Update last_grid & Continue Next Step
```

---

## 4. 変更対象ファイルと責務

| ファイル | 変更内容・責務 |
|---|---|
| [`src/acr_agi3/agent/llm/local_vlm.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/agent/llm/local_vlm.py) | GPU Guard（CUDA 不可時の `RuntimeError` 送出、`allow_cpu_fallback` 引数制御） |
| [`scripts/run_local_leaderboard.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/scripts/run_local_leaderboard.py) | 起動時 Preflight GPU Check、`--allow-cpu` フラグ追加 |
| [`src/acr_agi3/agent/adk_game_player.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/agent/adk_game_player.py) | クリック専用ゲームにおける A* 経路の無効化、0 ピクセル変化時の反証仮説アーカイブ、アクティブ仮説の記録 |
| [`src/acr_agi3/tools/memory_tools.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/tools/memory_tools.py) | `reset_episode()` における `hypothesis.refuted.*` 永続化と `hypothesis.active` 破棄の分離 |
| [`tests/test_local_vlm_gpu_guard.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/tests/test_local_vlm_gpu_guard.py) | GPU Guard および Preflight Check の単体テスト（3/3 passed） |
| [`tests/test_hypothesis_refutation_state_machine.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/tests/test_hypothesis_refutation_state_machine.py) | 反証仮説アーカイブ、リセット永続性、A* ブロックの単体テスト（3/3 passed） |

---

## 5. 検証結果 (Validation Summary)

1. **GPU Guard 単体テスト**:
   - `test_cuda_not_available_raises_runtime_error`: **PASSED**
   - `test_mock_mode_bypasses_cuda_check`: **PASSED**
   - `test_leaderboard_preflight_fails_when_cuda_missing`: **PASSED**
2. **ステートマシン単体テスト**:
   - `test_hypothesis_active_and_refuted_archiving`: **PASSED**
   - `test_refuted_hypotheses_preserved_on_reset`: **PASSED**
   - `test_click_only_game_blocks_backward_architect`: **PASSED**
3. **実機ゲーム環境（`s5i5` 5-step run on GPU）**:
   - 実行時間: 107 秒（GPU 正常稼働、従来の 14 時間ハング・CPU フォールバック解消）
   - モード遷移: `BACKWARD_ARCHITECT` の誤判定がなくなり、`CAUSAL_PROGRAMMER` として正しく探索が遂行された。
   - プロンプトの清潔度: プロンプトには目次（しおり）のみが渡され、注意散漫のない安定した推論を確認。
