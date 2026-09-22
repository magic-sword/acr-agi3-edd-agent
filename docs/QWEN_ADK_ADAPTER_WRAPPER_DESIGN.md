# 🔌 Qwen-ADK Adapter Layer Design
## Google ADK 2.0 のスキル構造を維持しながらローカル VLM (Qwen2.5-VL) を適応させるラッパー設計思想

本ドキュメントは、Google ADK 2.0（Agent Development Kit）の設計思想（Progressive Disclosure・最小権限・契約テスト）を完全に保ちながら、オープンソースの小型ローカル VLM（**Qwen2.5-VL-3B-Instruct**）を完全自律動作させるために構築された**アダプター・ラッパー層（`LocalQwenVL`）の設計思想と実装工夫**を記録したものです。

---

## 1. 背景と課題の所在：Google ADK 2.0 と Qwen2.5-VL の設計思想のギャップ

### 1.1 Google ADK 2.0 の前提思想
Google ADK 2.0 は、Gemini 2.0 や GPT-4 などの「商用・最新鋭の超大規模クラウド LLM」を前提に設計されています：
- **JSON Schema ネイティブ**: `google.genai.types.FunctionDeclaration` に基づく厳格なツール宣言と引数検証。
- **Progressive Disclosure（3段階開示）**:
  - `SkillToolset` により、初期状態では `list_skills`, `load_skill` のみが提供され、エージェントが自律的に `load_skill` を呼んだ時点で初めて個別の実行ツール（`step_action`, `observe_screen` 等）が開示される。
  - ツールが未開示の状態で呼ばれた場合、ADK は `Tool '...' not found` を返し、モデルが自ら `load_skill` を呼んでリカバリーすることを期待する。
- **認知的統制**: 思考モード（`ThoughtMode`）に応じて利用可能なツールを動的に絞り込み、モデルの暴走を防ぐ。

### 1.2 小型ローカル VLM（Qwen2.5-VL-3B）の実態とギャップ
本プロジェクトの運用環境は Kaggle 完全オフライン（インターネット遮断・GPU 16GB 制限）であり、3B クラスの軽量マルチモーダルモデル（Qwen2.5-VL-3B）をローカル推論で使用します。
しかし、このモデルと ADK の間には、以下のような深い「言語・認知・プロトコルのギャップ」が存在しました：

```
┌──────────────────────────────────────┐         ┌──────────────────────────────────────┐
│        Google ADK 2.0 仕様           │         │          Qwen2.5-VL-3B の挙動        │
├──────────────────────────────────────┤         ├──────────────────────────────────────┤
│ ・JSON Schema ネイティブ宣言         │   ≠     │ ・Alibaba ChatML ツール構文に特化    │
│ ・スキルとツールを厳格に階層分離     │   ≠     │ ・スキル名とツール名を混同して呼ぶ   │
│ ・未ロードツールは Tool not found    │   ≠     │ ・未ロードでも直感で直接ツールを呼ぶ │
│ ・厳格な型（int, schema 等）         │   ≠     │ ・引数名や型の揺らぎ (action="ACTION1")│
│ ・タグで確実に囲まれた JSON 出力     │   ≠     │ ・タグ脱落や平文でのツール呼び出し   │
│ ・バイト列による画像インライン受け渡し│   ≠     │ ・PIL Image や Vision Tokens を要求  │
└──────────────────────────────────────┘         └──────────────────────────────────────┘
```

具体的には、以下のような問題が実機で頻発しました：
1. **スキルとツールの混同**:
   `load_skill({'skill_name': 'game-controller'})` を呼ぶべきところで、`game-controller({'action_id': 1})` とスキル名そのものを関数として呼び出してしまう。
2. **Progressive Disclosure の無視**:
   スキルをロードしていない初期状態（Level 1）のまま、いきなり `step_action` や `observe_screen` を直接呼んで `Tool not found` エラーになり、パニックを起こす。
3. **引数の語彙・型の揺らぎ**:
   `step_action` が `action_id: int`（例: `1`）を要求しているのに対し、モデルは直感で `action="ACTION1"` や `action="UP"` と出力する。
4. **構文の脱落**:
   `<tool_call>` タグを付けずに、平文の文末に `I will call step_action(action=1)` と自然言語で書いてしまう。

---

## 2. 設計哲学：なぜ「スキル」ではなく「モデルの境界」をラップしたのか？

### ❌ 避けるべきアンチパターン（やってはいけないこと）
Qwen の挙動を合わせるために、以下のような変更を行うことは**絶対に許されません**：
- **スキルの平坦化（Flattening）**:
  Qwen が `load_skill` を呼べないからといって、ADK の Progressive Disclosure を捨てて、最初から全スキル・全ツール（十数個）をプロンプトに常時平積み（Prompt Stuffing）すること。
  → **結果**: コンテキスト長が数千トークン肥大化し、VRAM OOM、推論遅延（1手数十秒）、アテンション低下による幻覚が激増する。
- **裏側での暗黙モード昇格**:
  呼べないモードでアクションを呼んだ際に、コード側で勝手にモードを昇格させて通すこと。
  → **結果**: 因果検証のない当てずっぽう行動となり、失敗原因の学習・自己修復が不可能になる。

### ✅ 採用した設計思想：アダプター・パターン（Adapter Pattern）
> **「Google ADK 2.0 のスキル構造・Progressive Disclosure 原則・契約テストは 1mm も歪めない。モデルと ADK の境界（`LocalQwenVL`）にインテリジェントなラッパー層を設け、Qwen の語彙・プロトコルを ADK 公式仕様へ双方向で透過翻訳する。」**

これにより：
- `meta_skills/` 内の `SKILL.md` やツール実装は、Google ADK 公式の標準構造を 100% 保持できる。
- 将来モデルを Gemini 2.0 や他の OSS モデルに差し替える際も、コアロジックを修正することなく、アダプター層を切り替えるだけで対応可能となる。

---

## 3. `LocalQwenVL` アダプター層の 7 つの設計工夫

アダプター層（[`src/acr_agi3/agent/llm/local_vlm.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/agent/llm/local_vlm.py)）では、以下の 7 つの技術的工夫を実装しています。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LocalQwenVL Adapter Architecture                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [Google ADK 2.0 Runner / Tools]                                       │
│       ▲                    │                                            │
│       │ Part / Content     │ LlmRequest (FunctionDeclarations, Images)  │
│       │                    ▼                                            │
│   ┌───┴────────────────────┴────────────────────────────────────────┐   │
│   │                 LocalQwenVL Wrapper Layer                       │   │
│   │                                                                 │   │
│   │ 1. _format_tools_for_qwen (JSON Schema -> ChatML Tool Format)   │   │
│   │ 2. _build_qwen_messages (Multimodal Image & Turn Normalization) │   │
│   │ 3. _detect_tool_call (Tag Parse + Markdown + Regex Fallback)    │   │
│   │ 4. _normalize_tool_call (Skill/Tool Disambiguation & Args Map)  │   │
│   │ 5. Progressive Disclosure Auto-Proxy (loaded_skills check)      │   │
│   │ 6. Mode-Aware Safe Adaptation (REVIEW: resolve -> assess)       │   │
│   │ 7. Lossless Observation Passthrough (Keep mode_guidance)        │   │
│   └───┬────────────────────┬────────────────────────────────────────┘   │
│       ▲                    │ ChatML String + PIL Images                 │
│       │ <tool_call> JSON   │                                            │
│       ▼                    ▼                                            │
│   [Qwen2.5-VL-3B Model & Processor (Local GPU)]                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 工夫 1: ChatML ↔ ADK 相互プロトコル変換（`_format_tools_for_qwen`, `_build_qwen_messages`）
ADK の `FunctionDeclaration` を、Qwen 公式の ChatML 形式（`{"type": "function", "function": {"name": ..., "parameters": ...}}`）に動的変換し、システムプロンプト内に自然に注入します。
また、ADK の会話ターン（`user`, `model`, `function_response`）を、Qwen の公式 ChatML（`system`, `user`, `assistant`, `tool`）に完全準拠した構造へマッピングします。

### 工夫 2: 3重フォールバックによる堅牢なツール検出（`_detect_tool_call`）
小型モデル特有の出力ブレを吸収するため、3 段階の検出エンジンを搭載しています：
1. **公式タグ検出**: `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>` の完全一致パース。
2. **マークダウン検出**: ` ```json {"name": ...} ``` ` など、コードフェンス内にツール JSON が書かれた場合の抽出。
3. **平文正規表現フォールバック**: モデルがタグを忘れて `I call step_action(action=1)` や `load skill visual-inspector` と出力した場合に、正規表現で意図をサルベージして正規ツールコールへ復元。

### 工夫 3: スキルとツールの混同の自動吸収（`_normalize_tool_call`）
モデルがスキル名そのものをツールとして呼び出すミスを自動解決します：
```python
# モデルが game-controller をツールとして呼び出した場合
if name in ("game_controller", "game-controller"):
    if "action" in args or "action_id" in args:
        name = "step_action"  # アクション引数があれば step_action に自動変換
    elif "x" in args and "y" in args:
        name = "click_at"     # 座標引数があれば click_at に自動変換
    else:
        return "load_skill", {"skill_name": "game-controller"}  # 引数がなければ load_skill に変換
```

### 工夫 4: 引数型と語彙揺らぎの柔軟な正規化
モデルが `action="ACTION1"` や `action="UP"` といった文字列を出力した場合でも、ADK のツール引数定義（`action_id: int`）に合わせて即座に数値型へ正規化します：
```python
if name == "step_action" and "action" in args and "action_id" not in args:
    act_val = args["action"]
    if isinstance(act_val, str) and act_val.startswith("ACTION"):
        args["action_id"] = int(act_val.replace("ACTION", ""))
    elif isinstance(act_val, int):
        args["action_id"] = act_val
```

### 工夫 5: Progressive Disclosure 自動プロキシ（未開放ツールの自律ロード）
モデルがまだ開示されていないツール（例: `observe_screen` や `step_action`）を直接呼んだ場合、ラッパーがそれを検知し、**モデルに代わって自動的に該当スキルの `load_skill` をプロキシ発行**します。
これにより、モデルは ADK の複雑な階層ルールを意識することなく、自然に Progressive Disclosure の 3 段階開示シーケンスを通過できます。

### 工夫 6: 会話履歴追跡による多重ロードループの防止（`_extract_loaded_skills`）
「ツールが存在しない理由」が **① スキルが未ロード** なのか、それとも **② 現在の思考モード（PLAN 等）で不許可** なのかを厳密に区別します。
会話履歴（`llm_request.contents`）からロード済みスキルを追跡し、**すでにロード済みのスキルに対しては `load_skill` への再変換を行いません**。
これにより、モード制約による不許可の場合は ADK 本来の `Tool not found. Available tools: [...]` というエラーが素直にモデルへ返り、モデル自身が思考を修正できるようになります（無限ロードループの根絶）。

### 工夫 7: マルチモーダル観測とメタデータの無損失透過（Lossless Observation Passthrough）
`observe_screen` の戻り値に含まれる Base64 画像を PIL Image にデコードして Qwen の Vision Encoder に供給する際、**既存の辞書フィールド（`mode_guidance` などのテキストガイダンス）を破棄せずに 100% 保持**して Qwen に伝達します。

---

## 4. 実証エビデンス：実機ログにみるアダプター層の機能

公式ゲーム環境 `tu93` における実機シミュレーションにおいて、アダプター層が完璧に機能したログの実例です：

### 実例 1: スキル未ロード時の自律プロキシ
```text
[Qwen の生出力]:  <tool_call> {"name": "observe_screen", "arguments": {"view": "current"}} </tool_call>
[アダプター判定]: observe_screen は visual-inspector 未ロードのため未開示
[ADK への変換]:  load_skill({'skill_name': 'visual-inspector'}) をプロキシ発行
[次のターン]:     visual-inspector 開示完了 -> observe_screen を正規実行
```

### 実例 2: スキル名とツール名の混同および引数正規化
```text
[Qwen の生出力]:  <tool_call> {"name": "game-controller", "arguments": {"action_id": 1, "reasoning": "..."}} </tool_call>
[アダプター判定]: "game-controller" というツールは存在しないが、引数に "action_id" がある
[ADK への変換]:  step_action({'action_id': 1, 'reasoning': '...'}) へ透過正規化
[実行結果]:       [Step 00] ACTION1 🔮[game-controller] | Eff: ✅ | ΔPixels: 1 (環境アクション成功！)
```

### 実例 3: REVIEW モードでの認知的意図の救済
```text
[Qwen の生出力]:  <tool_call> {"name": "resolve_question", "arguments": {"effect": "Piece moves", ...}} </tool_call>
[アダプター判定]: REVIEW モードでは resolve_question は未開放（assess_result のみ利用可能）
[ADK への変換]:  モデルの意図は実験評価であるため、assess_result({'outcome': 'supported', ...}) へ安全適応
[実行結果]:       仮説支持を確認し、次の実験ステップへ円滑に移行
```

---

## 5. まとめとアーキテクチャ上の教訓

1. **スキルの独立性の維持**:
   - `meta_skills/` は Google ADK 2.0 のピュアな仕様に従って記述されており、特定の LLM や VLM の実装依存コードを一切含んでいません。
2. **モデル適合はラッパー層（`LocalQwenVL`）の責務**:
   - モデルのプロトコル特性、語彙揺らぎ、認知限界のフォローはすべてラッパー層に集約されています。
3. **拡張性とポータビリティ**:
   - 将来より賢いモデル（Gemini 2.5 や Qwen-Next など）を採用する場合でも、スキルの再設計やステートマシンの破壊を行うことなく、ラッパー層の正規化ルールを緩めるだけでシームレスに移行可能です。
