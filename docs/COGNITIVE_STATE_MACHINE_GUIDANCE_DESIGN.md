# 🧠 Cognitive State Machine Guidance & Autonomous Transition Design
## 5-Mode 認知ステートマシンにおける動的ガイダンス設計と自律遷移アーキテクチャ

本ドキュメントは、ARC-AGI-3 自律エージェント（`DeliberativeGamePlayer`）において、**なぜステートマシンの状態・目的・遷移条件を LLM に明示的に伝達するアーキテクチャが必要だったのか**、その設計思想・課題の根本原因・改善内容・実機シミュレーションによる検証結果を体系的にまとめたものです。

新しいセッションで作業する AI エージェント、および本オープンソースリポジトリを参照する開発者が、アーキテクチャの根幹思想を直ちに理解し、一貫した方針で機能拡張できるようにすることを目的としています。

---

## 1. 背景と課題の所在 (Why: なぜこの設計が必要だったのか)

### 1.1 ARC-AGI-3 と ARC-1/2 の決定的な違い
ARC-AGI-3 は、従来の ARC-1 / ARC-2 のような静的な入出力グリッド変換（`def transform(grid) -> grid`）ではありません。
**「未知の動的ゲーム環境において、状態観測からアフォーダンスを認識し、行動（Action: UP, DOWN, LEFT, RIGHT, CLICK 等）によってステージクリアを目指すインタラクティブなゲームプレイ」** です。

エージェントには、静的パターンマッチングではなく、**「未知の環境に対して仮説を立て、最小限の介入実験を行い、その因果関係を検証してルール化し、目標に向けて計画・実行する」** という科学的探究プロセス（仮説検証型認知ループ）が不可欠となります。

### 1.2 LLM（特に小型ローカル VLM）が直面した認知的ボトルネック
本プロジェクトでは Kaggle 完全オフライン環境（インターネット遮断・GPU 制限）を前提としており、エージェントはローカルの **Qwen2.5-VL-3B-Instruct** で動作します。
しかし、ステートマシンとツールを与えた初期段階において、モデルは以下のような認知的迷走や暴走を起こしました：

1. **PLAN モードでの「直感的早合点アクション」**:
   - 画面を観察してゴール（例: 右上の黄色いマス）を発見した瞬間、因果関係（どのボタンでキャラクターがどう動くか）が全く未学習であるにもかかわらず、直感でいきなり `step_action` を呼ぼうとする。
2. **CAUSAL モードでの「知識獲得の錯覚」**:
   - 不足している知識を特定して `need_causal_knowledge(question="...")` を呼んだ直後、モデル自身は「質問を投げた＝知識を得た」と錯覚し、未解決のまま `plan_actions`（計画立案）を呼んでしまう。
3. **REVIEW モードでの「実験検証のスキップ」**:
   - アクションを実行した直後の `REVIEW` モードで、画面の前後の差分を観察した直後、仮説が支持されたか否かの判定（`assess_result`）を行わずに、直接因果ルールの登録（`resolve_question`）へ飛ぼうとする。
4. **Tool Not Found によるパニックと無限リトライ**:
   - 現在の思考モードで許可されていないツールを呼んだ際、モデルはなぜ呼べないのかを理解できず、同じツール呼び出しを無限にリトライして推論トークンや API コール上限（24回）を浪費してしまう。

### 1.3 なぜ「裏側での自動モード昇格パッチ」は厳禁なのか？
開発初期において、「モデルが PLAN 中にアクションを呼んだら、裏側で自動的に EXECUTE モードに昇格させて通してしまえば動くのではないか？」というアドホックな発想が生じることがあります。
しかし、これは **プロジェクトの根幹方針に反する最大の悪手** です：
- **因果防壁の崩壊**: ルールも仮説もないまま動かすのは「ただの当てずっぽう（ランダムウォーク）」であり、失敗した際に何が原因だったのか診断・学習できなくなります。
- **認知モデルの破綻**: モデルの内的な思考状態と、裏側で勝手に書き換わった環境状態が乖離し、次ターン以降の推論が完全に破綻します。
- **EDD 原則違反**: 契約テスト（35件）で規定された「正例・負例に対する因果的保証」を裏切ることになります。

したがって、目指すべき本質的な解決策は、**「裏側の小細工を一切排除し、モデル自身が『今の自分の状態では何をすべきで、何が不足していればどちらに進むべきか』を過不足なく理解し、自律的に正規の遷移ツールを呼び出すアーキテクチャ」** の構築でした。

---

## 2. 設計思想 (Design Philosophy)

本アーキテクチャは、**Google ADK 2.0（Agent Development Kit）公式の Progressive Disclosure（段階的開示）** と、**5-Mode Cognitive State Machine（認知ステートマシン）** の完全な融合に基づいています。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      5-Mode Cognitive State Machine                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐   rules known    ┌───────────────┐                   │
│   │  1. [PLAN]   ├─────────────────►│ 4. [EXECUTE]  │                   │
│   └───┬──────────┘                  └───────┬───────┘                   │
│       │ rules unknown                       │ step_action / click_at    │
│       │ (need_causal_knowledge)             ▼                           │
│       ▼                             ┌───────────────┐                   │
│   ┌──────────────┐                  │  5. [REVIEW]  │◄── action done    │
│   │ 2. [CAUSAL]  │                  └───────┬───────┘                   │
│   └───┬──────────┘                          │ assess_result             │
│       │ need_experiment                     ▼                           │
│       ▼                             ┌───────────────┐                   │
│   ┌──────────────┐  step_action     │ 2. [CAUSAL]   │                   │
│   │3.[EXPERIMENT]├─────────────────►│   (resolve)   ├──► [PLAN] へ戻る  │
│   └──────────────┘                  └───────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 認知ステートマシンの5つの役割と境界

| モード (ThoughtMode) | 認知的目的 (Purpose) | 許可ツール (Allowed Tools) | 次の遷移条件と進路 |
| :--- | :--- | :--- | :--- |
| **1. PLAN** | 画面の全体俯瞰、ゴールの設定、学習済みルールに基づく計画立案 | `set_goal`, `need_causal_knowledge`, `plan_actions`, `continue_plan`, `reset_game` | **ルール未学習**: `need_causal_knowledge` で CAUSAL へ<br>**ルール既知**: `plan_actions` で EXECUTE へ |
| **2. CAUSAL** | 未知の因果関係（疑問）の解消、仮説立案 | `need_causal_knowledge`, `need_experiment`, `resolve_question`, `answer_visible_question`, `use_known_rules`, `reset_game` | **画面で自明**: `answer_visible_question`<br>**実験が必要**: `need_experiment` で EXPERIMENT へ |
| **3. EXPERIMENT** | 仮説を検証するための最小介入アクション（1回）の実行 | `need_causal_knowledge`, `step_action`, `click_at` | `step_action` 実行後、環境が1ステップ進み自動で **REVIEW** へ遷移 |
| **4. EXECUTE** | 計画された行動ステップの順次実行 | `need_causal_knowledge`, `step_action`, `click_at` | `step_action` 実行後、環境が1ステップ進み自動で **REVIEW** へ遷移 |
| **5. REVIEW** | アクション実行前後のフレーム比較・仮説の合否検証 | `assess_result` | `assess_result` 実行後、**CAUSAL**（ルール確定）または **PLAN** へ戻る |

---

## 3. 実施された具体的な改善内容 (Implementation Details)

### 3.1 構造化地図と動的現在地ガイダンス（`_build_mode_guidance`）
システムプロンプト（`instruction`）に全体地図を記載するだけでなく、**毎ターン・毎推論時における「現在地」「目的」「分岐条件」「次に呼ぶべき具体的ツール名」** を動的に構築して提示する仕組みを実装しました。

```python
def _build_mode_guidance(self) -> str:
    observed = self.screen.current_id in self.screen.viewed
    mode = self.state.mode

    if mode == ThoughtMode.PLAN:
        if not observed:
            return (
                "📍 Current State: [PLAN] (Screen Unobserved)\n"
                "🎯 Goal: Observe the board to inspect layout, entities, and colors.\n"
                "👉 Next Action: Call load_skill('visual-inspector') and observe_screen(view='current')."
            )
        if not self.state.rules:
            return (
                "📍 Current State: [PLAN] (Screen Observed, Zero Causal Rules Known)\n"
                "🎯 Goal: Since how actions affect the game is unknown, you cannot plan yet. You must investigate causality.\n"
                "👉 Next Action: Call need_causal_knowledge(question='Which action moves the piece or interacts with targets?') to enter CAUSAL mode.\n"
                "⚠️ Notice: You have 0 learned rules. 'plan_actions' and 'step_action' are NOT available yet."
            )
        return (
            "📍 Current State: [PLAN] (Rules Available)\n"
            f"📚 Learned Rules: {list(self.state.rules.keys())}\n"
            "🎯 Goal: Formulate an execution plan using your learned rules.\n"
            "👉 Next Action: Call plan_actions(subgoal=..., steps=[...], rule_ids=[...]) to advance to EXECUTE mode.\n"
            "⚠️ Notice: Environment actions (step_action/click_at) are NOT available directly in PLAN mode."
        )

    elif mode == ThoughtMode.CAUSAL:
        active_q = self.state.questions[-1].question if self.state.questions else "unknown causal relation"
        if not observed:
            return (
                f"📍 Current State: [CAUSAL] (Screen Unobserved)\n"
                f"❓ Open Question: {active_q}\n"
                "🎯 Goal: Observe the screen to check for visible answers.\n"
                "👉 Next Action: Call observe_screen(view='current')."
            )
        return (
            f"📍 Current State: [CAUSAL] (Screen Observed)\n"
            f"❓ Open Question: {active_q}\n"
            "🎯 Goal: Design a minimal experiment to test how an action works.\n"
            "👉 Next Action: Call need_experiment(hypothesis='Action 1 moves the piece', prediction='Piece moves one cell', alternative='Piece does not move') to enter EXPERIMENT mode.\n"
            "⚠️ Notice: Direct actions and plan_actions are NOT available in CAUSAL mode."
        )

    elif mode == ThoughtMode.EXPERIMENT:
        exp = self.state.experiment or {}
        hyp = exp.get("hypothesis", "active hypothesis")
        return (
            f"📍 Current State: [EXPERIMENT]\n"
            f"🧪 Active Hypothesis: {hyp}\n"
            "🎯 Goal: Execute the single test action for this experiment.\n"
            "👉 Next Action: Call load_skill('game-controller') and step_action(action_id=..., reasoning='...') or click_at(x=..., y=..., reasoning='...').\n"
            "   (If this hypothesis is no longer viable, call need_causal_knowledge to propose a different question)."
        )

    elif mode == ThoughtMode.REVIEW:
        if not observed:
            return (
                "📍 Current State: [REVIEW] (Screen Unobserved)\n"
                "🎯 Goal: Observe the consequence of the last action.\n"
                "👉 Next Action: Call observe_screen(view='both') to inspect changes before and after."
            )
        before_id = (self.state.pending or {}).get("frame_id", max(1, self.screen.current_id - 1))
        after_id = self.screen.current_id
        return (
            "📍 Current State: [REVIEW] (Screen Observed)\n"
            "🎯 Goal: Assess whether the observed outcome supports or refutes the hypothesis.\n"
            f"👉 Next Action: Call assess_result(outcome='supported', evidence='Observed movement/change', before_frame_id={before_id}, after_frame_id={after_id}).\n"
            "⚠️ Notice: 'resolve_question' is NOT available in REVIEW mode. You MUST call assess_result first!"
        )
```

### 3.2 ツール実行結果へのガイダンスリアルタイム注入（`_wrap_snapshot`）
Google ADK 2.0 の `Runner.run_async` は、モデルがツールを呼ぶと、ツール結果（`<tool_response>`）を即座にモデルに返して内部ループを継続します。
従来の設計では、新しいユーザープロンプトが挟まれない限りモデルは `_build_mode_guidance` を受け取ることができませんでした。

そこで、思考ツールおよび画面観察ツールの戻り値すべてを `_wrap_snapshot` でラップし、**ツールの実行結果 JSON の中に最新の `mode_guidance` を直接埋め込む** アーキテクチャを導入しました：

```python
def _wrap_snapshot(self, res: dict) -> dict:
    """ツールの実行結果に最新の Cognitive Guidance を埋め込み、モデルが次の行動を自律理解できるようにする."""
    if isinstance(res, dict):
        res["mode_guidance"] = self._build_mode_guidance()
    return res
```

これにより、モデルはどんなツールを呼んだ後でも、次のターンで何を行うべきかをリアルタイムに認識できるようになりました。

### 3.3 `observe_screen` 整形時のテキスト消失バグの解消
`local_vlm.py` において、マルチモーダル画像（Base64）を展開・クレンジングする際、辞書を `response = {"screen_observation": ...}` と新しく再生成していたため、せっかく付加された `mode_guidance` などのテキスト情報がすべて消去されていました。
これを `cleaned_response = dict(response)` と既存フィールドを全量保持するよう修正しました。

### 3.4 Progressive Disclosure 防護と多重ロード変換ループの抑止
未開放ツールが呼ばれた際に `load_skill` を代理発行する Progressive Disclosure 防護において、**「スキルが未ロードなのか、それとも現在のモードで不許可なのか」** を区別する仕組みを追加しました。
会話履歴（`llm_request.contents`）から `loaded_skills` を走査し、既にロード済みのスキルに対しては再変換を行わず、ADK 本来のエラーメッセージ（Available tools 一覧）をモデルへ直接返すことで、無限リトライループを根絶しました。

### 3.5 推論トークン長（`max_new_tokens`）の拡張
`max_new_tokens=128` では、モデルの簡潔な思考（CoT）の直後に出力される `<tool_call>` JSON が途中で切断（Truncated）され、パーサーが構文エラーを起こしていました。これを `max_new_tokens=512` へ拡張し、安定したツール呼び出しを実現しました。

---

## 4. 実機シミュレーションによる検証結果 (Verification Evidence)

公式リーダーボード評価スクリプト（`scripts/run_local_leaderboard.py`）を用い、難関公式ゲーム環境 `tu93` において実機動作を検証しました。

### 4.1 単体テスト（契約テスト 35件）
```bash
docker exec arc-agi3-dev pytest tests/test_deliberative_player.py tests/test_qwen_tool_calling.py
======================== 35 passed, 3 warnings in 1.39s ========================
```
- 全 35 件の契約テスト（モード遷移、Tool 制限、Progressive Disclosure、オフラインロード、マルチモーダル推論）が **1.39 秒で 100% PASS**。

### 4.2 公式リーダーボード実機シミュレーション（`tu93`, `-s 2`）
実機環境において、2ステップのアクション実行を **エラーゼロ・無限ループゼロで完全完走** しました。

```text
=== Step 00: 未知環境における仮説立案と実験実行 ===
[LocalQwenVL] Native Tool Call: load_skill({'skill_name': 'visual-inspector'})
[LocalQwenVL] Native Tool Call: observe_screen({'view': 'current'})
              -> ガイダンス提示:「学習済みルール 0件。計画不可。因果関係を調査せよ」
[LocalQwenVL] Native Tool Call: load_skill({'skill_name': 'causal-deliberation'})
[LocalQwenVL] Native Tool Call: need_causal_knowledge({'question': 'What is the winning condition?'})
              -> CAUSAL モードへ突入
[LocalQwenVL] Native Tool Call: need_experiment({'hypothesis': 'The switch at (10, 15) moves the piece', ...})
              -> EXPERIMENT モードへ突入
[LocalQwenVL] Native Tool Call: load_skill({'skill_name': 'game-controller'})
[LocalQwenVL] Native Tool Call: step_action({'action_id': 1, 'reasoning': 'Test whether this button moves the piece'})
              -> [Step 00] ACTION1 🔮[game-controller] | Eff: ✅ | ΔPixels: 1 (環境が前進！)

=== Step 01: 結果検証と次なる実験サイクルの自律実行 ===
              -> 新フレーム受信により自動で REVIEW モードへ突入
[LocalQwenVL] Native Tool Call: load_skill({'skill_name': 'visual-inspector'})
[LocalQwenVL] Native Tool Call: observe_screen({'view': 'both'})
              -> ガイダンス提示:「画面観察完了。before_frame_id=1, after_frame_id=2 で assess_result を呼べ」
[LocalQwenVL] Native Tool Call: load_skill({'skill_name': 'causal-deliberation'})
[LocalQwenVL] Native Tool Call: assess_result({'outcome': 'supported', 'before_frame_id': 1, 'after_frame_id': 2, ...})
              -> 仮説支持を確認し CAUSAL モードへ復帰
[LocalQwenVL] Native Tool Call: need_experiment({'hypothesis': 'Action 1 moves the piece', ...})
              -> 次の実験を立案し EXPERIMENT モードへ突入
[LocalQwenVL] Native Tool Call: load_skill({'skill_name': 'game-controller'})
[LocalQwenVL] Native Tool Call: step_action({'action_id': 1, 'reasoning': 'Test whether this button moves the piece left'})
              -> [Step 01] ACTION1 🔮[game-controller] | Eff: ✅ | ΔPixels: 2 (2回目のアクション成功！)

Result -> ❌ FAIL | tu93-0768757b (TU93) | Levels: 0/9 | Steps: 2 | Eff: 100.0% | ExplorationTimeout
```

指定した `-s 2`（最大2ステップ）の上限まで、1つのエラーも淀みもなく、ステートマシンの定義通りの完全な認知サイクルが実機上で自律的に実行されました。

---

## 5. 今後の開発・拡張ガイドライン (Guidelines for AI Agents & Developers)

今後本リポジトリで作業するすべてのエージェントおよび開発者は、以下の原則を厳守してください：

1. **暗黙のモード自動昇格パッチを記述しないこと**:
   - モデルが呼べないツールを呼んだ場合、裏側のコードでモードを勝手に書き換えて通してはなりません。必ずガイダンス（`_build_mode_guidance`）またはツールの戻り値を通じて、モデル自身に「今何が足りないか」を認識させてください。
2. **ツールのレスポンスには必ず認知ガイダンスを含めること**:
   - 新規に思考系ツールを追加する場合は、戻り値を `self._wrap_snapshot(...)` でラップし、モデルが直後に次の手を打てるようにしてください。
3. **Google ADK Progressive Disclosure の階層を守ること**:
   - スキルを不用意に常駐化させず、Level 1（カタログ）→ Level 2（指示本文）→ Level 3（実行ツール）の開示手順を遵守してください。
4. **契約テストの事前実行**:
   - コードを変更した際は、必ず以下のコマンドで 35 件の契約テストが 100% 通ることを確認してください：
     ```bash
     docker exec arc-agi3-dev pytest tests/test_deliberative_player.py tests/test_qwen_tool_calling.py
     ```

---

## 🔗 関連設計書
- **Qwen-ADK アダプター層設計書**: [`docs/QWEN_ADK_ADAPTER_WRAPPER_DESIGN.md`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/docs/QWEN_ADK_ADAPTER_WRAPPER_DESIGN.md)
  *(Google ADK 2.0 の仕様を維持しながら Qwen2.5-VL を適応させるラッパー層の設計工夫)*
- **Progressive Disclosure アーキテクチャ**: [`docs/PROGRESSIVE_DISCLOSURE_ARCHITECTURE.md`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/docs/PROGRESSIVE_DISCLOSURE_ARCHITECTURE.md)
- **AI エージェント運用ガイドライン**: [`AGENTS.md`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/AGENTS.md)

