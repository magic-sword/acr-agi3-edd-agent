# スキル公開・思考状態の監査（2026-09-22）

## 結論

公式 `MyAgent → DeliberativeGamePlayer` は、Agent に実行関数を直接列挙していない。
`Agent(tools=[SkillToolset])` とし、関数は `SkillToolset.additional_tools` に登録している。
`SKILL.md` のロード・本文提供・追加ツールの有効化は公式 ADK が担当する。
今回、状態に応じたツール公開制限とローカルVLMへ渡す引数定義の欠落を修正した。

## 根拠と確認環境

ユーザー指定の Google Developer Knowledge API MCP に対し、`search_documents` と
`get_documents` を呼び、[Google公式 Skills for ADK agents](https://adk.dev/skills/) を取得した。
この文書は L1 メタデータ、L2 手順本文、L3 追加リソースという段階的開示と、
`SkillToolset(skills=..., additional_tools=...)` を Agent に渡す構成を説明している。

実行環境 `arc-agi3-dev` のインストール済み `google-adk` は **2.9.0**。
以下の詳細は、そのバージョンの公式実装を直接読み、実行テストで確認した。

- `LoadSkillTool` は本文を返すとともにセッション内でスキルを有効化する。
- `SkillToolset` は有効化したスキルの `metadata.adk_additional_tools` を解決する。
- 公式 `tool_filter` が公開対象をさらに絞る。今回ここに思考状態の条件を渡した。
- コード実行器を設定していない現行プレイヤーには `run_skill_script` は公開されない。
  実際のゲーム操作は、スキルに宣言されたホスト側実行ツールを使用する。

独自の SKILL.md パーサーや段階的ローダーは追加していない。
公式APIの挙動確認はこのインストール済みバージョンについてのものであり、
すべてのADK 2.xバージョンに同じ挙動を保証するものではない。

## 見つかった問題と修正

### 1. ロード済みツールの公開が思考状態に追従しなかった

従来はスキルをロードすると、そのスキルのツールがセッション内で公開され続けた。
例えば PLAN で `assess_result` の定義も見える状態だった。
関数内の状態チェックにより不正実行は拒否されるが、モデルの選択肢と定義量は増える。

`SkillHarness.get_scoped_toolset` から公式 `tool_filter` を透過的に渡し、
`DeliberativeGamePlayer` の現在の思考状態に応じて公開するよう修正した。
公開条件は「宣言元スキルが有効化済み」かつ「現在の状態で使用可能」である。
元の実行時チェックも残している。

### 2. ローカルVLM用の変換でADKツールの引数が欠落した

ADK のスキル管理ツールは `parameters_json_schema` を使用するが、
`LocalQwenVL._declaration_to_schema` は `parameters` しか参照していなかった。
このため `load_skill` の `skill_name` などが、モデルに空の引数定義として届いていた。
スクリプト化したモデルは正しい引数を事前に知っているため、従来の動作テストでは検出できなかった。

両方の公式スキーマ形式に対応し、必須引数・型・制約を維持してローカルVLMに渡すよう修正した。

## 現行スキルと公開状態

3スキルとも `SKILL.md` と `scripts/`、`tests/` を持つ。
追加の `references/`・`assets/` は必要に応じて使用する任意リソースである。

| 思考状態 | causal-deliberation の実行ツール | game-controller の実行ツール |
|---|---|---|
| PLAN | set_goal, need_causal_knowledge, plan_actions, continue_plan | reset_game |
| CAUSAL | need_causal_knowledge, need_experiment, resolve_question, answer_visible_question, use_known_rules | reset_game |
| EXPERIMENT | need_causal_knowledge | step_action, click_at |
| EXECUTE | need_causal_knowledge | step_action, click_at |
| REVIEW | assess_result | なし |

`visual-inspector` の `observe_screen` は全状態で使用可能。ただし先に当該スキルのロードが必要。
`list_skills`、`load_skill`、`load_skill_resource` は全状態で公開する。

すべてのスキルをロードした場合でも、モデルに渡すツール定義数は
PLAN=9、CAUSAL=10、EXPERIMENT=7、EXECUTE=7、REVIEW=5。
未ロード時はスキル管理の3個のみ。数はツール定義数であり、実トークン数の測定ではない。

Python 側でファイルを読み込んで保持することと、LLM のプロンプトに本文を入れることは別である。
初期プロンプトには3スキルの名称を案内する短い指示があるが、本文と実行ツール定義は入らない。
`list_skills` でメタデータを取得し、`load_skill` の結果で本文が履歴に入り、
次の推論に必要なツール定義が公開されることを、実際のVLM入力から確認した。

## 残る範囲・制約

- 現行ランタイムが利用するのは上記3スキル。`backward-planner`、`epistemic-prober`、
  `macro-skill-compiler`、`contract-tester` 等の既存スキルを全部ロードする構成ではない。
  計画・推論・探索・レビューの手順は現在 `causal-deliberation` にまとめてある。
- 従って、各思考状態に独立した SKILL.md を持つ構成ではない。今回の制限は公開ツールに適用される。
  読み込み済みの本文や過去のツール結果は、同じセッションの履歴に残る。
- Gateway の次の画面では新規 ADK セッションを作るため、前のセッションのスキル有効化は引き継がない。
  長大な会話履歴を持ち越さない一方、必要なスキルは再ロードする。
- 因果ルール等の思考データは引き継ぐ。知識の増加によるコンテキスト量は、今回のツール定義削減とは別の課題。
- 旧 `ADKGamePlayer` には `tools=[toolset] + additional_tools` という直接公開が残っている。
  これは現行 `MyAgent` の経路では使われないが、旧クラスを直接利用すればその問題が再現する。
- 開始・ゲームオーバー時の初期化リセットは、モデルのスキル選択を待たずホストが実行ツールを呼ぶ。
  戦略的リセットとは別のライフサイクル処理である。

## 検証

`tests/test_deliberative_player.py` に9ケース追加した。実際のADK RunnerからローカルVLMへ
渡されるプロンプトを捕捉して確認し、関数一覧の目視確認だけで判定していない。

- 全5状態で、未ロード時には本文・実行ツールを出さない。
- ロードしたスキルの本文と、その状態で使用できるツールだけを出す。
- 状態遷移の次の推論で公開ツールが更新される。
- `parameters_json_schema` の必須引数・制約を保持する。
- スキルカタログに手順本文を混ぜない。
- 次セッションへスキルの有効化が漏れない。

実行コマンド:

```sh
docker exec arc-agi3-dev python -m pytest tests/ \
  meta_skills/causal-deliberation/tests/test_contract.py \
  meta_skills/visual-inspector/tests/ meta_skills/game-controller/tests/ \
  --import-mode=importlib -q --disable-warnings --maxfail=2
```

結果: **184 passed, 19 warnings**。警告は残っており、「全検証で警告ゼロ」とはしない。
今回 SKILL.md は変更していない。実モデルの判断精度・トークン使用量・スコア改善は未測定。
