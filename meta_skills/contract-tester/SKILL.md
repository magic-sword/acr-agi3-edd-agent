---
name: contract-tester
description: 新規生成されたスキルに対して契約テストを実行し、全勝判定とストレステストを行う EDD 防壁ゲートメタスキル。
version: 1.0.0
inputs:
  - name: skill_path
    type: str
    description: 評価対象スキルのディレクトリパス (generated_skills/task_xxx/)
outputs:
  - name: passed_gate
    type: bool
    description: 契約テスト全勝 (100% Pass) かどうか
  - name: test_report
    type: dict
    description: テスト実行結果、カバレッジ、失敗詳細
---

# Contract Tester Meta-Skill (EDD Gatekeeper)

## 概要
生成されたスキルが本番環境で安全に動作するかを厳密に判定する「防壁ゲート（Gating）」メタスキルです。

## 防壁ゲートの判定ルール
1. **全勝必須原則**:
   - 正例 3 件、負例 3 件のすべてのテストケースを 1 件の失敗もなくパスすること。
   - 1 件でも例外や形状不一致があれば、即座に不合格判定とし、推論パイプラインへの投入を拒否する。
2. **実行速度制約**:
   - 1 回の変換が 500ms 以内に完了すること（計算量爆発の検知）。
3. **合格時のアクション**:
   - 合格したスキルのみを推論実行エンジンにロードし、Test 入力に対する予測を許可する。
4. **不合格時のアクション**:
   - エラー詳細を `failure-diagnoser` メタスキルに転送し、自己修復ループを開始する。
