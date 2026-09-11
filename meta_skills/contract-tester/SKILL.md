---
name: contract-tester
description: 新規生成されたゲーム行動ポリシーに対して契約テストとシミュレーション実行を行い、安全性と目標到達度を判定する EDD 防壁ゲートメタスキル。
version: 2.0.0
inputs:
  - name: skill_path
    type: str
    description: 評価対象スキルのディレクトリパス (generated_skills/subgoal_xxx/)
  - name: test_env
    type: GameEnvironment
    description: テスト実行用ゲーム環境（またはモックシミュレータ）
outputs:
  - name: passed_gate
    type: bool
    description: 契約テスト全勝 (100% Pass) かつシミュレーション安全基準達成かどうか
  - name: test_report
    type: dict
    description: テスト実行結果（正例/負例合否、衝突回数、到達ステップ数、失敗理由）
---

# Contract Tester Meta-Skill (EDD Gatekeeper)

## 概要
生成されたゲーム行動ポリシーが未知のゲーム環境で安全かつ確実に機能するかを判定する「防壁ゲート（Gating）」メタスキルです。

## 防壁ゲートの判定ルール
1. **全勝必須原則 (Contract Tests 100% Pass)**:
   - 正例 3 件（目標への前進、迂回、インタラクト）、負例 3 件（壁衝突ゼロ、トラップ回避、例外安全）のすべてに 1 件の不合格もなくパスすること。
   - 1 件でも例外送出や禁止行動（壁への激突、トラップ進入）があれば、即座に不合格判定とする。
2. **シミュレーション安全制約**:
   - 許容ステップ数（通常 50〜100 ステップ）以内に目的のサブゴール（目標到達・キー回収など）を達成すること。
   - 同一セルでの振動ループ（Oscillation Loop）やスタックが検知された場合は即座に不合格。
3. **合格時のアクション**:
   - 防壁ゲートを通過したスキルのみをメイン実行エージェントのポリシーセットに採用し、本番環境のプレイ実行を許可する。
4. **不合格時のアクション**:
   - 衝突座標やスタック原因の詳細を `failure-diagnoser` メタスキルに転送し、自己修復ループを開始する。

