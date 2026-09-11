---
name: failure-diagnoser
description: 契約テスト失敗の原因を構造化解析し、スキル自動合成メタスキルへの修正フィードバックを生成する自己修復メタスキル。
version: 1.0.0
inputs:
  - name: test_report
    type: dict
    description: contract-tester からの失敗情報
outputs:
  - name: diagnostic_summary
    type: str
    description: エラー原因の要約（形状の不一致、境界値参照、色の混濁など）
  - name: repair_guidance
    type: str
    description: skill-synthesizer への修正指示
---

# Failure Diagnoser Meta-Skill (Evolver)

## 概要
テストに失敗したスキルを分析し、自律的な自己修復（Self-Repair Loop）をガイドするメタスキルです。

## 主な診断パターン
1. **形状不一致 (Shape Mismatch)**:
   - 例: 予測 `(3, 3)` に対し期待 `(9, 9)` → タイリング（`np.tile`）または拡大倍率（`np.repeat`）の修正指示。
2. **境界値外参照 (IndexError / Out of Bounds)**:
   - グリッド端でのループ上限判定やクリッピング処理（`np.clip`）の追加指示。
3. **色の塗り残し・混濁 (Color Mismatch)**:
   - 連結成分の走査アルゴリズム（8近傍 vs 4近傍）の変更指示。
