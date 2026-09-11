---
name: failure-diagnoser
description: 契約テストやゲームシミュレーション失敗（壁衝突スタック、トラップ接触、振動ループ）の原因を構造化解析し、行動ポリシー自己修復の修正指示を生成するメタスキル。
version: 2.0.0
inputs:
  - name: test_report
    type: dict
    description: contract-tester または環境シミュレーションからの失敗情報
outputs:
  - name: diagnostic_summary
    type: str
    description: 失敗原因の要約（壁衝突スタック、トラップ突入、振動ループ、ステップ超過）
  - name: repair_guidance
    type: str
    description: skill-synthesizer への修正指示（Waypoint 迂回経路設定、回避マージン確保など）
---

# Failure Diagnoser Meta-Skill (Evolver)

## 概要
テストまたはゲームプレイで失敗した行動ポリシーを分析し、自律的な自己修復（Self-Repair Loop）をガイドするメタスキルです。

## 主な診断パターンと修復指示
1. **壁・障害物衝突スタック (Wall Collision / Obstacle Deadlock)**:
   - 症状: 目標に向かう直線経路上に壁があり、その場から動けなくなる（衝突回数増加）。
   - 修復指示: 壁の法線方向（直交方向）に一時的な迂回目標（Waypoint）を設定し、壁のエッジを回り込むロジックを追加。
2. **危険ゾーン・トラップ接触 (Hazard / Trap Collision)**:
   - 症状: 危険物セルに隣接した際、回避できずに接触してゲームオーバー。
   - 修復指示: 危険物周辺 1 マスを「進入禁止コスト領域（Forbidden Zone）」としてマークし、行動選択から除外。
3. **振動ループ・デッドロック (Oscillation / Infinite Loop)**:
   - 症状: 直近の 2〜3 セル間を行ったり来たりし、ステップ数を浪費。
   - 修復指示: 直近の訪問座標履歴（Visited History）をポリシーに保持し、未訪問の隣接セルを優先選択する。
4. **ステップ数超過 (Timeout / Step Limit Exceeded)**:
   - 症状: 壁やトラップを避けるあまり、目標から遠ざかり続けて制限ステップに到達。
   - 修復指示: マンハッタン距離によるポテンシャル場（Heuristic Field）を強化し、目標方向への引き込み力を高める。

