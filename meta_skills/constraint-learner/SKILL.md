---
name: constraint-learner
description: 失敗した操作ログや手詰まり状態から、不可逆なデッドエンド（詰み）を回避するための禁止制約を自動学習するメタスキル。
version: 1.0.0
inputs:
  - name: failed_trajectories
    type: list[dict]
    description: 失敗・手詰まりに終わった操作シーケンスのログ
  - name: terminal_state
    type: numpy.ndarray
    description: 手詰まり・ゲームオーバーとなったグリッド状態
outputs:
  - name: deadend_constraints
    type: list[dict]
    description: 探索時に避けるべき禁止条件のリスト (例: コーナー押し込み禁止)
  - name: pruning_rules
    type: list[str]
    description: プランナーやコード生成器に与える枝刈りルール
---

# Constraint Learner Meta-Skill

## 概要
人間プレイヤーが「一度ブロックを角に押し込んで詰んだら、二度と同じミスをしない」という学習能力を模倣したメタスキルです。
失敗の経験から「やってはいけない状態（Dead-end States）」を抽出し、以降の推論やスキル生成における探索空間を劇的に削減します。

## 主な学習対象制約 (Dead-End Categories)
1. **不可逆な接触 (Irreversible Adhesion / Corner Trap)**:
   - オブジェクトが引く手段のない壁の角（Corner）に押し込まれ、自由度が 0 になった状態。
2. **色の混合・消失 (Irreversible Color Blend)**:
   - 回収すべき色が背景色に塗りつぶされて情報が消失した状態。
3. **境界突破 (Out-of-Grid Escape)**:
   - 画面外に移動して再取得不能になった状態。

## 成果物の受け渡し
抽出された禁止制約（`pruning_rules`）は、`subgoal-decomposer` の経路計画および `skill-synthesizer` の契約テスト（負例）に直ちに反映されます。
