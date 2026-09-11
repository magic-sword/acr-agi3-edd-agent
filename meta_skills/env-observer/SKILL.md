---
name: env-observer
description: 未知の ARC-AGI-3 ゲーム環境から、アフォーダンス（自機・壁・ターゲット・危険物）・不変量・因果力学（行動による状態遷移）を抽出するメタスキル。
version: 2.0.0
inputs:
  - name: observation
    type: numpy.ndarray
    description: 現在の環境観測グリッド配列 (H, W)
  - name: transition_history
    type: optional[list[dict]]
    description: 過去の行動遷移ログ [obs, action, next_obs, reward, done]
  - name: interaction_log
    type: optional[list[dict]]
    description: 人間 VCGT 思考解説ログまたはプレイ記録
outputs:
  - name: affordances
    type: dict
    description: アフォーダンス分類（agent, obstacles, goals, hazards, interactables）
  - name: invariants
    type: dict
    description: 保存される環境特性（通過不能な壁色、背景色、グリッド境界）
  - name: dynamics_rules
    type: list[str]
    description: 同定された因果規則（4近傍移動、壁衝突反発、キー取得によるドア解錠等）
---

# Environment Observer Meta-Skill

## 概要
未知の動的ゲーム環境に直面したとき、最初に行うべき「観察・アフォーダンス同定・因果推論」を司るメタスキルです。
静的な一括変換ではなく、**「何が操作可能な自機か」「何が通過不能な壁か」「どこがゴール/危険ゾーンか」** および **「行動によって環境がどう変化するか（Dynamics）」** を自律抽出します。

## 1. アフォーダンス同定 (Affordances - 人間 VCGT モデル準拠)
観測グリッドおよび操作ログから、構成要素を以下のゲーム役割に分類します：
* **Controlled Agent (自機/操作主体)**: 行動（UP, DOWN, LEFT, RIGHT 等）に連動して座標が変化する単一または複数のセル。
* **Static Obstacles (静的壁/障害物)**: 移動行動を行っても通過できず、自機の侵入を遮る境界セル。
* **Target / Goal (クリア目標)**: 自機が到達・接触することで報酬が得られ、ステージクリアとなる目標セル。
* **Hazards / Traps (危険物/ペナルティ)**: 接触するとゲームオーバーまたはペナルティが発生する回避対象セル。
* **Interactables (相互作用物/スイッチ/鍵)**: 接触することで他オブジェクト（扉や壁）の状態を変化させる要素。

## 2. 抽出対象の不変量と因果力学 (Invariants & Dynamics)
1. **空間・境界不変量**:
   - グリッド寸法（H, W）、背景色（通常最頻色）、画面端のループ有無。
2. **行動因果力学 (Action Causality)**:
   - 1行動あたりの移動量（通常 1 セル/ステップ）。
   - 壁に衝突した際の挙動（その場に留まる、または跳ね返る）。
   - アイテム接触時のトリガー効果（鍵取得で対応する色の扉が消失するなど）。

## 成果物の受け渡し
抽出されたアフォーダンス辞書と因果規則は、`subgoal-decomposer` および `skill-synthesizer` に渡され、安全で最短なゲームクリア行動ポリシーの生成に直結します。
