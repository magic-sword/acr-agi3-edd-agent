---
name: subgoal-decomposer
description: 人間 VCGT 思考構造に基づき、ゲームクリア目的（ゴール到達、鍵回収、ゲート解錠等）を独立して検証可能な中間マイルストーン (Subgoals) の列に階層分解するメタスキル。
version: 2.0.0
inputs:
  - name: observation
    type: numpy.ndarray
    description: 現在のゲーム環境観測グリッド配列 (H, W)
  - name: affordances
    type: dict
    description: env-observer から得られたアフォーダンス情報 (agent, obstacles, goals, items)
  - name: goal_description
    type: str
    description: 達成すべき最終ゲーム目的（例: 「鍵4を取得して扉5を開け、ゴール3に到達する」）
outputs:
  - name: subgoals
    type: list[dict]
    description: 順序付けられた中間目標のリスト (各マイルストーンの事前・終了条件・Waypoint付き)
  - name: task_hierarchy
    type: str
    description: VCGT 形式の思考分解ツリー (Goal -> Reasoning -> Steps -> Reflection)
---

# Subgoal Decomposer Meta-Skill

## 概要
人間プレイヤーの思考データセット (VCGT) で実証された、**「目的の階層的分解」** を司るメタスキルです。
複雑なゲーム環境をいきなり一度に解こうとせず、**独立して検証可能な中間マイルストーン（Subgoals）** に自律分解することで、探索空間の爆発を防ぎます。

## VCGT 階層構造
本メタスキルは、各サブゴールを以下の 4 層構造で定式化します：

1. **Objective (中間目標)**: 何を達成すべきか？
   - *例: 「操作ブロックをターゲットの列と同じX座標に揃える」*
2. **Reasoning Breakdown (推論・理由)**: なぜその目標が必要か？
   - *例: 「ターゲットに押し込むためには、先に直線上の射線に位置する必要があるため」*
3. **Preconditions (事前条件)**: 開始時点で満たすべき状態。
4. **Postconditions (終了条件)**: 達成されたとみなす検証条件（テストアサーション）。

## 成果物の受け渡し
分解された各サブゴールは、`skill-synthesizer` に渡され、各中間ステップを達成するためのマクロスキル（例: `align_horizontally`, `push_to_boundary`）がオンデマンド合成されます。
