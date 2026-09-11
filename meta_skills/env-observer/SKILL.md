---
name: env-observer
description: 未知の ARC-AGI-3 ゲーム環境から、不変量・幾何対称性・アフォーダンス（操作可能物/障害物）・因果力学を抽出するメタスキル。
version: 1.1.0
inputs:
  - name: train_pairs
    type: list[dict]
    description: 入出力グリッド配列のペアリスト
  - name: interaction_log
    type: optional[list[dict]]
    description: ゲーム操作ログまたは人間 VCGT 思考ログ
outputs:
  - name: invariants
    type: dict
    description: 変化しない属性（背景色、グリッド形状比率、固定アンカーなど）
  - name: affordances
    type: dict
    description: アフォーダンス分類（agent, interactable, obstacles, targets）
  - name: transformation_category
    type: str
    description: 幾何学的・力学的カテゴリ（translation, rotation, flood_fill, push_pull 等）
---

# Environment Observer Meta-Skill

## 概要
未知の環境・新しいゲームに直面したとき、最初に行うべき「観察と因果推論」を司るメタスキルです。
個別のパズルを解くのではなく、「何が変化し、何が保存されているか（Invariants）」および「何が操作可能で何が障害物か（Affordances）」を抽出します。

## 1. 抽出対象の不変量 (Invariants)
1. **空間不変量**:
   - 入力と出力の形状（H, W）の変化比率（同一、倍率拡大、タイリング、トリミング）。
2. **色彩不変量**:
   - 最多頻度色（通常は背景色）の維持または変化。
   - 新規出現色の有無、消失色の有無。
3. **物体・位相不変量**:
   - 連結成分の個数が保存されているか（物体の移動/回転）。
   - 閉領域が形成されているか（塗りつぶし/包含関係）。

## 2. アフォーダンス同定 (Affordances - 人間 VCGT モデル準拠)
操作ログや入出力差分から、グリッド上の構成要素を以下の役割に分類します：
* **Controlled Agent (自機/操作主体)**: プレイヤーの入力（矢印キー、クリック）に直接連動して位置や形状が変化する要素。
* **Interactable Objects (相互作用対象)**: 自機が接触・押す・引く・貫通することで位置や状態が変化するオブジェクト。
* **Static Obstacles (静的障害物/壁)**: 一切変化せず、他オブジェクトの通過を阻止する境界。
* **Target / Goals (目標地点/終了条件)**: 到達・配置することでゲームクリアや次のフェーズに進む領域。

## 成果物の受け渡し
抽出された不変量とアフォーダンス辞書は、`subgoal-decomposer` および `skill-synthesizer` メタスキルに渡され、階層的推論と新しい `SKILL.md` の設計指針となります。
