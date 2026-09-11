---
name: env-observer
description: 未知の ARC-AGI-3 ゲーム環境から、不変量・幾何対称性・因果関係を抽出するメタスキル。
version: 1.0.0
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
  - name: transformation_category
    type: str
    description: 幾何学的カテゴリ（rotation, reflection, scaling, flood_fill, translation 等）
---

# Environment Observer Meta-Skill

## 概要
未知の環境・新しいゲームに直面したとき、最初に行うべき「観察と因果推論」を司るメタスキルです。
個別のパズルを解くのではなく、「何が変化し、何が保存されているか（Invariants）」を抽出します。

## 抽出対象の不変量 (Invariants)
1. **空間不変量**:
   - 入力と出力の形状（H, W）の変化比率（同一、倍率拡大、タイリング、トリミング）。
2. **色彩不変量**:
   - 最多頻度色（通常は背景色）の維持または変化。
   - 新規出現色の有無、消失色の有無。
3. **物体・位相不変量**:
   - 連結成分の個数が保存されているか（物体の移動/回転）。
   - 閉領域が形成されているか（塗りつぶし/包含関係）。

## 成果物の受け渡し
抽出された不変量と変換カテゴリは、`skill-synthesizer` メタスキルに渡され、新しい `SKILL.md` の設計指針となります。
