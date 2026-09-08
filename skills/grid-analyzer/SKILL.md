---
name: grid-analyzer
description: ARC-AGI グリッドの形状、色ヒストグラム、幾何学的対称性（水平・垂直・対角）を静的に解析する決定論的スキル
version: 1.0.0
author: magic-sword
tier: 1
tools:
  - scripts/analyze.py
---

# Grid Analyzer Skill

## 概要
入力された ARC グリッド（2次元配列）を静的に解析し、推論エージェントがパターンや変換ルールを導出するための基礎特徴量を抽出します。

## 入力仕様
- `grid`: 2次元整数配列（0〜9の値、最大 30x30）

## 出力仕様 (JSON)
- `shape`: [行数, 列数]
- `num_colors`: ユニークな色の総数
- `colors`: 含まれる色一覧（昇順）
- `color_counts`: 各色の画素数辞書
- `symmetry`: 水平 (`horizontal`), 垂直 (`vertical`), 対角 (`diagonal`) の対称性フラグ (bool)

## 実行方法
```bash
python scripts/analyze.py --input "[[1, 2], [2, 1]]"
```
