---
name: skill-synthesizer
description: 観察されたゲームアフォーダンスやサブゴールから、契約テスト（正例3+負例3）付きの SKILL.md 仕様書とゲーム行動ポリシー（choose_action）を自動執筆するメタスキル。
version: 2.0.0
inputs:
  - name: affordances
    type: dict
    description: env-observer から得られたアフォーダンス（agent, obstacles, goals）
  - name: subgoal
    type: dict
    description: 達成すべき中間マイルストーン情報（目標座標、経由地点、アイテム等）
outputs:
  - name: skill_md
    type: str
    description: 生成された SKILL.md 仕様書（Markdown + YAML frontmatter）
  - name: policy_code
    type: str
    description: 決定論的行動選択関数 choose_action(obs, info) -> Action の Python 実装
  - name: test_code
    type: str
    description: 正例3件 + 負例3件のゲーム契約テストコード
---

# Skill Synthesizer Meta-Skill

## 概要
観察されたゲームアフォーダンスとサブゴールを元に、新しいゲーム行動スキルを「量産」するメタスキル工場（Skill Factory）です。
場当たり的なハードコードではなく、EDD (Evaluation-Driven Development) に完全に準拠した「自己完結した行動ポリシーモジュール」を自動執筆します。

## スキル生成の必須要件
1. **仕様ファースト (Markdown-First)**:
   - 必ず `SKILL.md` を生成し、行動インターフェースとアルゴリズムの前提条件（障害物色、目標色）を明記する。
2. **行動インターフェース**:
   - `def choose_action(obs: np.ndarray, info: dict | None = None) -> Action:` の標準インターフェースに従うこと。
   - `Action` は `Action.UP`, `Action.DOWN`, `Action.LEFT`, `Action.RIGHT`, `Action.INTERACT`, `Action.WAIT` のいずれかを返す。
3. **厳格な契約テスト (Contract Tests)**:
   - **正例（Positive Cases） 3件**:
     1. 目標が直線上にある場合、最短直線方向へ前進する。
     2. 進行方向に壁がある場合、壁を迂回する垂直方向へ舵を切る。
     3. 目標に隣接した場合、目標セルへ進入またはインタラクトする。
   - **負例（Negative Cases） 3件**:
     1. 壁（Obstacle）に向かって突進・衝突しない。
     2. 危険物・トラップ（Hazard）に向かって進入しない。
     3. グリッド境界外への移動や例外（IndexError）を発生させない。

