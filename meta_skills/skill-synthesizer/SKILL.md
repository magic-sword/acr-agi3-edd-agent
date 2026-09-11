---
name: skill-synthesizer
description: 観察された環境不変量から、契約テスト（正例3+負例3）付きの SKILL.md 仕様書と決定論的スクリプトを自動執筆するメタスキル。
version: 1.0.0
inputs:
  - name: invariants
    type: dict
    description: env-observer から得られた不変量・幾何変換情報
  - name: task_examples
    type: list[dict]
    description: 適合すべき入出力グリッド例
outputs:
  - name: skill_md
    type: str
    description: 生成された SKILL.md 仕様書（Markdown + YAML frontmatter）
  - name: script_code
    type: str
    description: 決定論的 Python 実装コード
  - name: test_code
    type: str
    description: 正例3件 + 負例3件の契約テストコード
---

# Skill Synthesizer Meta-Skill

## 概要
観察された幾何学的・論理的規則を元に、新しい具象スキルを「量産」するメタスキル工場（Skill Factory）です。
場当たり的なコード片ではなく、EDD (Evaluation Driven Development) に完全に準拠した「自己完結したスキルモジュール」を自動執筆します。

## スキル生成の必須要件
1. **仕様ファースト (Markdown-First)**:
   - 必ず `SKILL.md` を生成し、関数の入出力型とアルゴリズムの前提条件を明記する。
2. **厳格な契約テスト (Contract Tests)**:
   - **正例（Positive Cases） 3件**: 正常系での期待変換動作。
   - **負例（Negative Cases） 3件**: エッジケース（グリッド外はみ出し、単色グリッド、境界サイズ 0/1）に対する例外安全性のテスト。
3. **決定論的実装**:
   - ランダム要素を排し、高速かつ安定して動作する Python 関数 `def transform(grid: np.ndarray) -> np.ndarray:` を出力する。
