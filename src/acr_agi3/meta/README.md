# 🚨 重要：AI エージェント向け運用ガイドライン (src/acr_agi3/meta/)

本ディレクトリ (`src/acr_agi3/meta/`) は、**Google ADK 2.0 準拠のスキルハーネス（`SkillHarness`）およびオーケストレーター基盤専用** です。

---

### ⚠️ 禁止事項

* **本ディレクトリ内にメタスキルや具象スキルの実装コード（`observer.py`, `decomposer.py` 等）を直接作成・配置してはなりません。**
* スキルのロジックを本ディレクトリ内に直接書き込むと、Google ADK の 3段階 Progressive Disclosure（段階的開示）および EDD 検証ゲートが破壊され、ロジックの二重管理が発生します。

---

### ✅ 正当なスキル開発・配置場所

* **メタスキル（永続・思考エンジン）**:
  * すべて [`meta_skills/<skill_name>/`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/meta_skills/) 配下に配置してください。
  * 各スキルディレクトリは `SKILL.md`, `scripts/`, `references/`, `tests/` で自己充足し、MCP ツール `edd_validate_skill` でエラー 0 件・警告 0 件を維持する必要があります。
* **具象スキル（実行時生成・タスク解決用）**:
  * メタスキルが生成する個別ゲーム解法スキルは、すべて `generated_skills/<skill_name>/` 配下に配置してください（Git 管理対象外）。

---

### 📂 本ディレクトリに残されている正規ファイル

1. [`skill_harness.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/skill_harness.py):
   * `meta_skills/` 配下のフォルダ構造を走査し、Level 1 (Metadata) $\rightarrow$ Level 2 (SKILL.md) $\rightarrow$ Level 3 (scripts/) のオンデマンド展開と動的ロードを担当するコア基盤。
2. [`gestalt_planner.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/gestalt_planner.py):
   * `SkillHarness` を介して動的にスキルを運用するゲームプレイプランナー。
3. [`human_vcgt.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/human_vcgt.py):
   * 人間のプレイ記録と視覚的思考ログ（VCGT: Visual Concept Guided Thinking）のデータセット定義。
4. [`__init__.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/__init__.py):
   * `SkillHarness` 経由で `meta_skills/` から動的にクラスを解決し、外部コードへの後方互換性を提供する薄いエントリポイント。
