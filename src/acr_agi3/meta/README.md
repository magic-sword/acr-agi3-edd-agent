# 🚨 重要：AI エージェント向け運用ガイドライン (src/acr_agi3/meta/)

本ディレクトリ (`src/acr_agi3/meta/`) は、**Google ADK 2.0 準拠のスキルハーネス基盤（`SkillHarness`）専用** です。

---

### ⚠️ 禁止事項

* **本ディレクトリ内に個別スキルやプランナーの実装コードを直接作成・配置してはなりません。**
* プランナー（`MetaSkillHarnessPlanner`）やエージェントロジックは [`src/acr_agi3/agent/`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/agent/) に配置してください。
* スキルの定義・手順書・スクリプトは [`meta_skills/<skill_name>/`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/meta_skills/) に配置してください。

---

### 📂 本ディレクトリに残されている正規ファイル

1. [`skill_harness.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/skill_harness.py):
   * `meta_skills/` 配下のフォルダ構造を走査し、Level 1 (Metadata) $\rightarrow$ Level 2 (SKILL.md) $\rightarrow$ Level 3 (scripts/) のオンデマンド展開と Google ADK 2.0 公式 `SkillToolset` を生成する唯一のコアハーネス基盤。
2. [`__init__.py`](file:///home/prog/work/kaggle/acr-agi3-edd-agent/src/acr_agi3/meta/__init__.py):
   * `SkillHarness`, `Skill` のみを公開する極小エントリポイント。
