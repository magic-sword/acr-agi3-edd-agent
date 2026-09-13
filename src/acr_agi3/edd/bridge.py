"""EDD メタスキル連携ブリッジ (Bridge to FailureDiagnoser & Meta-Skills).

診断レポート (DiagnosticReport) を FailureDiagnoser やプロンプト合成器が
直接消費できる形式へ橋渡しする。
"""

from __future__ import annotations

from typing import Any, Dict

from acr_agi3.edd.analyzer import DiagnosticReport


class EDDMetaSkillBridge:
    """診断レポートをメタスキル修復入力へ変換するブリッジ."""

    @staticmethod
    def to_diagnoser_input(report: DiagnosticReport) -> Dict[str, Any]:
        """FailureDiagnoser にそのまま渡せる辞書を生成."""
        return {
            "category": report.dominant_failure_category,
            "root_cause": (
                f"Game {report.game_id} ({report.title}): Effective Action Ratio = {report.effective_ratio*100:.1f}%, "
                f"Max Stagnation = {report.max_consecutive_stagnation} steps, "
                f"Clicks = {report.click_stats.get('total_clicks', 0)} (hit rate = {report.click_stats.get('effective_rate', 0)*100:.1f}%)."
            ),
            "directive": report.primary_recommendation,
            "severity": "HIGH" if report.effective_ratio < 0.2 else "MEDIUM",
            "telemetry_summary": {
                "effective_ratio": report.effective_ratio,
                "max_stagnation": report.max_consecutive_stagnation,
                "action_stats": report.action_stats,
            },
        }
