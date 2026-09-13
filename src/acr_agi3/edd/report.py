"""EDD 診断レポートフォーマッター (Report Formatter).

コンソール表示および Markdown 出力用の整形を行う。
"""

from __future__ import annotations

from typing import List

from acr_agi3.edd.analyzer import DiagnosticReport


class EDDReportFormatter:
    """診断結果を人間可読な形式に整形するフォーマッター."""

    @staticmethod
    def format_console_summary(reports: List[DiagnosticReport]) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("🔬 [EDD DIAGNOSTIC TELEMETRY & ROOT-CAUSE ANALYSIS]")
        lines.append("=" * 78)
        lines.append(f"{'Game ID':<14} | {'Title':<6} | {'Eff Ratio':<10} | {'Max Stag':<9} | {'Failure Category':<26}")
        lines.append("-" * 78)

        for r in reports:
            eff_str = f"{r.effective_ratio*100:5.1f}%"
            lines.append(
                f"{r.game_id:<14} | {r.title:<6} | {eff_str:<10} | {r.max_consecutive_stagnation:4d} steps | {r.dominant_failure_category:<26}"
            )

        lines.append("-" * 78)
        lines.append("💡 Actionable Directives for Meta-Skills:")
        for r in reports[:5]:
            if r.dominant_failure_category != "COMPLETED":
                lines.append(f"  • [{r.game_id}]: {r.primary_recommendation}")

        lines.append("=" * 78)
        return "\n".join(lines)
