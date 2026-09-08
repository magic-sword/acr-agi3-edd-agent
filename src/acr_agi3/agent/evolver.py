"""EDD (Evaluation Driven Development) に基づくスキルの自己改善・診断モジュール."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticResult:
    """スキル評価失敗時の構造化診断情報."""

    skill_name: str
    failed_cases: List[Dict[str, Any]]
    suggested_fix: Optional[str] = None
    pass_rate: float = 0.0


class SkillEvolver:
    """スキルのテスト結果を分析し、自己修復ループを仲介するエボルバー."""

    def __init__(self) -> None:
        pass

    def diagnose_failure(
        self,
        skill_name: str,
        test_results: List[Dict[str, Any]],
    ) -> DiagnosticResult:
        """テスト実行結果から失敗ケースを抽出し、修正方針を提示する."""
        failed = [case for case in test_results if not case.get("is_correct", False)]
        total = len(test_results)
        pass_rate = (total - len(failed)) / total if total > 0 else 0.0

        suggestion = None
        if failed:
            suggestion = (
                f"スキル '{skill_name}' において {len(failed)} 件のケースが失敗しました。"
                "エッジケース（非正方グリッド、背景色の違い、多色混在）の境界条件を再検証してください。"
            )

        return DiagnosticResult(
            skill_name=skill_name,
            failed_cases=failed,
            suggested_fix=suggestion,
            pass_rate=pass_rate,
        )
