"""EDD (Evaluation-Driven Development) 診断・計測パッケージ.

エージェントの行動、環境の状態遷移、および因果関係をステップ単位で追跡し、
メタスキル (FailureDiagnoser, EnvObserver) へ構造化された診断フィードバックを提供する。
"""

from __future__ import annotations

from acr_agi3.edd.analyzer import DiagnosticAnalyzer, DiagnosticReport
from acr_agi3.edd.bridge import EDDMetaSkillBridge
from acr_agi3.edd.report import EDDReportFormatter
from acr_agi3.edd.telemetry import SessionTelemetry, StepTelemetry

__all__ = [
    "StepTelemetry",
    "SessionTelemetry",
    "DiagnosticAnalyzer",
    "DiagnosticReport",
    "EDDMetaSkillBridge",
    "EDDReportFormatter",
]
