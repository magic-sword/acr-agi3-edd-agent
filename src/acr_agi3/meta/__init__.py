"""ARC-AGI-3 メタ認知オーケストレーションパッケージ (Meta-Cognitive Layer).

本パッケージは Google ADK 2.0 準拠の SkillHarness を中核とし、
すべてのメタスキルロジックは meta_skills/ 配下のフォルダ構造から透過的にロードされます。
"""

from __future__ import annotations

from acr_agi3.meta.gestalt_planner import GestaltVCGTPlanner, MetaSkillHarnessPlanner
from acr_agi3.meta.human_vcgt import VCGTDataset, VCGTExplanation, VCGTRecord
from acr_agi3.meta.skill_harness import SkillHarness, SkillMetadata

# meta_skills/ フォルダ構造から動的インポート解決 (Single Source of Truth)
_harness = SkillHarness()

_obs_mod = _harness.get_skill_module("env-observer")
_syn_mod = _harness.get_skill_module("skill-synthesizer")
_dec_mod = _harness.get_skill_module("subgoal-decomposer")
_diag_mod = _harness.get_skill_module("failure-diagnoser")
_int_mod = _harness.get_skill_module("game-style-intuitor")

MetaObserver = _obs_mod.MetaObserver
DynamicAffordanceReport = _obs_mod.DynamicAffordanceReport
VisualObject = _obs_mod.VisualObject
AffordanceObject = _obs_mod.AffordanceObject
GameAffordanceReport = _obs_mod.GameAffordanceReport

MetaSkillSynthesizer = _syn_mod.MetaSkillSynthesizer
BaseSkillPolicy = _syn_mod.BaseSkillPolicy
AffordanceNavigationSkill = _syn_mod.AffordanceNavigationSkill
InteractiveClickSkill = _syn_mod.InteractiveClickSkill
FrontierExplorationSkill = _syn_mod.FrontierExplorationSkill

SubgoalDecomposer = _dec_mod.SubgoalDecomposer
Subgoal = _dec_mod.Subgoal
DecompositionPlan = _dec_mod.DecompositionPlan

FailureDiagnoser = _diag_mod.FailureDiagnoser
GameStyleIntuitor = _int_mod.GameStyleIntuitor

__all__ = [
    "SkillHarness",
    "SkillMetadata",
    "MetaSkillHarnessPlanner",
    "GestaltVCGTPlanner",
    "MetaObserver",
    "DynamicAffordanceReport",
    "VisualObject",
    "AffordanceObject",
    "GameAffordanceReport",
    "MetaSkillSynthesizer",
    "BaseSkillPolicy",
    "AffordanceNavigationSkill",
    "InteractiveClickSkill",
    "FrontierExplorationSkill",
    "SubgoalDecomposer",
    "Subgoal",
    "DecompositionPlan",
    "FailureDiagnoser",
    "GameStyleIntuitor",
    "VCGTDataset",
    "VCGTRecord",
    "VCGTExplanation",
]
