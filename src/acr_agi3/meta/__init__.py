"""ARC-AGI-3 メタ認知オーケストレーションパッケージ (Meta-Cognitive Layer)."""

from acr_agi3.meta.decomposer import DecompositionPlan, Subgoal, SubgoalDecomposer
from acr_agi3.meta.human_vcgt import VCGTDataset, VCGTExplanation, VCGTRecord
from acr_agi3.meta.observer import AffordanceObject, GameAffordanceReport, MetaObserver

__all__ = [
    "MetaObserver",
    "GameAffordanceReport",
    "AffordanceObject",
    "SubgoalDecomposer",
    "Subgoal",
    "DecompositionPlan",
    "VCGTDataset",
    "VCGTRecord",
    "VCGTExplanation",
]

