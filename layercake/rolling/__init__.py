"""Git-like rolling training primitives for LayerCake."""

from .rubric import TrainingRubric
from .commit import ModelCommit
from .registry import ModuleRegistry
from .gates import GateResult
from .trainer import RollingTrainer

__all__ = [
    "TrainingRubric",
    "ModelCommit",
    "ModuleRegistry",
    "GateResult",
    "RollingTrainer",
]
