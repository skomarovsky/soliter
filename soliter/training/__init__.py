"""Training systems: sleep-wake loop and EWC."""

from .ewc import EWCLoss
from .sleep_wake import SleepWakeTrainer, TrainingConfig

__all__ = [
    'EWCLoss',
    'SleepWakeTrainer',
    'TrainingConfig',
]
