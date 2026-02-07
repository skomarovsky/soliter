# soliter/memory/__init__.py
from .replay_buffer import EpistemicReplayBuffer, Transition, PruningStats
from .fisher_matrix import FisherInformationMatrix

# Backwards compatibility alias
ReplayBuffer = EpistemicReplayBuffer

__all__ = [
    'EpistemicReplayBuffer',
    'ReplayBuffer',  # alias
    'Transition',
    'PruningStats',
    'FisherInformationMatrix',
]