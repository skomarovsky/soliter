"""
Replay Buffer with Replay-Driven Pruning.

Biological model (SHY + CLS):
    During NREM sleep the hippocampus replays memories to the neocortex.
    Memories that the cortex already "knows" (low replay surprise) have
    been consolidated — their hippocampal trace fades.  Memories that
    still surprise the cortex need further replay and are retained.

Implementation:
    During consolidation replay the trainer scores every transition by
    "replay surprise" — how wrong the current network is about that
    memory.  After replay, the buffer prunes the LEAST surprising
    memories (bottom percentile).  No fixed thresholds, no gates —
    pruning is a ranking problem, not a classification problem.

    The amount of information lost is naturally small: early in training
    even the bottom 20% are surprising (little is truly redundant).
    Late in training most memories are redundant and the bottom 20%
    safely reclaims space.

Author: Stan (Project Soliter)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Generator
from dataclasses import dataclass, field
import random
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Single experience transition with replay metadata."""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool

    # Metadata (set during wake, updated during sleep replay)
    tick: int = 0                       # World tick when recorded
    replay_surprise: float = 1.0        # How surprising during last replay [0, inf)
                                        # High = cortex doesn't know this yet
                                        # Low  = cortex has consolidated this

    # Legacy fields kept for backward compat / monitoring
    uncertainty: float = 1.0
    td_error: float = 1.0
    fisher_protection: float = 0.0


@dataclass
class PruningStats:
    """Statistics from a single replay-driven pruning operation."""
    buffer_before: int = 0
    buffer_after: int = 0
    pruned: int = 0
    surprise_min: float = 0.0           # Lowest surprise (most consolidated)
    surprise_max: float = 0.0           # Highest surprise (least consolidated)
    surprise_mean: float = 0.0          # Mean surprise across buffer
    surprise_cutoff: float = 0.0        # Surprise of last-pruned transition
    consolidation_loss: float = 0.0     # Total surprise of pruned batch

    # Legacy fields for backward compat with test_phase4
    candidates: int = 0
    fisher_gated_out: int = 0
    rate_limited: int = 0


# ---------------------------------------------------------------------------
# Main buffer
# ---------------------------------------------------------------------------

class EpistemicReplayBuffer:
    """
    Experience replay buffer with replay-driven pruning.

    Pruning is a two-step process that happens during sleep:

    1. **Score** — The trainer replays every memory through the current
       network.  Each transition gets a `replay_surprise` score based on
       how wrong the network's predictions are (action error + value error).

    2. **Prune** — The buffer removes the bottom `prune_fraction` of
       transitions ranked by replay surprise.  These are the memories
       the cortex has already absorbed.

    No fixed thresholds.  No gates.  The pruning decision emerges from
    the replay itself — exactly as in biological sleep.

    Parameters:
        capacity:        Hard upper bound on buffer size.
        prune_fraction:  Fraction of buffer to prune each sleep cycle
                         (default 0.2 = bottom 20% by replay surprise).
        min_buffer_size: Absolute minimum — never prune below this.
                         Small (default 64 = one batch) to avoid the
                         artificial-floor problem, but prevents total
                         buffer death if surprise collapses to zero.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        prune_fraction: float = 0.2,
        min_buffer_size: int = 64,
        # Legacy params accepted but ignored (backward compat)
        prune_threshold_uncertainty: float = 0.1,
        prune_threshold_td_error: float = 0.05,
        prune_threshold_fisher: float = 0.01,
        max_prune_fraction: float = 0.30,
    ):
        self.capacity = capacity
        self.prune_fraction = prune_fraction
        self.min_buffer_size = min_buffer_size

        self.buffer: List[Transition] = []
        self.position = 0

        # Lifetime statistics
        self.total_added = 0
        self.total_pruned = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.total_added += 1

    def sample(self, batch_size: int) -> List[Transition]:
        """Uniformly sample a batch from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def iter_batches(
        self, batch_size: int
    ) -> Generator[Tuple[List[Transition], List[int]], None, None]:
        """
        Iterate over the ENTIRE buffer in batches.

        Yields (batch_transitions, batch_indices) tuples.
        Used during consolidation replay so every memory is scored.
        """
        indices = list(range(len(self.buffer)))
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            batch_transitions = [self.buffer[i] for i in batch_idx]
            yield batch_transitions, batch_idx

    # ------------------------------------------------------------------
    # Replay-driven pruning
    # ------------------------------------------------------------------

    def prune_by_surprise(self) -> PruningStats:
        """
        Remove the least-surprising memories from the buffer.

        Call this AFTER the trainer has set `replay_surprise` on every
        transition via the consolidated replay pass.

        Strategy: sort by replay_surprise ascending, prune the bottom
        `prune_fraction`.  These are the memories the network has
        already absorbed — the cortex "dreamed" them and wasn't
        surprised, so the hippocampal trace fades.

        Returns:
            PruningStats with full diagnostics.
        """
        stats = PruningStats(buffer_before=len(self.buffer))

        if len(self.buffer) <= self.min_buffer_size:
            stats.buffer_after = len(self.buffer)
            return stats

        # Gather surprise scores
        surprises = [t.replay_surprise for t in self.buffer]
        stats.surprise_min = min(surprises)
        stats.surprise_max = max(surprises)
        stats.surprise_mean = float(np.mean(surprises))

        # How many to prune?
        n_prune = int(len(self.buffer) * self.prune_fraction)

        # Respect absolute floor
        n_prune = min(n_prune, len(self.buffer) - self.min_buffer_size)
        n_prune = max(n_prune, 0)

        if n_prune == 0:
            stats.buffer_after = len(self.buffer)
            return stats

        # Rank by surprise: lowest first (most consolidated)
        ranked = sorted(
            range(len(self.buffer)),
            key=lambda i: self.buffer[i].replay_surprise,
        )

        # Prune the bottom n_prune
        prune_set = set(ranked[:n_prune])
        stats.surprise_cutoff = self.buffer[ranked[n_prune - 1]].replay_surprise

        # Diagnostic: total surprise lost
        pruned_surprises = [self.buffer[i].replay_surprise for i in prune_set]
        stats.consolidation_loss = float(np.sum(pruned_surprises))

        # Remove (rebuild without pruned indices)
        self.buffer = [
            t for i, t in enumerate(self.buffer)
            if i not in prune_set
        ]

        stats.pruned = n_prune
        stats.buffer_after = len(self.buffer)
        stats.candidates = len(surprises)  # Legacy compat
        self.total_pruned += n_prune

        # Reset circular position
        self.position = len(self.buffer) % self.capacity

        logger.info(
            f"Replay pruning: {stats.buffer_before} -> {stats.buffer_after} "
            f"(pruned {stats.pruned}, cutoff={stats.surprise_cutoff:.4f}, "
            f"mean={stats.surprise_mean:.4f})"
        )

        return stats

    # ------------------------------------------------------------------
    # Legacy API (backward compat — test_phase4, old trainer)
    # ------------------------------------------------------------------

    def prune_consolidated(self, **kwargs) -> PruningStats:
        """Legacy entry point — redirects to prune_by_surprise."""
        return self.prune_by_surprise()

    def update_uncertainties(self, *args, **kwargs) -> None:
        """Legacy no-op. Uncertainty is now computed during replay."""
        pass

    def update_td_errors(self, *args, **kwargs) -> None:
        """Legacy no-op. TD errors are now computed during replay."""
        pass

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Get comprehensive buffer statistics."""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'total_added': self.total_added,
                'total_pruned': self.total_pruned,
            }

        surprises = [t.replay_surprise for t in self.buffer]
        rewards = [t.reward for t in self.buffer]

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned,
            # Replay surprise distribution
            'avg_surprise': float(np.mean(surprises)),
            'min_surprise': float(np.min(surprises)),
            'max_surprise': float(np.max(surprises)),
            'median_surprise': float(np.median(surprises)),
            'std_surprise': float(np.std(surprises)),
            # Reward distribution in buffer
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            # Legacy keys (consumed by test_phase4 / trainer)
            'avg_uncertainty': float(np.mean(surprises)),
            'avg_td_error': float(np.mean(surprises)),
            'avg_fisher_protection': 0.0,
            'near_consolidated': sum(1 for s in surprises if s < np.median(surprises)),
            'gate1_uncertainty_pass': 0,
            'gate12_unc_td_pass': 0,
            'gate123_fully_consolidated': 0,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0