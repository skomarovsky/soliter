"""Tests for replay buffer."""

import pytest
import torch
import numpy as np
from soliter.memory.replay_buffer import ReplayBuffer, Transition


def test_replay_buffer_init():
    """Test replay buffer initialization."""
    buffer = ReplayBuffer(capacity=1000)
    
    assert len(buffer) == 0
    assert buffer.capacity == 1000


def test_push_and_sample():
    """Test adding and sampling transitions."""
    buffer = ReplayBuffer(capacity=100)
    
    # Add transitions
    for i in range(50):
        transition = Transition(
            state=torch.randn(41),
            action=torch.randn(3),
            reward=1.0,
            next_state=torch.randn(41),
            done=False,
            tick=i,
        )
        buffer.push(transition)
    
    assert len(buffer) == 50
    
    # Sample batch
    batch = buffer.sample(32)
    assert len(batch) == 32
    assert all(isinstance(t, Transition) for t in batch)


def test_circular_buffer():
    """Test buffer overwrites when full."""
    buffer = ReplayBuffer(capacity=10)
    
    # Fill buffer
    for i in range(15):
        transition = Transition(
            state=torch.randn(41),
            action=torch.randn(3),
            reward=1.0,
            next_state=torch.randn(41),
            done=False,
            tick=i,
        )
        buffer.push(transition)
    
    # Should only have 10 transitions
    assert len(buffer) == 10


def test_pruning():
    """Test epistemic pruning removes consolidated experiences."""
    buffer = ReplayBuffer(
        capacity=100,
        prune_threshold_uncertainty=0.2,
        prune_threshold_td_error=0.1,
    )
    
    # Add transitions with varying uncertainty
    for i in range(50):
        transition = Transition(
            state=torch.randn(41),
            action=torch.randn(3),
            reward=1.0,
            next_state=torch.randn(41),
            done=False,
            tick=i,
            uncertainty=0.1 if i < 25 else 0.5,  # First half: low uncertainty
            td_error=0.05 if i < 25 else 0.3,    # First half: low error
        )
        buffer.push(transition)
    
    # Prune consolidated (low uncertainty AND low error)
    pruned = buffer.prune_consolidated()
    
    # Should have pruned ~25 transitions
    assert pruned > 0
    assert len(buffer) < 50
