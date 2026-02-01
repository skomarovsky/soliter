"""Tests for Fisher Information Matrix."""

import pytest
import torch
from soliter.core.cfc_network import CfCBrain
from soliter.memory.fisher_matrix import FisherInformationMatrix


@pytest.fixture
def brain():
    """Create test brain."""
    return CfCBrain(sensory_size=41, inter_size=64, command_size=16, motor_size=3)


def test_fisher_initialization(brain):
    """Test Fisher matrix initializes correctly."""
    device = torch.device('cpu')
    fisher = FisherInformationMatrix(brain, device)
    
    # Should have entries for all parameters
    param_count = sum(1 for name, param in brain.named_parameters() if param.requires_grad)
    assert len(fisher.fisher_diagonal) == param_count
    assert len(fisher.optimal_weights) == param_count


def test_fisher_decay(brain):
    """Test Fisher decay mechanism."""
    device = torch.device('cpu')
    fisher = FisherInformationMatrix(brain, device)
    
    # Set some Fisher values
    for name in fisher.fisher_diagonal:
        fisher.fisher_diagonal[name].fill_(10.0)
    
    # Decay
    fisher.decay_fisher(decay_rate=0.5)
    
    # Values should be halved
    for name in fisher.fisher_diagonal:
        assert torch.allclose(fisher.fisher_diagonal[name], torch.full_like(fisher.fisher_diagonal[name], 5.0))


def test_ewc_loss(brain):
    """Test EWC loss computation."""
    device = torch.device('cpu')
    fisher = FisherInformationMatrix(brain, device)
    
    # Set Fisher values
    for name in fisher.fisher_diagonal:
        fisher.fisher_diagonal[name].fill_(1.0)
    
    # Store optimal weights
    fisher.update_optimal_weights(brain)
    
    # Compute loss (should be ~0 since weights haven't changed)
    loss = fisher.get_ewc_loss(brain, lambda_ewc=1000.0)
    assert loss.item() < 1.0
    
    # Modify weights
    for param in brain.parameters():
        param.data += 0.1
    
    # Loss should now be positive
    loss = fisher.get_ewc_loss(brain, lambda_ewc=1000.0)
    assert loss.item() > 0.0
