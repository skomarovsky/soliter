"""Unit tests for CfC network."""

import pytest
import torch
from soliter.core.cfc_network import CfCBrain, SimpleMLP
from soliter.core.ncp_wiring import create_soliter_wiring


def test_cfc_brain_initialization():
    """Test that CfC brain initializes correctly."""
    brain = CfCBrain(
        sensory_size=41,
        inter_size=256,
        command_size=64,
        motor_size=3,
    )
    
    assert brain.sensory_size == 41
    assert brain.inter_size == 256
    assert brain.command_size == 64
    assert brain.motor_size == 3


def test_cfc_forward_pass():
    """Test forward pass produces correct output shape."""
    brain = CfCBrain()
    brain.eval()
    
    # Single timestep input
    x = torch.randn(1, 41)
    output, _ = brain(x)
    
    assert output.shape == (1, 3)
    
    # Check motor output constraints
    assert 0 <= output[0, 0] <= 1  # velocity [0, 1]
    assert -1 <= output[0, 1] <= 1  # turn [-1, 1]
    assert 0 <= output[0, 2] <= 1  # sleep [0, 1]


def test_cfc_sequence_processing():
    """Test that CfC can process sequences."""
    brain = CfCBrain(return_sequences=True)
    brain.eval()
    
    # Sequence input
    x = torch.randn(2, 10, 41)  # batch_size=2, seq_len=10
    output, _ = brain(x)
    
    assert output.shape == (2, 10, 3)


def test_hidden_state_persistence():
    """Test that hidden state persists across calls."""
    brain = CfCBrain()
    brain.eval()
    
    x1 = torch.randn(1, 41)
    x2 = torch.randn(1, 41)
    
    # First forward pass
    brain.reset_hidden(batch_size=1)
    output1, _ = brain(x1)
    # Hidden state is a tuple (h, c) for mixed_memory
    hidden1 = tuple(h.clone() for h in brain.hidden_state)
    
    # Second forward pass (should use hidden1)
    output2, _ = brain(x2)
    
    # Hidden state should have changed
    assert not all(torch.allclose(h1, h2) for h1, h2 in zip(hidden1, brain.hidden_state))


def test_homeostatic_scaling():
    """Test homeostatic scaling mechanism."""
    brain = CfCBrain()
    
    # Get initial weight norm
    initial_norm = brain.get_weight_norm()
    
    # Apply scaling (simulate high activity)
    brain.mean_activity.fill_(0.8)  # Neurons too active
    scale_factor = brain.apply_homeostatic_scaling(
        target_activity=0.5,
        scaling_rate=0.1
    )
    
    # Scale factor should be < 1 (downscaling)
    assert scale_factor < 1.0
    
    # Weight norm should have decreased
    new_norm = brain.get_weight_norm()
    assert new_norm < initial_norm


def test_mlp_baseline():
    """Test MLP baseline for comparison."""
    mlp = SimpleMLP(input_size=41, hidden_size=256, output_size=3)
    mlp.eval()
    
    x = torch.randn(1, 41)
    output, _ = mlp(x)
    
    assert output.shape == (1, 3)
    assert 0 <= output[0, 0] <= 1  # velocity
    assert -1 <= output[0, 1] <= 1  # turn
    assert 0 <= output[0, 2] <= 1  # sleep


def test_ncp_wiring():
    """Test NCP wiring creation."""
    wiring = create_soliter_wiring()
    
    # _inter_neurons is a list of neuron indices, check count
    assert len(wiring._inter_neurons) == 256
    assert len(wiring._command_neurons) == 64
    assert len(wiring._motor_neurons) == 3
    
    # Check basic properties
    assert wiring.units == 323  # 256 + 64 + 3
    assert wiring.output_dim == 3  # motor neurons
    
    # Note: adjacency_matrix might be sparse/empty in NCP
    # The actual connections are handled internally
    print(f"NCP wiring created with {wiring.units} total units")
