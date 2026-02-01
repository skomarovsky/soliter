"""Unit tests for Soliter agent."""

import pytest
import torch
import numpy as np
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.core.cfc_network import CfCBrain


@pytest.fixture
def agent():
    """Create a test agent."""
    brain = CfCBrain()
    config = VitalsConfig()
    device = torch.device('cpu')
    return SoliterAgent(brain, config, device)


def test_agent_initialization(agent):
    """Test agent initializes with correct values."""
    assert agent.is_alive
    assert not agent.is_sleeping
    assert agent.energy == 100.0
    assert agent.hydration == 100.0
    assert agent.temperature == 37.0
    assert agent.wakefulness == 1.0


def test_vitals_decay(agent):
    """Test that vitals decay over time."""
    initial_energy = agent.energy
    initial_wakefulness = agent.wakefulness
    
    # Simulate movement for 10 ticks
    for _ in range(10):
        agent.update_vitals(
            velocity=1.0,
            ambient_temperature=20.0,
            dt=1.0
        )
    
    assert agent.energy < initial_energy
    assert agent.wakefulness < initial_wakefulness


def test_motor_atrophy(agent):
    """Test that low energy reduces max speed."""
    initial_speed = agent.get_max_speed()
    
    agent.energy = 50.0  # Half energy
    reduced_speed = agent.get_max_speed()
    
    assert reduced_speed == pytest.approx(initial_speed * 0.5)


def test_thermal_stiffness(agent):
    """Test that low temperature reduces turn rate."""
    initial_turn = agent.get_turn_rate()
    
    agent.temperature = 20.0  # Cold
    reduced_turn = agent.get_turn_rate()
    
    assert reduced_turn < initial_turn


def test_death_from_starvation(agent):
    """Test that energy reaching 0 causes death."""
    agent.energy = 1.0
    
    # Deplete energy
    agent.update_vitals(velocity=10.0, ambient_temperature=20.0, dt=10.0)
    
    assert not agent.is_alive
    assert agent.cause_of_death == "starvation"


def test_resource_consumption(agent):
    """Test consuming resources restores vitals."""
    agent.energy = 50.0
    agent.consume_resource('food', 30.0)
    assert agent.energy == 80.0
    
    agent.hydration = 60.0
    agent.consume_resource('water', 20.0)
    assert agent.hydration == 80.0


def test_sleep_cycle(agent):
    """Test sleep entry and exit."""
    agent.wakefulness = 0.2
    
    agent.enter_sleep()
    assert agent.is_sleeping
    
    agent.exit_sleep()
    assert not agent.is_sleeping
    assert agent.wakefulness == 1.0


def test_movement(agent):
    """Test agent movement updates position."""
    initial_pos = agent.position.copy()
    
    agent.move(velocity=2.0, turn=0.0, dt=1.0)
    
    # Position should have changed
    assert not np.allclose(agent.position, initial_pos)


def test_state_save_load(agent):
    """Test saving and loading agent state."""
    # Modify state
    agent.energy = 75.0
    agent.position = np.array([100.0, 50.0])
    agent.total_ticks = 1000
    
    # Save state
    state = agent.get_state_dict()
    
    # Create new agent and load state
    brain = CfCBrain()
    config = VitalsConfig()
    device = torch.device('cpu')
    new_agent = SoliterAgent(brain, config, device)
    new_agent.load_state_dict(state)
    
    # Check state matches
    assert new_agent.energy == 75.0
    assert np.allclose(new_agent.position, np.array([100.0, 50.0]))
    assert new_agent.total_ticks == 1000
