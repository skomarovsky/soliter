"""Tests for resource system."""

import pytest
import numpy as np
from soliter.environment.resources import (
    Feeder, Fountain, Heater, ResourceConfig, create_default_resources
)


def test_feeder_availability():
    """Test feeder availability varies with season."""
    config = ResourceConfig(position=np.array([500.0, 500.0]))
    feeder = Feeder(config)
    
    seasonal_period = 20000
    
    # Summer (high availability)
    tick_summer = 5000  # phase ~ π/2
    assert feeder.is_available(tick_summer, seasonal_period)
    
    # Winter (low availability)
    tick_winter = 15000  # phase ~ 3π/2
    # Might still be available (> 0.1 threshold) but lower strength
    strength_summer = feeder.get_availability_strength(tick_summer, seasonal_period)
    strength_winter = feeder.get_availability_strength(tick_winter, seasonal_period)
    assert strength_summer > strength_winter


def test_fountain_availability():
    """Test fountain availability has two peaks per year."""
    config = ResourceConfig(position=np.array([500.0, 500.0]))
    fountain = Fountain(config)
    
    seasonal_period = 20000
    
    # Check multiple points throughout year
    availabilities = []
    for tick in range(0, seasonal_period, seasonal_period // 8):
        strength = fountain.get_availability_strength(tick, seasonal_period)
        availabilities.append(strength)
    
    # Should have variation (not constant)
    assert max(availabilities) > min(availabilities) + 0.5


def test_heater_always_available():
    """Test heater is always available."""
    config = ResourceConfig(position=np.array([500.0, 500.0]))
    heater = Heater(config)
    
    seasonal_period = 20000
    
    # Check at different times
    assert heater.is_available(0, seasonal_period)
    assert heater.is_available(10000, seasonal_period)
    assert heater.is_available(19999, seasonal_period)


def test_heater_night_radius():
    """Test heater radius shrinks at night."""
    config = ResourceConfig(position=np.array([500.0, 500.0]), radius=40.0)
    heater = Heater(config, night_radius_multiplier=0.5)
    
    # Day radius
    day_radius = heater.get_effective_radius(is_night=False)
    assert day_radius == 40.0
    
    # Night radius
    night_radius = heater.get_effective_radius(is_night=True)
    assert night_radius == 20.0  # 40 * 0.5


def test_agent_in_range():
    """Test agent range detection."""
    config = ResourceConfig(position=np.array([500.0, 500.0]), radius=30.0)
    resource = Feeder(config)
    
    # Agent close
    agent_pos_close = np.array([510.0, 510.0])
    assert resource.is_agent_in_range(agent_pos_close)
    
    # Agent far
    agent_pos_far = np.array([600.0, 600.0])
    assert not resource.is_agent_in_range(agent_pos_far)


def test_create_default_resources():
    """Test default resource creation."""
    resources = create_default_resources(1000, 1000)
    
    assert len(resources['feeders']) == 5
    assert len(resources['fountains']) == 5
    assert len(resources['heaters']) == 5
    
    # Check positions are within bounds
    for resource_list in resources.values():
        for resource in resource_list:
            assert 0 <= resource.position[0] <= 1000
            assert 0 <= resource.position[1] <= 1000
