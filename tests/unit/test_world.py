"""Tests for world simulation."""

import pytest
import numpy as np
from soliter.environment.world import World, WorldConfig, Season, TimeOfDay


def test_world_initialization():
    """Test world initializes correctly."""
    world = World()
    
    assert world.tick == 0
    assert world.year == 0
    assert world.day == 0
    assert world.get_season() == Season.SPRING


def test_world_step():
    """Test world advances time."""
    world = World()
    
    world.step()
    assert world.tick == 1
    
    # Advance a full day
    for _ in range(2000 - 1):
        world.step()
    
    assert world.tick == 2000
    assert world.day == 1


def test_seasonal_cycle():
    """Test seasonal phases."""
    world = World()
    
    # Spring (phase ~ 0)
    assert world.get_season() == Season.SPRING
    
    # Summer (phase ~ π/2)
    world.tick = 5000  # 1/4 year
    assert world.get_season() == Season.SUMMER
    
    # Fall (phase ~ π)
    world.tick = 10000  # 1/2 year
    assert world.get_season() == Season.FALL
    
    # Winter (phase ~ 3π/2)
    world.tick = 15000  # 3/4 year
    assert world.get_season() == Season.WINTER


def test_temperature_variation():
    """Test temperature varies with season and time of day."""
    config = WorldConfig(
        base_temperature=37.0,
        seasonal_amplitude=30.0,
        diurnal_amplitude=10.0
    )
    world = World(config)
    
    # Summer noon should be hot
    world.tick = 5000  # Summer
    world.tick += 1000  # Noon
    temp_summer_noon = world.get_ambient_temperature()
    
    # Winter midnight should be cold
    world.tick = 15000  # Winter
    temp_winter_midnight = world.get_ambient_temperature()
    
    assert temp_summer_noon > temp_winter_midnight
    assert temp_summer_noon > config.base_temperature
    assert temp_winter_midnight < config.base_temperature


def test_light_level():
    """Test light level varies with time of day."""
    world = World()
    
    # Midnight (phase ~ 0)
    world.tick = 0
    assert world.get_light_level() < 0.1
    
    # Noon (phase ~ π/2, when sin is maximum)
    world.tick = 500  # Quarter day = π/2 phase
    assert world.get_light_level() > 0.9
    
    # Night detection
    world.tick = 0
    assert world.is_night()  # Midnight
    world.tick = 500
    assert not world.is_night()  # Noon


def test_position_wrapping():
    """Test toroidal world wrapping."""
    world = World(WorldConfig(width=1000, height=1000))
    
    # Position outside bounds
    pos = np.array([1100.0, -50.0])
    wrapped = world.wrap_position(pos)
    
    assert 0 <= wrapped[0] < 1000
    assert 0 <= wrapped[1] < 1000
    assert wrapped[0] == 100.0  # 1100 % 1000
    assert wrapped[1] == 950.0  # -50 % 1000


def test_state_persistence():
    """Test world state save/load."""
    world = World()
    
    # Advance time
    for _ in range(5000):
        world.step()
    
    # Save state
    state = world.get_state_dict()
    
    # Create new world and load state
    world2 = World()
    world2.load_state_dict(state)
    
    assert world2.tick == world.tick
    assert world2.year == world.year
    assert world2.day == world.day
