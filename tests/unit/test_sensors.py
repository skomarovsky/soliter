"""Tests for sensor system."""

import pytest
import numpy as np
import torch
from soliter.environment.sensors import SensorSystem, SensorConfig
from soliter.environment.physics import Physics
from soliter.environment.resources import create_default_resources


def test_sensor_initialization():
    """Test sensor system initializes correctly."""
    sensors = SensorSystem()
    
    assert sensors.config.num_rays == 36
    assert sensors.physics is not None


def test_sensor_readings_shape():
    """Test sensor readings have correct shape."""
    sensors = SensorSystem()
    resources = create_default_resources()
    
    agent_position = np.array([500.0, 500.0])
    agent_vitals = {
        'energy': 100.0,
        'hydration': 100.0,
        'temperature': 37.0,
        'wakefulness': 1.0,
    }
    
    readings = sensors.get_sensor_readings(
        agent_position,
        agent_vitals,
        resources,
        sensor_noise=0.0
    )
    
    assert readings.shape == (41,)  # 4 vitals + 36 rays + 1 touch
    assert readings.dtype == torch.float32


def test_vitals_normalization():
    """Test vitals are properly normalized."""
    sensors = SensorSystem()
    resources = create_default_resources()
    
    agent_position = np.array([500.0, 500.0])
    agent_vitals = {
        'energy': 50.0,
        'hydration': 75.0,
        'temperature': 37.0,
        'wakefulness': 0.8,
    }
    
    readings = sensors.get_sensor_readings(
        agent_position,
        agent_vitals,
        resources
    )
    
    assert abs(readings[0].item() - 0.5) < 0.01  # Energy
    assert abs(readings[1].item() - 0.75) < 0.01  # Hydration
    assert abs(readings[2].item() - 0.37) < 0.01  # Temperature
    assert abs(readings[3].item() - 0.8) < 0.01  # Wakefulness


def test_sensor_noise():
    """Test sensor noise affects readings."""
    sensors = SensorSystem()
    resources = create_default_resources()
    
    agent_position = np.array([500.0, 500.0])
    agent_vitals = {
        'energy': 100.0,
        'hydration': 100.0,
        'temperature': 37.0,
        'wakefulness': 1.0,
    }
    
    # No noise
    readings_clean = sensors.get_sensor_readings(
        agent_position,
        agent_vitals,
        resources,
        sensor_noise=0.0
    )
    
    # With noise
    readings_noisy = sensors.get_sensor_readings(
        agent_position,
        agent_vitals,
        resources,
        sensor_noise=0.2
    )
    
    # Vitals should be the same (internal sensors)
    assert torch.allclose(readings_clean[:4], readings_noisy[:4])
    
    # External sensors should differ
    # (Note: might occasionally be the same due to randomness, but very unlikely)
    assert not torch.allclose(readings_clean[4:], readings_noisy[4:], atol=0.05)


def test_touch_detection():
    """Test touch sensor activates near resources."""
    sensors = SensorSystem(SensorConfig(touch_threshold=15.0))
    resources = create_default_resources()
    
    # Get a resource position
    feeder_pos = resources['feeders'][0].position
    
    # Agent very close to feeder
    agent_position = feeder_pos + np.array([5.0, 0.0])
    agent_vitals = {
        'energy': 100.0,
        'hydration': 100.0,
        'temperature': 37.0,
        'wakefulness': 1.0,
    }
    
    readings = sensors.get_sensor_readings(
        agent_position,
        agent_vitals,
        resources
    )
    
    touch = readings[40].item()
    assert touch > 0.5  # Touch sensor activated
