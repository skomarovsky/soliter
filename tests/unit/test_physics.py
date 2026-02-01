"""Tests for physics engine."""

import pytest
import numpy as np
from soliter.environment.physics import Physics, RaycastResult


def test_circle_collision():
    """Test circle collision detection."""
    physics = Physics()
    
    pos1 = np.array([100.0, 100.0])
    pos2 = np.array([110.0, 100.0])
    
    # Overlapping circles
    assert physics.check_circle_collision(pos1, 10.0, pos2, 10.0)
    
    # Non-overlapping circles
    assert not physics.check_circle_collision(pos1, 5.0, pos2, 4.0)


def test_raycast_hit():
    """Test raycast detects obstacles."""
    physics = Physics()
    
    origin = np.array([100.0, 100.0])
    angle = 0.0  # Pointing right
    
    # Obstacle in path
    obstacles = [
        (np.array([150.0, 100.0]), 10.0, 'obstacle')
    ]
    
    result = physics.raycast(origin, angle, 200.0, obstacles)
    
    assert result.hit
    assert result.hit_type == 'obstacle'
    assert 30 < result.distance < 50  # Should hit around 40 units away


def test_raycast_miss():
    """Test raycast when no obstacles."""
    physics = Physics()
    
    origin = np.array([100.0, 100.0])
    angle = 0.0
    
    # No obstacles
    result = physics.raycast(origin, angle, 200.0, [])
    
    assert not result.hit
    assert result.distance == 200.0


def test_raycast_360():
    """Test 360-degree raycasting."""
    physics = Physics()
    
    origin = np.array([500.0, 500.0])
    num_rays = 36
    
    # Single obstacle
    obstacles = [
        (np.array([600.0, 500.0]), 20.0, 'obstacle')
    ]
    
    results = physics.raycast_360(origin, num_rays, 200.0, obstacles)
    
    assert len(results) == num_rays
    
    # At least one ray should hit
    hits = sum(1 for r in results if r.hit)
    assert hits > 0


def test_position_wrapping():
    """Test toroidal position wrapping."""
    physics = Physics(1000, 1000)
    
    # Outside bounds
    pos = np.array([1100.0, -50.0])
    wrapped = physics.wrap_position(pos)
    
    assert wrapped[0] == 100.0
    assert wrapped[1] == 950.0


def test_wrapped_distance():
    """Test distance calculation in toroidal space."""
    physics = Physics(1000, 1000)
    
    # Direct distance
    pos1 = np.array([100.0, 100.0])
    pos2 = np.array([200.0, 100.0])
    dist = physics.get_wrapped_distance(pos1, pos2)
    assert abs(dist - 100.0) < 0.1
    
    # Wrapped distance (closer through boundary)
    pos1 = np.array([50.0, 50.0])
    pos2 = np.array([950.0, 50.0])
    dist = physics.get_wrapped_distance(pos1, pos2)
    assert dist < 200  # Should wrap around
