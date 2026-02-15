#!/usr/bin/env python3
"""
Diagnostic script to test if sensors are working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from soliter.environment import (
    World, WorldConfig,
    create_default_resources,
    SensorSystem, SensorConfig,
)

# Create world
world_config = WorldConfig(width=200, height=200)
world = World(world_config)

# Create resources
resources = create_default_resources(200, 200)

print("=" * 80)
print("SENSOR DIAGNOSTIC")
print("=" * 80)

# Check resources were created
print(f"\n📦 RESOURCES CREATED:")
print(f"  Feeders:   {len(resources.get('feeders', []))}")
print(f"  Fountains: {len(resources.get('fountains', []))}")
print(f"  Heaters:   {len(resources.get('heaters', []))}")

# Check resource positions
print(f"\n📍 RESOURCE POSITIONS:")
for i, feeder in enumerate(resources.get('feeders', [])):
    print(f"  Feeder {i}: {feeder.position}, capacity={feeder.current_capacity:.1f}/{feeder.max_capacity:.1f}, can_consume={feeder.can_consume()}")

for i, fountain in enumerate(resources.get('fountains', [])):
    print(f"  Fountain {i}: {fountain.position}, capacity={fountain.current_capacity:.1f}/{fountain.max_capacity:.1f}, can_consume={fountain.can_consume()}")

for i, heater in enumerate(resources.get('heaters', [])):
    print(f"  Heater {i}: {heater.position}, capacity={heater.current_capacity:.1f}/{heater.max_capacity:.1f}, can_consume={heater.can_consume()}")

# Create sensor system
sensor_config = SensorConfig()
sensors = SensorSystem(config=sensor_config)

# Test agent at different positions
test_positions = [
    np.array([100.0, 100.0]),  # Center
    np.array([50.0, 50.0]),    # Near typical resource
    np.array([10.0, 10.0]),    # Corner
]

print(f"\n🔍 SENSOR READINGS AT TEST POSITIONS:")
print("=" * 80)

for pos in test_positions:
    print(f"\nAgent at position: {pos}")
    
    # Get sensor readings
    vitals = {'energy': 50.0, 'hydration': 50.0, 'temperature': 37.0, 'wakefulness': 1.0}
    drive_vector = np.array([0.5, 0.5, 0.0, 0.0])  # Moderate hunger/thirst
    
    sensor_readings = sensors.get_sensor_readings(
        agent_position=pos,
        agent_vitals=vitals,
        resources=resources,
        sensor_noise=0.0,
        drive_vector=drive_vector,
        world_width=200.0,
        world_height=200.0,
    )
    
    # Extract gradient components
    gradients = sensor_readings[41:47].numpy()
    food_gx, food_gy = gradients[0], gradients[1]
    water_gx, water_gy = gradients[2], gradients[3]
    heat_gx, heat_gy = gradients[4], gradients[5]
    
    print(f"  Gradient sensors:")
    print(f"    Food:  ({food_gx:+.3f}, {food_gy:+.3f}) magnitude: {np.sqrt(food_gx**2 + food_gy**2):.3f}")
    print(f"    Water: ({water_gx:+.3f}, {water_gy:+.3f}) magnitude: {np.sqrt(water_gx**2 + water_gy**2):.3f}")
    print(f"    Heat:  ({heat_gx:+.3f}, {heat_gy:+.3f}) magnitude: {np.sqrt(heat_gx**2 + heat_gy**2):.3f}")
    
    # Check if any gradient is non-zero
    if np.any(np.abs(gradients) > 0.001):
        print(f"  ✅ Gradients detected!")
    else:
        print(f"  ❌ NO GRADIENTS - sensors broken or resources too far!")
        
        # Find nearest resource
        min_dist = float('inf')
        nearest = None
        for feeder in resources.get('feeders', []):
            dist = np.linalg.norm(pos - feeder.position)
            if dist < min_dist:
                min_dist = dist
                nearest = ('feeder', feeder.position)
        
        print(f"  Nearest resource: {nearest[0]} at {nearest[1]}, distance={min_dist:.1f}")
        print(f"  Detection radius: 30.0")
        if min_dist > 30:
            print(f"  ⚠️  TOO FAR! Agent can't sense resources from this position")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
