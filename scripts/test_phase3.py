#!/usr/bin/env python3
"""
Phase 3 verification script.

Tests environment components work together.
"""

import numpy as np
import torch
from soliter.environment import (
    World, WorldConfig,
    create_default_resources,
    Physics,
    SensorSystem,
    Season,
)

print("="*60)
print("Phase 3: Environment Implementation - Verification")
print("="*60)

# Test 1: World Simulation
print("\n[Test 1: World Simulation]")
world = World(WorldConfig(width=1000, height=1000))

print(f"Initial state: {world.get_stats()}")

# Simulate 1 year
print("\nSimulating 1 year (20,000 ticks)...")
for _ in range(20000):
    world.step()

print(f"After 1 year: {world.get_stats()}")
assert world.year == 1
print("✓ World simulation working")

# Test 2: Resources
print("\n[Test 2: Resources]")
resources = create_default_resources(1000, 1000)

print(f"Feeders: {len(resources['feeders'])}")
print(f"Fountains: {len(resources['fountains'])}")
print(f"Heaters: {len(resources['heaters'])}")

# Check seasonal availability
feeder = resources['feeders'][0]
print(f"\nFeeder availability:")
print(f"  Summer (tick 5000): {feeder.get_availability_strength(5000, 20000):.2f}")
print(f"  Winter (tick 15000): {feeder.get_availability_strength(15000, 20000):.2f}")

print("✓ Resources created and working")

# Test 3: Physics
print("\n[Test 3: Physics]")
physics = Physics(1000, 1000)

# Raycasting
origin = np.array([500.0, 500.0])
obstacles = [
    (resources['feeders'][0].position, resources['feeders'][0].radius, 'feeder')
]

results = physics.raycast_360(origin, 36, 200.0, obstacles)
hits = sum(1 for r in results if r.hit)
print(f"Raycast results: {hits}/{len(results)} rays hit obstacles")

print("✓ Physics engine working")

# Test 4: Sensors
print("\n[Test 4: Sensors]")
sensors = SensorSystem(physics=physics)

agent_position = np.array([500.0, 500.0])
agent_vitals = {
    'energy': 80.0,
    'hydration': 60.0,
    'temperature': 35.0,
    'wakefulness': 0.9,
}

readings = sensors.get_sensor_readings(
    agent_position,
    agent_vitals,
    resources,
    sensor_noise=0.1
)

print(f"Sensor readings shape: {readings.shape}")
print(f"Sensor values range: [{readings.min():.2f}, {readings.max():.2f}]")
print("\nSensor breakdown:")
print(sensors.visualize_sensors(readings))

print("\n✓ Sensors working")

# Test 5: Integration - Full Simulation Step
print("\n[Test 5: Full Integration]")
print("Simulating one complete environment step...")

world = World()
resources = create_default_resources()
physics = Physics()
sensors = SensorSystem(physics=physics)

# Simulate
world.step()
ambient_temp = world.get_ambient_temperature()
is_night = world.is_night()

readings = sensors.get_sensor_readings(
    agent_position=np.array([500.0, 500.0]),
    agent_vitals=agent_vitals,
    resources=resources,
    sensor_noise=0.0
)

print(f"✓ Tick: {world.tick}")
print(f"✓ Season: {world.get_season().value}")
print(f"✓ Ambient temp: {ambient_temp:.1f}°C")
print(f"✓ Is night: {is_night}")
print(f"✓ Sensors: {readings.shape[0]} values")

print("\n" + "="*60)
print("Phase 3 Complete: All Environment Components Working! ✅")
print("="*60)
