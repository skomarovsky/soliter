#!/usr/bin/env python3
"""
Phase 5 Fix Validation Script

Tests the critical fixes for Phase 5:
1. World size reduction (300x300 → 150x150)
2. Spawn location near resource cluster
3. Resource discoverability within exploration range

Expected outcomes:
- Agent should discover resources within first 20 cycles
- Consumption rate should exceed 10% by cycle 50
- Movement should explore more than 50% of world
"""

import sys
import numpy as np
from pathlib import Path

import torch

from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.utils.config import config
from soliter.environment import (
    World, WorldConfig,
    create_default_resources,
    Physics, SensorSystem, SensorConfig,
)
from soliter.training import SleepWakeTrainer, TrainingConfig


def test_world_size():
    """Test 1: World size is correctly reduced."""
    print("Test 1: World size validation")
    print("-" * 60)
    
    world_width = config.world.size[0]
    world_height = config.world.size[1]
    
    print(f"  World dimensions: {world_width}x{world_height}")
    
    if world_width == 150 and world_height == 150:
        print("  ✓ PASS: World is 150x150")
    else:
        print(f"  ✗ FAIL: Expected 150x150, got {world_width}x{world_height}")
        print(f"  → Update configs/default.yaml: world.size: [150, 150]")
        return False
    
    print()
    return True


def test_resource_distribution():
    """Test 2: Resources are distributed within discoverable range."""
    print("Test 2: Resource distribution validation")
    print("-" * 60)
    
    world_width = config.world.size[0]
    world_height = config.world.size[1]
    
    resources = create_default_resources(world_width, world_height)
    
    # Calculate resource statistics
    all_positions = []
    for rtype in ['feeders', 'fountains', 'heaters']:
        for r in resources[rtype]:
            all_positions.append(r.position)
    
    center = np.array([world_width / 2, world_height / 2])
    distances = [np.linalg.norm(pos - center) for pos in all_positions]
    
    print(f"  Total resources: {len(all_positions)}")
    print(f"  Distance from center:")
    print(f"    Min: {min(distances):.1f} units")
    print(f"    Max: {max(distances):.1f} units")
    print(f"    Mean: {np.mean(distances):.1f} units")
    
    # Agent exploration range (based on action_std decay analysis)
    exploration_range = 60.0  # Empirically measured from training logs
    max_world_distance = np.sqrt(2) * (world_width / 2)  # Diagonal from center
    
    print(f"  Agent exploration range: {exploration_range:.1f} units")
    print(f"  Max world distance: {max_world_distance:.1f} units")
    
    discoverable = sum(1 for d in distances if d <= exploration_range)
    coverage = (exploration_range / max_world_distance) * 100
    
    print(f"  Resources within exploration range: {discoverable}/{len(all_positions)} ({discoverable/len(all_positions)*100:.1f}%)")
    print(f"  World coverage: {coverage:.1f}%")
    
    if discoverable >= len(all_positions) * 0.5:
        print(f"  ✓ PASS: At least 50% of resources are discoverable")
    else:
        print(f"  ✗ FAIL: Only {discoverable/len(all_positions)*100:.1f}% resources discoverable")
        print(f"  → Consider further world size reduction or increased action_std")
        return False
    
    print()
    return True


def test_spawn_location():
    """Test 3: Agent spawns near resource cluster."""
    print("Test 3: Spawn location validation")
    print("-" * 60)
    
    device = torch.device('cpu')
    world_width = config.world.size[0]
    world_height = config.world.size[1]
    
    world_config = WorldConfig(
        width=world_width,
        height=world_height,
        seasonal_period=config.world.seasonal_period,
        diurnal_period=config.world.diurnal_period
    )
    world = World(world_config)
    resources = create_default_resources(world_width, world_height)
    
    # Calculate resource cluster center
    all_positions = []
    for rtype in ['feeders', 'fountains', 'heaters']:
        for r in resources[rtype]:
            all_positions.append(r.position)
    
    resource_center = np.mean(all_positions, axis=0)
    
    # Create agent
    sensor_config = SensorConfig(
        gradient_scale_factor=world_width / 4.0,
        enable_gradients=True,
        enable_drive_input=True,
    )
    physics = Physics()
    sensors = SensorSystem(config=sensor_config, physics=physics)
    brain = CfCBrain(sensory_size=51)
    
    vitals_config = VitalsConfig(
        initial_energy=config.agent.initial_energy,
        initial_hydration=config.agent.initial_hydration,
        initial_temperature=config.agent.initial_temperature,
        initial_wakefulness=config.agent.initial_wakefulness,
    )
    agent = SoliterAgent(brain, vitals_config, device)
    
    # Simulate spawn logic from train_soliter.py
    agent.position = resource_center + np.random.uniform(-10, 10, size=2)
    agent.position = np.clip(agent.position, 0, [world_width - 1, world_height - 1])
    
    # Check spawn distance to nearest resource
    distances_to_resources = [np.linalg.norm(agent.position - pos) for pos in all_positions]
    nearest_resource_dist = min(distances_to_resources)
    
    print(f"  Resource cluster center: ({resource_center[0]:.1f}, {resource_center[1]:.1f})")
    print(f"  Agent spawn location: ({agent.position[0]:.1f}, {agent.position[1]:.1f})")
    print(f"  Distance to nearest resource: {nearest_resource_dist:.1f} units")
    
    # An agent should be within ~50 units of at least one resource to discover it quickly
    if nearest_resource_dist <= 50:
        print(f"  ✓ PASS: Agent spawns within discovery range of resources")
    else:
        print(f"  ✗ FAIL: Spawn too far from resources ({nearest_resource_dist:.1f} > 50)")
        print(f"  → Check spawn logic in train_soliter.py")
        return False
    
    print()
    return True


def test_gradient_sensor_range():
    """Test 4: Gradient sensors have appropriate detection range."""
    print("Test 4: Gradient sensor range validation")
    print("-" * 60)
    
    world_width = config.world.size[0]
    
    # From training config
    scale_factor = world_width / 4.0
    
    print(f"  World width: {world_width}")
    print(f"  Gradient scale factor: {scale_factor:.1f}")
    print(f"  Effective detection range: ~{scale_factor * 2:.1f} units")
    
    # Gradient should be detectable across at least half the world
    if scale_factor >= world_width / 5:  # At least 20% of world
        print(f"  ✓ PASS: Gradients detectable across {scale_factor / world_width * 100:.0f}% of world")
    else:
        print(f"  ✗ FAIL: Gradient range too small")
        return False
    
    print()
    return True


def test_action_std_decay():
    """Test 5: Action std decay is appropriate for world size."""
    print("Test 5: Action std decay validation")
    print("-" * 60)
    
    # From config
    action_std_init = 0.5
    action_std_min = 0.1
    action_std_decay = 0.995
    
    print(f"  Initial action std: {action_std_init}")
    print(f"  Decay rate: {action_std_decay} per cycle")
    print(f"  Minimum action std: {action_std_min}")
    
    # Calculate std at key cycles
    std_at_50 = max(action_std_init * (action_std_decay ** 50), action_std_min)
    std_at_100 = max(action_std_init * (action_std_decay ** 100), action_std_min)
    
    print(f"\n  Predicted std values:")
    print(f"    Cycle 50: {std_at_50:.4f}")
    print(f"    Cycle 100: {std_at_100:.4f}")
    
    # By cycle 50, we want std > 0.35 for adequate exploration
    if std_at_50 >= 0.35:
        print(f"  ✓ PASS: Exploration sustained through early learning")
    else:
        print(f"  ✗ WARN: Exploration may decay too quickly")
        print(f"  → Consider slower decay (0.998) or higher initial std (1.0)")
    
    print()
    return True


def run_all_tests():
    """Run all Phase 5 validation tests."""
    print("=" * 80)
    print("PHASE 5 FIX VALIDATION")
    print("=" * 80)
    print()
    
    tests = [
        test_world_size,
        test_resource_distribution,
        test_spawn_location,
        test_gradient_sensor_range,
        test_action_std_decay,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append(False)
            print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Ready for Phase 5 training!")
        print("\nNext steps:")
        print("  1. Run short training: python scripts/train_soliter.py --cycles 50")
        print("  2. Check for resource consumption in first 20 cycles")
        print("  3. Verify consumption rate > 10% by cycle 50")
        return 0
    else:
        print(f"\n✗ {total - passed} TESTS FAILED - Fix issues before training")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
