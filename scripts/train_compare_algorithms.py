#!/usr/bin/env python3
"""
Algorithm Comparison: REINFORCE+Baseline vs Hebbian vs Evolutionary Strategy

Runs all three algorithms for N cycles each and compares results.

Usage:
    uv run python scripts/train_compare_algorithms.py --cycles 50 --fps 60
    uv run python scripts/train_compare_algorithms.py --cycles 30 --fps 0  # no viz
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pygame

from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.core.ncp_drive_modulated_brain import (
    DriveModulatedNCPBrain,
    create_drive_modulated_ncp_brain
)
from soliter.core.drive_system import DriveSystem, DriveConfig
from soliter.environment import World, WorldConfig, create_default_resources
from soliter.environment.drive_modulated_sensors import (
    DriveModulatedSensorSystem, DriveModulatedSensorConfig
)

# Import all three trainers
from soliter.training.trainer_reinforce import REINFORCETrainer, REINFORCEConfig
from soliter.training.trainer_hebbian import HebbianTrainer, HebbianConfig
from soliter.training.trainer_es import ESTrainer, ESConfig

# Visualization
from soliter.visualization import ProfessionalVisualizer


def make_agent_and_trainers(device, world_width, world_height):
    """Create fresh brain, agent, sensors, drive system for one run."""
    from soliter.utils.config import config as global_config
    from soliter.environment import Physics

    # World
    world_config = WorldConfig(width=world_width, height=world_height)
    world = World(world_config)
    physics = Physics()
    
    # Resources
    resources = create_default_resources(world_width, world_height)
    
    # Brain
    brain = create_drive_modulated_ncp_brain(
        sensory_size=49, inter_size=256, command_size=64,
        motor_size=3, num_drives=4, device=device,
    )
    
    # Agent
    vitals_config = VitalsConfig(
        initial_energy=global_config.agent.initial_energy,
        initial_hydration=global_config.agent.initial_hydration,
        initial_temperature=global_config.agent.initial_temperature,
        initial_wakefulness=global_config.agent.initial_wakefulness,
    )
    agent = SoliterAgent(brain, vitals_config, device)
    
    # Spawn near resource center
    all_pos = [r.position for rtype in ['feeders','fountains','heaters'] 
               for r in resources[rtype]]
    if all_pos:
        center = np.mean(all_pos, axis=0)
        agent.position = np.clip(
            center + np.random.uniform(-20, 20, size=2),
            0, [world_width - 1, world_height - 1]
        )
    agent.heading = np.random.uniform(0, 2 * np.pi)

    # Sensors
    sensor_config = DriveModulatedSensorConfig(
        num_rays=36,
        max_ray_distance=200.0,
        gradient_scale_factor=40.0,  # Fixed value - was world_width/4 (too large for 400×400!)
    )
    sensors = DriveModulatedSensorSystem(config=sensor_config, physics=physics)
    
    # Drive system
    drive_system = DriveSystem(DriveConfig())
    
    return agent, world, resources, sensors, drive_system


def run_algorithm(
    trainer_class, trainer_config, algorithm_name,
    num_cycles, steps_per_cycle, world_width, world_height,
    device, fps, viz=None,
):
    """Run one algorithm for num_cycles and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {algorithm_name}")
    print(f"{'='*60}")
    
    agent, world, resources, sensors, drive_system = make_agent_and_trainers(
        device, world_width, world_height
    )
    
    trainer = trainer_class(
        agent=agent,
        world=world,
        drive_system=drive_system,
        sensors=sensors,
        config=trainer_config,
        device=device,
    )
    
    results = {
        'name': algorithm_name,
        'consumptions_per_cycle': [],
        'deaths_per_cycle': [],
        'total_consumptions': 0,
        'total_deaths': 0,
    }
    
    total_consumptions = 0
    total_deaths = 0
    start_time = time.time()
    
    for cycle in range(1, num_cycles + 1):
        # Track energy at cycle start
        energy_start = agent.energy
        
        # New cycle: reset position and brain, but NOT vitals!
        # (Vitals carry over from previous cycle - this is intentional for learning)
        agent.position = np.array([
            np.random.uniform(50, world_width - 50),
            np.random.uniform(50, world_height - 50),
        ], dtype=np.float32)
        agent.heading = np.random.uniform(0, 2 * np.pi)
        agent.brain.reset_hidden()
        drive_system.reset()
        world = World(WorldConfig(width=world_width, height=world_height))
        
        cycle_consumptions = 0
        cycle_deaths = 0
        cycle_velocities = []  # Track velocities
        
        for step in range(steps_per_cycle):
            # Track sudden energy jumps
            if step % 200 == 0:
                print(f"    Step {step:4d}: E={agent.energy:.1f}, H={agent.hydration:.1f}, Cons={cycle_consumptions}")
            
            # Check quit
            if viz is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return results
                    if event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                            return results
            
            world.step()
            
            # Agent acts (may die during this!)
            reward, done, details = trainer.wake_step(resources)
            
            # Track velocity
            cycle_velocities.append(agent.last_velocity)
            
            # NOW check if agent died during wake_step
            if not agent.is_alive:
                cycle_deaths += 1
                
                # DEBUG: Print death info
                print(f"    💀 Death {cycle_deaths}: {agent.cause_of_death}, "
                      f"Energy={agent.energy:.1f}, Hydration={agent.hydration:.1f}, "
                      f"Temp={agent.temperature:.1f}°C")
                
                # CRITICAL: Don't respawn at death location (especially if at wall!)
                # Respawn in a safe area away from edges
                safe_x = np.random.uniform(world_width * 0.2, world_width * 0.8)
                safe_y = np.random.uniform(world_height * 0.2, world_height * 0.8)
                agent.reset(position=np.array([safe_x, safe_y], dtype=np.float32), after_death=True)
                
                agent.heading = np.random.uniform(0, 2 * np.pi)
                agent.brain.reset_hidden()
                drive_system.reset()
            
            if trainer._consumed_this_tick is not None:
                cycle_consumptions += 1
            
            # Visualize
            if viz is not None and fps > 0:
                viz.draw(agent, resources, world, trainer, cycle, step,
                        total_consumptions + cycle_consumptions, details)
                viz.clock.tick(fps)
        
        total_consumptions += cycle_consumptions
        total_deaths += cycle_deaths
        
        results['consumptions_per_cycle'].append(cycle_consumptions)
        results['deaths_per_cycle'].append(cycle_deaths)
        
        # Track resource depletion
        feeders_avg = sum(f.get_depletion_ratio() for f in resources['feeders']) / len(resources['feeders'])
        fountains_avg = sum(f.get_depletion_ratio() for f in resources['fountains']) / len(resources['fountains'])
        heaters_avg = sum(h.get_depletion_ratio() for h in resources['heaters']) / len(resources['heaters'])
        
        avg_velocity = sum(cycle_velocities) / len(cycle_velocities) if cycle_velocities else 0
        energy_end = agent.energy
        energy_delta = energy_end - energy_start
        
        elapsed = time.time() - start_time
        print(f"  Cycle {cycle:3d}: Cons={cycle_consumptions:4d}, "
              f"Deaths={cycle_deaths:3d}, Total={total_consumptions:5d}, "
              f"Resources: F={feeders_avg:.2f} W={fountains_avg:.2f} H={heaters_avg:.2f}, "
              f"Vel={avg_velocity:.2f}, E:{energy_start:.0f}→{energy_end:.0f}({energy_delta:+.0f})")
    
    results['total_consumptions'] = total_consumptions
    results['total_deaths'] = total_deaths
    results['wall_time'] = time.time() - start_time
    
    # Resource state at end
    print(f"\n  📊 Final Resource State:")
    print(f"     Feeders:   {[f'{f.get_depletion_ratio():.2f}' for f in resources['feeders'][:5]]}")
    print(f"     Fountains: {[f'{f.get_depletion_ratio():.2f}' for f in resources['fountains'][:5]]}")
    print(f"     Heaters:   {[f'{h.get_depletion_ratio():.2f}' for h in resources['heaters'][:3]]}")
    print(f"     Total available: Food={sum(f.current_capacity for f in resources['feeders']):.1f}, "
          f"Water={sum(f.current_capacity for f in resources['fountains']):.1f}, "
          f"Heat={sum(h.current_capacity for h in resources['heaters']):.1f}")
    
    # Print diagnostics
    diag = trainer.get_diagnostics()
    if diag['weight_changes']:
        weight_change = diag['weight_changes'][-1]
        print(f"\n  Weight change: {weight_change:.8f}")
    if diag['drive_modulation_weights']:
        final_weights = np.array(diag['drive_modulation_weights'][-1])
        print(f"  Drive weights (food group): {final_weights[0].mean():.4f} ± {final_weights[0].std():.4f}")
    
    # Agent state
    print(f"\n  🤖 Final Agent State:")
    print(f"     Position: ({agent.position[0]:.1f}, {agent.position[1]:.1f})")
    print(f"     Velocity: {agent.last_velocity:.3f}, Turn: {agent.last_turn:.3f}")
    print(f"     Energy: {agent.energy:.1f}, Hydration: {agent.hydration:.1f}, Temp: {agent.temperature:.1f}°C")
    print(f"     Total movement: {agent.total_ticks} ticks")
    
    return results


def print_comparison(all_results: List[Dict], num_cycles: int):
    """Print side-by-side comparison of all algorithms."""
    print(f"\n\n{'='*70}")
    print("ALGORITHM COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<30} " + " ".join(f"{r['name']:<20}" for r in all_results))
    print("-" * 70)
    
    metrics = [
        ("Total Consumptions", lambda r: r['total_consumptions']),
        ("Total Deaths", lambda r: r['total_deaths']),
        ("Avg Consumptions/Cycle", lambda r: r['total_consumptions'] / num_cycles),
        ("Max Cycle Consumptions", lambda r: max(r['consumptions_per_cycle'])),
        ("Zero Cycles", lambda r: sum(1 for c in r['consumptions_per_cycle'] if c == 0)),
        ("Wall Time (s)", lambda r: r.get('wall_time', 0)),
    ]
    
    for name, fn in metrics:
        row = f"{name:<30}"
        for r in all_results:
            val = fn(r)
            if isinstance(val, float):
                row += f" {val:<20.1f}"
            else:
                row += f" {val:<20}"
        print(row)
    
    print(f"\n{'='*70}")
    print("CYCLE-BY-CYCLE COMPARISON (consumptions)")
    print(f"{'='*70}")
    
    header = f"{'Cycle':<8}" + " ".join(f"{r['name']:<20}" for r in all_results)
    print(header)
    print("-" * 70)
    
    for i in range(num_cycles):
        row = f"{i+1:<8}"
        for r in all_results:
            cons = r['consumptions_per_cycle'][i] if i < len(r['consumptions_per_cycle']) else 0
            row += f" {cons:<20}"
        print(row)
    
    print(f"\n{'='*70}")
    
    # Winner
    best = max(all_results, key=lambda r: r['total_consumptions'])
    print(f"🏆 WINNER: {best['name']} ({best['total_consumptions']} consumptions)")
    
    # Trend analysis
    print("\nTREND ANALYSIS (first 10 vs last 10 cycles):")
    for r in all_results:
        cons = r['consumptions_per_cycle']
        if len(cons) >= 20:
            first10 = sum(cons[:10]) / 10
            last10 = sum(cons[-10:]) / 10
            trend = "↑ IMPROVING" if last10 > first10 * 1.1 else (
                    "↓ DECLINING" if last10 < first10 * 0.9 else "→ STABLE")
            print(f"  {r['name']}: {first10:.1f} → {last10:.1f}  {trend}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--no-viz', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_width, world_height = 400, 400  # Increased from 200×200 for learning challenge
    
    print(f"{'='*70}")
    print("ALGORITHM COMPARISON: REINFORCE vs HEBBIAN vs EVOLUTIONARY STRATEGY")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Cycles per algorithm: {args.cycles}")
    print(f"Steps per cycle: {args.steps}")
    print(f"World: {world_width}×{world_height}")
    print(f"{'='*70}\n")
    
    # Visualization (shared across algorithms)
    viz = None
    if not args.no_viz and args.fps > 0:
        viz = ProfessionalVisualizer(world_width, world_height, scale=2)  # Scale=2 for 400×400 world
    
    all_results = []
    
    try:
        # 1. REINFORCE with Baseline
        results_reinforce = run_algorithm(
            trainer_class=REINFORCETrainer,
            trainer_config=REINFORCEConfig(),
            algorithm_name="REINFORCE+Baseline",
            num_cycles=args.cycles,
            steps_per_cycle=args.steps,
            world_width=world_width,
            world_height=world_height,
            device=device,
            fps=args.fps,
            viz=viz,
        )
        all_results.append(results_reinforce)
        
        # 2. Hebbian
        results_hebbian = run_algorithm(
            trainer_class=HebbianTrainer,
            trainer_config=HebbianConfig(),
            algorithm_name="Hebbian",
            num_cycles=args.cycles,
            steps_per_cycle=args.steps,
            world_width=world_width,
            world_height=world_height,
            device=device,
            fps=args.fps,
            viz=viz,
        )
        all_results.append(results_hebbian)
        
        # 3. Evolutionary Strategy
        results_es = run_algorithm(
            trainer_class=ESTrainer,
            trainer_config=ESConfig(),
            algorithm_name="Evol.Strategy",
            num_cycles=args.cycles,
            steps_per_cycle=args.steps,
            world_width=world_width,
            world_height=world_height,
            device=device,
            fps=args.fps,
            viz=viz,
        )
        all_results.append(results_es)
        
    finally:
        if viz:
            viz.close()
    
    # Print comparison
    print_comparison(all_results, args.cycles)
    
    # Save results
    output_dir = Path("experiments/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = output_dir / f"comparison_{timestamp}.json"
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Results saved: {result_path}")


if __name__ == '__main__':
    main()
