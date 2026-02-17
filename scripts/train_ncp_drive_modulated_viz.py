#!/usr/bin/env python3
"""
Training script for Drive-Modulated NCP Brain.

This uses the UPGRADED NCP library with internal drive modulation.

Combines:
  - NCP sparse wiring (C. elegans)
  - CfC continuous-time dynamics
  - Drive modulation (homeostatic)
  - End-to-end learning

Usage:
  python scripts/train_ncp_drive_modulated_viz.py --cycles 20 --fps 60
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import pygame
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict
from datetime import datetime

from soliter.core.ncp_drive_modulated_brain import create_drive_modulated_ncp_brain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.utils.config import config
from soliter.environment import (
    World, WorldConfig,
    create_default_resources,
    Physics,
)
from soliter.environment.drive_modulated_sensors import DriveModulatedSensorSystem, DriveModulatedSensorConfig
from soliter.training.ncp_drive_modulated_trainer import NCPDriveModulatedTrainer, NCPDriveModulatedTrainerConfig
from soliter.core.drive_system import DriveSystem, DriveConfig

# Import visualization from existing script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Use existing visualization classes
from soliter.visualization import ProfessionalVisualizer


# ===================================================================
# LOGGING STRUCTURES
# ===================================================================

@dataclass
class TickSnapshot:
    """Snapshot of state at a single tick."""
    tick: int
    cycle: int
    x: float
    y: float
    heading: float
    velocity: float
    energy: float
    hydration: float
    temperature: float
    wakefulness: float
    reward: float
    ambient_temp: float
    season: str
    is_night: bool
    touching_resource: bool
    surprise: float = 0.0
    hunger: float = 0.0
    thirst: float = 0.0
    cold: float = 0.0
    curiosity: float = 0.0
    satisfaction: float = 0.0
    discomfort: float = 0.0
    consumed: str = ''


@dataclass
class DeathEvent:
    """Record of an agent death."""
    tick: int
    cycle: int
    cause: str
    x: float
    y: float
    energy: float
    hydration: float
    temperature: float
    life_duration: int


@dataclass
class ResourceSnapshot:
    """Snapshot of a resource position."""
    resource_type: str
    index: int
    x: float
    y: float
    detection_radius: float
    consumption_radius: float
    restore_rate: float


@dataclass
class TrainingLog:
    """Complete training log - saved as JSON."""
    start_time: str = ""
    config: Dict = field(default_factory=dict)
    world_config: Dict = field(default_factory=dict)
    device: str = ""
    resources: List[Dict] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)
    deaths: List[Dict] = field(default_factory=list)
    sleeps: List[Dict] = field(default_factory=list)
    total_ticks: int = 0
    total_cycles: int = 0
    total_deaths: int = 0
    wall_time_seconds: float = 0.0


def log_resources(resources: Dict) -> List[Dict]:
    """Snapshot all resource positions."""
    entries = []
    for rtype, rlist in resources.items():
        singular = rtype.rstrip('s')
        for i, r in enumerate(rlist):
            entries.append(asdict(ResourceSnapshot(
                resource_type=singular,
                index=i,
                x=float(r.position[0]),
                y=float(r.position[1]),
                detection_radius=float(r.detection_radius),
                consumption_radius=float(r.consumption_radius),
                restore_rate=float(r.restore_rate),
            )))
    return entries


def save_training_log(log: TrainingLog, output_dir: Path):
    """Save training log as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'training_{timestamp}.json'
    
    with open(log_path, 'w') as f:
        json.dump(asdict(log), f, indent=2)
    
    print(f"\n✅ Training log saved: {log_path}")
    return log_path



def main():
    parser = argparse.ArgumentParser(description="Train with NCP drive-modulated brain")
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--wake-steps', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='experiments/ncp_drive_modulated',
                        help='Output directory for logs')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"🧠 NCP DRIVE-MODULATED BRAIN TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Cycles: {args.cycles}")
    print(f"Wake steps per cycle: {args.wake_steps}")
    print(f"\nArchitecture:")
    print(f"  ✅ NCP sparse wiring (C. elegans inspired)")
    print(f"  ✅ CfC continuous-time dynamics")
    print(f"  ✅ Internal drive modulation")
    print(f"  ✅ 49 sensors (with light + ambient temp)")
    print(f"  ✅ 256 interneurons (4 drive groups)")
    print(f"  ✅ 64 command neurons")
    print(f"  ✅ 3 motor outputs")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time_seconds = time.time()
    
    # Initialize training log
    training_log = TrainingLog(
        start_time=datetime.now().isoformat(),
        device=str(device),
        config={
            'cycles': args.cycles,
            'wake_steps': args.wake_steps,
            'architecture': 'NCP Drive-Modulated',
        },
        world_config={
            'width': config.world.size[0],
            'height': config.world.size[1],
        }
    )
    
    # Setup world
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
    physics = Physics()
    
    training_log.resources = log_resources(resources)
    
    # DRIVE-MODULATED SENSORS (49 channels)
    sensor_config = DriveModulatedSensorConfig(
        num_rays=36,
        max_ray_distance=200.0,
        gradient_scale_factor=world_width / 4.0,
    )
    sensors = DriveModulatedSensorSystem(config=sensor_config, physics=physics)
    
    # NCP DRIVE-MODULATED BRAIN!
    print("Creating NCP Drive-Modulated Brain...")
    brain = create_drive_modulated_ncp_brain(
        sensory_size=49,
        inter_size=256,   # 4 groups of 64
        command_size=64,
        motor_size=3,
        num_drives=4,
        device=device,
    )
    
    # Print wiring stats
    stats = brain.get_wiring_stats()
    print(f"\n🕸️  NCP Wiring Statistics:")
    print(f"  Total neurons: {stats['total_units']}")
    print(f"  Interneurons: {stats['inter_neurons']}")
    print(f"  Command: {stats['command_neurons']}")
    print(f"  Motor: {stats['motor_neurons']}")
    print(f"  Connections: {stats['total_connections']}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print()
    
    # Create agent
    vitals_config = VitalsConfig(
        initial_energy=config.agent.initial_energy,
        initial_hydration=config.agent.initial_hydration,
        initial_temperature=config.agent.initial_temperature,
        initial_wakefulness=config.agent.initial_wakefulness,
    )
    agent = SoliterAgent(brain, vitals_config, device)
    
    # Initial spawn near resources
    all_resource_positions = []
    for rtype in ['feeders', 'fountains', 'heaters']:
        for r in resources[rtype]:
            all_resource_positions.append(r.position)
    
    if all_resource_positions:
        resource_center = np.mean(all_resource_positions, axis=0)
        agent.position = resource_center + np.random.uniform(-10, 10, size=2)
        agent.position = np.clip(agent.position, 0, [world_width - 1, world_height - 1])
    
    # Initialize drive system
    drive_config = DriveConfig()
    drive_system = DriveSystem(drive_config)
    
    # Initialize NCP DRIVE-MODULATED TRAINER
    trainer_config = NCPDriveModulatedTrainerConfig(
        learning_rate=0.001,
        action_std_init=0.3,
        entropy_coef=0.01,
    )
    
    trainer = NCPDriveModulatedTrainer(
        agent=agent,
        world=world,
        drive_system=drive_system,
        sensors=sensors,
        config=trainer_config,
        device=device,
    )
    
    # Visualization with professional visualizer
    viz = ProfessionalVisualizer(world_width, world_height, scale=4)
    
    print(f"\n{'='*80}")
    print(f"TRAINING WITH VISUALIZATION")
    print(f"{'='*80}")
    print(f"Controls: ESC or Q to quit")
    print(f"FPS: {args.fps if args.fps > 0 else 'Unlimited'}")
    print(f"{'='*80}\n")
    
    total_consumptions = 0
    last_reward_details = {}
    running = True
    
    try:
        for cycle in range(1, args.cycles + 1):
            cycle_consumptions = 0
            cycle_start_life = 0
            
            for step in range(args.wake_steps):
                if not running:
                    break
                
                # Update world
                world.step()
                
                # Update all resources (recovery)
                for resource_list in resources.values():
                    for resource in resource_list:
                        resource.update(world.tick)
                
                # Training step
                reward, done, reward_details = trainer.wake_step(resources)
                last_reward_details = reward_details
                
                # Update brain (REINFORCE learning)
                trainer.update(
                    reward=reward,
                    value=torch.tensor(reward_details['value'], device=device),
                    log_prob=torch.tensor(reward_details['log_prob'], device=device),
                    done=done,
                )
                
                if reward_details.get('consumption_bonus', 0) > 0:
                    cycle_consumptions += 1
                    total_consumptions += 1
                
                # Log every 100 ticks
                if (world.tick % 100) == 0:
                    drives = drive_system.get_drives(
                        agent.energy,
                        agent.hydration,
                        agent.temperature
                    )
                    consumed_what = ''
                    if reward_details.get('consumption_bonus', 0) > 0:
                        consumed_what = reward_details.get('consumed_type', '')
                    
                    training_log.snapshots.append(asdict(TickSnapshot(
                        tick=world.tick,
                        cycle=cycle,
                        x=float(agent.position[0]),
                        y=float(agent.position[1]),
                        heading=float(agent.heading),
                        velocity=float(agent.last_velocity),
                        energy=float(agent.energy),
                        hydration=float(agent.hydration),
                        temperature=float(agent.temperature),
                        wakefulness=float(agent.wakefulness),
                        reward=float(reward),
                        ambient_temp=float(world.get_ambient_temperature()),
                        season=world.get_season().name,
                        is_night=bool(world.is_night()),
                        touching_resource=False,
                        surprise=float(reward_details.get('surprise', 0.0)),
                        hunger=float(drives.get('hunger', 0)),
                        thirst=float(drives.get('thirst', 0)),
                        cold=float(drives.get('cold', 0)),
                        curiosity=float(drives.get('curiosity', 0)),
                        satisfaction=float(reward_details.get('satisfaction', 0.0)),
                        discomfort=float(reward_details.get('discomfort', 0.0)),
                        consumed=consumed_what,
                    )))
                
                # Visualization
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]):
                        running = False
                        break
                
                # Draw
                viz.draw(
                    agent, resources, world,
                    trainer,  # Pass trainer object
                    cycle,
                    step,
                    total_consumptions,
                    last_reward_details,  # Not reward_details!
                )
                
                if args.fps > 0:
                    viz.clock.tick(args.fps)
                
                # Check if agent died
                if done:
                    # Save death position
                    death_position = agent.position.copy()
                    
                    training_log.deaths.append(asdict(DeathEvent(
                        tick=world.tick,
                        cycle=cycle,
                        cause='starvation/dehydration/hypothermia',
                        x=float(agent.position[0]),
                        y=float(agent.position[1]),
                        energy=float(agent.energy),
                        hydration=float(agent.hydration),
                        temperature=float(agent.temperature),
                        life_duration=world.tick - cycle_start_life,
                    )))
                    
                    # Respawn at same position where agent died
                    agent.reset(
                        position=death_position,
                        after_death=True,
                    )
                    agent.heading = np.random.uniform(0, 2 * np.pi)  # Random heading
                    drive_system.reset()
                    cycle_start_life = world.tick
            
            if not running:
                break
            
            # Cycle summary
            print(f"Cycle {cycle:3d}: Consumptions={cycle_consumptions:2d}, "
                  f"Total={total_consumptions:3d}, Deaths={len(training_log.deaths):2d}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        viz.close()
        
        # Finalize log
        training_log.total_ticks = world.tick
        training_log.total_cycles = cycle
        training_log.total_deaths = len(training_log.deaths)
        training_log.wall_time_seconds = time.time() - start_time_seconds
        
        # Save log
        log_path = save_training_log(training_log, output_dir)
        
        # Print learning diagnostics
        trainer.print_diagnostics()
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total cycles: {cycle}")
        print(f"Total consumptions: {total_consumptions}")
        print(f"Total deaths: {len(training_log.deaths)}")
        print(f"Wall time: {training_log.wall_time_seconds:.1f}s")
        print(f"Log saved: {log_path}")
        print(f"{'='*80}\n")
        
        print(f"\n🧠 NCP DRIVE-MODULATED ARCHITECTURE NOTES:")
        print(f"  ✅ NCP sparse wiring (C. elegans)")
        print(f"  ✅ CfC continuous-time dynamics")
        print(f"  ✅ Internal drive modulation (no external hacks!)")
        print(f"  ✅ Winner-take-all emerged from brain dynamics")
        print(f"  ✅ Drives modulated interneurons internally")
        print(f"  ✅ 49 sensors with light + ambient temp")
        print(f"  ✅ End-to-end learning with REINFORCE")


if __name__ == '__main__':
    main()
