#!/usr/bin/env python3
"""
Phase 5b: Full Training Script with Biological Drive System.

Runs the agent with gradient sensors + drive-based internal reward.
Logs drive states, satisfaction, consumption events alongside
all previous metrics.

Output: JSON log file + summary, plottable with plot_training.py
"""

import argparse
import json
import time
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import torch

from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent
from soliter.utils.config import config
from soliter.environment import (
    World, WorldConfig,
    create_default_resources,
    Physics, SensorSystem, SensorConfig,
)
from soliter.training import SleepWakeTrainer, TrainingConfig


# ===================================================================
# DATA STRUCTURES FOR LOGGING
# ===================================================================

@dataclass
class TickSnapshot:
    """Per-tick data point (logged every N ticks)."""
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
    surprise: float
    # NEW: drive states
    hunger: float
    thirst: float
    cold: float
    curiosity: float
    # NEW: reward breakdown
    satisfaction: float
    discomfort: float
    consumed: str  # 'food', 'water', 'heat', or ''


@dataclass
class DeathEvent:
    """Logged when agent dies."""
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
class SleepEvent:
    """Logged per sleep cycle."""
    tick: int
    cycle: int
    buffer_size: int
    pruned: int
    surprise_min: float
    surprise_mean: float
    surprise_max: float
    surprise_cutoff: float
    policy_loss: float
    value_loss: float
    entropy: float
    action_std: float
    gated_out_this_cycle: int
    # NEW: drive system stats
    total_consumptions: int


@dataclass
class ResourceSnapshot:
    """Static resource positions (logged once at start)."""
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


# ===================================================================
# TRAINING LOOP
# ===================================================================

def log_resources(resources: Dict) -> List[Dict]:
    """Snapshot all resource positions (called once)."""
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


def run_training(args):
    """Main training loop."""

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Create components with config values, using command-line args only if explicitly set
    # (default command-line value of 1000 should be ignored in favor of config)
    world_width = config.world.size[0] if args.world_size == 1000 else args.world_size
    world_height = config.world.size[1] if args.world_size == 1000 else args.world_size
    
    world_config = WorldConfig(
        width=world_width,
        height=world_height,
        seasonal_period=config.world.seasonal_period,
        diurnal_period=config.world.diurnal_period
    )
    world = World(world_config)
    resources = create_default_resources(world_width, world_height)
    physics = Physics()

    # Enhanced sensors (51 channels: 41 original + 6 gradients + 4 drives)
    sensor_config = SensorConfig(
        gradient_scale_factor=world_width / 4.0,
        enable_gradients=True,
        enable_drive_input=True,
    )
    sensors = SensorSystem(config=sensor_config, physics=physics)

    # Brain with 51 inputs (was 41)
    brain = CfCBrain(sensory_size=51)
    from soliter.agents.soliter_agent import VitalsConfig
    
    vitals_config = VitalsConfig(
        initial_energy=config.agent.initial_energy,
        initial_hydration=config.agent.initial_hydration,
        initial_temperature=config.agent.initial_temperature,
        initial_wakefulness=config.agent.initial_wakefulness,
        energy_decay_base=config.agent.energy_decay_base,
        hydration_decay_base=config.agent.hydration_decay_base,
        temperature_decay_rate=config.agent.temperature_decay_rate,
        wakefulness_decay_rate=config.agent.wakefulness_decay_rate,
        base_speed=config.agent.base_speed,
        base_turn_rate=config.agent.base_turn_rate,
        sleep_threshold=config.agent.sleep_threshold,
    )
    agent = SoliterAgent(brain, vitals_config, device)

    # Initial spawn: Use resource cluster center
    # Find resource cluster center
    all_resource_positions = []
    for rtype in ['feeders', 'fountains', 'heaters']:
        for r in resources[rtype]:
            all_resource_positions.append(r.position)
    
    if all_resource_positions:
        resource_center = np.mean(all_resource_positions, axis=0)
        # Initial spawn near resources for first life
        agent.position = resource_center + np.random.uniform(-10, 10, size=2)
        agent.position = np.clip(agent.position, 0, [world_width - 1, world_height - 1])
        print(f"Agent initial spawn at: ({agent.position[0]:.1f}, {agent.position[1]:.1f})")
        print(f"Resource cluster center: ({resource_center[0]:.1f}, {resource_center[1]:.1f})")
    else:
        # Fallback to center
        center = np.array([world_width / 2.0, world_height / 2.0])
        agent.position = center.copy()
        resource_center = center  # For respawn logic
    
    # Track last death location for natural respawn
    last_death_location = agent.position.copy()

    training_config = TrainingConfig(
        wake_duration=config.training.wake_duration or 10000,
        learning_rate=config.training.learning_rate,
        batch_size=config.memory.batch_size,
        sleep_epochs=config.training.sleep_epochs,
        surprise_gating=getattr(config.training, 'surprise_gating', True),
        surprise_momentum=getattr(config.training, 'surprise_momentum', 0.95),
        surprise_percentile=getattr(config.training, 'surprise_percentile', 0.3),
        prune_fraction=getattr(config.memory, 'prune_fraction', 0.2),
        lambda_ewc=config.memory.ewc_lambda,
        fisher_decay=config.memory.fisher_decay,
        action_std_init=getattr(config.training, 'action_std_init', 0.5),
        action_std_min=getattr(config.training, 'action_std_min', 0.1),
        action_std_decay=getattr(config.training, 'action_std_decay', 0.995),
        buffer_capacity=getattr(config.memory, 'buffer_max_size', 1000000),
    )

    trainer = SleepWakeTrainer(
        agent=agent,
        world=world,
        sensors=sensors,
        physics=physics,
        config=training_config,
        device=device,
    )

    # Initialize log
    log = TrainingLog(
        start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        config={
            'wake_steps': args.wake_steps,
            'cycles': args.cycles,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'sleep_epochs': args.sleep_epochs,
            'surprise_gating': args.surprise_gating,
            'surprise_momentum': args.surprise_momentum,
            'surprise_percentile': args.surprise_percentile,
            'prune_fraction': args.prune_fraction,
            'lambda_ewc': args.lambda_ewc,
            'fisher_decay': args.fisher_decay,
            'action_std_init': args.action_std_init,
            'action_std_min': args.action_std_min,
            'action_std_decay': args.action_std_decay,
            'world_size': world_width,
            'sensor_channels': 51,
        },
        world_config={
            'width': world_width,
            'height': world_height,
            'seasonal_period': world_config.seasonal_period,
            'diurnal_period': world_config.diurnal_period,
        },
        device=str(device),
        resources=log_resources(resources),
    )

    # Print resource map
    print(f"\n{'='*70}")
    print(f"WORLD: {world_width}x{world_height} | "
          f"{args.cycles} cycles x {args.wake_steps} steps")
    print(f"SENSORS: 51 channels (4 vitals + 36 rays + 1 touch + 6 gradients + 4 drives)")
    print(f"REWARD: Biological drive system (no shaped reward)")
    print(f"{'='*70}")
    print(f"\nResource positions:")
    for r in log.resources:
        print(f"  {r['resource_type']:>8} #{r['index']}: "
              f"({r['x']:.0f}, {r['y']:.0f}) r={r['radius']:.0f}")
    print(f"\nAgent start: ({agent.position[0]:.0f}, {agent.position[1]:.0f})")
    print(f"{'='*70}\n")

    # Training loop
    start_wall = time.time()
    global_tick = 0
    last_death_tick = 0
    gated_out_last_cycle = 0
    cycle_reward = 0.0
    last_reward_details = {}

    # Header
    print(f"{'Cyc':>4} {'D':>3} {'Cause':>10} {'Pos':>15} "
          f"{'Reward':>8} {'Satisf':>8} {'Discomf':>8} {'Consumed':>8} "
          f"{'Buf':>6} {'ActStd':>8}")
    print("-" * 110)

    for cycle in range(1, args.cycles + 1):
        cycle_reward = 0.0
        cycle_deaths = 0
        cycle_death_cause = ""
        cycle_consumptions = 0
        gated_before = trainer.total_gated_out

        for step in range(args.wake_steps):
            global_tick += 1

            # WAKE STEP (now returns reward_details)
            reward, done, reward_details = trainer.wake_step(resources)
            world.step()
            cycle_reward += reward
            last_reward_details = reward_details

            if reward_details.get('consumption_bonus', 0) > 0:
                cycle_consumptions += 1

            # LOG SNAPSHOT
            if global_tick % args.log_interval == 0:
                snap = TickSnapshot(
                    tick=global_tick,
                    cycle=cycle,
                    x=float(agent.position[0]),
                    y=float(agent.position[1]),
                    heading=float(getattr(agent, 'heading', 0.0)),
                    velocity=float(getattr(agent, 'last_velocity', 0.0)),
                    energy=float(agent.energy),
                    hydration=float(agent.hydration),
                    temperature=float(agent.temperature),
                    wakefulness=float(agent.wakefulness),
                    reward=float(reward),
                    ambient_temp=float(world.get_ambient_temperature()),
                    season=world.get_season().value,
                    is_night=world.is_night(),
                    touching_resource=trainer._consumed_this_tick is not None,
                    surprise=float(trainer.running_surprise) if args.surprise_gating else 0.0,
                    hunger=float(reward_details.get('hunger_drive', 0)),
                    thirst=float(reward_details.get('thirst_drive', 0)),
                    cold=float(reward_details.get('cold_drive', 0)),
                    curiosity=float(reward_details.get('curiosity_drive', 0)),
                    satisfaction=float(reward_details.get('satisfaction', 0)),
                    discomfort=float(reward_details.get('discomfort', 0)),
                    consumed=str(trainer._consumed_this_tick or ''),
                )
                log.snapshots.append(asdict(snap))

            # DEATH
            if done:
                cycle_deaths += 1

                if agent.energy <= 0:
                    cause = "starvation"
                elif agent.hydration <= 0:
                    cause = "dehydration"
                elif agent.temperature <= 0 or agent.temperature >= 100:
                    cause = "hypothermia" if agent.temperature <= 0 else "hyperthermia"
                else:
                    cause = "unknown"

                cycle_death_cause = cause
                life_dur = global_tick - last_death_tick

                death = DeathEvent(
                    tick=global_tick,
                    cycle=cycle,
                    cause=cause,
                    x=float(agent.position[0]),
                    y=float(agent.position[1]),
                    energy=float(agent.energy),
                    hydration=float(agent.hydration),
                    temperature=float(agent.temperature),
                    life_duration=life_dur,
                )
                log.deaths.append(asdict(death))
                last_death_tick = global_tick
                
                # Store death location for respawn
                last_death_location = agent.position.copy()

                # BIOLOGICAL RESPAWN: Agent respawns near where it died
                # This is more natural - the body doesn't teleport to resource center
                # The agent must learn to navigate from wherever it fails
                agent.reset()
                # Respawn near death location with variation (simulates new life nearby)
                agent.position = last_death_location + np.random.uniform(-20, 20, size=2)
                agent.position = np.clip(agent.position, 0, [world_width - 1, world_height - 1])
                trainer.drive_system.reset()

        # SLEEP
        sleep_stats = trainer.sleep_cycle()

        prn = sleep_stats['transitions_pruned']
        gated_this_cycle = trainer.total_gated_out - gated_before

        sleep_event = SleepEvent(
            tick=global_tick,
            cycle=cycle,
            buffer_size=sleep_stats['buffer_size'],
            pruned=prn.pruned,
            surprise_min=prn.surprise_min,
            surprise_mean=prn.surprise_mean,
            surprise_max=prn.surprise_max,
            surprise_cutoff=prn.surprise_cutoff,
            policy_loss=sleep_stats['policy_loss'],
            value_loss=sleep_stats['value_loss'],
            entropy=sleep_stats['entropy'],
            action_std=sleep_stats['action_std'],
            gated_out_this_cycle=gated_this_cycle,
            total_consumptions=trainer.drive_system.total_consumption_events,
        )
        log.sleeps.append(asdict(sleep_event))

        # Print cycle summary
        death_str = f"{cycle_deaths}" if cycle_deaths > 0 else "."
        cause_str = cycle_death_cause[:10] if cycle_death_cause else ""
        pos_str = f"({agent.position[0]:.0f},{agent.position[1]:.0f})"
        consumed_str = f"{cycle_consumptions}" if cycle_consumptions > 0 else "."

        print(f"{cycle:4d} {death_str:>3} {cause_str:>10} {pos_str:>15} "
              f"{cycle_reward:8.1f} "
              f"{last_reward_details.get('satisfaction', 0):8.3f} "
              f"{last_reward_details.get('discomfort', 0):8.3f} "
              f"{consumed_str:>8} "
              f"{sleep_stats['buffer_size']:6d} "
              f"{sleep_stats['action_std']:8.4f}")

    # Finalize
    wall_time = time.time() - start_wall
    log.total_ticks = global_tick
    log.total_cycles = args.cycles
    log.total_deaths = len(log.deaths)
    log.wall_time_seconds = wall_time

    # Save log
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"training_{timestamp}.json"

    with open(log_path, 'w') as f:
        json.dump(asdict(log), f, indent=2, default=str)

    # Save checkpoint
    ckpt_path = out_dir / f"checkpoint_{timestamp}.pt"
    trainer.save_checkpoint(str(ckpt_path))

    # Print summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Ticks: {global_tick:,}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Deaths: {len(log.deaths)}")
    if log.deaths:
        causes = {}
        for d in log.deaths:
            causes[d['cause']] = causes.get(d['cause'], 0) + 1
        for c, n in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"    {c}: {n} ({100*n/len(log.deaths):.0f}%)")
    print(f"  Resource consumptions: {trainer.drive_system.total_consumption_events}")
    print(f"  Wall time: {wall_time:.1f}s ({global_tick/wall_time:.0f} ticks/s)")
    print(f"  Log: {log_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"\nPlot with:")
    print(f"  python scripts/plot_training.py {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Soliter agent (v2 with drives)")

    # Only keep arguments that aren't in the config file
    parser.add_argument('--cycles', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='experiments')
    parser.add_argument('--wake-steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sleep-epochs', type=int, default=5)
    parser.add_argument('--surprise-gating', action='store_true', default=True)
    parser.add_argument('--surprise-momentum', type=float, default=0.95)
    parser.add_argument('--surprise-percentile', type=float, default=0.3)
    parser.add_argument('--prune-fraction', type=float, default=0.2)
    parser.add_argument('--lambda-ewc', type=float, default=155000.0)
    parser.add_argument('--fisher-decay', type=float, default=0.77)
    parser.add_argument('--action-std-init', type=float, default=0.5)
    parser.add_argument('--action-std-min', type=float, default=0.1)
    parser.add_argument('--action-std-decay', type=float, default=0.995)
    parser.add_argument('--world-size', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=1000)
    
    args = parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()