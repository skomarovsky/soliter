#!/usr/bin/env python3
"""
Training script with real-time pygame visualization.

Shows:
- Agent position and heading
- Resources (color-coded: green=food, blue=water, red=heat)
- Vitals bars
- Drive states
- Current gradient directions

NOW WITH COMPLETE LOGGING!
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
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from datetime import datetime

from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
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
    hunger: float
    thirst: float
    cold: float
    curiosity: float
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


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
CYAN = (0, 255, 255)


# ===================================================================
# LOGGING HELPERS
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


def save_training_log(log: TrainingLog, output_dir: Path):
    """Save training log as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'training_{timestamp}.json'
    
    with open(log_path, 'w') as f:
        json.dump(asdict(log), f, indent=2)
    
    print(f"\n✅ Training log saved: {log_path}")
    return log_path


class Visualizer:
    def __init__(self, world_width, world_height, scale=4):
        pygame.init()
        self.scale = scale
        self.width = world_width * scale
        self.height = world_height * scale + 150  # Extra space for stats
        self.world_height = world_height * scale
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Soliter Agent - Live Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
    def draw(self, agent, resources, world, trainer, cycle, step, total_consumptions, last_reward_details):
        self.screen.fill(BLACK)
        
        # Draw resources with depletion visualization
        # Two circles: detection radius (thin) and consumption radius (thicker)
        for feeder in resources.get('feeders', []):
            pos = (int(feeder.position[0] * self.scale), int(feeder.position[1] * self.scale))
            
            # Depletion ratio: 1.0 = full (bright), 0.0 = empty (dim)
            depletion = feeder.get_depletion_ratio()
            color_intensity = int(255 * depletion)
            color = (0, color_intensity, 0)  # Fade from dark to bright green
            
            # Detection radius (very thin, light green) - gradient sensing range
            detection_radius = int(feeder.get_detection_radius(False) * self.scale)
            pygame.draw.circle(self.screen, (150, 255, 150), pos, detection_radius, 1)
            
            # Consumption radius (thicker, bright green) - MUST BE THIS CLOSE
            consumption_radius = int(feeder.get_consumption_radius(False) * self.scale)
            pygame.draw.circle(self.screen, GREEN, pos, consumption_radius, 3)
            
            # Resource dot (the actual resource) - SMALL and visible
            pygame.draw.circle(self.screen, GREEN, pos, 5)
            
            # Show depletion percentage as text
            if depletion < 0.3:  # Only show when getting low
                text = self.small_font.render(f'{int(depletion*100)}%', True, ORANGE)
                self.screen.blit(text, (pos[0] + 8, pos[1] - 8))
            
        for fountain in resources.get('fountains', []):
            pos = (int(fountain.position[0] * self.scale), int(fountain.position[1] * self.scale))
            
            depletion = fountain.get_depletion_ratio()
            color_intensity = int(255 * depletion)
            color = (0, int(color_intensity * 0.4), color_intensity)  # Fade blue
            
            # Detection radius (thin, light blue)
            detection_radius = int(fountain.get_detection_radius(False) * self.scale)
            pygame.draw.circle(self.screen, (150, 200, 255), pos, detection_radius, 1)
            
            # Consumption radius (thick, bright blue)
            consumption_radius = int(fountain.get_consumption_radius(False) * self.scale)
            pygame.draw.circle(self.screen, BLUE, pos, consumption_radius, 3)
            
            pygame.draw.circle(self.screen, BLUE, pos, 5)
            
            if depletion < 0.3:
                text = self.small_font.render(f'{int(depletion*100)}%', True, ORANGE)
                self.screen.blit(text, (pos[0] + 8, pos[1] - 8))
            
        for heater in resources.get('heaters', []):
            pos = (int(heater.position[0] * self.scale), int(heater.position[1] * self.scale))
            
            depletion = heater.get_depletion_ratio()
            color_intensity = int(255 * depletion)
            color = (color_intensity, 0, 0)  # Fade from dark to bright red
            
            # Detection radius (thin, pink)
            detection_radius = int(heater.get_detection_radius(False) * self.scale)
            pygame.draw.circle(self.screen, (255, 150, 150), pos, detection_radius, 1)
            
            # Consumption radius (thick, bright red)
            consumption_radius = int(heater.get_consumption_radius(False) * self.scale)
            pygame.draw.circle(self.screen, RED, pos, consumption_radius, 3)
            
            pygame.draw.circle(self.screen, RED, pos, 5)
            
            if depletion < 0.3:
                text = self.small_font.render(f'{int(depletion*100)}%', True, ORANGE)
                self.screen.blit(text, (pos[0] + 8, pos[1] - 8))
        
        # Draw agent - OUTLINE ONLY with direction arrow
        agent_pos = (int(agent.position[0] * self.scale), int(agent.position[1] * self.scale))
        
        # Agent body - OUTLINE ONLY (not filled)
        color = WHITE if agent.is_alive else GRAY
        agent_radius = int(agent.radius * self.scale)
        pygame.draw.circle(self.screen, color, agent_pos, agent_radius, 2)  # thickness=2, hollow
        
        # Agent heading (direction arrow) - THICKER AND LONGER
        heading_len = 25
        heading_end = (
            agent_pos[0] + int(heading_len * np.cos(agent.heading)),
            agent_pos[1] + int(heading_len * np.sin(agent.heading))
        )
        pygame.draw.line(self.screen, YELLOW, agent_pos, heading_end, 4)  # Thicker arrow
        
        # Stats panel
        y_offset = self.world_height + 10
        
        # Cycle info
        text = self.font.render(f"Cycle: {cycle}  Step: {step}  Consumptions: {total_consumptions}", True, WHITE)
        self.screen.blit(text, (10, y_offset))
        
        # Vitals
        y_offset += 30
        vital_width = 200
        bar_height = 15
        
        # Energy
        energy_pct = agent.energy / 100.0
        pygame.draw.rect(self.screen, GRAY, (10, y_offset, vital_width, bar_height))
        pygame.draw.rect(self.screen, GREEN, (10, y_offset, int(vital_width * energy_pct), bar_height))
        text = self.small_font.render(f"Energy: {agent.energy:.0f}", True, WHITE)
        self.screen.blit(text, (220, y_offset))
        
        y_offset += 20
        
        # Hydration
        hydration_pct = agent.hydration / 100.0
        pygame.draw.rect(self.screen, GRAY, (10, y_offset, vital_width, bar_height))
        pygame.draw.rect(self.screen, BLUE, (10, y_offset, int(vital_width * hydration_pct), bar_height))
        text = self.small_font.render(f"Hydration: {agent.hydration:.0f}", True, WHITE)
        self.screen.blit(text, (220, y_offset))
        
        y_offset += 20
        
        # Temperature
        temp_pct = agent.temperature / 100.0
        pygame.draw.rect(self.screen, GRAY, (10, y_offset, vital_width, bar_height))
        pygame.draw.rect(self.screen, RED, (10, y_offset, int(vital_width * temp_pct), bar_height))
        text = self.small_font.render(f"Temp: {agent.temperature:.1f}°C", True, WHITE)
        self.screen.blit(text, (220, y_offset))
        
        y_offset += 20
        
        # Wakefulness
        wake_pct = agent.wakefulness
        pygame.draw.rect(self.screen, GRAY, (10, y_offset, vital_width, bar_height))
        pygame.draw.rect(self.screen, CYAN, (10, y_offset, int(vital_width * wake_pct), bar_height))
        text = self.small_font.render(f"Wake: {agent.wakefulness:.2f}", True, WHITE)
        self.screen.blit(text, (220, y_offset))
        
        # Drive states (right side)
        if last_reward_details:
            x_offset = 450
            y_offset = self.world_height + 10
            
            text = self.font.render("Drive States:", True, WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            
            y_offset += 30
            
            drives = [
                ('Hunger', last_reward_details.get('hunger_drive', 0), ORANGE),
                ('Thirst', last_reward_details.get('thirst_drive', 0), BLUE),
                ('Cold', last_reward_details.get('cold_drive', 0), CYAN),
            ]
            
            for drive_name, drive_val, drive_color in drives:
                pygame.draw.rect(self.screen, GRAY, (x_offset, y_offset, vital_width, bar_height))
                pygame.draw.rect(self.screen, drive_color, (x_offset, y_offset, int(vital_width * drive_val), bar_height))
                text = self.small_font.render(f"{drive_name}: {drive_val:.2f}", True, WHITE)
                self.screen.blit(text, (x_offset + 210, y_offset))
                y_offset += 20
        
        pygame.display.flip()
        
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
        return True
    
    def close(self):
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Train Soliter with visualization")
    parser.add_argument('--cycles', type=int, default=50)
    parser.add_argument('--wake-steps', type=int, default=2000)
    parser.add_argument('--fps', type=int, default=60, help='Visualization FPS (0=unlimited)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='experiments/viz_training',
                        help='Output directory for training logs')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    
    # Initialize logging
    output_dir = Path(args.output_dir)
    start_time_seconds = time.time()
    
    training_log = TrainingLog(
        start_time=datetime.now().isoformat(),
        device=str(device),
        config={
            'cycles': args.cycles,
            'wake_steps': args.wake_steps,
        },
        world_config={
            'width': config.world.size[0],
            'height': config.world.size[1],
        }
    )
    
    # Setup
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
    
    # Log resources
    training_log.resources = log_resources(resources)
    
    sensor_config = SensorConfig(
        gradient_scale_factor=world_width / 4.0,
        enable_gradients=True,
        enable_drive_input=True,
    )
    sensors = SensorSystem(config=sensor_config, physics=physics)
    
    brain = CfCBrain(sensory_size=51)
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
    
    last_death_location = agent.position.copy()
    
    training_config = TrainingConfig(
        wake_duration=args.wake_steps,
        learning_rate=config.training.learning_rate,
        batch_size=config.memory.batch_size,
        sleep_epochs=config.training.sleep_epochs,
        lambda_ewc=config.memory.ewc_lambda,
        fisher_decay=config.memory.fisher_decay,
        buffer_capacity=config.memory.buffer_max_size,
    )
    
    trainer = SleepWakeTrainer(agent, world, sensors, physics, training_config, device)
    
    # Visualization
    viz = Visualizer(world_width, world_height, scale=4)
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH VISUALIZATION")
    print(f"{'='*70}")
    print(f"Controls: ESC or Q to quit")
    print(f"FPS: {args.fps if args.fps > 0 else 'Unlimited'}")
    print(f"{'='*70}\n")
    
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
                
                # Training step
                reward, done, reward_details = trainer.wake_step(resources)
                world.step()
                last_reward_details = reward_details
                
                if reward_details.get('consumption_bonus', 0) > 0:
                    cycle_consumptions += 1
                    total_consumptions += 1
                
                # Log every 100 ticks
                if (world.tick % 100) == 0:
                    drives = trainer.drive_system.get_drives(
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
                        hunger=float(drives['hunger']),
                        thirst=float(drives['thirst']),
                        cold=float(drives['cold']),
                        curiosity=float(drives['curiosity']),
                        satisfaction=float(reward_details.get('satisfaction', 0.0)),
                        discomfort=float(reward_details.get('discomfort', 0.0)),
                        consumed=consumed_what,
                    )))
                
                # Visualization
                viz.draw(agent, resources, world, trainer, cycle, step + 1, 
                        total_consumptions, last_reward_details)
                
                if args.fps > 0:
                    viz.clock.tick(args.fps)
                
                running = viz.check_events()
                
                # Death
                if done:
                    last_death_location = agent.position.copy()
                    life_duration = world.tick - cycle_start_life
                    
                    # Log death
                    training_log.deaths.append(asdict(DeathEvent(
                        tick=world.tick,
                        cycle=cycle,
                        cause=agent.cause_of_death or 'unknown',
                        x=float(agent.position[0]),
                        y=float(agent.position[1]),
                        energy=float(agent.energy),
                        hydration=float(agent.hydration),
                        temperature=float(agent.temperature),
                        life_duration=life_duration,
                    )))
                    
                    agent.reset()
                    agent.position = last_death_location + np.random.uniform(-20, 20, size=2)
                    agent.position = np.clip(agent.position, 0, [world_width - 1, world_height - 1])
                    trainer.drive_system.reset()
                    cycle_start_life = world.tick
            
            if not running:
                break
            
            # Sleep - log sleep event
            print(f"Cycle {cycle:3d}: Consumptions={cycle_consumptions:2d}, "
                  f"Total={total_consumptions:3d}, Buffer={len(trainer.replay_buffer):5d}")
            
            sleep_metrics = trainer.sleep_cycle()
            
            # Log sleep
            training_log.sleeps.append(asdict(SleepEvent(
                tick=world.tick,
                cycle=cycle,
                buffer_size=len(trainer.replay_buffer),
                pruned=sleep_metrics.get('pruned', 0),
                surprise_min=float(sleep_metrics.get('surprise_min', 0.0)),
                surprise_mean=float(sleep_metrics.get('surprise_mean', 0.0)),
                surprise_max=float(sleep_metrics.get('surprise_max', 0.0)),
                surprise_cutoff=float(sleep_metrics.get('surprise_cutoff', 0.0)),
                policy_loss=float(sleep_metrics.get('policy_loss', 0.0)),
                value_loss=float(sleep_metrics.get('value_loss', 0.0)),
                entropy=float(sleep_metrics.get('entropy', 0.0)),
                action_std=float(np.exp(trainer.action_log_std.mean().item())),
                gated_out_this_cycle=sleep_metrics.get('gated_out', 0),
                total_consumptions=total_consumptions,
            )))
    
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
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total cycles: {cycle}")
        print(f"Total consumptions: {total_consumptions}")
        print(f"Total deaths: {len(training_log.deaths)}")
        print(f"Wall time: {training_log.wall_time_seconds:.1f}s")
        print(f"Log saved: {log_path}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
