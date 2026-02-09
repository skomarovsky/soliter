#!/usr/bin/env python3
"""
Training script with real-time pygame visualization.

Shows:
- Agent position and heading
- Resources (color-coded: green=food, blue=water, red=heat)
- Vitals bars
- Drive states
- Current gradient directions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import pygame
import time

from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.utils.config import config
from soliter.environment import (
    World, WorldConfig,
    create_default_resources,
    Physics, SensorSystem, SensorConfig,
)
from soliter.training import SleepWakeTrainer, TrainingConfig


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
        
        # Draw resources
        for feeder in resources.get('feeders', []):
            pos = (int(feeder.position[0] * self.scale), int(feeder.position[1] * self.scale))
            radius = int(feeder.radius * self.scale)
            pygame.draw.circle(self.screen, GREEN, pos, radius, 2)
            pygame.draw.circle(self.screen, GREEN, pos, 3)
            
        for fountain in resources.get('fountains', []):
            pos = (int(fountain.position[0] * self.scale), int(fountain.position[1] * self.scale))
            radius = int(fountain.radius * self.scale)
            pygame.draw.circle(self.screen, BLUE, pos, radius, 2)
            pygame.draw.circle(self.screen, BLUE, pos, 3)
            
        for heater in resources.get('heaters', []):
            pos = (int(heater.position[0] * self.scale), int(heater.position[1] * self.scale))
            radius = int(heater.radius * self.scale)
            pygame.draw.circle(self.screen, RED, pos, radius, 2)
            pygame.draw.circle(self.screen, RED, pos, 3)
        
        # Draw agent
        agent_pos = (int(agent.position[0] * self.scale), int(agent.position[1] * self.scale))
        
        # Agent body
        color = WHITE if agent.is_alive else GRAY
        pygame.draw.circle(self.screen, color, agent_pos, int(agent.radius * self.scale))
        
        # Agent heading (direction arrow)
        heading_len = 20
        heading_end = (
            agent_pos[0] + int(heading_len * np.cos(agent.heading)),
            agent_pos[1] + int(heading_len * np.sin(agent.heading))
        )
        pygame.draw.line(self.screen, YELLOW, agent_pos, heading_end, 3)
        
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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    
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
                
                # Visualization
                viz.draw(agent, resources, world, trainer, cycle, step + 1, 
                        total_consumptions, last_reward_details)
                
                if args.fps > 0:
                    viz.clock.tick(args.fps)
                
                running = viz.check_events()
                
                # Death
                if done:
                    last_death_location = agent.position.copy()
                    agent.reset()
                    agent.position = last_death_location + np.random.uniform(-20, 20, size=2)
                    agent.position = np.clip(agent.position, 0, [world_width - 1, world_height - 1])
                    trainer.drive_system.reset()
            
            if not running:
                break
            
            # Sleep
            print(f"Cycle {cycle:3d}: Consumptions={cycle_consumptions:2d}, "
                  f"Total={total_consumptions:3d}, Buffer={len(trainer.replay_buffer):5d}")
            trainer.sleep_cycle()
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        viz.close()
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total cycles: {cycle}")
        print(f"Total consumptions: {total_consumptions}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
