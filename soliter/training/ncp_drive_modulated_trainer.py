"""
Trainer for Drive-Modulated NCP Brain.

Uses REINFORCE policy gradient with the upgraded NCP architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from ..core.ncp_drive_modulated_brain import DriveModulatedNCPBrain
from ..agents.soliter_agent import SoliterAgent
from ..environment import World
from ..environment.drive_modulated_sensors import DriveModulatedSensorSystem
from ..core.drive_system import DriveSystem


@dataclass
class NCPDriveModulatedTrainerConfig:
    """Configuration for NCP drive-modulated trainer."""
    learning_rate: float = 0.001
    action_std_init: float = 0.3
    action_std_min: float = 0.05
    action_std_decay: float = 0.9999
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0


class NCPDriveModulatedTrainer:
    """
    Trainer for NCP brain with internal drive modulation.
    
    Key difference from standard trainer:
      - Brain has internal drive modulation (no external hacks!)
      - Drives passed to brain.forward(sensors, drives)
      - NCP handles sparse connectivity automatically
    """
    
    def __init__(
        self,
        agent: SoliterAgent,
        world: World,
        drive_system: DriveSystem,
        sensors: DriveModulatedSensorSystem,
        config: NCPDriveModulatedTrainerConfig,
        device: torch.device,
    ):
        self.agent = agent
        self.world = world
        self.drive_system = drive_system
        self.sensors = sensors
        self.config = config
        self.device = device
        
        # Verify agent has drive-modulated NCP brain
        assert isinstance(agent.brain, DriveModulatedNCPBrain), \
            "Agent must have DriveModulatedNCPBrain!"
        
        # Value head (for baseline)
        self.value_head = nn.Sequential(
            nn.Linear(49, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)
        
        # Optimizer for brain + value head
        all_params = (
            list(self.agent.brain.parameters()) +
            list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)
        
        # Action standard deviation (exploration)
        self.action_log_std = nn.Parameter(
            torch.ones(3, device=device) * np.log(config.action_std_init)
        )
        self.optimizer.add_param_group({'params': [self.action_log_std]})
        
        # Learning diagnostics
        self.learning_diagnostics = {
            'total_updates': 0,
            'weight_changes': [],
            'losses': [],
            'gradient_norms': [],
            'drive_modulation_weights': [],  # Track how drives evolve
        }
        
        # Store initial weights
        self.initial_weights = {
            name: param.clone().detach()
            for name, param in self.agent.brain.named_parameters()
        }
        
        # Track consumption
        self._consumed_this_tick = None
    
    def wake_step(
        self,
        resources: Dict,
    ) -> Tuple[float, bool, dict]:
        """
        Single wake step with NCP drive-modulated brain.
        
        Returns:
            reward: Scalar reward
            done: Whether episode ended
            details: Info dict
        """
        # Snapshot vitals BEFORE action
        self.drive_system.snapshot_vitals(
            self.agent.energy,
            self.agent.hydration,
            self.agent.temperature,
        )
        
        # Get drive vector (4 values)
        drives_array = self.drive_system.get_drive_vector(
            self.agent.energy,
            self.agent.hydration,
            self.agent.temperature,
        )
        
        # Get RAW sensors (49 channels, NO drives!)
        sensor_noise = self.agent.get_sensor_noise()
        sensors = self._get_sensors(resources, sensor_noise)
        
        # Convert to tensors
        sensors_tensor = torch.tensor(sensors, dtype=torch.float32, device=self.device)
        drives_tensor = torch.tensor(drives_array, dtype=torch.float32, device=self.device)
        
        # Select action (NCP with internal drive modulation!)
        action, value, log_prob = self._select_action(sensors_tensor, drives_tensor)
        velocity = action[0].item()
        turn = action[1].item()
        should_sleep = action[2].item() > 0.5
        
        # Execute movement
        world_bounds = (self.world.config.width, self.world.config.height)
        self.agent.move(velocity, turn, allow_turning=True, world_bounds=world_bounds, dt=1.0)
        
        # Update vitals
        ambient_temp = self.world.get_ambient_temperature()
        self.agent.update_vitals(velocity, ambient_temp, turn=turn, dt=1.0)
        
        # Check resource consumption
        self._consumed_this_tick = None
        self._check_resource_consumption(resources)
        
        # Enter sleep if needed
        if should_sleep and self.agent.wakefulness < 0.3:
            self.agent.enter_sleep()
        
        # Check death
        done = not self.agent.is_alive
        
        # Compute reward (ONLY from drive system, NO external penalties!)
        reward, reward_details = self.drive_system.compute_internal_reward(
            energy=self.agent.energy,
            hydration=self.agent.hydration,
            temperature=self.agent.temperature,
            consumed_resource=self._consumed_this_tick,
            is_dead=done,
        )
        
        return reward, done, {
            **reward_details,
            'velocity': velocity,
            'turn': turn,
            'action': action.cpu().numpy(),
            'value': value.item(),
            'log_prob': log_prob.item(),
        }
    
    def _get_sensors(
        self,
        resources: Dict,
        sensor_noise: float,
    ) -> np.ndarray:
        """Get RAW sensor readings (49 channels)."""
        light_level = self.world.get_light_level()
        ambient_temp = self.world.get_ambient_temperature()
        
        sensors = self.sensors.get_sensor_readings(
            agent_position=self.agent.position,
            agent_heading=self.agent.heading,
            agent_vitals={
                'energy': self.agent.energy,
                'hydration': self.agent.hydration,
                'temperature': self.agent.temperature,
                'wakefulness': self.agent.wakefulness,
            },
            resources=resources,
            sensor_noise=sensor_noise,
            world_width=self.world.config.width,
            world_height=self.world.config.height,
            light_level=light_level,
            ambient_temp=ambient_temp,
        )
        
        return sensors.cpu().numpy()
    
    def _select_action(
        self,
        sensors: torch.Tensor,
        drives: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using NCP brain with internal drive modulation.
        
        The magic: Drives modulate interneurons INSIDE the brain!
        """
        # Brain forward pass (drive modulation happens internally!)
        mean_action, _ = self.agent.brain(sensors, drives)
        
        # Create Gaussian distribution
        action_std = torch.exp(self.action_log_std)
        action_dist = torch.distributions.Normal(mean_action, action_std)
        
        # Sample action
        sampled_action = action_dist.sample()
        
        # Compute log probability
        log_prob = action_dist.log_prob(sampled_action).sum()
        
        # Constrain to valid ranges
        action_constrained = torch.stack([
            torch.sigmoid(sampled_action[0]),      # velocity [0, 1]
            torch.tanh(sampled_action[1]) * 0.5,   # turn [-0.5, 0.5] rad
            torch.sigmoid(sampled_action[2]),      # sleep [0, 1]
        ])
        
        # Get value estimate
        value = self.value_head(sensors)
        
        return action_constrained, value, log_prob
    
    def _check_resource_consumption(self, resources: Dict):
        """Check if agent consumed a resource."""
        self._consumed_this_tick = None
        seasonal_period = self.world.config.seasonal_period
        
        # Check food
        for feeder in resources.get('feeders', []):
            if feeder.is_agent_in_consumption_range(self.agent.position):
                if feeder.can_consume():
                    amount = feeder.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('food', amount)
                        self._consumed_this_tick = 'food'
                        return
        
        # Check water
        for fountain in resources.get('fountains', []):
            if fountain.is_agent_in_consumption_range(self.agent.position):
                if fountain.can_consume():
                    amount = fountain.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('water', amount)
                        self._consumed_this_tick = 'water'
                        return
        
        # Check heat
        for heater in resources.get('heaters', []):
            if heater.is_agent_in_consumption_range(self.agent.position):
                if heater.can_consume():
                    amount = heater.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('heat', amount)
                        self._consumed_this_tick = 'heat'
                        return
    
    def update(
        self,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ):
        """Update brain using REINFORCE."""
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # Compute advantage
        advantage = reward_tensor - value.detach()
        
        # Policy loss (REINFORCE)
        policy_loss = -log_prob * advantage
        
        # Value loss
        value_loss = ((value - reward_tensor) ** 2)
        
        # Entropy bonus
        action_std = torch.exp(self.action_log_std)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * action_std ** 2).sum()
        entropy_bonus = -self.config.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.config.value_coef * value_loss + entropy_bonus
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.agent.brain.parameters(),
            self.config.max_grad_norm
        )
        
        # Update
        self.optimizer.step()
        
        # Decay exploration
        with torch.no_grad():
            self.action_log_std.data = torch.max(
                self.action_log_std.data * self.config.action_std_decay,
                torch.ones_like(self.action_log_std.data) * np.log(self.config.action_std_min)
            )
        
        # Track diagnostics
        self.learning_diagnostics['total_updates'] += 1
        self.learning_diagnostics['losses'].append(total_loss.item())
        
        # Track drive modulation weights (how drives evolve!)
        if hasattr(self.agent.brain.cfc, 'drive_modulation_weights'):
            drive_weights = self.agent.brain.cfc.drive_modulation_weights.detach().cpu().numpy()
            self.learning_diagnostics['drive_modulation_weights'].append(drive_weights.copy())
        
        # Weight change
        total_weight_change = 0.0
        for name, param in self.agent.brain.named_parameters():
            if name in self.initial_weights:
                change = (param - self.initial_weights[name]).abs().mean().item()
                total_weight_change += change
        self.learning_diagnostics['weight_changes'].append(total_weight_change)
        
        # Gradient norm
        grad_norm = 0.0
        for param in self.agent.brain.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)
        self.learning_diagnostics['gradient_norms'].append(grad_norm)
    
    def print_diagnostics(self):
        """Print learning diagnostics including NCP-specific stats."""
        if len(self.learning_diagnostics['losses']) == 0:
            return
        
        print(f"\n{'='*80}")
        print("NCP DRIVE-MODULATED LEARNING DIAGNOSTICS")
        print(f"{'='*80}")
        
        # Wiring stats
        stats = self.agent.brain.get_wiring_stats()
        print(f"\nNCP Wiring:")
        print(f"  Total units: {stats['total_units']}")
        print(f"  Interneurons: {stats['inter_neurons']}")
        print(f"  Command neurons: {stats['command_neurons']}")
        print(f"  Motor neurons: {stats['motor_neurons']}")
        print(f"  Connections: {stats['total_connections']}")
        print(f"  Sparsity: {stats['sparsity']:.1%}")
        
        # Learning stats
        print(f"\nLearning:")
        print(f"  Total updates: {self.learning_diagnostics['total_updates']}")
        
        losses = self.learning_diagnostics['losses']
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        
        weight_changes = self.learning_diagnostics['weight_changes']
        print(f"  Weight change: {weight_changes[-1]:.6f}")
        
        gradient_norms = self.learning_diagnostics['gradient_norms']
        print(f"  Gradient norm: {np.mean(gradient_norms):.4f}")
        
        # Drive modulation weights evolution
        if len(self.learning_diagnostics['drive_modulation_weights']) > 0:
            drive_weights = self.learning_diagnostics['drive_modulation_weights'][-1]
            print(f"\nDrive Modulation Weights (learned):")
            print(f"  Food group: {drive_weights[0].mean():.3f} ± {drive_weights[0].std():.3f}")
            print(f"  Water group: {drive_weights[1].mean():.3f} ± {drive_weights[1].std():.3f}")
            print(f"  Heat group: {drive_weights[2].mean():.3f} ± {drive_weights[2].std():.3f}")
            print(f"  Explore group: {drive_weights[3].mean():.3f} ± {drive_weights[3].std():.3f}")
        
        print(f"{'='*80}\n")
