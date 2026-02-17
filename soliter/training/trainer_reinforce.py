"""
Trainer 1: REINFORCE with Running Baseline.

Key improvements over naive REINFORCE:
- Running mean baseline (reduces variance dramatically)
- Reward normalization per update
- No value head needed - baseline is just a scalar mean
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from ..core.ncp_drive_modulated_brain import DriveModulatedNCPBrain
from ..agents.soliter_agent import SoliterAgent
from ..environment import World
from ..environment.drive_modulated_sensors import DriveModulatedSensorSystem
from ..core.drive_system import DriveSystem


@dataclass
class REINFORCEConfig:
    learning_rate: float = 0.003
    action_std_init: float = 0.5
    action_std_min: float = 0.05
    action_std_decay: float = 0.9999
    baseline_momentum: float = 0.99    # EMA momentum for baseline
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    reward_scale: float = 1.0


class REINFORCETrainer:
    """
    REINFORCE with exponential moving average baseline.
    
    The baseline b(s) reduces variance:
      loss = -log_prob * (reward - baseline)
    
    Baseline = running EMA of rewards:
      baseline = momentum * baseline + (1 - momentum) * reward
    
    This is much more stable than naive REINFORCE!
    """
    
    name = "REINFORCE+Baseline"
    
    def __init__(self, agent, world, drive_system, sensors, config, device):
        self.agent = agent
        self.world = world
        self.drive_system = drive_system
        self.sensors = sensors
        self.config = config
        self.device = device
        
        assert isinstance(agent.brain, DriveModulatedNCPBrain)
        
        # Optimizer (brain only, no value head!)
        self.optimizer = torch.optim.Adam(
            self.agent.brain.parameters(), lr=config.learning_rate
        )
        
        # Action exploration
        self.action_log_std = torch.nn.Parameter(
            torch.ones(3, device=device) * np.log(config.action_std_init)
        )
        self.optimizer.add_param_group({'params': [self.action_log_std]})
        
        # Running baseline (starts at 0)
        self.baseline = 0.0
        self.baseline_initialized = False
        
        # Diagnostics
        self.learning_diagnostics = {
            'total_updates': 0, 'losses': [], 'baseline_values': [],
            'weight_changes': [], 'gradient_norms': [],
            'drive_modulation_weights': [],
        }
        self.initial_weights = {
            name: param.clone().detach()
            for name, param in self.agent.brain.named_parameters()
        }
        self._consumed_this_tick = None

    def wake_step(self, resources: Dict) -> Tuple[float, bool, dict]:
        """Single step: sense → act → reward → update."""
        # Sensors
        sensor_np = self._get_sensors(resources)
        sensor_t = torch.FloatTensor(sensor_np).to(self.device)
        
        # Drives
        drive_vector = self.drive_system.get_drive_vector(
            self.agent.energy, self.agent.hydration, self.agent.temperature
        )
        drives = torch.FloatTensor(drive_vector).to(self.device)
        
        # Select action
        action, log_prob = self._select_action(sensor_t, drives)
        
        # Apply action
        velocity = float(action[0].item())
        turn = float(action[1].item())
        sleep_signal = float(action[2].item())
        
        self.agent.move(
            velocity=velocity * 2.0,
            turn=turn,
            world_bounds=(self.world.config.width, self.world.config.height),
        )
        
        # Check consumption
        self._check_resource_consumption(resources)
        
        # Update vitals
        self.agent.update_vitals(
            self.world.get_ambient_temperature(),
            self.world.config.seasonal_period,
        )
        
        # Compute reward
        reward, details = self.drive_system.compute_internal_reward(
            energy=self.agent.energy,
            hydration=self.agent.hydration,
            temperature=self.agent.temperature,
            consumed_resource=self._consumed_this_tick,
            is_dead=not self.agent.is_alive,
        )
        
        # Update brain
        self.update(reward, log_prob)
        
        done = not self.agent.is_alive
        return reward, done, details

    def update(self, reward: float, log_prob: torch.Tensor):
        """Update with REINFORCE + running baseline."""
        # Initialize baseline with first reward
        if not self.baseline_initialized:
            self.baseline = reward
            self.baseline_initialized = True
        
        # Compute advantage (reward - baseline)
        advantage = reward - self.baseline
        
        # Update baseline (EMA)
        self.baseline = (
            self.config.baseline_momentum * self.baseline +
            (1 - self.config.baseline_momentum) * reward
        )
        
        # Policy loss
        advantage_t = torch.tensor(advantage, dtype=torch.float32, device=self.device)
        policy_loss = -log_prob * advantage_t
        
        # Entropy bonus (encourages exploration)
        action_std = torch.exp(self.action_log_std)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * action_std ** 2).sum()
        
        total_loss = policy_loss - self.config.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.brain.parameters()) + [self.action_log_std],
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Decay exploration
        with torch.no_grad():
            self.action_log_std.data = torch.max(
                self.action_log_std.data * self.config.action_std_decay,
                torch.full_like(self.action_log_std.data, np.log(self.config.action_std_min))
            )
        
        self._track_diagnostics(total_loss)

    def _select_action(self, sensors, drives):
        mean_action, _ = self.agent.brain(sensors, drives)
        action_std = torch.exp(self.action_log_std)
        dist = torch.distributions.Normal(mean_action, action_std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum()
        action = torch.stack([
            torch.sigmoid(sampled[0]),
            torch.tanh(sampled[1]) * 0.5,
            torch.sigmoid(sampled[2]),
        ])
        return action, log_prob

    def _get_sensors(self, resources):
        return self.sensors.get_sensor_readings(
            agent_position=self.agent.position,
            agent_heading=self.agent.heading,
            agent_vitals={
                'energy': self.agent.energy,
                'hydration': self.agent.hydration,
                'temperature': self.agent.temperature,
                'wakefulness': self.agent.wakefulness,
            },
            resources=resources,
            sensor_noise=0.0,
            world_width=self.world.config.width,
            world_height=self.world.config.height,
            light_level=self.world.get_light_level(),
            ambient_temp=self.world.get_ambient_temperature(),
        ).cpu().numpy()

    def _check_resource_consumption(self, resources):
        self._consumed_this_tick = None
        seasonal_period = self.world.config.seasonal_period
        for feeder in resources.get('feeders', []):
            if feeder.is_agent_in_consumption_range(self.agent.position):
                if feeder.can_consume():
                    amount = feeder.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('food', amount)
                        self._consumed_this_tick = 'food'
                        return
        for fountain in resources.get('fountains', []):
            if fountain.is_agent_in_consumption_range(self.agent.position):
                if fountain.can_consume():
                    amount = fountain.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('water', amount)
                        self._consumed_this_tick = 'water'
                        return
        for heater in resources.get('heaters', []):
            if heater.is_agent_in_consumption_range(self.agent.position):
                if heater.can_consume():
                    amount = heater.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('heat', amount)
                        self._consumed_this_tick = 'heat'
                        return

    def _track_diagnostics(self, loss):
        self.learning_diagnostics['total_updates'] += 1
        self.learning_diagnostics['losses'].append(loss.item())
        self.learning_diagnostics['baseline_values'].append(self.baseline)
        if hasattr(self.agent.brain.cfc, 'drive_modulation_weights'):
            dw = self.agent.brain.cfc.drive_modulation_weights.detach().cpu().numpy()
            self.learning_diagnostics['drive_modulation_weights'].append(dw.copy())
        total_change = sum(
            (param - self.initial_weights[name]).abs().mean().item()
            for name, param in self.agent.brain.named_parameters()
            if name in self.initial_weights
        )
        self.learning_diagnostics['weight_changes'].append(total_change)
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.agent.brain.parameters() if p.grad is not None
        ) ** 0.5
        self.learning_diagnostics['gradient_norms'].append(grad_norm)

    def sleep_cycle(self):
        pass

    def get_diagnostics(self):
        return self.learning_diagnostics
