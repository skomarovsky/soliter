"""
Trainer 3: Evolutionary Strategy (ES).

Gradient-free optimization. No backprop needed!

Algorithm (OpenAI ES, Salimans 2017):
1. At each update step, sample N perturbations of weights
2. Evaluate each perturbation (forward pass only)
3. Compute weighted average of perturbations by reward
4. Update weights in direction of highest-reward perturbations

Key advantage: Works perfectly with sparse rewards!
Doesn't need differentiable reward signal.
Naturally explores the weight space.

For real-time: Use single-perturbation "online ES":
- Keep a running perturbation
- Scale it by reward signal
- Periodically reset perturbation

This is fast enough to run every step.
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
class ESConfig:
    sigma: float = 0.02              # Perturbation noise magnitude
    learning_rate: float = 0.01      # How fast to move weights
    update_every: int = 50           # Steps between weight updates
    reward_scale: float = 1.0        # Scale reward signal
    action_std: float = 0.3          # Action exploration noise
    antithetic: bool = True          # Use +noise and -noise pairs (reduces variance!)


class ESTrainer:
    """
    Online Evolutionary Strategy trainer.
    
    Uses antithetic sampling (paired +noise/-noise) to reduce variance.
    
    Runs as:
    - Every N steps, compare +noise vs -noise performance
    - Update weights in direction of better-performing perturbation
    - Clear perturbation and repeat
    
    Much better credit assignment than REINFORCE:
    - Compares two trajectories directly
    - No log-prob issues
    - Works with any reward, even binary
    """
    
    name = "Evolutionary Strategy"
    
    def __init__(self, agent, world, drive_system, sensors, config, device):
        self.agent = agent
        self.world = world
        self.drive_system = drive_system
        self.sensors = sensors
        self.config = config
        self.device = device
        
        assert isinstance(agent.brain, DriveModulatedNCPBrain)
        
        # Store base weights (what we're actually updating)
        self.base_weights = {
            name: param.clone().detach()
            for name, param in agent.brain.named_parameters()
            if param.requires_grad
        }
        
        # Current noise perturbation
        self.current_noise = self._generate_noise()
        self.noise_sign = 1.0  # +1 or -1 (antithetic)
        
        # Buffer for collecting rewards per perturbation
        self.positive_rewards = []
        self.negative_rewards = []
        self.step_count = 0
        self.phase = 'positive'  # Currently evaluating + or - perturbation
        
        # Action exploration
        self.action_std = config.action_std
        
        # Apply initial perturbation
        self._apply_perturbation(sign=1.0)
        
        # Diagnostics
        self.learning_diagnostics = {
            'total_updates': 0, 'losses': [], 'weight_changes': [],
            'gradient_norms': [], 'drive_modulation_weights': [],
        }
        self.initial_weights = {
            name: param.clone().detach()
            for name, param in self.agent.brain.named_parameters()
        }
        self._consumed_this_tick = None

    def _generate_noise(self):
        """Generate random perturbation of same shape as all parameters."""
        return {
            name: torch.randn_like(param) * self.config.sigma
            for name, param in self.agent.brain.named_parameters()
            if param.requires_grad
        }

    def _apply_perturbation(self, sign: float):
        """Apply base_weights + sign * noise to brain."""
        with torch.no_grad():
            for name, param in self.agent.brain.named_parameters():
                if name in self.base_weights:
                    param.data.copy_(
                        self.base_weights[name] + sign * self.current_noise[name]
                    )

    def _update_weights(self):
        """
        ES weight update using antithetic rewards.
        
        If + perturbation got more reward than - perturbation:
          Move weights in + direction
        """
        pos_avg = np.mean(self.positive_rewards) if self.positive_rewards else 0.0
        neg_avg = np.mean(self.negative_rewards) if self.negative_rewards else 0.0
        
        # Fitness difference (+ vs -)
        fitness_diff = pos_avg - neg_avg
        
        # Update base weights in direction of + noise scaled by fitness
        with torch.no_grad():
            for name in self.base_weights:
                noise = self.current_noise[name]
                # ES gradient estimate: (fitness_diff / 2*sigma) * noise
                update = (self.config.learning_rate * fitness_diff / 
                         (2.0 * self.config.sigma)) * noise
                self.base_weights[name].add_(update)
        
        # Clear rewards and generate new noise
        self.positive_rewards.clear()
        self.negative_rewards.clear()
        self.current_noise = self._generate_noise()
        
        # Start new + phase
        self.phase = 'positive'
        self._apply_perturbation(sign=1.0)
        
        self._track_diagnostics(abs(fitness_diff))

    def wake_step(self, resources: Dict) -> Tuple[float, bool, dict]:
        """Single step with ES."""
        # Sensors
        sensor_np = self._get_sensors(resources)
        sensor_t = torch.FloatTensor(sensor_np).to(self.device)
        
        # Drives
        drive_vector = self.drive_system.get_drive_vector(
            self.agent.energy, self.agent.hydration, self.agent.temperature
        )
        drives = torch.FloatTensor(drive_vector).to(self.device)
        
        # Forward pass (no gradient needed!)
        with torch.no_grad():
            action_mean, _ = self.agent.brain(sensor_t, drives)
        
        # Exploration noise on actions
        noise = torch.randn_like(action_mean) * self.action_std
        action_raw = action_mean + noise
        
        velocity = float(torch.sigmoid(action_raw[0]).item())
        turn = float(torch.tanh(action_raw[1]).item() * 0.5)
        
        # Apply action
        self.agent.move(
            velocity=velocity * 2.0,
            turn=turn,
            world_bounds=(self.world.config.width, self.world.config.height),
        )
        
        # Check consumption
        self._check_resource_consumption(resources)
        
        # Update vitals
        self.agent.update_vitals(
            velocity=self.agent.last_velocity,
            ambient_temperature=self.world.get_ambient_temperature(),
            turn=self.agent.last_turn,
            dt=1.0,
        )
        
        # Compute reward
        reward, details = self.drive_system.compute_internal_reward(
            energy=self.agent.energy,
            hydration=self.agent.hydration,
            temperature=self.agent.temperature,
            consumed_resource=self._consumed_this_tick,
            is_dead=not self.agent.is_alive,
        )
        
        # Collect reward for current perturbation phase
        if self.phase == 'positive':
            self.positive_rewards.append(reward)
        else:
            self.negative_rewards.append(reward)
        
        self.step_count += 1
        
        # Halfway through: switch to negative perturbation
        if self.config.antithetic:
            halfway = self.config.update_every // 2
            if self.step_count % self.config.update_every == halfway:
                if self.phase == 'positive':
                    self.phase = 'negative'
                    self._apply_perturbation(sign=-1.0)
        
        # End of cycle: update weights
        if self.step_count % self.config.update_every == 0:
            self._update_weights()
        
        done = not self.agent.is_alive
        return reward, done, details

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
        
        # Only consume if actually needed (not already full/satisfied)
        needs_food = self.agent.energy < 80.0  # Eat when getting low
        needs_water = self.agent.hydration < 80.0  # Drink when getting low
        needs_heat = self.agent.temperature < 34.0  # Too cold (normal is 37°C)
        needs_cooling = self.agent.temperature > 40.0  # Too hot (danger at 45°C)
        
        for feeder in resources.get('feeders', []):
            if not needs_food:
                continue  # Skip if already full!
            if feeder.is_agent_in_consumption_range(self.agent.position):
                if feeder.can_consume():
                    amount = feeder.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('food', amount)
                        self._consumed_this_tick = 'food'
                        return
        
        for fountain in resources.get('fountains', []):
            if not (needs_water or needs_cooling):
                continue  # Skip if already hydrated and not overheating!
            if fountain.is_agent_in_consumption_range(self.agent.position):
                if fountain.can_consume():
                    amount = fountain.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('water', amount)
                        self._consumed_this_tick = 'water'
                        return
        
        for heater in resources.get('heaters', []):
            if not needs_heat:
                continue  # Skip if already warm!
            if heater.is_agent_in_consumption_range(self.agent.position):
                if heater.can_consume():
                    amount = heater.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('heat', amount)
                        self._consumed_this_tick = 'heat'
                        return

    
    def _track_diagnostics(self, fitness):
        self.learning_diagnostics['total_updates'] += 1
        self.learning_diagnostics['losses'].append(-fitness)
        if hasattr(self.agent.brain.cfc, 'drive_modulation_weights'):
            dw = self.agent.brain.cfc.drive_modulation_weights.detach().cpu().numpy()
            self.learning_diagnostics['drive_modulation_weights'].append(dw.copy())
        total_change = sum(
            (base - self.initial_weights[name]).abs().mean().item()
            for name, base in self.base_weights.items()
            if name in self.initial_weights
        )
        self.learning_diagnostics['weight_changes'].append(total_change)
        self.learning_diagnostics['gradient_norms'].append(0.0)

    def sleep_cycle(self):
        pass

    def get_diagnostics(self):
        return self.learning_diagnostics
