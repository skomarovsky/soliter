"""
Trainer 2: Hebbian Learning.

Biologically plausible: no backprop!

Rule: When neuron A fires and reward follows,
strengthen connection A→B.

Implementation:
  Δw_ij = lr * pre_i * post_j * reward_signal

This is a neuromodulated Hebbian rule:
  - reward > 0: strengthen active connections (LTP)
  - reward < 0: weaken active connections (LTD)
  - No reward: no change

Perfect for continuous sparse-reward navigation
because it's local and online - no episode boundaries needed.

Biological basis: Dopaminergic modulation of
Hebbian plasticity (Schultz 1997, Reynolds 2001).
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
class HebbianConfig:
    learning_rate: float = 0.0001      # Small - Hebbian can be unstable
    reward_decay: float = 0.95         # Eligibility trace decay
    weight_decay: float = 0.0001       # Prevent runaway weights (forgetting)
    reward_scale: float = 0.1          # Scale reward signal
    action_std: float = 0.3            # Fixed exploration noise


class HebbianTrainer:
    """
    Neuromodulated Hebbian learning for NCP brain.
    
    Algorithm:
    1. Forward pass: record activations at each layer
    2. Compute reward signal
    3. Update each weight using local Hebbian rule:
       Δw = lr * pre * post * reward_eligibility
    4. Apply weight decay (homeostatic)
    
    Key: Uses eligibility traces so past activations
    can be credited with delayed rewards.
    """
    
    name = "Hebbian"
    
    def __init__(self, agent, world, drive_system, sensors, config, device):
        self.agent = agent
        self.world = world
        self.drive_system = drive_system
        self.sensors = sensors
        self.config = config
        self.device = device
        
        assert isinstance(agent.brain, DriveModulatedNCPBrain)
        
        # Fixed action noise (no learning here)
        self.action_std = config.action_std
        
        # Eligibility traces: one per learnable weight matrix
        self.eligibility_traces = {}
        for name, param in agent.brain.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self.eligibility_traces[name] = torch.zeros_like(param.data)
        
        # Hook to capture activations during forward pass
        self._pre_activations = {}
        self._post_activations = {}
        self._register_hooks()
        
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
        self._last_reward = 0.0

    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        brain = self.agent.brain
        
        # Hook on the CfC to capture activations
        def make_hook(layer_name):
            def hook(module, inp, output):
                # Store input (pre-synaptic) and output (post-synaptic)
                if isinstance(inp, tuple) and len(inp) > 0:
                    if isinstance(inp[0], torch.Tensor):
                        self._pre_activations[layer_name] = inp[0].detach()
                if isinstance(output, torch.Tensor):
                    self._post_activations[layer_name] = output.detach()
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    self._post_activations[layer_name] = output[0].detach()
            return hook
        
        # Register hooks on CfC layers
        if hasattr(brain.cfc, 'rnn_cell'):
            for i, layer in enumerate(brain.cfc.rnn_cell._layers):
                layer.register_forward_hook(make_hook(f'layer_{i}'))

    def wake_step(self, resources: Dict) -> Tuple[float, bool, dict]:
        """Single step with Hebbian learning."""
        # Sensors
        sensor_np = self._get_sensors(resources)
        sensor_t = torch.FloatTensor(sensor_np).to(self.device)
        
        # Drives
        drive_vector = self.drive_system.get_drive_vector(
            self.agent.energy, self.agent.hydration, self.agent.temperature
        )
        drives = torch.FloatTensor(drive_vector).to(self.device)
        
        # Forward pass (hooks capture activations)
        with torch.no_grad():
            action_mean, _ = self.agent.brain(sensor_t, drives)
        
        # Add exploration noise
        noise = torch.randn_like(action_mean) * self.action_std
        action_raw = action_mean + noise
        
        # Constrain actions
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
        
        # Hebbian update
        self._hebbian_update(sensor_t, reward)
        
        done = not self.agent.is_alive
        return reward, done, details

    def _hebbian_update(self, sensor_input: torch.Tensor, reward: float):
        """
        Apply neuromodulated Hebbian learning.
        
        Δw_ij = lr * pre_i * post_j * reward_eligibility
        
        Using eligibility traces for credit assignment.
        """
        reward_signal = reward * self.config.reward_scale
        
        with torch.no_grad():
            for name, param in self.agent.brain.named_parameters():
                if not param.requires_grad or param.dim() < 2:
                    continue
                
                # Update eligibility trace using current activations
                # For each weight, the trace is pre * post
                if name in self.eligibility_traces:
                    trace = self.eligibility_traces[name]
                    
                    # Compute Hebbian signal from activations
                    # Match weight matrix dimensions to activations
                    hebbian_signal = self._compute_hebbian_signal(
                        name, param, sensor_input
                    )
                    
                    if hebbian_signal is not None:
                        # Decay old trace, add new
                        trace.mul_(self.config.reward_decay)
                        trace.add_(hebbian_signal)
                        
                        # Apply: Δw = lr * trace * reward
                        delta_w = self.config.learning_rate * trace * reward_signal
                        
                        # Weight decay (homeostatic - prevents runaway)
                        decay = self.config.weight_decay * param.data
                        
                        # Update weight
                        param.data.add_(delta_w - decay)
        
        self._track_diagnostics(reward_signal)

    def _compute_hebbian_signal(self, name, param, sensor_input):
        """Compute pre * post signal for a weight matrix."""
        # Get layer activations from hooks
        layer_key = None
        if 'layer_0' in name:
            layer_key = 'layer_0'
        elif 'layer_1' in name:
            layer_key = 'layer_1'
        elif 'layer_2' in name:
            layer_key = 'layer_2'
        
        if layer_key is None:
            # drive_modulation_weights - use sensor input and output
            if 'drive_modulation' in name:
                # Simple: use magnitude as signal
                return torch.ones_like(param.data) * 0.01
            return None
        
        pre = self._pre_activations.get(layer_key)
        post = self._post_activations.get(layer_key)
        
        if pre is None or post is None:
            return None
        
        # Ensure 1D
        pre_flat = pre.flatten()
        post_flat = post.flatten()
        
        # Match dimensions to weight matrix
        target_shape = param.data.shape
        
        # Weight is [out, in], compute outer product
        if pre_flat.shape[0] == target_shape[1] and post_flat.shape[0] == target_shape[0]:
            signal = torch.outer(post_flat, pre_flat)
            return signal.clamp(-1.0, 1.0)
        
        return None

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

    def _track_diagnostics(self, reward_signal):
        self.learning_diagnostics['total_updates'] += 1
        self.learning_diagnostics['losses'].append(-reward_signal)
        if hasattr(self.agent.brain.cfc, 'drive_modulation_weights'):
            dw = self.agent.brain.cfc.drive_modulation_weights.detach().cpu().numpy()
            self.learning_diagnostics['drive_modulation_weights'].append(dw.copy())
        total_change = sum(
            (param - self.initial_weights[name]).abs().mean().item()
            for name, param in self.agent.brain.named_parameters()
            if name in self.initial_weights
        )
        self.learning_diagnostics['weight_changes'].append(total_change)
        self.learning_diagnostics['gradient_norms'].append(0.0)

    def sleep_cycle(self):
        pass

    def get_diagnostics(self):
        return self.learning_diagnostics
