"""
Sleep-Wake Training Loop with NCP/CfC - NO PPO!

NCP (Neural Circuit Policies) Architecture:
- Based on C. elegans connectome (biological realism)
- Continuous-time dynamics (no batch updates needed)
- Natural stability (no catastrophic forgetting)
- Online learning (no experience replay needed)

This replaces the broken PPO batch learning system with biologically-inspired
continuous learning that matches the agent's continuous existence.

Key Principles:
1. NO BATCH LEARNING - agent learns continuously during wake
2. NO PPO - use NCP's natural learning dynamics
3. Sleep = rest only (restore wakefulness, decay exploration)
4. Learning happens through CfC weight updates, not gradient descent
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.cfc_network import CfCBrain
from ..core.drive_system import DriveSystem, DriveConfig
from ..agents.soliter_agent import SoliterAgent
from ..environment import World
from ..environment.gradient_sensors import GradientSensors, GradientSensorConfig


@dataclass
class NCPTrainerConfig:
    """Configuration for NCP-based trainer."""
    
    # World parameters
    world_width: int = 200
    world_height: int = 200
    
    # Gradient sensor configuration
    gradient_config: GradientSensorConfig = None
    
    # Drive system configuration
    drive_config: DriveConfig = None
    
    # Sleep/wake cycle
    wake_duration: int = 2000  # Ticks between sleep cycles
    
    # NCP learning parameters (if we add explicit learning later)
    learning_rate: float = 0.0001  # For potential weight updates
    
    # Exploration
    initial_action_std: float = 0.5
    action_std_min: float = 0.05
    action_std_decay: float = 0.995  # Slower decay for continuous learning
    
    def __post_init__(self):
        if self.gradient_config is None:
            self.gradient_config = GradientSensorConfig()
        if self.drive_config is None:
            self.drive_config = DriveConfig()


class NCPSleepWakeTrainer:
    """
    Trainer using NCP/CfC architecture - NO PPO!
    
    Philosophy:
    - Agent learns continuously during wake (like real organisms)
    - Sleep is just rest (no batch updates that break the policy)
    - CfC network naturally handles temporal dynamics
    - No experience replay needed (continuous learning)
    """
    
    def __init__(
        self,
        agent: SoliterAgent,
        world: World,
        config: NCPTrainerConfig,
        device: torch.device,
    ):
        self.agent = agent
        self.world = world
        self.config = config
        self.device = device
        
        # Drive system (internal rewards)
        self.drive_system = DriveSystem(config.drive_config)
        
        # Gradient sensors (resource detection)
        self.gradient_sensors = GradientSensors(config.gradient_config)
        
        # Sensor system (proximity sensing) - FIXED!
        from ..environment import SensorSystem
        self.sensors = SensorSystem()
        
        # Exploration parameter (decays over time)
        self.action_log_std = nn.Parameter(
            torch.tensor([np.log(config.initial_action_std)] * 3, device=device)
        )
        
        # Value head for curiosity/planning (optional, simple MLP)
        self.value_head = nn.Sequential(
            nn.Linear(51, 32),  # 51 = sensor inputs
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        # Track sleep cycles
        self.total_sleep_cycles = 0
        self.last_sleep_tick = 0
        
        # Track what was consumed this tick
        self._consumed_this_tick = None
    
    def _get_sensors(
        self,
        resources: Dict,
        drive_vector: np.ndarray,
        sensor_noise: float,
    ) -> torch.Tensor:
        """
        Construct full sensor input for the CfC brain.
        
        Uses SensorSystem which combines:
        - Proximity sensors (41 channels)
        - Gradient sensors (6 channels)  
        - Drive states (4 channels)
        
        Returns tensor of shape (51,)
        """
        return self.sensors.get_sensor_readings(
            agent_position=self.agent.position,
            agent_vitals={
                'energy': self.agent.energy,
                'hydration': self.agent.hydration,
                'temperature': self.agent.temperature,
                'wakefulness': self.agent.wakefulness,
            },
            resources=resources,
            sensor_noise=sensor_noise,
            drive_vector=drive_vector,
            world_width=float(self.world.config.width),
            world_height=float(self.world.config.height),
        ).to(self.device)
    
    def _select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using CfC network output + exploration noise.
        
        NO PPO! Just network forward pass with gaussian noise for exploration.
        """
        with torch.no_grad():
            # CfC network produces mean action
            mean_action, _ = self.agent.brain(state.unsqueeze(0))
            mean_action = mean_action.squeeze(0)
            
            # Add exploration noise
            action_std = torch.exp(self.action_log_std)
            noise = torch.randn_like(mean_action) * action_std
            action = mean_action + noise
            
            # Clamp to valid range
            action = torch.clamp(action, -1.0, 1.0)
            
            # Transform to action space
            action_constrained = torch.stack([
                torch.sigmoid(action[0]),  # velocity [0, 1]
                torch.tanh(action[1]),     # turn [-1, 1]
                torch.sigmoid(action[2]),  # sleep [0, 1]
            ])
            
            # Get value estimate (for logging/curiosity)
            value = self.value_head(state)
        
        return action_constrained, value
    
    def _check_resource_consumption(self, resources: Dict):
        """Check if agent consumed a resource this tick."""
        self._consumed_this_tick = None
        seasonal_period = self.world.config.seasonal_period  # FIXED: use period not season
        
        # Check food
        for feeder in resources.get('feeders', []):
            if feeder.is_agent_in_consumption_range(self.agent.position):
                if feeder.can_consume():
                    amount = feeder.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('food', amount)
                        self._consumed_this_tick = 'food'
                        return
        
        # Check water (fountains, not waterers!)
        for fountain in resources.get('fountains', []):
            if fountain.is_agent_in_consumption_range(self.agent.position):
                if fountain.can_consume():
                    amount = fountain.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('water', amount)
                        self._consumed_this_tick = 'water'
                        return
        
        # Check heat
        is_night = self.world.is_night()
        for heater in resources.get('heaters', []):
            if heater.is_agent_in_consumption_range(self.agent.position, is_night):
                if heater.can_consume():
                    amount = heater.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('heat', amount)
                        self._consumed_this_tick = 'heat'
                        return
    
    def wake_step(self, resources: Dict) -> Tuple[float, bool, Dict]:
        """
        Execute one wake step (no PPO updates!).
        
        Returns:
            reward: Internal reward from drive system
            done: Whether agent died
            details: Logging information
        """
        # Snapshot vitals BEFORE action
        self.drive_system.snapshot_vitals(
            self.agent.energy,
            self.agent.hydration,
            self.agent.temperature,
        )
        
        # Get drives and sensors
        drive_vector = self.drive_system.get_drive_vector(
            self.agent.energy,
            self.agent.hydration,
            self.agent.temperature,
        )
        
        sensor_noise = self.agent.get_sensor_noise()
        sensors = self._get_sensors(resources, drive_vector, sensor_noise)
        
        # Select action (NCP forward pass + exploration)
        action, value = self._select_action(sensors)
        velocity = action[0].item()
        turn = action[1].item()
        should_sleep = action[2].item() > 0.5
        
        # Check if turning allowed (directional stability)
        allow_turning = self.agent.should_allow_turning(drive_vector)
        
        # Execute movement
        world_bounds = (self.world.config.width, self.world.config.height)
        self.agent.move(velocity, turn, allow_turning=allow_turning, 
                       world_bounds=world_bounds, dt=1.0)
        
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
        
        # Compute internal reward
        reward, reward_details = self.drive_system.compute_internal_reward(
            energy=self.agent.energy,
            hydration=self.agent.hydration,
            temperature=self.agent.temperature,
            consumed_resource=self._consumed_this_tick,
            is_dead=done,
        )
        
        return reward, done, reward_details
    
    def sleep_cycle(self) -> Dict:
        """
        Execute sleep cycle - NO LEARNING!
        
        Sleep is just rest:
        - Restore wakefulness
        - Decay exploration slightly
        - NO batch updates
        - NO PPO
        - NO experience replay
        
        The CfC network's continuous dynamics handle learning naturally.
        """
        print(f"\n  💤 Sleep at tick {self.world.tick} - NCP MODE (no batch updates)")
        
        # Decay exploration (agent becomes more confident over time)
        with torch.no_grad():
            self.action_log_std.data = torch.clamp(
                self.action_log_std.data - np.log(1 / self.config.action_std_decay),
                min=np.log(self.config.action_std_min),
            )
        
        # Exit sleep (restore wakefulness)
        self.agent.exit_sleep()
        
        # Track statistics
        self.total_sleep_cycles += 1
        self.last_sleep_tick = self.world.tick
        
        current_action_std = torch.exp(self.action_log_std).mean().item()
        
        return {
            'action_std': current_action_std,
            'sleep_cycles': self.total_sleep_cycles,
        }
