"""
Soliter Agent: Embodied agent with physiological vitals and CfC brain.

The agent maintains four vital parameters (Energy, Hydration, Temperature,
Wakefulness) and must balance them to survive. The CfC brain controls
movement and sleep decisions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..core.cfc_network import CfCBrain


@dataclass
class VitalsConfig:
    """Configuration for agent vitals."""
    # Initial values
    initial_energy: float = 100.0
    initial_hydration: float = 100.0
    initial_temperature: float = 37.0  # Celsius
    initial_wakefulness: float = 1.0
    
    # Decay rates
    energy_decay_base: float = 0.01
    hydration_decay_base: float = 0.008
    temperature_decay_rate: float = 0.05
    wakefulness_decay_rate: float = 0.0001
    
    # Movement parameters
    base_speed: float = 2.0
    base_turn_rate: float = 0.1
    
    # Sleep threshold
    sleep_threshold: float = 0.3


class SoliterAgent:
    """
    The Soliter agent: a vehicle with vitals controlled by a CfC brain.
    
    The agent exists in a 2D continuous space and must manage four vitals:
    - Energy: Fuels movement
    - Hydration: Fuels cooling
    - Temperature: Must be maintained
    - Wakefulness: Meta-resource, only restored by sleep
    
    Death occurs if any vital reaches 0.
    
    Args:
        brain: CfC brain network
        config: Vitals configuration
        device: Torch device
    """
    
    def __init__(
        self,
        brain: CfCBrain,
        config: VitalsConfig,
        device: torch.device,
    ):
        self.brain = brain
        self.config = config
        self.device = device
        
        # Physical state
        self.position = np.array([0.0, 0.0])
        self.rotation = 0.0  # radians
        self.radius = 10.0  # collision radius
        
        # Movement tracking (for logging)
        self.heading = 0.0  # Current heading in radians
        self.last_velocity = 0.0  # Last velocity magnitude
        self.last_delta = np.array([0.0, 0.0])  # Last movement vector
        
        # Vitals
        self.energy = config.initial_energy
        self.hydration = config.initial_hydration
        self.temperature = config.initial_temperature
        self.wakefulness = config.initial_wakefulness
        
        # State tracking
        self.is_alive = True
        self.is_sleeping = False
        self.total_ticks = 0
        self.total_sleep_cycles = 0
        
        # Statistics
        self.lifetime_reward = 0.0
        self.cause_of_death: Optional[str] = None
        
        # Initialize brain hidden state
        self.brain.reset_hidden(batch_size=1, device=device)
    
    def reset(self, position: Optional[np.ndarray] = None) -> None:
        """Reset agent to initial state."""
        self.position = position if position is not None else np.array([0.0, 0.0])
        self.rotation = 0.0
        
        self.energy = self.config.initial_energy
        self.hydration = self.config.initial_hydration
        self.temperature = self.config.initial_temperature
        self.wakefulness = self.config.initial_wakefulness
        
        self.is_alive = True
        self.is_sleeping = False
        self.total_ticks = 0
        self.total_sleep_cycles = 0
        self.lifetime_reward = 0.0
        self.cause_of_death = None
        
        self.brain.reset_hidden(batch_size=1, device=self.device)
    
    def get_vitals_tensor(self) -> torch.Tensor:
        """Get vitals as normalized tensor [0, 1]."""
        return torch.tensor([
            self.energy / 100.0,
            self.hydration / 100.0,
            self.temperature / 100.0,  # Normalized to ~[0, 1] range
            self.wakefulness,
        ], dtype=torch.float32, device=self.device)
    
    def get_max_speed(self) -> float:
        """Get current maximum speed (affected by energy - motor atrophy)."""
        return self.config.base_speed * (self.energy / 100.0)
    
    def get_turn_rate(self) -> float:
        """Get current turn rate (affected by temperature - thermal stiffness)."""
        return self.config.base_turn_rate * (self.temperature / 100.0)
    
    def get_sensor_noise(self) -> float:
        """Get current sensor noise level (cognitive fog)."""
        # Noise increases as hydration and wakefulness decrease
        clarity = (self.hydration / 100.0) * self.wakefulness
        return 1.0 - clarity
    
    def select_action(self, sensors: torch.Tensor) -> Tuple[float, float, bool]:
        """
        Use brain to select action based on sensors.
        
        Args:
            sensors: Sensor inputs (already includes vitals)
            
        Returns:
            velocity: Forward velocity [0, 1] (scaled by max_speed)
            turn: Turn rate [-1, 1] (scaled by turn_rate)
            should_sleep: Whether to trigger sleep
        """
        with torch.no_grad():
            motor_output, _ = self.brain(sensors.unsqueeze(0))
            motor_output = motor_output.squeeze(0)
        
        velocity = motor_output[0].item()
        turn = motor_output[1].item()
        sleep_signal = motor_output[2].item()
        
        # Apply impairments
        velocity *= (self.energy / 100.0)  # Motor atrophy
        turn *= (self.temperature / 100.0)  # Thermal stiffness
        
        # Scale by base parameters
        velocity *= self.config.base_speed
        turn *= self.config.base_turn_rate
        
        # Determine if should sleep
        # Sleep if: wakefulness low OR brain strongly signals sleep
        should_sleep = (
            self.wakefulness < self.config.sleep_threshold or
            sleep_signal > 0.7
        )
        
        return velocity, turn, should_sleep
    
    def update_vitals(
        self,
        velocity: float,
        ambient_temperature: float,
        dt: float = 1.0,
    ) -> None:
        """
        Update vitals based on actions and environment.
        
        Args:
            velocity: Current movement speed
            ambient_temperature: Environmental temperature
            dt: Time step
        """
        if not self.is_alive:
            return
        
        # Energy decay (faster when moving)
        energy_decay = self.config.energy_decay_base * (1 + velocity ** 2)
        self.energy -= energy_decay * dt
        
        # Hydration decay (faster when hot)
        temp_stress = max(0, self.temperature - ambient_temperature)
        hydration_decay = self.config.hydration_decay_base * (1 + temp_stress)
        self.hydration -= hydration_decay * dt
        
        # Temperature decay toward ambient
        temp_diff = self.temperature - ambient_temperature
        self.temperature -= self.config.temperature_decay_rate * temp_diff * dt
        
        # Wakefulness decay (linear)
        if not self.is_sleeping:
            self.wakefulness -= self.config.wakefulness_decay_rate * dt
        
        # Clamp vitals
        self.energy = max(0.0, min(100.0, self.energy))
        self.hydration = max(0.0, min(100.0, self.hydration))
        self.temperature = max(0.0, min(100.0, self.temperature))
        self.wakefulness = max(0.0, min(1.0, self.wakefulness))
        
        # Check for death
        self._check_death()
        
        self.total_ticks += 1
    
    def consume_resource(self, resource_type: str, amount: float) -> None:
        """
        Consume a resource to restore vitals.
        
        Args:
            resource_type: 'food', 'water', or 'heat'
            amount: Amount to restore
        """
        if resource_type == 'food':
            self.energy = min(100.0, self.energy + amount)
        elif resource_type == 'water':
            self.hydration = min(100.0, self.hydration + amount)
        elif resource_type == 'heat':
            self.temperature = min(100.0, self.temperature + amount)
    
    def enter_sleep(self) -> None:
        """Enter sleep state."""
        self.is_sleeping = True
        self.total_sleep_cycles += 1
    
    def exit_sleep(self) -> None:
        """Exit sleep state and restore wakefulness."""
        self.is_sleeping = False
        self.wakefulness = 1.0
    
    def move(self, velocity: float, turn: float, dt: float = 1.0) -> None:
        """
        Update position based on velocity and turn rate.
        
        Args:
            velocity: Forward velocity
            turn: Turn rate (radians per tick)
            dt: Time step
        """
        if self.is_sleeping or not self.is_alive:
            self.last_velocity = 0.0
            self.last_delta = np.array([0.0, 0.0])
            self.heading = self.rotation
            return
        
        # Update rotation
        self.rotation += turn * dt
        self.rotation = self.rotation % (2 * np.pi)
        
        # Track heading and velocity for logging
        self.heading = self.rotation
        self.last_velocity = velocity
        
        # Update position
        dx = velocity * np.cos(self.rotation) * dt
        dy = velocity * np.sin(self.rotation) * dt
        self.last_delta = np.array([dx, dy])
        self.position += self.last_delta
    
    def _check_death(self) -> None:
        """Check if any vital has reached 0 (death condition)."""
        if self.energy <= 0:
            self.is_alive = False
            self.cause_of_death = "starvation"
        elif self.hydration <= 0:
            self.is_alive = False
            self.cause_of_death = "dehydration"
        elif self.temperature <= 0:
            self.is_alive = False
            self.cause_of_death = "hypothermia"
        elif self.temperature >= 100:
            self.is_alive = False
            self.cause_of_death = "hyperthermia"
    
    def get_state_dict(self) -> Dict:
        """Get complete state for checkpointing."""
        return {
            'position': self.position.copy(),
            'rotation': self.rotation,
            'vitals': {
                'energy': self.energy,
                'hydration': self.hydration,
                'temperature': self.temperature,
                'wakefulness': self.wakefulness,
            },
            'is_alive': self.is_alive,
            'is_sleeping': self.is_sleeping,
            'total_ticks': self.total_ticks,
            'total_sleep_cycles': self.total_sleep_cycles,
            'lifetime_reward': self.lifetime_reward,
            'cause_of_death': self.cause_of_death,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self.position = state['position'].copy()
        self.rotation = state['rotation']
        
        vitals = state['vitals']
        self.energy = vitals['energy']
        self.hydration = vitals['hydration']
        self.temperature = vitals['temperature']
        self.wakefulness = vitals['wakefulness']
        
        self.is_alive = state['is_alive']
        self.is_sleeping = state['is_sleeping']
        self.total_ticks = state['total_ticks']
        self.total_sleep_cycles = state['total_sleep_cycles']
        self.lifetime_reward = state['lifetime_reward']
        self.cause_of_death = state['cause_of_death']
