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
    # CRITICAL: Lower turn rate prevents wild spinning when exploring
    # Old: 0.1 rad/tick = 5.7°/tick → 570°/100 ticks (excessive!)
    # New: 0.03 rad/tick = 1.7°/tick → 170°/100 ticks (more realistic)
    base_turn_rate: float = 0.03  # Reduced from 0.1 to prevent thrashing
    
    # Turn momentum - smooth out rapid direction changes
    turn_momentum: float = 0.7  # 70% previous turn + 30% new turn
    
    # BIOLOGICAL: Directional stability (like vacuum cleaners!)
    # Direction should only change when there's a REASON
    # Not random oscillation from network noise
    direction_stability_threshold: float = 0.15  # Drive must change >15% to allow turning
    direction_change_cooldown: int = 50  # Minimum ticks between direction changes
    
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
        self.last_turn = 0.0  # Last turn rate (for momentum smoothing)
        
        # Directional stability (genetic memory - like vacuum cleaners!)
        self.last_drive_state = np.array([0.0, 0.0, 0.0, 0.0])  # [hunger, thirst, cold, curiosity]
        self.ticks_since_direction_change = 0
        self.stable_direction = True
        
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
    
    def reset(self, position: Optional[np.ndarray] = None, after_death: bool = False) -> None:
        """
        Reset agent to initial state.
        
        Args:
            position: Starting position (if None, use [0, 0])
            after_death: If True, agent respawns with minimal resources (death penalty)
        """
        self.position = position if position is not None else np.array([0.0, 0.0])
        self.rotation = 0.0
        self.last_turn = 0.0  # Reset turn momentum
        
        # Reset directional stability
        self.last_drive_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.ticks_since_direction_change = 0
        
        if after_death:
            # DEATH PENALTY: Respawn with barely enough to survive
            # Prevents death exploitation - dying should NOT be a strategy!
            # Agent gets 1/4 of initial resources - just enough to start searching
            self.energy = self.config.initial_energy * 0.25  # 25% energy
            self.hydration = self.config.initial_hydration * 0.25  # 25% hydration
            self.temperature = self.config.initial_temperature  # Normal temp (37°C)
            self.wakefulness = 0.5  # Groggy after death
        else:
            # Normal reset (first spawn)
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
        turn: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        """
        Update vitals based on actions and environment.
        
        THERMAL SYSTEM (REVISED):
        - Movement generates heat (slow accumulation)
        - Asymmetric warming/cooling (easier to warm than cool)
        - Time to critical: 80-140 ticks (not 10-14)
        
        Args:
            velocity: Current movement speed
            ambient_temperature: Environmental temperature
            turn: Turn rate (rotation cost)
            dt: Time step
        """
        if not self.is_alive:
            return
        
        # ═══════════════════════════════════════════════════════════
        # ENERGY SYSTEM
        # ═══════════════════════════════════════════════════════════
        # Energy decay (faster when moving OR turning)
        movement_cost = velocity ** 2
        turning_cost = abs(turn) * 0.5  # Turning costs energy
        energy_decay = self.config.energy_decay_base * (1 + movement_cost + turning_cost)
        self.energy -= energy_decay * dt
        
        # ═══════════════════════════════════════════════════════════
        # HYDRATION SYSTEM
        # ═══════════════════════════════════════════════════════════
        # Hydration decay (faster when hot)
        temp_stress = max(0, (self.temperature - 37.0) / 10.0)  # Normalized stress
        hydration_decay = self.config.hydration_decay_base * (1 + temp_stress * 0.5)
        self.hydration -= hydration_decay * dt
        
        # ═══════════════════════════════════════════════════════════
        # THERMAL SYSTEM (NEW - SLOW DYNAMICS)
        # ═══════════════════════════════════════════════════════════
        
        # 1. METABOLIC HEAT GENERATION (slow)
        metabolic_base = 0.005  # Base metabolism (°C/tick)
        
        # Resting reduces metabolism
        if velocity < 0.1 and abs(turn) < 0.05:
            metabolic_base *= 0.5  # Half heat when resting
        
        # Movement generates heat (quadratic)
        movement_heat = (velocity ** 2) * 0.01  # °C/tick
        
        # Rotation generates heat (linear)
        rotation_heat = abs(turn) * 0.005  # °C/tick
        
        total_heat_production = metabolic_base + movement_heat + rotation_heat
        
        # 2. THERMAL EXCHANGE WITH ENVIRONMENT (symmetric & slower)
        temp_diff = self.temperature - ambient_temperature
        
        # SYMMETRIC thermal exchange (was asymmetric, too harsh)
        # Both warming and cooling use same rate
        thermal_exchange = 0.004 * temp_diff  # Symmetric rate
        
        # 3. NET TEMPERATURE CHANGE
        delta_temp = total_heat_production - thermal_exchange
        self.temperature += delta_temp * dt
        
        # ═══════════════════════════════════════════════════════════
        # WAKEFULNESS SYSTEM
        # ═══════════════════════════════════════════════════════════
        if not self.is_sleeping:
            self.wakefulness -= self.config.wakefulness_decay_rate * dt
        
        # ═══════════════════════════════════════════════════════════
        # CLAMP VITALS
        # ═══════════════════════════════════════════════════════════
        self.energy = max(0.0, min(100.0, self.energy))
        self.hydration = max(0.0, min(100.0, self.hydration))
        self.temperature = max(0.0, min(150.0, self.temperature))
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
            # NEW: Water provides cooling (evaporative cooling effect)
            cooling_effect = amount * 0.15  # 15% of water amount cools body
            self.temperature = max(0.0, self.temperature - cooling_effect)
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
    
    def should_allow_turning(self, current_drives: np.ndarray) -> bool:
        """
        BIOLOGICAL: Direction stability - like vacuum cleaners!
        
        Only allow turning when there's a REASON:
        1. Drives changed significantly (new goal)
        2. Been going straight long enough (cooldown expired)
        3. Near obstacle/wall
        
        This prevents constant oscillation from network noise.
        
        Args:
            current_drives: [hunger, thirst, cold, curiosity]
            
        Returns:
            True if turning is allowed, False if should go straight
        """
        # Always allow turning if cooldown expired
        if self.ticks_since_direction_change >= self.config.direction_change_cooldown:
            return True
        
        # Check if any drive changed significantly
        drive_changes = np.abs(current_drives - self.last_drive_state)
        max_drive_change = np.max(drive_changes)
        
        if max_drive_change > self.config.direction_stability_threshold:
            # Significant drive change → new goal → allow turning
            self.last_drive_state = current_drives.copy()
            self.ticks_since_direction_change = 0
            return True
        
        # No significant change → maintain direction
        self.ticks_since_direction_change += 1
        return False
    
    def move(self, velocity: float, turn: float, allow_turning: bool = True, world_bounds: Tuple[float, float] = None, dt: float = 1.0) -> None:
        """
        Update position based on velocity and turn rate.
        
        Args:
            velocity: Forward velocity
            turn: Turn rate (radians per tick)
            allow_turning: If False, suppress turning (directional stability)
            world_bounds: (width, height) of world for boundary enforcement
            dt: Time step
        """
        if self.is_sleeping or not self.is_alive:
            self.last_velocity = 0.0
            self.last_delta = np.array([0.0, 0.0])
            self.last_turn = 0.0
            self.heading = self.rotation
            return
        
        # BIOLOGICAL: Directional stability
        # If not allowed to turn, suppress turn command (go straight)
        if not allow_turning:
            turn = 0.0  # Override network output - maintain direction!
        
        # CRITICAL: Apply turn momentum to smooth out rapid direction changes
        # Biological organisms have inertia - can't instantly reverse direction
        # This prevents wild thrashing/spinning behavior
        smoothed_turn = self.config.turn_momentum * self.last_turn + (1 - self.config.turn_momentum) * turn
        self.last_turn = smoothed_turn
        
        # Update rotation with smoothed turn
        self.rotation += smoothed_turn * dt
        self.rotation = self.rotation % (2 * np.pi)
        
        # Track heading and velocity for logging
        self.heading = self.rotation
        self.last_velocity = velocity
        
        # Update position
        dx = velocity * np.cos(self.rotation) * dt
        dy = velocity * np.sin(self.rotation) * dt
        self.last_delta = np.array([dx, dy])
        self.position += self.last_delta
        
        # CRITICAL: Enforce world boundaries (prevent escape)
        if world_bounds is not None:
            world_width, world_height = world_bounds
            self.position[0] = np.clip(self.position[0], 0, world_width - 1)
            self.position[1] = np.clip(self.position[1], 0, world_height - 1)
    
    def _check_death(self) -> None:
        """
        Check if any vital has reached critical threshold (death condition).
        
        Temperature zones:
        - Below 15°C: Hypothermia death
        - 15-20°C: Danger (very cold)
        - 20-42°C: Safe range
        - 42-45°C: Danger (very hot)
        - Above 45°C: Hyperthermia death
        """
        if self.energy <= 0:
            self.is_alive = False
            self.cause_of_death = "starvation"
        elif self.hydration <= 0:
            self.is_alive = False
            self.cause_of_death = "dehydration"
        elif self.temperature <= 15.0:  # Hypothermia threshold (was 0)
            self.is_alive = False
            self.cause_of_death = "hypothermia"
        elif self.temperature >= 45.0:  # Hyperthermia threshold (was 100)
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
