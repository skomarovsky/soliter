"""
Sensor system for the Soliter agent.

Provides sensory input combining:
- Internal vitals (4 values)
- Visual raycasting (36 directions)
- Touch sensor (1 value)

Total: 41 sensor inputs

Sensors can be noisy/hallucinatory based on agent's wakefulness and hydration.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .physics import Physics, RaycastResult
from .resources import Resource, Feeder, Fountain, Heater


@dataclass
class SensorConfig:
    """Configuration for sensor system."""
    num_rays: int = 36  # Number of raycast directions
    max_ray_distance: float = 200.0  # Maximum vision distance
    touch_threshold: float = 15.0  # Distance for touch sensor
    
    # Noise parameters
    base_noise: float = 0.01  # Minimum sensor noise
    max_noise: float = 0.3  # Maximum sensor noise when impaired


class SensorSystem:
    """
    Manages all sensory input for the agent.
    
    The agent perceives the world through:
    1. Vitals (4): energy, hydration, temperature, wakefulness
    2. Vision (36): raycasts in all directions
    3. Touch (1): collision detection
    """
    
    def __init__(
        self,
        config: SensorConfig = None,
        physics: Physics = None,
    ):
        self.config = config or SensorConfig()
        self.physics = physics or Physics()
        
    def get_sensor_readings(
        self,
        agent_position: np.ndarray,
        agent_vitals: Dict[str, float],
        resources: Dict[str, List[Resource]],
        sensor_noise: float = 0.0,
    ) -> torch.Tensor:
        """
        Get all sensor readings for the agent.
        
        Args:
            agent_position: Agent's current position [x, y]
            agent_vitals: Dictionary with 'energy', 'hydration', 'temperature', 'wakefulness'
            resources: Dictionary with 'feeders', 'fountains', 'heaters' lists
            sensor_noise: Noise level [0, 1] from agent impairment
            
        Returns:
            Tensor of shape (41,) with all sensor values
        """
        # 1. Internal vitals (4 values, normalized to [0, 1])
        vitals_array = np.array([
            agent_vitals['energy'] / 100.0,
            agent_vitals['hydration'] / 100.0,
            agent_vitals['temperature'] / 100.0,
            agent_vitals['wakefulness'],
        ], dtype=np.float32)
        
        # 2. Visual raycasting (36 values, normalized distances)
        ray_readings = self._get_raycast_readings(
            agent_position,
            resources,
            sensor_noise
        )
        
        # 3. Touch sensor (1 value, binary)
        touch_reading = self._get_touch_reading(
            agent_position,
            resources
        )
        
        # Combine all sensors
        all_sensors = np.concatenate([
            vitals_array,  # 4
            ray_readings,  # 36
            [touch_reading]  # 1
        ])
        
        # Add noise to sensors (except vitals which are internal)
        if sensor_noise > 0:
            noise = np.random.normal(0, sensor_noise, size=all_sensors.shape)
            # Only add noise to external sensors (rays and touch)
            all_sensors[4:] += noise[4:]
            all_sensors = np.clip(all_sensors, 0.0, 1.0)
        
        return torch.tensor(all_sensors, dtype=torch.float32)
    
    def _get_raycast_readings(
        self,
        agent_position: np.ndarray,
        resources: Dict[str, List[Resource]],
        sensor_noise: float,
    ) -> np.ndarray:
        """
        Perform raycasting in all directions.
        
        Returns array of shape (36,) with normalized distances to obstacles.
        Value of 1.0 means nothing detected (max distance).
        Value of 0.0 means obstacle very close.
        """
        # Build obstacle list
        obstacles = []
        
        for feeder in resources.get('feeders', []):
            obstacles.append((feeder.position, feeder.radius, 'feeder'))
        
        for fountain in resources.get('fountains', []):
            obstacles.append((fountain.position, fountain.radius, 'fountain'))
        
        for heater in resources.get('heaters', []):
            # Use current effective radius (might be smaller at night)
            obstacles.append((heater.position, heater.radius, 'heater'))
        
        # Cast rays
        raycast_results = self.physics.raycast_360(
            origin=agent_position,
            num_rays=self.config.num_rays,
            max_distance=self.config.max_ray_distance,
            obstacles=obstacles
        )
        
        # Convert to normalized distances
        readings = np.zeros(self.config.num_rays, dtype=np.float32)
        
        for i, result in enumerate(raycast_results):
            if result.hit:
                # Normalize distance to [0, 1]
                # Close objects → 0, far objects → 1
                normalized_dist = result.distance / self.config.max_ray_distance
                readings[i] = normalized_dist
            else:
                # No hit → maximum distance
                readings[i] = 1.0
        
        # Hallucinations when impaired (high noise)
        if sensor_noise > 0.2:  # Significant impairment
            # Randomly "see" things that aren't there
            hallucination_prob = sensor_noise * 0.5
            hallucinate = np.random.random(self.config.num_rays) < hallucination_prob
            readings[hallucinate] = np.random.random(hallucinate.sum())
        
        return readings
    
    def _get_touch_reading(
        self,
        agent_position: np.ndarray,
        resources: Dict[str, List[Resource]],
    ) -> float:
        """
        Check if agent is touching any resource.
        
        Returns 1.0 if touching something, 0.0 otherwise.
        """
        for resource_list in resources.values():
            for resource in resource_list:
                distance = np.linalg.norm(agent_position - resource.position)
                if distance < resource.radius + self.config.touch_threshold:
                    return 1.0
        
        return 0.0
    
    def visualize_sensors(
        self,
        sensor_values: torch.Tensor,
    ) -> str:
        """
        Create a text visualization of sensor readings.
        
        Useful for debugging.
        """
        lines = []
        lines.append("=== Sensor Readings ===")
        
        # Vitals
        lines.append(f"Energy:      {sensor_values[0]:.2f}")
        lines.append(f"Hydration:   {sensor_values[1]:.2f}")
        lines.append(f"Temperature: {sensor_values[2]:.2f}")
        lines.append(f"Wakefulness: {sensor_values[3]:.2f}")
        
        # Raycasts (show summary)
        ray_readings = sensor_values[4:40].numpy()
        lines.append(f"\nRaycasts ({self.config.num_rays}):")
        lines.append(f"  Min distance: {ray_readings.min():.2f}")
        lines.append(f"  Avg distance: {ray_readings.mean():.2f}")
        lines.append(f"  Max distance: {ray_readings.max():.2f}")
        lines.append(f"  Detections:   {(ray_readings < 0.9).sum()}/{self.config.num_rays}")
        
        # Touch
        touch = sensor_values[40].item()
        lines.append(f"\nTouch: {'YES' if touch > 0.5 else 'NO'}")
        
        return "\n".join(lines)
