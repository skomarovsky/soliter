"""
Enhanced Sensor System for Soliter Agent.

Manages all sensory input including:
    1. Internal vitals (4): energy, hydration, temperature, wakefulness
    2. Vision raycasts (36): distance to obstacles in all directions
    3. Touch sensor (1): binary collision detection
    4. Gradient sensors (6): directional signals toward food, water, heat
    5. Drive state (4): hunger, thirst, cold, curiosity intensities

Total: 51 sensor channels (was 41 before gradients + drives)

Place in: soliter/environment/sensors.py
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

from .physics import Physics
from .resources import Resource
from .gradient_sensors import GradientSensors, GradientSensorConfig


@dataclass
class SensorConfig:
    """Configuration for the sensor system."""
    # Raycasting
    num_rays: int = 36              # Number of raycast directions
    max_ray_distance: float = 200.0 # Maximum vision distance
    touch_threshold: float = 15.0   # Distance for touch sensor

    # Noise parameters
    base_noise: float = 0.01        # Minimum sensor noise
    max_noise: float = 0.3          # Maximum sensor noise when impaired

    # Gradient sensors (NEW)
    gradient_scale_factor: float = 250.0  # Distance where signal strength = 50%
    enable_gradients: bool = True
    enable_drive_input: bool = True


class SensorSystem:
    """
    Manages all sensory input for the agent.

    The agent perceives the world through:
    1. Vitals (4): energy, hydration, temperature, wakefulness
    2. Vision (36): raycasts in all directions
    3. Touch (1): collision detection
    4. Gradients (6): smell/humidity/heat direction toward resources (NEW)
    5. Drives (4): internal drive states — hunger, thirst, cold, curiosity (NEW)

    Total: 51 sensor channels
    """

    def __init__(
        self,
        config: SensorConfig = None,
        physics: Physics = None,
    ):
        self.config = config or SensorConfig()
        self.physics = physics or Physics()

        # Gradient sensors
        self.gradient_sensors = GradientSensors(
            GradientSensorConfig(
                scale_factor=self.config.gradient_scale_factor,
                toroidal=True,
            )
        )

    def get_sensor_readings(
        self,
        agent_position: np.ndarray,
        agent_vitals: Dict[str, float],
        resources: Dict[str, List[Resource]],
        sensor_noise: float = 0.0,
        drive_vector: np.ndarray = None,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
    ) -> torch.Tensor:
        """
        Get all sensor readings for the agent.

        Args:
            agent_position: Agent's current position [x, y]
            agent_vitals: Dictionary with 'energy', 'hydration',
                          'temperature', 'wakefulness'
            resources: Dictionary with 'feeders', 'fountains', 'heaters' lists
            sensor_noise: Noise level [0, 1] from agent impairment
            drive_vector: Array of shape (4,) with [hunger, thirst, cold, curiosity]
                          or None (zeros used if None)
            world_width: World width for toroidal gradient computation
            world_height: World height for toroidal gradient computation

        Returns:
            Tensor of shape (51,) with all sensor values:
                [0:4]   - Vitals: energy, hydration, temperature, wakefulness
                [4:40]  - Raycasts: 36 distance readings
                [40]    - Touch: binary contact
                [41:47] - Gradients: food_gx, food_gy, water_gx, water_gy, heat_gx, heat_gy
                [47:51] - Drives: hunger, thirst, cold, curiosity
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
            agent_position, resources, sensor_noise
        )

        # 3. Touch sensor (1 value)
        touch = self._get_touch_reading(agent_position, resources)

        # 4. Gradient sensors (6 values) — smell/humidity/heat sensing
        # NEW: Pass drive states to modulate gradient sensitivity (attention)
        if self.config.enable_gradients:
            # Extract drive states from drive_vector if available
            drive_states_dict = None
            if drive_vector is not None and len(drive_vector) >= 3:
                drive_states_dict = {
                    'hunger': float(drive_vector[0]),
                    'thirst': float(drive_vector[1]),
                    'cold': float(drive_vector[2]),
                }
            
            gradient_readings = self.gradient_sensors.compute_gradients(
                agent_position=agent_position,
                resources=resources,
                world_width=world_width,
                world_height=world_height,
                sensor_noise=sensor_noise,
                drive_states=drive_states_dict,
            )
        else:
            gradient_readings = np.zeros(6, dtype=np.float32)

        # 5. Drive state (4 values) — agent feels its own needs
        if self.config.enable_drive_input and drive_vector is not None:
            drive_readings = drive_vector.astype(np.float32)
        else:
            drive_readings = np.zeros(4, dtype=np.float32)

        # Combine all sensors
        all_readings = np.concatenate([
            vitals_array,                   # [0:4]   - 4 values
            ray_readings,                   # [4:40]  - 36 values
            np.array([touch], np.float32),  # [40]    - 1 value
            gradient_readings,              # [41:47] - 6 values
            drive_readings,                 # [47:51] - 4 values
        ])

        # Apply noise to external sensors (not vitals, not drives)
        if sensor_noise > 0.01:
            noise = np.random.normal(
                0, sensor_noise * self.config.max_noise,
                size=len(all_readings)
            ).astype(np.float32)
            # Vitals (0:4) are internal — no noise
            noise[:4] = 0
            # Drives (47:51) are internal — no noise
            noise[47:51] = 0
            all_readings += noise
            # Clip: raycasts to [0,1], gradients can be [-1,1], drives to [0,1]
            all_readings[:41] = np.clip(all_readings[:41], 0.0, 1.0)
            all_readings[41:47] = np.clip(all_readings[41:47], -1.0, 1.0)
            all_readings[47:51] = np.clip(all_readings[47:51], 0.0, 1.0)

        return torch.tensor(all_readings, dtype=torch.float32)

    def _get_raycast_readings(
        self,
        agent_position: np.ndarray,
        resources: Dict[str, List[Resource]],
        sensor_noise: float = 0.0,
    ) -> np.ndarray:
        """
        Cast rays in all directions to detect resources.

        Returns array of shape (36,) with normalized distances to obstacles.
        Value of 1.0 means nothing detected (max distance).
        Value of 0.0 means obstacle very close.
        """
        # Build obstacle list
        # Use detection_radius for raycasts (visual/sensing range)
        obstacles = []

        for feeder in resources.get('feeders', []):
            obstacles.append((feeder.position, feeder.detection_radius, 'feeder'))

        for fountain in resources.get('fountains', []):
            obstacles.append((fountain.position, fountain.detection_radius, 'fountain'))

        for heater in resources.get('heaters', []):
            obstacles.append((heater.position, heater.detection_radius, 'heater'))

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
                normalized_dist = result.distance / self.config.max_ray_distance
                readings[i] = normalized_dist
            else:
                readings[i] = 1.0

        # Hallucinations when impaired (high noise)
        if sensor_noise > 0.2:
            hallucination_prob = sensor_noise * 0.5
            hallucinate = np.random.random(self.config.num_rays) < hallucination_prob
            readings[hallucinate] = np.random.random(hallucinate.sum()).astype(np.float32)

        return readings

    def _get_touch_reading(
        self,
        agent_position: np.ndarray,
        resources: Dict[str, List[Resource]],
    ) -> float:
        """
        Check if agent is touching any resource.
        
        Uses consumption_radius (close contact range).
        Returns 1.0 if touching something, 0.0 otherwise.
        """
        for resource_list in resources.values():
            for resource in resource_list:
                distance = np.linalg.norm(agent_position - resource.position)
                if distance < resource.consumption_radius + self.config.touch_threshold:
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
        v = sensor_values.numpy() if isinstance(sensor_values, torch.Tensor) else sensor_values

        lines = ["=== Sensor Readings (51 channels) ==="]
        lines.append(f"Energy:      {v[0]:.2f}")
        lines.append(f"Hydration:   {v[1]:.2f}")
        lines.append(f"Temperature: {v[2]:.2f}")
        lines.append(f"Wakefulness: {v[3]:.2f}")

        rays = v[4:40]
        detections = np.sum(rays < 1.0)
        lines.append(f"Raycasts (36):")
        lines.append(f"  Min distance: {rays.min():.2f}")
        lines.append(f"  Avg distance: {rays.mean():.2f}")
        lines.append(f"  Max distance: {rays.max():.2f}")
        lines.append(f"  Detections:   {int(detections)}/36")

        lines.append(f"Touch: {'YES' if v[40] > 0.5 else 'NO'}")

        lines.append(f"Gradients:")
        lines.append(f"  Food:  ({v[41]:+.3f}, {v[42]:+.3f})")
        lines.append(f"  Water: ({v[43]:+.3f}, {v[44]:+.3f})")
        lines.append(f"  Heat:  ({v[45]:+.3f}, {v[46]:+.3f})")

        lines.append(f"Drives:")
        lines.append(f"  Hunger:    {v[47]:.3f}")
        lines.append(f"  Thirst:    {v[48]:.3f}")
        lines.append(f"  Cold:      {v[49]:.3f}")
        lines.append(f"  Curiosity: {v[50]:.3f}")

        return "\n".join(lines)