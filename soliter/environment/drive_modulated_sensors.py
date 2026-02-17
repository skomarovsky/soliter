"""
Sensor System for Drive-Modulated CfC Brain.

Key differences from standard sensor system:
    1. Does NOT include drive states in sensor vector (they're passed separately)
    2. Does NOT amplify gradients by drives (done inside brain)
    3. Returns RAW sensor data: 49 channels instead of 51

Channels:
    [0:6]   - Vitals: energy, hydration, body_temp, wakefulness, light_level, ambient_temp
    [6:42]  - Raycasts: 36 directions
    [42:43] - Touch: binary collision
    [43:49] - Gradients: food_x, food_y, water_x, water_y, heat_x, heat_y (RAW!)

Total: 49 channels (NO drives, but WITH light and ambient temp!)
"""

import numpy as np
import torch
from typing import Dict, List
from dataclasses import dataclass

from .physics import Physics
from .resources import Resource
from .gradient_sensors import GradientSensors, GradientSensorConfig


@dataclass
class DriveModulatedSensorConfig:
    """Configuration for drive-modulated sensor system."""
    # Raycasting
    num_rays: int = 36
    max_ray_distance: float = 80.0  # Reduced from 200.0 to prevent wall-following
    touch_threshold: float = 15.0
    
    # Noise parameters
    base_noise: float = 0.01
    max_noise: float = 0.3
    
    # Gradient sensors
    gradient_scale_factor: float = 50.0
    gradient_falloff_exponent: float = 0.7


class DriveModulatedSensorSystem:
    """
    Sensor system for drive-modulated brain.
    
    Returns 47 channels of RAW sensory data (no drives, no amplification).
    Drives are passed separately to the brain for internal modulation.
    """
    
    def __init__(
        self,
        config: DriveModulatedSensorConfig = None,
        physics: Physics = None,
    ):
        self.config = config or DriveModulatedSensorConfig()
        self.physics = physics or Physics()
        
        # Gradient sensors (NO toroidal, NO drive amplification!)
        self.gradient_sensors = GradientSensors(
            GradientSensorConfig(
                scale_factor=self.config.gradient_scale_factor,
                falloff_exponent=self.config.gradient_falloff_exponent,
                toroidal=False,  # Hard walls
            )
        )
    
    def get_sensor_readings(
        self,
        agent_position: np.ndarray,
        agent_heading: float,
        agent_vitals: Dict[str, float],
        resources: Dict[str, List[Resource]],
        sensor_noise: float = 0.0,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        obstacles: List = None,
        light_level: float = 1.0,  # NEW: from world.get_light_level()
        ambient_temp: float = 25.0,  # NEW: from world.get_ambient_temperature()
    ) -> torch.Tensor:
        """
        Get RAW sensor readings (49 channels, NO drives).
        
        Args:
            agent_position: Agent position [x, y]
            agent_heading: Agent heading in radians
            agent_vitals: Dict with 'energy', 'hydration', 'temperature', 'wakefulness'
            resources: Dict with 'feeders', 'fountains', 'heaters'
            sensor_noise: Noise level [0, 1]
            world_width: World width
            world_height: World height
            obstacles: List of obstacles for raycasting
            light_level: Light level [0, 1] (0=night, 1=day)
            ambient_temp: Ambient temperature in Celsius
            
        Returns:
            Tensor of shape (49,) with RAW sensors (NO drives!)
        """
        sensors = []
        
        # 1. VITALS (6 channels) - EXPANDED!
        sensors.extend([
            agent_vitals['energy'] / 100.0,        # [0, 1]
            agent_vitals['hydration'] / 100.0,     # [0, 1]
            agent_vitals['temperature'] / 100.0,   # Body temp [0, 1]
            agent_vitals['wakefulness'],           # [0, 1]
            light_level,                           # NEW: [0, 1] (0=night, 1=day)
            ambient_temp / 50.0,                   # NEW: normalized (~0.5 at 25°C)
        ])
        
        # 2. RAYCASTS (36 channels)
        raycasts = self._compute_raycasts(
            agent_position,
            agent_heading,
            resources,
            obstacles,
        )
        sensors.extend(raycasts.tolist())
        
        # 3. TOUCH (1 channel)
        touch = self._compute_touch(agent_position, resources, obstacles)
        sensors.append(touch)
        
        # 4. GRADIENTS (6 channels) - RAW, NO DRIVE AMPLIFICATION!
        gradients = self.gradient_sensors.compute_gradients(
            agent_position=agent_position,
            resources=resources,
            world_width=world_width,
            world_height=world_height,
            sensor_noise=sensor_noise,
            drive_states=None,  # NO DRIVE AMPLIFICATION!
        )
        sensors.extend(gradients.tolist())
        
        # Convert to tensor
        sensor_tensor = torch.tensor(sensors, dtype=torch.float32)
        
        # Add noise if needed
        if sensor_noise > 0:
            noise = torch.randn_like(sensor_tensor) * sensor_noise * self.config.max_noise
            sensor_tensor = sensor_tensor + noise
            sensor_tensor = torch.clamp(sensor_tensor, -1.0, 1.0)
        
        return sensor_tensor
    
    def _compute_raycasts(
        self,
        agent_position: np.ndarray,
        agent_heading: float,
        resources: Dict[str, List[Resource]],
        obstacles: List = None,
    ) -> np.ndarray:
        """
        Compute raycasts in all directions.
        
        Returns:
            Array of shape (36,) with normalized distances [0, 1]
            0 = far, 1 = close
        """
        num_rays = self.config.num_rays
        max_distance = self.config.max_ray_distance
        
        distances = np.zeros(num_rays, dtype=np.float32)
        
        for i in range(num_rays):
            # Ray angle
            angle = agent_heading + (2 * np.pi * i / num_rays)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Cast ray to find nearest object
            min_dist = max_distance
            
            # Check resources
            for resource_type, resource_list in resources.items():
                for resource in resource_list:
                    dist = self._ray_circle_intersection(
                        agent_position,
                        direction,
                        resource.position,
                        resource.detection_radius,
                    )
                    if dist is not None and dist < min_dist:
                        min_dist = dist
            
            # Check obstacles (if any)
            if obstacles:
                for obstacle in obstacles:
                    # Obstacle collision detection (simplified)
                    pass
            
            # Normalize: 0 = far, 1 = close
            distances[i] = 1.0 - (min_dist / max_distance)
        
        return distances
    
    def _ray_circle_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        circle_center: np.ndarray,
        circle_radius: float,
    ) -> float:
        """
        Compute ray-circle intersection distance.
        
        Returns:
            Distance to intersection, or None if no intersection
        """
        # Vector from ray origin to circle center
        oc = ray_origin - circle_center
        
        # Quadratic equation coefficients
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - circle_radius ** 2
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None  # No intersection
        
        # Compute nearest intersection
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        
        if t < 0:
            return None  # Intersection behind ray
        
        return t
    
    def _compute_touch(
        self,
        agent_position: np.ndarray,
        resources: Dict[str, List[Resource]],
        obstacles: List = None,
    ) -> float:
        """
        Compute touch sensor (binary collision detection).
        
        Returns:
            1.0 if touching something, 0.0 otherwise
        """
        threshold = self.config.touch_threshold
        
        # Check resources
        for resource_type, resource_list in resources.items():
            for resource in resource_list:
                dist = np.linalg.norm(agent_position - resource.position)
                if dist < threshold:
                    return 1.0
        
        # Check obstacles
        if obstacles:
            for obstacle in obstacles:
                # Simplified obstacle collision
                pass
        
        return 0.0


def create_drive_modulated_sensors(
    config: DriveModulatedSensorConfig = None,
    physics: Physics = None,
) -> DriveModulatedSensorSystem:
    """
    Factory function to create drive-modulated sensor system.
    
    Returns:
        Configured DriveModulatedSensorSystem instance
    """
    return DriveModulatedSensorSystem(config=config, physics=physics)
