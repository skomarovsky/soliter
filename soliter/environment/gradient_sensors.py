"""
Gradient Sensor System for Soliter Agent.

Adds biologically-inspired gradient sensing — the agent can "smell" food,
"sense" humidity, and "feel" heat from a distance. Each gradient provides
a 2D directional signal pointing toward the nearest resource of that type,
with intensity proportional to proximity.

Biological basis:
    - Chemotaxis: bacteria follow chemical gradients (Berg & Brown, 1972)
    - Olfaction: insects follow pheromone plumes (Cardé & Willis, 2008)
    - Infrared sensing: pit vipers detect thermal gradients (Gracheva et al., 2010)
    - Hygroreception: insects sense humidity gradients (Tichy & Loftus, 1996)

Sensor channels added (6 total):
    food_gradient_x, food_gradient_y    — direction + strength toward nearest feeder
    water_gradient_x, water_gradient_y  — direction + strength toward nearest fountain
    heat_gradient_x, heat_gradient_y    — direction + strength toward nearest heater

Signal model:
    direction = unit_vector(agent → nearest_resource)
    strength  = 1 / (1 + distance / scale_factor)    [inverse distance, like diffusion]
    gradient  = direction * strength

    At distance 0:     strength = 1.0 (maximum)
    At scale_factor:   strength = 0.5
    At 2*scale_factor: strength = 0.33
    At infinity:       strength → 0 (but never exactly zero)

Integration:
    Old sensors: [vitals(4), raycasts(36), touch(1)] = 41
    New sensors: [vitals(4), raycasts(36), touch(1), gradients(6), drives(4)] = 51
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GradientSensorConfig:
    """Configuration for gradient sensing."""

    # Scale factor for inverse-distance falloff
    # At this distance, signal strength = 0.5
    # For 1000×1000 world: 250 means signal is detectable from anywhere
    # For 200×200 world: 50 would be appropriate
    scale_factor: float = 250.0

    # Noise parameters
    # Gradient sensing is impaired by low wakefulness/hydration
    # (dehydrated animals have impaired olfaction)
    noise_scale: float = 0.1

    # Whether to use toroidal (wrapping) distance
    toroidal: bool = True


class GradientSensors:
    """
    Computes gradient signals toward nearest resources.

    Each resource type produces a 2D gradient vector that tells the agent:
    - WHICH DIRECTION the nearest resource is (unit vector)
    - HOW CLOSE it is (magnitude via inverse distance)

    The agent doesn't need to "see" the resource — it can smell/sense it.
    """

    def __init__(self, config: GradientSensorConfig = None):
        self.config = config or GradientSensorConfig()

    def compute_gradients(
        self,
        agent_position: np.ndarray,
        resources: Dict[str, list],
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        sensor_noise: float = 0.0,
        season_availability: Dict[str, bool] = None,
        drive_states: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Compute gradient vectors toward nearest resources.
        
        NEW: Gradients are amplified by corresponding drive states.
        When hungry, food gradients become more salient (attention mechanism).

        Args:
            agent_position: [x, y] agent position
            resources: Dict with 'feeders', 'fountains', 'heaters' lists
            world_width: World width for toroidal distance
            world_height: World height for toroidal distance
            sensor_noise: Impairment level [0, 1]
            season_availability: Optional dict {'feeders': True/False, ...}
                If provided, unavailable resources produce zero gradient.
            drive_states: Optional dict {'hunger': float, 'thirst': float, 'cold': float}
                If provided, amplifies corresponding gradients (biological attention)

        Returns:
            np.ndarray of shape (6,): [food_gx, food_gy, water_gx, water_gy, heat_gx, heat_gy]
            Each pair is a direction vector scaled by proximity.
            Values in approximately [-1, 1] (before drive amplification).
        """
        gradients = np.zeros(6, dtype=np.float32)

        # CRITICAL: Only sense resources that have capacity (not depleted)
        # Filter out depleted resources - no point sensing what you can't consume!
        
        # Food gradient (channels 0-1)
        available_feeders = [f for f in resources.get('feeders', []) if f.can_consume()]
        food_grad = self._nearest_gradient(
            agent_position,
            available_feeders,  # Only non-depleted feeders
            world_width, world_height,
        )
        gradients[0:2] = food_grad

        # Water gradient (channels 2-3)
        available_fountains = [f for f in resources.get('fountains', []) if f.can_consume()]
        water_grad = self._nearest_gradient(
            agent_position,
            available_fountains,  # Only non-depleted fountains
            world_width, world_height,
        )
        gradients[2:4] = water_grad

        # Heat gradient (channels 4-5)
        available_heaters = [h for h in resources.get('heaters', []) if h.can_consume()]
        heat_grad = self._nearest_gradient(
            agent_position,
            available_heaters,  # Only non-depleted heaters
            world_width, world_height,
        )
        gradients[4:6] = heat_grad

        # DRIVE-MODULATED ATTENTION: Amplify gradients based on need
        # When hungry, food gradients become more salient (selective attention)
        # Biological basis: Hungry animals have enhanced olfaction for food
        if drive_states is not None:
            hunger = drive_states.get('hunger', 0.0)
            thirst = drive_states.get('thirst', 0.0)
            cold = drive_states.get('cold', 0.0)
            
            # Amplification: 1.0 (no drive) → 3.0 (max drive)
            # This makes weak gradients stronger when needed
            food_amp = 1.0 + 2.0 * hunger
            water_amp = 1.0 + 2.0 * thirst
            heat_amp = 1.0 + 2.0 * cold
            
            gradients[0:2] *= food_amp
            gradients[2:4] *= water_amp
            gradients[4:6] *= heat_amp

        # Apply noise (impaired senses)
        if sensor_noise > 0:
            noise = np.random.normal(0, sensor_noise * self.config.noise_scale, size=6)
            gradients += noise.astype(np.float32)

        return gradients

    def _nearest_gradient(
        self,
        agent_pos: np.ndarray,
        resource_list: list,
        world_w: float,
        world_h: float,
    ) -> np.ndarray:
        """
        Compute gradient toward the nearest resource in a list.

        Returns: [gx, gy] — direction * strength
        """
        if not resource_list:
            return np.zeros(2, dtype=np.float32)

        best_dist = float('inf')
        best_direction = np.zeros(2)

        for resource in resource_list:
            r_pos = resource.position

            # Compute displacement (with toroidal wrapping)
            if self.config.toroidal:
                dx, dy = self._toroidal_displacement(
                    agent_pos, r_pos, world_w, world_h
                )
            else:
                dx = r_pos[0] - agent_pos[0]
                dy = r_pos[1] - agent_pos[1]

            dist = np.sqrt(dx**2 + dy**2)

            if dist < best_dist:
                best_dist = dist
                if dist > 1e-6:
                    best_direction = np.array([dx / dist, dy / dist])
                else:
                    best_direction = np.zeros(2)

        # Inverse distance strength
        strength = 1.0 / (1.0 + best_dist / self.config.scale_factor)

        return (best_direction * strength).astype(np.float32)

    @staticmethod
    def _toroidal_displacement(
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        width: float,
        height: float,
    ) -> Tuple[float, float]:
        """
        Compute shortest displacement from A to B in toroidal space.

        Returns (dx, dy) pointing from A toward B via the shortest path.
        """
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]

        # Wrap to shortest path
        if abs(dx) > width / 2:
            dx = dx - np.sign(dx) * width
        if abs(dy) > height / 2:
            dy = dy - np.sign(dy) * height

        return float(dx), float(dy)
