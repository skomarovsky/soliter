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
    # ADAPTIVE: Should be ~20% of world size for good coverage
    # For 200×200 world: 40 (was 250 - too weak!)
    # For 1000×1000 world: 200
    # At this distance, signal strength = 0.5
    scale_factor: float = 50.0  # Default for medium worlds
    
    # Strength exponent (< 1.0 makes falloff slower)
    # 1.0 = linear falloff
    # 0.7 = slower falloff (stronger at medium distances)
    falloff_exponent: float = 0.7

    # Noise parameters
    # Gradient sensing is impaired by low wakefulness/hydration
    # (dehydrated animals have impaired olfaction)
    noise_scale: float = 0.1

    # Whether to use toroidal (wrapping) distance
    toroidal: bool = False  # Changed to False - hard walls!


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

        # WALL AVOIDANCE: Add repulsive gradient from walls
        # When agent is near a wall with no resource gradients, push it away
        wall_avoidance = self._compute_wall_avoidance(
            agent_position, world_width, world_height
        )
        
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

        # NOTE: Drive-based prioritization moved to DriveSystem!
        # Cleaner separation: gradients = spatial info, drives = motivation
        # DriveSystem.get_drive_vector() applies commitment boost/suppression

        # WALL ESCAPE: If no strong resource gradient, add wall avoidance
        # This prevents agent from getting stuck rotating at walls
        total_gradient_strength = np.linalg.norm(gradients)
        if total_gradient_strength < 0.5:  # Weak or no resource gradients
            # Add wall avoidance to help escape
            # Apply to all resource gradient channels
            gradients[0:2] += wall_avoidance  # food
            gradients[2:4] += wall_avoidance  # water
            gradients[4:6] += wall_avoidance  # heat

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
        
        BIOLOGICAL REALISM: Gradients only exist WITHIN detection range!
        Beyond detection range, agent gets NO directional signal.
        This forces exploration behavior (like bacteria/animals).
        
        Returns: [gx, gy] — direction * strength, or [0, 0] if nothing in range
        """
        if not resource_list:
            return np.zeros(2, dtype=np.float32)

        best_dist = float('inf')
        best_direction = np.zeros(2)
        best_detection_radius = 0.0

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
            
            # Get detection radius for this resource
            detection_radius = resource.get_detection_radius()

            # CRITICAL: Only sense resources WITHIN detection range!
            # Beyond this, agent gets NO signal → must explore randomly
            if dist <= detection_radius and dist < best_dist:
                best_dist = dist
                best_detection_radius = detection_radius
                if dist > 1e-6:
                    best_direction = np.array([dx / dist, dy / dist])
                else:
                    best_direction = np.zeros(2)

        # If nothing in range, return zero gradient (no signal!)
        if best_dist == float('inf'):
            return np.zeros(2, dtype=np.float32)

        # Inverse distance strength (stronger when closer)
        # Strength drops from 1.0 (at resource) to 0.0 (at detection boundary)
        strength = 1.0 - (best_dist / best_detection_radius)
        strength = max(0.0, min(1.0, strength))  # Clamp [0, 1]

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
    
    def _compute_wall_avoidance(
        self,
        agent_pos: np.ndarray,
        world_width: float,
        world_height: float,
    ) -> np.ndarray:
        """
        Compute repulsive gradient from walls.
        
        When agent is near a wall, push it back toward the center.
        This prevents agent from getting stuck rotating at walls.
        
        Returns:
            2D gradient vector pointing away from nearest wall
        """
        x, y = agent_pos[0], agent_pos[1]
        
        # Wall proximity threshold (activate when within this distance)
        wall_threshold = 15.0  # units from wall
        
        # Compute distance to each wall
        dist_left = x
        dist_right = world_width - x
        dist_bottom = y  
        dist_top = world_height - y
        
        # Find minimum distance to any wall
        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
        
        # Only activate if near a wall
        if min_dist > wall_threshold:
            return np.zeros(2, dtype=np.float32)
        
        # Compute avoidance vector (points away from nearest wall)
        avoidance = np.zeros(2, dtype=np.float32)
        
        # Strength increases as agent gets closer to wall
        # strength = 1.0 at wall, 0.0 at threshold
        strength = 1.0 - (min_dist / wall_threshold)
        strength = max(0.0, min(1.0, strength))
        
        # Push away from the nearest wall
        if min_dist == dist_left:
            # Near left wall → push right
            avoidance[0] = strength * 0.8
        elif min_dist == dist_right:
            # Near right wall → push left
            avoidance[0] = -strength * 0.8
        elif min_dist == dist_bottom:
            # Near bottom wall → push up
            avoidance[1] = strength * 0.8
        elif min_dist == dist_top:
            # Near top wall → push down
            avoidance[1] = -strength * 0.8
        
        return avoidance
