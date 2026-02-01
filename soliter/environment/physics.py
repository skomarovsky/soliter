"""
Physics engine for the Soliter world.

Handles:
- Collision detection between agents and resources
- Raycasting for agent sensors
- Boundary wrapping (toroidal world)
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RaycastResult:
    """Result of a raycast operation."""
    hit: bool
    distance: float
    hit_type: Optional[str] = None  # 'feeder', 'fountain', 'heater', 'boundary'
    hit_position: Optional[np.ndarray] = None


class Physics:
    """Physics engine for collision detection and raycasting."""
    
    def __init__(self, world_width: int = 1000, world_height: int = 1000):
        self.world_width = world_width
        self.world_height = world_height
    
    def check_circle_collision(
        self,
        pos1: np.ndarray,
        radius1: float,
        pos2: np.ndarray,
        radius2: float,
    ) -> bool:
        """
        Check if two circles collide.
        
        Args:
            pos1: Center of first circle [x, y]
            radius1: Radius of first circle
            pos2: Center of second circle [x, y]
            radius2: Radius of second circle
            
        Returns:
            True if circles overlap
        """
        distance = np.linalg.norm(pos1 - pos2)
        return distance < (radius1 + radius2)
    
    def raycast(
        self,
        origin: np.ndarray,
        angle: float,
        max_distance: float,
        obstacles: List[Tuple[np.ndarray, float, str]] = None,
    ) -> RaycastResult:
        """
        Cast a ray from origin in given direction.
        
        Args:
            origin: Starting position [x, y]
            angle: Direction in radians
            max_distance: Maximum raycast distance
            obstacles: List of (position, radius, type) tuples
            
        Returns:
            RaycastResult with hit information
        """
        # Ray direction
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        closest_hit = None
        closest_distance = max_distance
        
        # Check obstacles
        if obstacles:
            for obs_pos, obs_radius, obs_type in obstacles:
                # Ray-circle intersection
                # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
                
                to_circle = obs_pos - origin
                proj_length = np.dot(to_circle, direction)
                
                # Skip if circle is behind ray
                if proj_length < 0:
                    continue
                
                # Closest point on ray to circle center
                closest_point = origin + proj_length * direction
                distance_to_center = np.linalg.norm(obs_pos - closest_point)
                
                # Check if ray intersects circle
                if distance_to_center <= obs_radius:
                    # Calculate intersection distance
                    # Using Pythagorean theorem
                    offset = np.sqrt(max(0, obs_radius**2 - distance_to_center**2))
                    hit_distance = proj_length - offset
                    
                    if 0 < hit_distance < closest_distance:
                        closest_distance = hit_distance
                        hit_pos = origin + hit_distance * direction
                        closest_hit = RaycastResult(
                            hit=True,
                            distance=hit_distance,
                            hit_type=obs_type,
                            hit_position=hit_pos
                        )
        
        # Check world boundaries (toroidal, but we'll report them)
        # Calculate distance to edges
        if angle >= 0:  # Moving right
            dist_to_right = (self.world_width - origin[0]) / np.cos(angle) if np.cos(angle) > 0 else max_distance
        else:
            dist_to_right = max_distance
            
        if not closest_hit or dist_to_right < closest_distance:
            # Could add boundary detection here if needed
            pass
        
        if closest_hit:
            return closest_hit
        else:
            return RaycastResult(
                hit=False,
                distance=max_distance,
                hit_type=None,
                hit_position=None
            )
    
    def raycast_360(
        self,
        origin: np.ndarray,
        num_rays: int,
        max_distance: float,
        obstacles: List[Tuple[np.ndarray, float, str]] = None,
    ) -> List[RaycastResult]:
        """
        Cast rays in all directions around a point.
        
        Args:
            origin: Center position
            num_rays: Number of rays to cast (evenly distributed)
            max_distance: Maximum distance for each ray
            obstacles: List of obstacles
            
        Returns:
            List of RaycastResults, one per ray
        """
        results = []
        
        for i in range(num_rays):
            angle = (i / num_rays) * 2 * np.pi
            result = self.raycast(origin, angle, max_distance, obstacles)
            results.append(result)
        
        return results
    
    def wrap_position(self, position: np.ndarray) -> np.ndarray:
        """
        Wrap position to stay within world bounds (toroidal).
        
        Args:
            position: [x, y] position
            
        Returns:
            Wrapped position
        """
        wrapped = position.copy()
        wrapped[0] = wrapped[0] % self.world_width
        wrapped[1] = wrapped[1] % self.world_height
        return wrapped
    
    def get_wrapped_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Get shortest distance between two points in toroidal space.
        
        Args:
            pos1: First position [x, y]
            pos2: Second position [x, y]
            
        Returns:
            Shortest distance considering wrapping
        """
        # Calculate differences in each dimension
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Consider wrapping
        dx = min(dx, self.world_width - dx)
        dy = min(dy, self.world_height - dy)
        
        return np.sqrt(dx**2 + dy**2)
