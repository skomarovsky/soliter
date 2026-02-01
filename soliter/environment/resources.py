"""
Resources in the world: Feeders, Fountains, and Heaters.

Resources restore agent vitals but have different availability patterns:
- Feeders: Available in Summer/Fall (growing season)
- Fountains: Available in Spring/Summer (wet season)
- Heaters: Always available but smaller radius at night
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ResourceConfig:
    """Base configuration for resources."""
    position: np.ndarray
    radius: float = 20.0
    restore_rate: float = 10.0  # Amount restored per tick


class Resource(ABC):
    """Base class for all resources in the world."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.position = config.position
        self.radius = config.radius
        self.restore_rate = config.restore_rate
        
    @abstractmethod
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Check if resource is currently available."""
        pass
    
    @abstractmethod
    def get_restore_amount(self) -> float:
        """Get amount of vital restored per consumption."""
        pass
    
    def get_effective_radius(self, is_night: bool = False) -> float:
        """Get current effective radius (may change with time)."""
        return self.radius
    
    def is_agent_in_range(self, agent_position: np.ndarray, is_night: bool = False) -> bool:
        """Check if agent is close enough to consume resource."""
        distance = np.linalg.norm(self.position - agent_position)
        return distance <= self.get_effective_radius(is_night)


class Feeder(Resource):
    """
    Food source that restores Energy.
    
    Availability: sin²(seasonal_phase)
    - High in Summer (phase ~ π/2)
    - Low in Winter (phase ~ 3π/2)
    """
    
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Feeders available based on seasonal cycle."""
        availability = self.get_availability_strength(world_tick, seasonal_period)
        # Consider available if > 0.1 (10% threshold)
        return availability > 0.1
    
    def get_availability_strength(self, world_tick: int, seasonal_period: int) -> float:
        """Get strength of availability [0, 1]."""
        phase = (world_tick % seasonal_period) / seasonal_period * 2 * np.pi
        # (1 + sin(phase))/2 gives: Spring=0.5, Summer=1.0, Fall=0.5, Winter=0.0
        return (1 + np.sin(phase)) / 2
    
    def get_restore_amount(self) -> float:
        """Food restores energy."""
        return self.restore_rate


class Fountain(Resource):
    """
    Water source that restores Hydration.
    
    Availability: sin²(2 × seasonal_phase)
    - High in Spring and Fall (phase ~ π/4, 3π/4)
    - Low in Summer and Winter (phase ~ π/2, 3π/2)
    """
    
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Fountains available based on doubled seasonal cycle."""
        phase = (world_tick % seasonal_period) / seasonal_period * 2 * np.pi
        # sin²(2×phase) gives two wet seasons per year
        availability = np.sin(2 * phase) ** 2
        return availability > 0.1
    
    def get_availability_strength(self, world_tick: int, seasonal_period: int) -> float:
        """Get strength of availability [0, 1]."""
        phase = (world_tick % seasonal_period) / seasonal_period * 2 * np.pi
        return np.sin(2 * phase) ** 2
    
    def get_restore_amount(self) -> float:
        """Water restores hydration."""
        return self.restore_rate


class Heater(Resource):
    """
    Heat source that restores Temperature.
    
    Availability: Always available
    Radius: Shrinks at night (agents must be closer to stay warm)
    """
    
    def __init__(self, config: ResourceConfig, night_radius_multiplier: float = 0.5):
        super().__init__(config)
        self.night_radius_multiplier = night_radius_multiplier
    
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Heaters are always available."""
        return True
    
    def get_effective_radius(self, is_night: bool = False) -> float:
        """Radius shrinks at night."""
        if is_night:
            return self.radius * self.night_radius_multiplier
        return self.radius
    
    def get_restore_amount(self) -> float:
        """Heat restores temperature."""
        return self.restore_rate


def create_default_resources(world_width: int = 1000, world_height: int = 1000) -> dict:
    """
    Create a default set of resources scattered in the world.
    
    Returns:
        Dictionary with 'feeders', 'fountains', 'heaters' lists
    """
    np.random.seed(42)  # Reproducible placement
    
    # Create 5 of each resource type
    num_each = 5
    
    feeders = []
    fountains = []
    heaters = []
    
    for i in range(num_each):
        # Random positions, avoiding edges
        margin = 100
        
        # Feeder
        feeder_pos = np.array([
            np.random.uniform(margin, world_width - margin),
            np.random.uniform(margin, world_height - margin)
        ])
        feeders.append(Feeder(ResourceConfig(
            position=feeder_pos,
            radius=30.0,
            restore_rate=15.0
        )))
        
        # Fountain
        fountain_pos = np.array([
            np.random.uniform(margin, world_width - margin),
            np.random.uniform(margin, world_height - margin)
        ])
        fountains.append(Fountain(ResourceConfig(
            position=fountain_pos,
            radius=25.0,
            restore_rate=12.0
        )))
        
        # Heater
        heater_pos = np.array([
            np.random.uniform(margin, world_width - margin),
            np.random.uniform(margin, world_height - margin)
        ])
        heaters.append(Heater(
            ResourceConfig(
                position=heater_pos,
                radius=40.0,
                restore_rate=8.0
            ),
            night_radius_multiplier=0.5
        ))
    
    return {
        'feeders': feeders,
        'fountains': fountains,
        'heaters': heaters,
    }
