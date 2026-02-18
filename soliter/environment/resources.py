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
    detection_radius: float = 20.0  # How far agent can sense/see resource (for gradients)
    consumption_radius: float = 5.0  # How close agent must be to actually consume
    restore_rate: float = 10.0  # Amount restored per tick
    max_capacity: float = 100.0  # Maximum stored resource
    recovery_rate: float = 0.1  # Units recovered per tick when not being consumed (SLOW!)
    recovery_cooldown: int = 200  # Ticks before recovery starts after consumption (LONG!)


class Resource(ABC):
    """Base class for all resources in the world."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.position = config.position
        self.detection_radius = config.detection_radius  # For sensing/gradients
        self.consumption_radius = config.consumption_radius  # For actual consumption
        self.restore_rate = config.restore_rate
        self.max_capacity = config.max_capacity
        self.recovery_rate = config.recovery_rate
        self.recovery_cooldown = config.recovery_cooldown  # NEW: configurable cooldown
        
        # Depletion tracking
        self.current_capacity = config.max_capacity  # Start full
        self.last_consumption_tick = -1000  # Last time agent consumed from this
        
    @abstractmethod
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Check if resource is currently available."""
        pass
    
    @abstractmethod
    def get_restore_amount(self) -> float:
        """Get amount of vital restored per consumption."""
        pass
    
    def update(self, world_tick: int) -> None:
        """
        Update resource state (recovery over time).
        
        Resources slowly recover capacity when not being consumed.
        Recovery is slower if recently consumed (prevents infinite camping).
        """
        if self.current_capacity < self.max_capacity:
            # Ticks since last consumption
            ticks_since_consumption = world_tick - self.last_consumption_tick
            
            # Recovery starts after cooldown period (configurable)
            if ticks_since_consumption > self.recovery_cooldown:
                # Recover capacity gradually
                self.current_capacity = min(
                    self.max_capacity,
                    self.current_capacity + self.recovery_rate
                )
    
    def can_consume(self) -> bool:
        """Check if resource has enough capacity to be consumed."""
        return self.current_capacity >= self.restore_rate
    
    def consume(self, world_tick: int, seasonal_period: int = 20000) -> float:
        """
        Consume from resource, depleting its capacity.
        
        Applies seasonal strength multiplier to restore amount.
        
        Returns: Actual amount restored (scaled by season and depletion)
        """
        if not self.can_consume():
            return 0.0
        
        # Get seasonal strength multiplier [0.5, 1.0]
        seasonal_strength = self.get_availability_strength(world_tick, seasonal_period)
        
        # Base restore amount, scaled by season
        seasonal_restore = self.restore_rate * seasonal_strength
        
        # Consume from capacity (limited by what's available)
        actual_amount = min(seasonal_restore, self.current_capacity)
        self.current_capacity -= actual_amount
        self.last_consumption_tick = world_tick
        
        return actual_amount
    
    def get_depletion_ratio(self) -> float:
        """Get current depletion ratio [0, 1] where 1 = full, 0 = empty."""
        return self.current_capacity / self.max_capacity
    
    def get_detection_radius(self, is_night: bool = False) -> float:
        """Get current detection radius (for sensing/gradients)."""
        return self.detection_radius
    
    def get_consumption_radius(self, is_night: bool = False) -> float:
        """Get current consumption radius (how close must be to consume)."""
        return self.consumption_radius
    
    def is_agent_in_consumption_range(self, agent_position: np.ndarray, is_night: bool = False) -> bool:
        """Check if agent is close enough to consume resource (STRICT)."""
        distance = np.linalg.norm(self.position - agent_position)
        return distance <= self.get_consumption_radius(is_night)
    
    def is_agent_in_detection_range(self, agent_position: np.ndarray, is_night: bool = False) -> bool:
        """Check if agent can detect/sense resource (for gradients)."""
        distance = np.linalg.norm(self.position - agent_position)
        return distance <= self.get_detection_radius(is_night)


class Feeder(Resource):
    """
    Food source that restores Energy.
    
    Availability: Always available
    Seasonal Variation: Full strength in summer, half strength in winter
    """
    
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Feeders always available (but strength varies seasonally)."""
        return True  # Always available
    
    def get_availability_strength(self, world_tick: int, seasonal_period: int) -> float:
        """
        Get seasonal strength multiplier [0.5, 1.0].
        
        Summer (phase ≈ π/2): 1.0 (full strength)
        Winter (phase ≈ 3π/2): 0.5 (half strength)
        Spring/Fall: 0.75 (medium)
        """
        phase = (world_tick % seasonal_period) / seasonal_period * 2 * np.pi
        # (1 + sin(phase))/2 gives: Spring=0.5, Summer=1.0, Fall=0.5, Winter=0.0
        # We want: Winter=0.5, Summer=1.0
        # So: 0.5 + 0.5*(1+sin(phase))/2 = 0.5 + 0.25*(1+sin(phase))
        return 0.5 + 0.25 * (1 + np.sin(phase))
    
    def get_restore_amount(self) -> float:
        """Food restores energy (scaled by season)."""
        # Will be multiplied by seasonal strength in consume()
        return self.restore_rate


class Fountain(Resource):
    """
    Water source that restores Hydration.
    
    Availability: Always available
    Seasonal Variation: Two wet/dry seasons per year (half strength in dry seasons)
    """
    
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Fountains always available (but strength varies seasonally)."""
        return True  # Always available
    
    def get_availability_strength(self, world_tick: int, seasonal_period: int) -> float:
        """
        Get seasonal strength multiplier [0.5, 1.0].
        
        Wet seasons (Spring/Fall): 1.0 (full strength)
        Dry seasons (Summer/Winter): 0.5 (half strength)
        """
        phase = (world_tick % seasonal_period) / seasonal_period * 2 * np.pi
        # sin²(2×phase) gives two peaks per year
        # Range [0, 1] → convert to [0.5, 1.0]
        raw_strength = np.sin(2 * phase) ** 2
        return 0.5 + 0.5 * raw_strength
    
    def get_restore_amount(self) -> float:
        """Water restores hydration (scaled by season)."""
        return self.restore_rate


class Heater(Resource):
    """
    Heat source that restores Temperature.
    
    Availability: Always available
    Consumption Radius: Shrinks at night (agents must be closer to stay warm)
    """
    
    def __init__(self, config: ResourceConfig, night_radius_multiplier: float = 0.5):
        super().__init__(config)
        self.night_radius_multiplier = night_radius_multiplier
    
    def is_available(self, world_tick: int, seasonal_period: int) -> bool:
        """Heaters are always available."""
        return True
    
    def get_availability_strength(self, world_tick: int, seasonal_period: int) -> float:
        """Heaters always provide full strength (no seasonal variation)."""
        return 1.0  # Always full strength
    
    def get_consumption_radius(self, is_night: bool = False) -> float:
        """Consumption radius shrinks at night (must be very close)."""
        if is_night:
            return self.consumption_radius * self.night_radius_multiplier
        return self.consumption_radius
    
    def get_restore_amount(self) -> float:
        """Heat restores temperature."""
        return self.restore_rate


def create_default_resources(world_width: int = 1000, world_height: int = 1000) -> dict:
    '''Create a default set of resources scattered in the world.'''
    np.random.seed(42)  # Reproducible placement
    
    # Extreme scarcity for 400×400 world - only 2 of each resource!
    num_each = 2  # Was 3 - reduced further for real survival pressure
    
    feeders = []
    fountains = []
    heaters = []
    
    # Scale margin to world size (10% of smallest dimension, min 5)
    margin = max(5.0, min(world_width, world_height) * 0.1)
    
    for i in range(num_each):
        # Feeder - Food depletes fast, recovers VERY slow
        # IMPORTANT: Small consumption radius forces movement!
        # IMPORTANT: Slow recovery prevents wandering over same depleted resources
        feeder_pos = np.array([
            np.random.uniform(margin, world_width - margin),
            np.random.uniform(margin, world_height - margin)
        ])
        feeders.append(Feeder(ResourceConfig(
            position=feeder_pos,
            detection_radius=30.0,    # Can sense from 30 units (gradient)
            consumption_radius=5.0,   # Must be within 5 units to eat (STRICT!)
            restore_rate=15.0,
            max_capacity=200.0,
            recovery_rate=0.5,        # INCREASED 10x: 0.5 units/tick = 1000 units/cycle recovery
            recovery_cooldown=300     # LONG COOLDOWN: 300 ticks before recovery starts
        )))
        
        # Fountain - Water depletes slower, recovers slow
        fountain_pos = np.array([
            np.random.uniform(margin, world_width - margin),
            np.random.uniform(margin, world_height - margin)
        ])
        fountains.append(Fountain(ResourceConfig(
            position=fountain_pos,
            detection_radius=25.0,    # Can sense from 25 units
            consumption_radius=5.0,   # Must be within 5 units to drink
            restore_rate=12.0,
            max_capacity=240.0,
            recovery_rate=0.8,        # INCREASED 10x: 0.8 units/tick = 1600 units/cycle recovery
            recovery_cooldown=250
        )))
        
        # Heater - Heat depletes fast, forces movement
        heater_pos = np.array([
            np.random.uniform(margin, world_width - margin),
            np.random.uniform(margin, world_height - margin)
        ])
        heaters.append(Heater(
            ResourceConfig(
                position=heater_pos,
                detection_radius=40.0,    # Can sense from 40 units
                consumption_radius=8.0,   # Must be within 8 units for heat
                restore_rate=8.0,
                max_capacity=120.0,
                recovery_rate=0.3,        # INCREASED 10x: 0.3 units/tick = 600 units/cycle recovery
                recovery_cooldown=200
            ),
            night_radius_multiplier=0.5  # At night: 8 * 0.5 = 4 units (very close!)
        ))
    
    return {
        'feeders': feeders,
        'fountains': fountains,
        'heaters': heaters,
    }