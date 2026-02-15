"""
World simulation with seasonal and diurnal cycles.

The world operates on two overlapping cycles:
- Seasonal: 20,000 ticks = 1 year (Summer → Fall → Winter → Spring)
- Diurnal: 2,000 ticks = 1 day (Day → Night)

These cycles create dynamic temperature patterns that affect agent survival.
"""

import numpy as np
from typing import Tuple
from enum import Enum
from dataclasses import dataclass


class Season(Enum):
    """Seasons of the year."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class TimeOfDay(Enum):
    """Time of day."""
    DAY = "day"
    NIGHT = "night"
    DAWN = "dawn"
    DUSK = "dusk"


@dataclass
class WorldConfig:
    """Configuration for world simulation."""
    width: int = 1000
    height: int = 1000
    
    # Cycle lengths (in ticks)
    seasonal_period: int = 20000  # 1 year
    diurnal_period: int = 2000    # 1 day
    
    # Temperature parameters (REVISED - less extreme)
    base_temperature: float = 30.0    # Moderate baseline
    seasonal_amplitude: float = 10.0  # ±10°C seasonal (was 15, too harsh)
    diurnal_amplitude: float = 5.0    # ±5°C day/night (was 10)
    # Result: Summer 2pm = 45°C, Winter 2am = 15°C (more survivable)
    
    # Night definition (fraction of day that's night)
    night_fraction: float = 0.4


class World:
    """
    Simulates the passage of time and environmental conditions.
    
    The world tracks seasonal and diurnal cycles, computing ambient
    temperature based on time of year and time of day.
    """
    
    def __init__(self, config: WorldConfig = None):
        self.config = config or WorldConfig()
        
        # Time tracking
        self.tick = 0
        self.year = 0
        self.day = 0
        
    def step(self) -> None:
        """Advance time by one tick."""
        self.tick += 1
        
        # Update year counter
        if self.tick % self.config.seasonal_period == 0:
            self.year += 1
        
        # Update day counter
        if self.tick % self.config.diurnal_period == 0:
            self.day += 1
    
    def get_seasonal_phase(self) -> float:
        """
        Get current seasonal phase [0, 2π].
        
        Phase = 0 → Spring equinox
        Phase = π/2 → Summer solstice (hottest)
        Phase = π → Fall equinox
        Phase = 3π/2 → Winter solstice (coldest)
        """
        return (self.tick % self.config.seasonal_period) / self.config.seasonal_period * 2 * np.pi
    
    def get_diurnal_phase(self) -> float:
        """
        Get current diurnal phase [0, 2π].
        
        Phase = 0 → Midnight
        Phase = π/2 → Dawn/Sunrise
        Phase = π → Noon (hottest)
        Phase = 3π/2 → Dusk/Sunset
        """
        return (self.tick % self.config.diurnal_period) / self.config.diurnal_period * 2 * np.pi
    
    def get_season(self) -> Season:
        """Get current season."""
        phase = self.get_seasonal_phase()
        
        if phase < np.pi / 2:
            return Season.SPRING
        elif phase < np.pi:
            return Season.SUMMER
        elif phase < 3 * np.pi / 2:
            return Season.FALL
        else:
            return Season.WINTER
    
    def get_time_of_day(self) -> TimeOfDay:
        """Get current time of day."""
        phase = self.get_diurnal_phase()
        
        # Define transitions (with hysteresis zones)
        dawn_start = np.pi / 2 - 0.3
        dawn_end = np.pi / 2 + 0.3
        dusk_start = 3 * np.pi / 2 - 0.3
        dusk_end = 3 * np.pi / 2 + 0.3
        
        if dawn_start < phase < dawn_end:
            return TimeOfDay.DAWN
        elif dusk_start < phase < dusk_end:
            return TimeOfDay.DUSK
        elif dawn_end <= phase < dusk_start:
            return TimeOfDay.DAY
        else:
            return TimeOfDay.NIGHT
    
    def is_night(self) -> bool:
        """Check if it's currently night time."""
        # Night when light level is low
        return self.get_light_level() < 0.3
    
    def get_ambient_temperature(self) -> float:
        """
        Calculate ambient temperature based on season and time of day.
        
        Temperature = base + seasonal_effect + diurnal_effect
        
        Returns:
            Temperature in Celsius
        """
        # Seasonal component (Summer hot, Winter cold)
        seasonal_phase = self.get_seasonal_phase()
        seasonal_effect = self.config.seasonal_amplitude * np.sin(seasonal_phase)
        
        # Diurnal component (Noon hot, Midnight cold)
        diurnal_phase = self.get_diurnal_phase()
        diurnal_effect = self.config.diurnal_amplitude * np.sin(diurnal_phase)
        
        return self.config.base_temperature + seasonal_effect + diurnal_effect
    
    def get_light_level(self) -> float:
        """
        Get current light level [0, 1].
        
        0 = pitch black (midnight)
        1 = full daylight (noon)
        """
        diurnal_phase = self.get_diurnal_phase()
        # Light peaks at noon (phase = π)
        return max(0.0, np.sin(diurnal_phase))
    
    def wrap_position(self, position: np.ndarray) -> np.ndarray:
        """
        Wrap position to stay within world bounds (toroidal topology).
        
        Args:
            position: [x, y] position
            
        Returns:
            Wrapped position within [0, width) × [0, height)
        """
        wrapped = position.copy()
        wrapped[0] = wrapped[0] % self.config.width
        wrapped[1] = wrapped[1] % self.config.height
        return wrapped
    
    def get_state_dict(self) -> dict:
        """Get world state for checkpointing."""
        return {
            'tick': self.tick,
            'year': self.year,
            'day': self.day,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load world state from checkpoint."""
        self.tick = state['tick']
        self.year = state['year']
        self.day = state['day']
    
    def get_stats(self) -> dict:
        """Get current world statistics."""
        return {
            'tick': self.tick,
            'year': self.year,
            'day': self.day,
            'season': self.get_season().value,
            'time_of_day': self.get_time_of_day().value,
            'ambient_temp': self.get_ambient_temperature(),
            'light_level': self.get_light_level(),
            'is_night': self.is_night(),
        }
