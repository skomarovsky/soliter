"""
Biological Drive System for Soliter Agent.

Replaces external reward shaping with internal homeostatic drives.
The agent "feels" its needs and experiences satisfaction when needs are met.

Architecture:
    Drive States (internal pressure):
        hunger_drive    = f(energy deficit)      — urgency to find food
        thirst_drive    = f(hydration deficit)    — urgency to find water
        cold_drive      = f(temperature deviation) — urgency to find warmth
        curiosity_drive = f(sensory monotony)     — urge to explore

    Satisfaction Signal (= internal reward):
        When a drive is REDUCED (need met), satisfaction = drive_before - drive_after
        When drives are HIGH (needs unmet), continuous discomfort penalty

    This replaces _calculate_shaped_reward() entirely.
    The agent learns from its own body, not from an omniscient scorer.

Biological basis:
    - Homeostatic drives: Cannon (1932), homeostasis theory
    - Drive reduction as reinforcement: Hull (1943)
    - Dopamine as prediction error on drive reduction: Schultz (1997)
    - Curiosity as intrinsic motivation: Schmidhuber (1991), Oudeyer & Kaplan (2007)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DriveConfig:
    """Configuration for the biological drive system."""

    # Setpoints (optimal values — no drive when at setpoint)
    energy_setpoint: float = 80.0       # Energy level where hunger = 0
    hydration_setpoint: float = 80.0    # Hydration level where thirst = 0
    temperature_optimal: float = 37.0   # Optimal body temperature
    temperature_range: float = 30.0     # Range before max cold drive

    # Drive curve parameters
    # Drives use power law: drive = max(0, 1 - value/setpoint)^exponent
    # Higher exponent = steeper urgency near depletion
    drive_exponent: float = 2.0         # Quadratic urgency (biological)

    # Drive weights (relative importance)
    hunger_weight: float = 1.0
    thirst_weight: float = 1.0
    cold_weight: float = 1.5           # Cold kills fastest, higher priority
    curiosity_weight: float = 0.3       # Gentle exploration pressure

    # Satisfaction scaling
    # satisfaction = (drive_before - drive_after) * satisfaction_scale
    satisfaction_scale: float = 5.0     # Amplify satisfaction for faster learning

    # Discomfort cost (continuous penalty for unmet drives)
    # Each tick with high drive costs: drive_intensity * discomfort_rate
    discomfort_rate: float = 0.1

    # Death penalty (massive negative — drive system's "ultimate failure")
    death_penalty: float = -20.0

    # Curiosity parameters
    curiosity_window: int = 50          # Ticks of sensory history for novelty
    curiosity_decay: float = 0.95       # Exponential decay for sensory memory

    # Resource consumption bonus
    # Extra satisfaction when actually consuming a resource
    # (on top of drive reduction from vitals change)
    consumption_bonus: float = 2.0


class DriveSystem:
    """
    Biological drive system that generates internal reward signals.

    The agent's "feelings" — hunger, thirst, cold, curiosity —
    drive behavior through satisfaction (drive reduction) and
    discomfort (sustained unmet needs).

    Usage:
        drive_sys = DriveSystem()

        # Before action:
        drive_sys.snapshot_vitals(energy, hydration, temperature)

        # After action:
        reward = drive_sys.compute_internal_reward(
            energy, hydration, temperature,
            sensor_readings, consumed_resource, is_dead
        )

        # Get drive states as sensor input:
        drives = drive_sys.get_drive_vector()  # [hunger, thirst, cold, curiosity]
    """

    def __init__(self, config: DriveConfig = None):
        self.config = config or DriveConfig()

        # Previous vitals (for computing deltas)
        self._prev_energy: float = 100.0
        self._prev_hydration: float = 100.0
        self._prev_temperature: float = 37.0

        # Previous drives (for computing satisfaction)
        self._prev_hunger: float = 0.0
        self._prev_thirst: float = 0.0
        self._prev_cold: float = 0.0

        # Curiosity state
        self._sensory_history: list = []
        self._sensory_ema: float = 0.0   # Exponential moving avg of sensory variance
        self._curiosity_drive: float = 0.5  # Start with moderate curiosity

        # Statistics
        self.total_satisfaction: float = 0.0
        self.total_discomfort: float = 0.0
        self.total_consumption_events: int = 0

    # ── Drive Computation ──────────────────────────────────────

    def _compute_hunger(self, energy: float) -> float:
        """Hunger drive: quadratic urgency as energy drops below setpoint."""
        deficit = max(0.0, 1.0 - energy / self.config.energy_setpoint)
        return deficit ** self.config.drive_exponent

    def _compute_thirst(self, hydration: float) -> float:
        """Thirst drive: quadratic urgency as hydration drops."""
        deficit = max(0.0, 1.0 - hydration / self.config.hydration_setpoint)
        return deficit ** self.config.drive_exponent

    def _compute_cold(self, temperature: float) -> float:
        """Cold/heat drive: urgency from temperature deviation."""
        deviation = abs(temperature - self.config.temperature_optimal)
        normalized = min(1.0, deviation / self.config.temperature_range)
        return normalized ** self.config.drive_exponent

    def _compute_curiosity(self, sensor_readings: np.ndarray = None) -> float:
        """
        Curiosity drive: builds when sensory input is monotonous.

        Uses variance of recent sensor readings as novelty proxy.
        Low variance = boring = high curiosity drive.
        """
        if sensor_readings is not None:
            # Compute instantaneous sensory novelty
            # (variance of current reading vs recent history)
            current_hash = float(np.std(sensor_readings))
            self._sensory_history.append(current_hash)

            # Keep window bounded
            if len(self._sensory_history) > self.config.curiosity_window:
                self._sensory_history = self._sensory_history[-self.config.curiosity_window:]

            if len(self._sensory_history) >= 2:
                # Variance of the sensory hash sequence
                recent_variance = np.std(self._sensory_history)
                # Update EMA
                alpha = 1.0 - self.config.curiosity_decay
                self._sensory_ema = self.config.curiosity_decay * self._sensory_ema + alpha * recent_variance
            else:
                self._sensory_ema = 0.1  # Default moderate novelty

        # Curiosity = inverse of novelty (bored when nothing changes)
        # Clamped to [0, 1]
        novelty = min(1.0, self._sensory_ema * 10.0)  # Scale so typical variance → ~0.5
        self._curiosity_drive = max(0.0, min(1.0, 1.0 - novelty))
        return self._curiosity_drive

    # ── Public Interface ───────────────────────────────────────

    def get_drives(self, energy: float, hydration: float,
                   temperature: float, sensor_readings: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all current drive intensities.

        Returns dict with hunger, thirst, cold, curiosity in [0, 1].
        """
        return {
            'hunger': self._compute_hunger(energy),
            'thirst': self._compute_thirst(hydration),
            'cold': self._compute_cold(temperature),
            'curiosity': self._compute_curiosity(sensor_readings),
        }

    def get_drive_vector(self, energy: float, hydration: float,
                         temperature: float, sensor_readings: np.ndarray = None) -> np.ndarray:
        """
        Get drive states as a numpy vector for sensor input.

        Returns: array of shape (4,) — [hunger, thirst, cold, curiosity]
        Normalized to [0, 1].
        """
        drives = self.get_drives(energy, hydration, temperature, sensor_readings)
        return np.array([
            drives['hunger'],
            drives['thirst'],
            drives['cold'],
            drives['curiosity'],
        ], dtype=np.float32)

    def snapshot_vitals(self, energy: float, hydration: float, temperature: float):
        """
        Call BEFORE the agent acts. Stores current state for delta computation.
        """
        self._prev_energy = energy
        self._prev_hydration = hydration
        self._prev_temperature = temperature

        # Store current drives
        self._prev_hunger = self._compute_hunger(energy)
        self._prev_thirst = self._compute_thirst(hydration)
        self._prev_cold = self._compute_cold(temperature)

    def compute_internal_reward(
        self,
        energy: float,
        hydration: float,
        temperature: float,
        sensor_readings: np.ndarray = None,
        consumed_resource: str = None,   # 'food', 'water', 'heat', or None
        is_dead: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the internal reward from drive state changes.

        Call AFTER the agent acts and vitals are updated.

        Returns:
            reward: Total internal reward (float)
            details: Breakdown dict for logging
        """
        if is_dead:
            details = {
                'satisfaction': 0.0,
                'discomfort': 0.0,
                'consumption_bonus': 0.0,
                'curiosity_reward': 0.0,
                'death_penalty': self.config.death_penalty,
                'total': self.config.death_penalty,
            }
            return self.config.death_penalty, details

        # ── Current drives ─────────────────────────────────────
        hunger_now = self._compute_hunger(energy)
        thirst_now = self._compute_thirst(hydration)
        cold_now = self._compute_cold(temperature)
        curiosity_now = self._compute_curiosity(sensor_readings)

        # ── Satisfaction: drive REDUCTION feels good ───────────
        hunger_satisfaction = max(0.0, self._prev_hunger - hunger_now)
        thirst_satisfaction = max(0.0, self._prev_thirst - thirst_now)
        cold_satisfaction = max(0.0, self._prev_cold - cold_now)

        satisfaction = (
            hunger_satisfaction * self.config.hunger_weight
            + thirst_satisfaction * self.config.thirst_weight
            + cold_satisfaction * self.config.cold_weight
        ) * self.config.satisfaction_scale

        # ── Discomfort: unmet drives hurt ──────────────────────
        total_drive_pressure = (
            hunger_now * self.config.hunger_weight
            + thirst_now * self.config.thirst_weight
            + cold_now * self.config.cold_weight
        )
        discomfort = -total_drive_pressure * self.config.discomfort_rate

        # ── Consumption bonus: extra kick for actually eating/drinking
        consumption_reward = 0.0
        if consumed_resource is not None:
            consumption_reward = self.config.consumption_bonus
            self.total_consumption_events += 1

        # ── Curiosity reward: novelty reduces boredom drive ────
        # When curiosity was high and sensory input changes → small reward
        curiosity_change = max(0.0, self._prev_cold - curiosity_now)  # Intentional: reuse pattern
        # Actually: curiosity satisfaction from novelty
        prev_curiosity = self._curiosity_drive  # Before update
        curiosity_reward = 0.0
        if sensor_readings is not None:
            # Reward for experiencing something new
            # (curiosity drive was high, now it dropped = we explored)
            if self._sensory_ema > 0.05:
                curiosity_reward = self.config.curiosity_weight * 0.1  # Gentle constant push

        # Small base survival signal (alive = slightly positive)
        # This prevents total reward from being overwhelmingly negative at start
        alive_bonus = 0.02

        # ── Total internal reward ──────────────────────────────
        total = satisfaction + discomfort + consumption_reward + curiosity_reward + alive_bonus

        # Update stats
        self.total_satisfaction += satisfaction
        self.total_discomfort += abs(discomfort)

        # Store for next step
        self._prev_hunger = hunger_now
        self._prev_thirst = thirst_now
        self._prev_cold = cold_now

        details = {
            'satisfaction': satisfaction,
            'discomfort': discomfort,
            'consumption_bonus': consumption_reward,
            'curiosity_reward': curiosity_reward,
            'alive_bonus': alive_bonus,
            'total': total,
            'hunger_drive': hunger_now,
            'thirst_drive': thirst_now,
            'cold_drive': cold_now,
            'curiosity_drive': curiosity_now,
        }

        return total, details

    def reset(self):
        """Reset drives for new life (after death/respawn)."""
        self._prev_energy = 100.0
        self._prev_hydration = 100.0
        self._prev_temperature = 37.0
        self._prev_hunger = 0.0
        self._prev_thirst = 0.0
        self._prev_cold = 0.0
        self._sensory_history.clear()
        self._sensory_ema = 0.0
        self._curiosity_drive = 0.5

    def get_stats(self) -> Dict[str, float]:
        """Get cumulative statistics."""
        return {
            'total_satisfaction': self.total_satisfaction,
            'total_discomfort': self.total_discomfort,
            'total_consumption_events': self.total_consumption_events,
        }
