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

    def _compute_curiosity(self, hunger: float, thirst: float, cold: float) -> float:
        """
        Curiosity drive: Builds over time, suppressed by surprise and survival needs.
        
        BIOLOGICAL PRINCIPLE: Energy Conservation
        - Curiosity builds when bored (nothing happening)
        - Surprise temporarily satisfies curiosity (resets it)
        - Survival needs suppress curiosity
        - No drives → no movement (conserve energy!)
        
        Args:
            hunger: Hunger drive [0, 1]
            thirst: Thirst drive [0, 1]
            cold: Cold drive [0, 1]
            
        Returns:
            Curiosity drive [0, 1]
        """
        # Maximum survival drive (most urgent need)
        max_survival_drive = max(hunger, thirst, cold)
        
        # If survival critical (>75%), curiosity HEAVILY suppressed
        if max_survival_drive > 0.75:
            survival_suppression = 0.0  # Complete suppression
        elif max_survival_drive > 0.5:
            # Gradual suppression 50-75%
            survival_suppression = (0.75 - max_survival_drive) / 0.25
        else:
            # Minimal suppression when needs met
            survival_suppression = 1.0
        
        # Curiosity builds over time (boredom)
        # Decays after surprise events
        # This is the "itch to explore" that grows when nothing interesting happens
        self._curiosity_drive = min(1.0, max(0.0, self._curiosity_drive * survival_suppression))
        
        return self._curiosity_drive
    
    def increase_curiosity(self, amount: float = 0.01):
        """
        Gradually increase curiosity over time (boredom builds).
        Call this each tick when nothing interesting happens.
        """
        self._curiosity_drive = min(1.0, self._curiosity_drive + amount)
    
    def reduce_curiosity_from_surprise(self, surprise: float):
        """
        Surprise satisfies curiosity temporarily.
        High surprise → curiosity drops significantly.
        
        Args:
            surprise: Surprise magnitude [0, 1]
        """
        # Surprise reduces curiosity proportionally
        reduction = surprise * 0.5  # Surprise satisfies up to 50% of curiosity
        self._curiosity_drive = max(0.0, self._curiosity_drive - reduction)

    # ── Public Interface ───────────────────────────────────────

    def get_drives(self, energy: float, hydration: float,
                   temperature: float, sensor_readings: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all current drive intensities.

        Returns dict with hunger, thirst, cold, curiosity in [0, 1].
        """
        # Compute survival drives first
        hunger = self._compute_hunger(energy)
        thirst = self._compute_thirst(hydration)
        cold = self._compute_cold(temperature)
        
        # Curiosity is inversely proportional to survival drives
        curiosity = self._compute_curiosity(hunger, thirst, cold)
        
        return {
            'hunger': hunger,
            'thirst': thirst,
            'cold': cold,
            'curiosity': curiosity,
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
        
        # Curiosity builds over time if nothing interesting happens
        # Surprise (big reward changes) temporarily satisfies it
        self.increase_curiosity(amount=0.005)  # Slow buildup (boredom)
        curiosity_now = self._compute_curiosity(hunger_now, thirst_now, cold_now)

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
            # Consumption is surprising! Reduces curiosity temporarily
            self.reduce_curiosity_from_surprise(surprise=0.3)

        # ── Curiosity reward: small bonus for exploring ────
        # Curiosity drive provides gentle exploration pressure
        curiosity_reward = curiosity_now * self.config.curiosity_weight * 0.1
        
        # ── Wasteful movement penalty ────
        # BIOLOGICAL: No drives = should rest (conserve energy!)
        # If all drives are low AND agent moves, penalize it
        max_drive = max(hunger_now, thirst_now, cold_now, curiosity_now)
        wasteful_movement_penalty = 0.0
        if max_drive < 0.3:  # All drives low
            # Agent should rest, not wander aimlessly
            # This penalty teaches energy conservation
            wasteful_movement_penalty = -0.05
        
        # ── Surprise detection: big reward changes ────
        # If reward changes dramatically from previous tick, agent was "surprised"
        # This temporarily satisfies curiosity
        current_reward_mag = abs(satisfaction + consumption_reward)
        if hasattr(self, '_prev_reward_mag'):
            surprise = abs(current_reward_mag - self._prev_reward_mag)
            if surprise > 0.5:  # Significant surprise
                self.reduce_curiosity_from_surprise(surprise=min(1.0, surprise))
        self._prev_reward_mag = current_reward_mag

        # Small base survival signal (alive = slightly positive)
        alive_bonus = 0.02

        # ── Total internal reward ──────────────────────────────
        total = (satisfaction + discomfort + consumption_reward + 
                curiosity_reward + wasteful_movement_penalty + alive_bonus)

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
            'consumed_type': consumed_resource if consumed_resource else '',  # FIX: Add consumed type!
            'curiosity_reward': curiosity_reward,
            'wasteful_movement': wasteful_movement_penalty,
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
        self._curiosity_drive = 0.0  # Reset curiosity
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
