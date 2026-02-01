"""
Sleep-Wake Training Loop.

Orchestrates the complete learning cycle:
- Wake: Agent interacts with environment, collects experiences
- Sleep: Consolidation, homeostatic scaling, epistemic pruning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..core.cfc_network import CfCBrain
from ..agents.soliter_agent import SoliterAgent
from ..environment import World, SensorSystem, Physics
from ..memory.replay_buffer import ReplayBuffer, Transition
from ..memory.fisher_matrix import FisherInformationMatrix
from .ewc import EWCLoss


# PyTorch 2.6+ security: allowlist numpy for checkpoint loading
try:
    import torch.serialization
    torch.serialization.add_safe_globals([
        np.ndarray,
        np.dtype,
    ])
    if hasattr(np, '_core') and hasattr(np._core, 'multiarray'):
        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
    if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
except (AttributeError, TypeError):
    pass


@dataclass
class TrainingConfig:
    """Configuration for sleep-wake training."""
    # Wake phase
    wake_duration: int = 10000  # Ticks before sleep
    learning_rate: float = 0.0001
    batch_size: int = 32
    gradient_clip: float = 1.0
    
    # Sleep phase
    sleep_epochs: int = 5  # Consolidation epochs
    sleep_trigger_saturation: float = 0.2  # % neurons saturated
    sleep_trigger_buffer: float = 0.9  # Buffer utilization
    
    # Homeostatic scaling
    target_activity: float = 0.5
    scaling_rate: float = 0.01
    
    # EWC
    lambda_ewc: float = 155000.0
    fisher_decay: float = 0.77
    
    # Buffer
    buffer_capacity: int = 1_000_000
    prune_uncertainty_threshold: float = 0.1
    prune_td_threshold: float = 0.05
    
    # Uncertainty estimation
    uncertainty_num_samples: int = 10
    uncertainty_perturbation_scale: float = 0.1
    
    # Checkpointing
    checkpoint_interval: int = 10  # Days
    checkpoint_dir: str = "checkpoints"


class SleepWakeTrainer:
    """
    Main training orchestrator for Soliter.
    
    Manages the complete learning lifecycle including:
    - Environment interaction
    - Experience collection
    - Sleep-wake transitions
    - Homeostatic regulation
    - Fisher matrix updates
    """
    
    def __init__(
        self,
        agent: SoliterAgent,
        world: World,
        sensors: SensorSystem,
        physics: Physics,
        config: TrainingConfig = None,
        device: torch.device = None,
    ):
        self.agent = agent
        self.world = world
        self.sensors = sensors
        self.physics = physics
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cpu')
        
        # Move agent brain to device
        self.agent.brain.to(self.device)
        
        # Initialize brain hidden state after moving to device
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
        
        # Initialize memory systems
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            prune_threshold_uncertainty=self.config.prune_uncertainty_threshold,
            prune_threshold_td_error=self.config.prune_td_threshold,
        )
        
        self.fisher_matrix = FisherInformationMatrix(
            model=self.agent.brain,
            device=self.device,
        )
        
        self.ewc_loss = EWCLoss(
            fisher_matrix=self.fisher_matrix,
            lambda_ewc=self.config.lambda_ewc,
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.agent.brain.parameters(),
            lr=self.config.learning_rate
        )
        
        # State tracking
        self.total_wake_ticks = 0
        self.total_sleep_cycles = 0
        self.last_sleep_tick = 0
        
        # Statistics
        self.stats = {
            'total_reward': 0.0,
            'episodes': 0,
            'deaths': 0,
            'average_lifespan': 0.0,
        }
    
    def wake_step(
        self,
        resources: Dict,
    ) -> Tuple[float, bool]:
        """
        Execute one wake step: sense → act → learn.
        
        Returns:
            reward: Immediate reward
            done: Whether episode ended (death)
        """
        # Get sensor readings
        sensor_noise = self.agent.get_sensor_noise()
        sensors = self.sensors.get_sensor_readings(
            agent_position=self.agent.position,
            agent_vitals={
                'energy': self.agent.energy,
                'hydration': self.agent.hydration,
                'temperature': self.agent.temperature,
                'wakefulness': self.agent.wakefulness,
            },
            resources=resources,
            sensor_noise=sensor_noise,
        ).to(self.device)
        
        # Agent selects action
        velocity, turn, should_sleep = self.agent.select_action(sensors)
        
        # Execute action
        self.agent.move(velocity, turn, dt=1.0)
        self.agent.position = self.physics.wrap_position(self.agent.position)
        
        # Update vitals
        ambient_temp = self.world.get_ambient_temperature()
        self.agent.update_vitals(velocity, ambient_temp, dt=1.0)
        
        # Check resource consumption
        self._check_resource_consumption(resources)
        
        # Check sleep trigger
        if should_sleep and self.agent.wakefulness < 0.3:
            self.agent.enter_sleep()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next state
        next_sensors = self.sensors.get_sensor_readings(
            agent_position=self.agent.position,
            agent_vitals={
                'energy': self.agent.energy,
                'hydration': self.agent.hydration,
                'temperature': self.agent.temperature,
                'wakefulness': self.agent.wakefulness,
            },
            resources=resources,
            sensor_noise=sensor_noise,
        ).to(self.device)
        
        # Store transition
        transition = Transition(
            state=sensors,
            action=torch.tensor([velocity, turn, float(should_sleep)]),
            reward=reward,
            next_state=next_sensors,
            done=not self.agent.is_alive,
            tick=self.world.tick,
        )
        self.replay_buffer.push(transition)
        
        # Learn from experience
        if len(self.replay_buffer) >= self.config.batch_size:
            self._learn_from_replay()
        
        self.total_wake_ticks += 1
        self.stats['total_reward'] += reward
        
        return reward, not self.agent.is_alive
    
    def _check_resource_consumption(self, resources: Dict) -> None:
        """Check if agent is near resources and consume them."""
        seasonal_period = self.world.config.seasonal_period
        is_night = self.world.is_night()
        
        # Check feeders
        for feeder in resources.get('feeders', []):
            if feeder.is_available(self.world.tick, seasonal_period):
                if feeder.is_agent_in_range(self.agent.position, is_night):
                    self.agent.consume_resource('food', feeder.get_restore_amount())
        
        # Check fountains
        for fountain in resources.get('fountains', []):
            if fountain.is_available(self.world.tick, seasonal_period):
                if fountain.is_agent_in_range(self.agent.position, is_night):
                    self.agent.consume_resource('water', fountain.get_restore_amount())
        
        # Check heaters
        for heater in resources.get('heaters', []):
            if heater.is_available(self.world.tick, seasonal_period):
                if heater.is_agent_in_range(self.agent.position, is_night):
                    self.agent.consume_resource('heat', heater.get_restore_amount())
    
    def _calculate_reward(self) -> float:
        """
        Calculate survival reward.
        
        Reward = +1 for staying alive
                 +bonus for maintaining vitals
                 -10 for death
        """
        if not self.agent.is_alive:
            return -10.0
        
        # Base survival reward
        reward = 1.0
        
        # Bonus for healthy vitals
        vitals_health = (
            self.agent.energy / 100.0 +
            self.agent.hydration / 100.0 +
            abs(37.0 - self.agent.temperature) / 37.0 +  # Closer to 37°C is better
            self.agent.wakefulness
        ) / 4.0
        
        reward += vitals_health * 0.1
        
        return reward
    
    def _learn_from_replay(self) -> None:
        """Learn from replay buffer (experience replay)."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.stack([t.action for t in batch]).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], device=self.device)
        
        # Forward pass
        predicted_actions, _ = self.agent.brain(states)
        
        # Simple supervised loss: predict actions that led to high rewards
        # (In full RL this would be policy gradient or Q-learning)
        weights = torch.sigmoid(rewards)  # Positive rewards → higher weight
        task_loss = (weights.unsqueeze(1) * (predicted_actions - actions) ** 2).mean()
        
        # Add EWC penalty
        total_loss = self.ewc_loss.compute_loss(self.agent.brain, task_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.agent.brain.parameters(),
            self.config.gradient_clip
        )
        
        self.optimizer.step()
        
        # Reset hidden state to single-agent (batch_size=1)
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
    
    def sleep_cycle(self) -> Dict:
        """
        Execute one complete sleep cycle.
        
        Returns:
            Statistics from sleep cycle
        """
        print(f"\n💤 Entering Sleep at tick {self.world.tick}")
        
        # Step 1: Consolidation (NREM)
        print("  [NREM] Consolidating memories...")
        for epoch in range(self.config.sleep_epochs):
            if len(self.replay_buffer) >= self.config.batch_size:
                self._learn_from_replay()
        
        # Step 2: Update Fisher matrix
        print("  [Computing Fisher Information...]")
        self._update_fisher_matrix()
        
        # Step 3: Homeostatic Synaptic Scaling (REM)
        print("  [REM] Applying homeostatic scaling...")
        scale_factor = self.agent.brain.apply_homeostatic_scaling(
            target_activity=self.config.target_activity,
            scaling_rate=self.config.scaling_rate,
        )
        
        # Step 4: Epistemic Pruning with Fisher-informed uncertainty
        print("  [Pruning] Computing Fisher-informed uncertainties...")
        self.replay_buffer.update_uncertainties(
            self.agent.brain,
            num_samples=self.config.uncertainty_num_samples,
            fisher_matrix=self.fisher_matrix,
            perturbation_scale=self.config.uncertainty_perturbation_scale,
        )
        self.replay_buffer.update_td_errors(self.agent.brain)
        
        print("  [Pruning] Removing consolidated memories...")
        pruned = self.replay_buffer.prune_consolidated()
        
        # Step 5: Fisher decay
        self.fisher_matrix.decay_fisher(self.config.fisher_decay)
        
        # Step 6: Update optimal weights
        self.fisher_matrix.update_optimal_weights(self.agent.brain)
        
        # Exit sleep
        self.agent.exit_sleep()
        self.total_sleep_cycles += 1
        self.last_sleep_tick = self.world.tick
        
        # Gather statistics
        buffer_stats = self.replay_buffer.get_stats()
        stats = {
            'scale_factor': scale_factor,
            'transitions_pruned': pruned,
            'buffer_size': len(self.replay_buffer),
            'fisher_stats': self.fisher_matrix.get_stats(),
            'avg_uncertainty': buffer_stats.get('avg_uncertainty', 1.0),
            'avg_td_error': buffer_stats.get('avg_td_error', 1.0),
            'near_consolidated': buffer_stats.get('near_consolidated', 0),
        }
        
        print(f"  ✓ Sleep complete: pruned {pruned}, buffer {len(self.replay_buffer)}")
        print(f"    Avg uncertainty: {stats['avg_uncertainty']:.4f}, Avg TD: {stats['avg_td_error']:.4f}")
        return stats
    
    def _update_fisher_matrix(self) -> None:
        """Update Fisher Information Matrix from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Create dataloader from buffer
        batch = self.replay_buffer.sample(min(1000, len(self.replay_buffer)))
        states = torch.stack([t.state for t in batch]).to(self.device)
        
        # Simple dataloader
        dataloader = [(states[i:i+32],) for i in range(0, len(states), 32)]
        
        self.fisher_matrix.compute_fisher(
            self.agent.brain,
            dataloader,
            num_samples=1000
        )
    
    def should_sleep(self) -> bool:
        """Check if sleep should be triggered."""
        # Trigger 1: Saturation
        mean_activity = self.agent.brain.get_mean_activity()
        saturated = ((mean_activity < 0.1) | (mean_activity > 0.9)).float().mean()
        if saturated > self.config.sleep_trigger_saturation:
            return True
        
        # Trigger 2: Buffer full
        if len(self.replay_buffer) / self.config.buffer_capacity > self.config.sleep_trigger_buffer:
            return True
        
        # Trigger 3: Time-based (every wake_duration ticks)
        if self.total_wake_ticks - self.last_sleep_tick >= self.config.wake_duration:
            return True
        
        return False
    
    def save_checkpoint(self, path: str) -> None:
        """Save complete training state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'agent_brain': self.agent.brain.state_dict(),
            'agent_state': self.agent.get_state_dict(),
            'world_state': self.world.get_state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'fisher_matrix': {
                'fisher_diagonal': self.fisher_matrix.fisher_diagonal,
                'optimal_weights': self.fisher_matrix.optimal_weights,
            },
            'stats': self.stats,
            'config': self.config,
            'total_wake_ticks': self.total_wake_ticks,
            'total_sleep_cycles': self.total_sleep_cycles,
            'last_sleep_tick': self.last_sleep_tick,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load complete training state."""
        checkpoint = torch.load(path, weights_only=False)
        
        self.agent.brain.load_state_dict(checkpoint['agent_brain'])
        self.agent.load_state_dict(checkpoint['agent_state'])
        self.world.load_state_dict(checkpoint['world_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.fisher_matrix.fisher_diagonal = checkpoint['fisher_matrix']['fisher_diagonal']
        self.fisher_matrix.optimal_weights = checkpoint['fisher_matrix']['optimal_weights']
        self.stats = checkpoint['stats']
        
        # Load additional state if present (backwards compatibility)
        if 'total_wake_ticks' in checkpoint:
            self.total_wake_ticks = checkpoint['total_wake_ticks']
        if 'total_sleep_cycles' in checkpoint:
            self.total_sleep_cycles = checkpoint['total_sleep_cycles']
        if 'last_sleep_tick' in checkpoint:
            self.last_sleep_tick = checkpoint['last_sleep_tick']
        
        # Reset hidden state after loading
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
