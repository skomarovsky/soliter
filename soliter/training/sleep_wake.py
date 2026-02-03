"""
Sleep-Wake Training Loop with Proper Reinforcement Learning.

Orchestrates the complete learning cycle:
- Wake: Agent interacts with environment, collects experiences
- Sleep: Consolidation, homeostatic scaling, epistemic pruning

Uses PPO (Proximal Policy Optimization) for stable policy learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Dict, Tuple, List
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
    learning_rate: float = 0.0003  # PPO typical LR
    batch_size: int = 64
    gradient_clip: float = 0.5
    
    # PPO hyperparameters
    ppo_epochs: int = 4  # PPO update epochs per batch
    ppo_clip: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    
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
    prune_uncertainty_threshold: float = 0.3  # Stricter - only prune truly consolidated
    prune_td_threshold: float = 0.5  # Stricter - must predict well
    min_buffer_size: int = 1000  # NEVER prune below this - prevents collapse
    
    # Uncertainty estimation
    uncertainty_num_samples: int = 10
    uncertainty_perturbation_scale: float = 0.1
    
    # Exploration
    action_std_init: float = 0.5  # Initial action standard deviation
    action_std_min: float = 0.1  # Minimum action std (annealed)
    action_std_decay: float = 0.995  # Decay per sleep cycle
    
    # Checkpointing
    checkpoint_interval: int = 10  # Days
    checkpoint_dir: str = "checkpoints"


class ValueHead(nn.Module):
    """
    Value function head for PPO.
    
    Estimates V(s) - the expected return from a state.
    """
    
    def __init__(self, input_size: int = 41, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOMemory:
    """
    Memory buffer for PPO rollouts.
    
    Stores complete trajectories for advantage estimation.
    """
    
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.next_states: List[torch.Tensor] = []
    
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
        next_state: torch.Tensor,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def __len__(self):
        return len(self.states)
    
    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert stored data to batched tensors."""
        return {
            'states': torch.stack(self.states).to(device),
            'actions': torch.stack(self.actions).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'values': torch.stack(self.values).squeeze(-1).to(device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'next_states': torch.stack(self.next_states).to(device),
        }


class SleepWakeTrainer:
    """
    Main training orchestrator for Soliter with PPO-based learning.
    
    Manages the complete learning lifecycle including:
    - Environment interaction with proper exploration
    - PPO policy updates for stable learning
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
        
        # Value function for PPO
        self.value_head = ValueHead(
            input_size=self.agent.brain.sensory_size,
            hidden_size=128,
        ).to(self.device)
        
        # Action standard deviation (learnable exploration)
        self.action_log_std = nn.Parameter(
            torch.ones(3, device=self.device) * np.log(self.config.action_std_init)
        )
        
        # Initialize memory systems
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            prune_threshold_uncertainty=self.config.prune_uncertainty_threshold,
            prune_threshold_td_error=self.config.prune_td_threshold,
        )
        
        self.ppo_memory = PPOMemory()
        
        self.fisher_matrix = FisherInformationMatrix(
            model=self.agent.brain,
            device=self.device,
        )
        
        self.ewc_loss = EWCLoss(
            fisher_matrix=self.fisher_matrix,
            lambda_ewc=self.config.lambda_ewc,
        )
        
        # Combined optimizer for policy + value
        self.optimizer = optim.Adam([
            {'params': self.agent.brain.parameters(), 'lr': self.config.learning_rate},
            {'params': self.value_head.parameters(), 'lr': self.config.learning_rate},
            {'params': [self.action_log_std], 'lr': self.config.learning_rate},
        ])
        
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
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'avg_reward_per_cycle': 0.0,
        }
        
        # Reward normalization (running stats)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def _get_action_distribution(
        self, 
        mean_actions: torch.Tensor
    ) -> Normal:
        """Create action distribution for exploration."""
        action_std = torch.exp(self.action_log_std)
        return Normal(mean_actions, action_std)
    
    def _select_action_with_exploration(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action with exploration noise.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            # Get mean action from policy
            mean_action, _ = self.agent.brain(state.unsqueeze(0))
            mean_action = mean_action.squeeze(0)
            
            # Get value estimate
            value = self.value_head(state)
            
            # Sample from distribution
            dist = self._get_action_distribution(mean_action)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            # Clamp actions to valid range
            action = torch.clamp(action, -1.0, 1.0)
            
            # Apply motor constraints (same as in CfCBrain)
            action_constrained = torch.stack([
                torch.sigmoid(action[0]),  # velocity [0, 1]
                torch.tanh(action[1]),     # turn [-1, 1]
                torch.sigmoid(action[2]),  # sleep [0, 1]
            ])
        
        return action_constrained, log_prob, value
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns (for value function target)
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Append next_value for bootstrapping
        values_extended = torch.cat([values, next_value.unsqueeze(0)])
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values_extended[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def _update_reward_stats(self, reward: float):
        """Update running reward statistics for normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(
            (self.reward_std ** 2 * (self.reward_count - 1) + delta * delta2) / self.reward_count
        )
        self.reward_std = max(self.reward_std, 1e-8)  # Prevent division by zero
    
    def wake_step(
        self,
        resources: Dict,
    ) -> Tuple[float, bool]:
        """
        Execute one wake step: sense → act (with exploration) → learn.
        
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
        
        # Select action with exploration
        action, log_prob, value = self._select_action_with_exploration(sensors)
        
        velocity = action[0].item()
        turn = action[1].item()
        should_sleep = action[2].item() > 0.5
        
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
        
        # Calculate shaped reward
        reward = self._calculate_shaped_reward(velocity)
        
        # Update reward statistics
        self._update_reward_stats(reward)
        
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
        
        done = not self.agent.is_alive
        
        # Store in PPO memory
        self.ppo_memory.push(
            state=sensors,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
            next_state=next_sensors,
        )
        
        # Store in replay buffer (for Fisher/uncertainty)
        transition = Transition(
            state=sensors,
            action=action,
            reward=reward,
            next_state=next_sensors,
            done=done,
            tick=self.world.tick,
        )
        self.replay_buffer.push(transition)
        
        self.total_wake_ticks += 1
        self.stats['total_reward'] += reward
        
        return reward, done
    
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
    
    def _calculate_shaped_reward(self, velocity: float) -> float:
        """
        Calculate shaped reward to encourage good behavior.
        
        Reward components:
        - Survival: +1.0 per tick alive
        - Vitals health: +0.5 for maintaining good vitals
        - Resource seeking: +2.0 for consuming resources
        - Energy efficiency: -0.1 * velocity (penalize wasteful movement)
        - Death: -10.0
        """
        if not self.agent.is_alive:
            return -10.0
        
        reward = 0.0
        
        # Survival bonus (scaled down)
        reward += 0.1
        
        # Vitals health bonus (encourage maintaining vitals)
        energy_health = self.agent.energy / 100.0
        hydration_health = self.agent.hydration / 100.0
        temp_health = 1.0 - abs(37.0 - self.agent.temperature) / 20.0  # Optimal at 37°C
        temp_health = max(0, temp_health)
        wakefulness_health = self.agent.wakefulness
        
        vitals_reward = (energy_health + hydration_health + temp_health + wakefulness_health) / 4.0
        reward += vitals_reward * 0.5
        
        # Penalty for low vitals (urgent signal)
        if self.agent.energy < 20:
            reward -= 0.5 * (1 - self.agent.energy / 20)
        if self.agent.hydration < 20:
            reward -= 0.5 * (1 - self.agent.hydration / 20)
        
        # Energy efficiency penalty
        reward -= 0.05 * velocity
        
        # Exploration bonus (small reward for movement)
        reward += 0.02 * velocity
        
        return reward
    
    def _ppo_update(self) -> Dict[str, float]:
        """
        Perform PPO policy update.
        
        Returns:
            Dictionary of loss components
        """
        if len(self.ppo_memory) < self.config.batch_size:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        # Get batch data
        batch = self.ppo_memory.get_batch(self.device)
        
        # Compute next value for GAE
        with torch.no_grad():
            next_value = self.value_head(batch['next_states'][-1])
        
        # Compute advantages
        advantages, returns = self._compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones'],
            next_value.squeeze(),
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        batch_size = len(self.ppo_memory)
        mini_batch_size = min(self.config.batch_size, batch_size)
        
        for _ in range(self.config.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_states = batch['states'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass
                mean_actions, _ = self.agent.brain(mb_states)
                values = self.value_head(mb_states).squeeze(-1)
                
                # Get new log probs
                dist = self._get_action_distribution(mean_actions)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.ppo_clip, 
                    1 + self.config.ppo_clip
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # EWC loss
                ewc_loss = self.ewc_loss.compute_loss(self.agent.brain, torch.tensor(0.0))
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.value_loss_coef * value_loss 
                    - self.config.entropy_coef * entropy
                    + ewc_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.brain.parameters()) + 
                    list(self.value_head.parameters()) + 
                    [self.action_log_std],
                    self.config.gradient_clip
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Reset hidden state after batch update
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
        
        # Clear PPO memory
        self.ppo_memory.clear()
        
        num_updates = self.config.ppo_epochs * (batch_size // mini_batch_size + 1)
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def sleep_cycle(self) -> Dict:
        """
        Execute one complete sleep cycle.
        
        Returns:
            Statistics from sleep cycle
        """
        print(f"\n💤 Entering Sleep at tick {self.world.tick}")
        
        # Step 0: PPO Update (learn from wake phase experiences)
        print("  [Learning] PPO policy update...")
        ppo_stats = self._ppo_update()
        
        # Step 1: Consolidation (NREM) - additional replay
        print("  [NREM] Consolidating memories...")
        for epoch in range(self.config.sleep_epochs):
            if len(self.replay_buffer) >= self.config.batch_size:
                self._consolidation_replay()
        
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
        # Protect against over-pruning - keep minimum buffer size
        max_to_prune = max(0, len(self.replay_buffer) - self.config.min_buffer_size)
        pruned = self.replay_buffer.prune_consolidated(max_prune=max_to_prune)
        
        # Step 5: Fisher decay
        self.fisher_matrix.decay_fisher(self.config.fisher_decay)
        
        # Step 6: Update optimal weights
        self.fisher_matrix.update_optimal_weights(self.agent.brain)
        
        # Step 7: Decay exploration (anneal action std)
        with torch.no_grad():
            self.action_log_std.data = torch.clamp(
                self.action_log_std.data - np.log(1 / self.config.action_std_decay),
                min=np.log(self.config.action_std_min),
            )
        
        # Exit sleep
        self.agent.exit_sleep()
        self.total_sleep_cycles += 1
        self.last_sleep_tick = self.world.tick
        
        # Gather statistics
        buffer_stats = self.replay_buffer.get_stats()
        current_action_std = torch.exp(self.action_log_std).mean().item()
        
        stats = {
            'scale_factor': scale_factor,
            'transitions_pruned': pruned,
            'buffer_size': len(self.replay_buffer),
            'fisher_stats': self.fisher_matrix.get_stats(),
            'avg_uncertainty': buffer_stats.get('avg_uncertainty', 1.0),
            'avg_td_error': buffer_stats.get('avg_td_error', 1.0),
            'near_consolidated': buffer_stats.get('near_consolidated', 0),
            'policy_loss': ppo_stats['policy_loss'],
            'value_loss': ppo_stats['value_loss'],
            'entropy': ppo_stats['entropy'],
            'action_std': current_action_std,
        }
        
        # Update running stats
        self.stats['policy_loss'] = ppo_stats['policy_loss']
        self.stats['value_loss'] = ppo_stats['value_loss']
        self.stats['entropy'] = ppo_stats['entropy']
        
        print(f"  ✓ Sleep complete: pruned {pruned}, buffer {len(self.replay_buffer)}")
        print(f"    Avg uncertainty: {stats['avg_uncertainty']:.4f}, Avg TD: {stats['avg_td_error']:.4f}")
        print(f"    Policy loss: {ppo_stats['policy_loss']:.4f}, Value loss: {ppo_stats['value_loss']:.4f}")
        print(f"    Entropy: {ppo_stats['entropy']:.4f}, Action std: {current_action_std:.4f}")
        
        return stats
    
    def _consolidation_replay(self) -> None:
        """Replay from buffer during consolidation (NREM-like)."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.stack([t.action for t in batch]).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        
        # Normalize rewards
        rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        
        # Forward pass
        mean_actions, _ = self.agent.brain(states)
        values = self.value_head(states).squeeze(-1)
        
        # Simple policy loss: encourage actions that led to high rewards
        # Weight by advantage (reward as proxy since we don't have full trajectory)
        advantages = rewards - values.detach()
        
        dist = self._get_action_distribution(mean_actions)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, rewards)
        
        # EWC penalty
        ewc_loss = self.ewc_loss.compute_loss(self.agent.brain, torch.tensor(0.0))
        
        total_loss = policy_loss + 0.5 * value_loss + ewc_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.brain.parameters()) + 
            list(self.value_head.parameters()),
            self.config.gradient_clip
        )
        self.optimizer.step()
        
        # Reset hidden state
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
    
    def _update_fisher_matrix(self) -> None:
        """Update Fisher Information Matrix from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        batch = self.replay_buffer.sample(min(1000, len(self.replay_buffer)))
        states = torch.stack([t.state for t in batch]).to(self.device)
        
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
        
        # Trigger 3: Time-based
        if self.total_wake_ticks - self.last_sleep_tick >= self.config.wake_duration:
            return True
        
        return False
    
    def save_checkpoint(self, path: str) -> None:
        """Save complete training state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'agent_brain': self.agent.brain.state_dict(),
            'value_head': self.value_head.state_dict(),
            'action_log_std': self.action_log_std.data,
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
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load complete training state."""
        checkpoint = torch.load(path, weights_only=False)
        
        self.agent.brain.load_state_dict(checkpoint['agent_brain'])
        self.value_head.load_state_dict(checkpoint['value_head'])
        self.action_log_std.data = checkpoint['action_log_std']
        self.agent.load_state_dict(checkpoint['agent_state'])
        self.world.load_state_dict(checkpoint['world_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.fisher_matrix.fisher_diagonal = checkpoint['fisher_matrix']['fisher_diagonal']
        self.fisher_matrix.optimal_weights = checkpoint['fisher_matrix']['optimal_weights']
        self.stats = checkpoint['stats']
        
        # Load additional state
        if 'total_wake_ticks' in checkpoint:
            self.total_wake_ticks = checkpoint['total_wake_ticks']
        if 'total_sleep_cycles' in checkpoint:
            self.total_sleep_cycles = checkpoint['total_sleep_cycles']
        if 'last_sleep_tick' in checkpoint:
            self.last_sleep_tick = checkpoint['last_sleep_tick']
        if 'reward_mean' in checkpoint:
            self.reward_mean = checkpoint['reward_mean']
        if 'reward_std' in checkpoint:
            self.reward_std = checkpoint['reward_std']
        if 'reward_count' in checkpoint:
            self.reward_count = checkpoint['reward_count']
        
        # Reset hidden state
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)