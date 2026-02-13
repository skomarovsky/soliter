"""
Sleep-Wake Training Loop with Biological Drive System.

Orchestrates the complete learning cycle:
- Wake: Agent interacts with environment, collects experiences
  - Internal reward from drive system (hunger/thirst/cold/curiosity)
  - Gradient sensors for resource-directed navigation
- Sleep: Consolidation, homeostatic scaling, epistemic pruning

Uses PPO (Proximal Policy Optimization) for stable policy learning.

Key changes from v1:
  - DriveSystem replaces _calculate_shaped_reward()
  - GradientSensors provide 6 new directional channels
  - Drive states provide 4 new internal channels (51 total inputs)
  - action_log_std REMOVED from optimizer (bug fix - was causing explosion)
  - _check_resource_consumption tracks what was consumed for drive satisfaction
  - wake_step() returns reward_details dict for logging

Place in: soliter/training/sleep_wake.py
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
from ..core.drive_system import DriveSystem, DriveConfig
from ..agents.soliter_agent import SoliterAgent
from ..environment import World, SensorSystem, Physics
from ..environment.gradient_sensors import GradientSensors, GradientSensorConfig
from ..memory.replay_buffer import EpistemicReplayBuffer, Transition, PruningStats
from ..memory.fisher_matrix import FisherInformationMatrix
from .ewc import EWCLoss


# PyTorch 2.6+ security: allowlist numpy for checkpoint loading
try:
    import torch.serialization
    torch.serialization.add_safe_globals([np.ndarray, np.dtype])
    if hasattr(np, '_core') and hasattr(np._core, 'multiarray'):
        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
    if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
except (AttributeError, TypeError):
    pass


@dataclass
class TrainingConfig:
    """Configuration for sleep-wake training."""
    wake_duration: int = 10000
    learning_rate: float = 0.0003
    batch_size: int = 64
    gradient_clip: float = 0.5
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    sleep_epochs: int = 5
    sleep_trigger_saturation: float = 0.2
    sleep_trigger_buffer: float = 0.9
    target_activity: float = 0.5
    scaling_rate: float = 0.01
    lambda_ewc: float = 155000.0
    fisher_decay: float = 0.77
    buffer_capacity: int = 1_000_000
    prune_fraction: float = 0.2
    min_buffer_size: int = 64
    surprise_gating: bool = True
    surprise_momentum: float = 0.95
    surprise_percentile: float = 0.3
    action_std_init: float = 0.5
    action_std_min: float = 0.1
    action_std_decay: float = 0.995
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"


class ValueHead(nn.Module):
    """Value function head for PPO. Estimates V(s)."""

    def __init__(self, input_size: int = 51, hidden_size: int = 128):
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
    """Memory buffer for PPO rollouts."""

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.next_states: List[torch.Tensor] = []

    def push(self, state, action, log_prob, reward, value, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear(self):
        for lst in [self.states, self.actions, self.log_probs,
                    self.rewards, self.values, self.dones, self.next_states]:
            lst.clear()

    def __len__(self):
        return len(self.states)

    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
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
    Main training orchestrator with biological drive system.

    Key difference from v1: action_log_std is NOT in the optimizer.
    Reward comes from internal DriveSystem, not external shaped reward.
    Sensors include gradient channels (6) and drive states (4) = 51 total.
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

        self.agent.brain.to(self.device)
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)

        self.value_head = ValueHead(
            input_size=self.agent.brain.sensory_size,
            hidden_size=128,
        ).to(self.device)

        # action_log_std: NOT in optimizer. Managed by explicit decay only.
        self.action_log_std = nn.Parameter(
            torch.ones(3, device=self.device) * np.log(self.config.action_std_init)
        )

        self.replay_buffer = EpistemicReplayBuffer(
            capacity=self.config.buffer_capacity,
            prune_fraction=self.config.prune_fraction,
            min_buffer_size=self.config.min_buffer_size,
        )
        self.ppo_memory = PPOMemory()

        self.fisher_matrix = FisherInformationMatrix(
            model=self.agent.brain, device=self.device,
        )
        self.ewc_loss = EWCLoss(
            fisher_matrix=self.fisher_matrix,
            lambda_ewc=self.config.lambda_ewc,
        )

        # CRITICAL FIX: action_log_std NOT in optimizer
        self.optimizer = optim.Adam([
            {'params': self.agent.brain.parameters(), 'lr': self.config.learning_rate},
            {'params': self.value_head.parameters(), 'lr': self.config.learning_rate},
        ])

        # Biological drive system (replaces shaped reward)
        self.drive_system = DriveSystem(DriveConfig())

        # Gradient sensors (smell/heat sensing)
        self.gradient_sensors = GradientSensors(
            GradientSensorConfig(
                scale_factor=self.world.config.width / 4.0,
                toroidal=True,
            )
        )

        self._consumed_this_tick: Optional[str] = None

        self.total_wake_ticks = 0
        self.total_sleep_cycles = 0
        self.last_sleep_tick = 0

        self.stats = {
            'total_reward': 0.0, 'episodes': 0, 'deaths': 0,
            'average_lifespan': 0.0, 'policy_loss': 0.0,
            'value_loss': 0.0, 'entropy': 0.0, 'avg_reward_per_cycle': 0.0,
        }

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        self.running_surprise = 0.0
        self._surprise_history: List[float] = []
        self._surprise_threshold = 0.0
        self.total_gated_out = 0

    def _get_action_distribution(self, mean_actions: torch.Tensor) -> Normal:
        """Create action distribution for exploration."""
        action_std = torch.exp(self.action_log_std)
        return Normal(mean_actions, action_std)

    def _select_action_with_exploration(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action with exploration noise."""
        with torch.no_grad():
            mean_action, _ = self.agent.brain(state.unsqueeze(0))
            mean_action = mean_action.squeeze(0)
            value = self.value_head(state)
            dist = self._get_action_distribution(mean_action)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            action = torch.clamp(action, -1.0, 1.0)
            action_constrained = torch.stack([
                torch.sigmoid(action[0]),
                torch.tanh(action[1]),
                torch.sigmoid(action[2]),
            ])
        return action_constrained, log_prob, value, mean_action

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        values_ext = torch.cat([values, next_value.unsqueeze(0)])
        for t in reversed(range(len(rewards))):
            nv = next_value if t == len(rewards) - 1 else values_ext[t + 1]
            delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        return advantages, advantages + values

    def _update_reward_stats(self, reward: float):
        """Update running reward statistics for normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(
            (self.reward_std ** 2 * (self.reward_count - 1) + delta * delta2)
            / self.reward_count
        )
        self.reward_std = max(self.reward_std, 1e-8)

    def _get_sensors(self, resources, drive_vector, sensor_noise):
        """Helper: build 51-channel sensor tensor."""
        return self.sensors.get_sensor_readings(
            agent_position=self.agent.position,
            agent_vitals={
                'energy': self.agent.energy,
                'hydration': self.agent.hydration,
                'temperature': self.agent.temperature,
                'wakefulness': self.agent.wakefulness,
            },
            resources=resources,
            sensor_noise=sensor_noise,
            drive_vector=drive_vector,
            world_width=float(self.world.config.width),
            world_height=float(self.world.config.height),
        ).to(self.device)

    def wake_step(self, resources: Dict) -> Tuple[float, bool, Dict]:
        """
        Execute one wake step with biological drive reward.

        Returns: (reward, done, reward_details)
        """
        # Snapshot vitals BEFORE action
        self.drive_system.snapshot_vitals(
            self.agent.energy, self.agent.hydration, self.agent.temperature,
        )

        drive_vector = self.drive_system.get_drive_vector(
            self.agent.energy, self.agent.hydration, self.agent.temperature,
        )

        sensor_noise = self.agent.get_sensor_noise()
        sensors = self._get_sensors(resources, drive_vector, sensor_noise)

        action, log_prob, value, mean_action = self._select_action_with_exploration(sensors)
        velocity = action[0].item()
        turn = action[1].item()
        should_sleep = action[2].item() > 0.5

        # Execute
        world_bounds = (self.world.config.width, self.world.config.height)
        self.agent.move(velocity, turn, world_bounds=world_bounds, dt=1.0)
        # Position is now clipped in move() - no wrapping
        ambient_temp = self.world.get_ambient_temperature()
        self.agent.update_vitals(velocity, ambient_temp, dt=1.0)

        self._consumed_this_tick = None
        self._check_resource_consumption(resources)

        if should_sleep and self.agent.wakefulness < 0.3:
            self.agent.enter_sleep()

        done = not self.agent.is_alive

        # Internal reward from drive system
        reward, reward_details = self.drive_system.compute_internal_reward(
            energy=self.agent.energy,
            hydration=self.agent.hydration,
            temperature=self.agent.temperature,
            sensor_readings=sensors.cpu().numpy(),
            consumed_resource=self._consumed_this_tick,
            is_dead=done,
        )
        self._update_reward_stats(reward)

        # Next state
        next_drive = self.drive_system.get_drive_vector(
            self.agent.energy, self.agent.hydration, self.agent.temperature,
        )
        next_sensors = self._get_sensors(resources, next_drive, sensor_noise)

        # PPO memory
        self.ppo_memory.push(sensors, action, log_prob, reward, value, done, next_sensors)

        # Replay buffer with surprise gating
        transition = Transition(
            state=sensors, action=action, reward=reward,
            next_state=next_sensors, done=done, tick=self.world.tick,
        )

        if self.config.surprise_gating:
            with torch.no_grad():
                instant_surprise = float((mean_action - action).pow(2).sum().item())
            theta = self.config.surprise_momentum
            self.running_surprise = theta * self.running_surprise + (1 - theta) * instant_surprise
            self._surprise_history.append(self.running_surprise)
            if len(self._surprise_history) >= 100:
                sorted_hist = sorted(self._surprise_history)
                pct_idx = int(len(sorted_hist) * self.config.surprise_percentile)
                self._surprise_threshold = sorted_hist[pct_idx]
                self._surprise_history = self._surprise_history[-500:]
            store = (
                self.running_surprise >= self._surprise_threshold
                or done
                or len(self.replay_buffer) < self.config.batch_size * 4
            )
            if store:
                transition.replay_surprise = self.running_surprise
                self.replay_buffer.push(transition)
            else:
                self.total_gated_out += 1
        else:
            self.replay_buffer.push(transition)

        self.total_wake_ticks += 1
        self.stats['total_reward'] += reward
        return reward, done, reward_details

    def _check_resource_consumption(self, resources: Dict) -> None:
        """Check resources and track what was consumed for drive system."""
        seasonal_period = self.world.config.seasonal_period
        is_night = self.world.is_night()
        
        # Update all resources (recovery over time)
        for resource_list in resources.values():
            for resource in resource_list:
                resource.update(self.world.tick)

        # Check consumption for each resource type
        # IMPORTANT: Now uses STRICT consumption_radius, not detection_radius
        for feeder in resources.get('feeders', []):
            if feeder.is_available(self.world.tick, seasonal_period):
                if feeder.is_agent_in_consumption_range(self.agent.position, is_night):
                    if feeder.can_consume():  # Check if resource has capacity
                        actual_amount = feeder.consume(self.world.tick, seasonal_period)
                        if actual_amount > 0:
                            self.agent.consume_resource('food', actual_amount)
                            self._consumed_this_tick = 'food'

        for fountain in resources.get('fountains', []):
            if fountain.is_available(self.world.tick, seasonal_period):
                if fountain.is_agent_in_consumption_range(self.agent.position, is_night):
                    if fountain.can_consume():
                        actual_amount = fountain.consume(self.world.tick, seasonal_period)
                        if actual_amount > 0:
                            self.agent.consume_resource('water', actual_amount)
                            self._consumed_this_tick = 'water'

        for heater in resources.get('heaters', []):
            if heater.is_available(self.world.tick, seasonal_period):
                if heater.is_agent_in_consumption_range(self.agent.position, is_night):
                    if heater.can_consume():
                        actual_amount = heater.consume(self.world.tick, seasonal_period)
                        if actual_amount > 0:
                            self.agent.consume_resource('heat', actual_amount)
                            self._consumed_this_tick = 'heat'

    def _ppo_update(self) -> Dict[str, float]:
        """Perform PPO policy update."""
        if len(self.ppo_memory) < self.config.batch_size:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        batch = self.ppo_memory.get_batch(self.device)
        with torch.no_grad():
            next_value = self.value_head(batch['next_states'][-1])

        advantages, returns = self._compute_gae(
            batch['rewards'], batch['values'], batch['dones'], next_value.squeeze(),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pl = total_vl = total_ent = 0
        batch_size = len(self.ppo_memory)
        mbs = min(self.config.batch_size, batch_size)

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, mbs):
                mb_idx = indices[start:start + mbs]
                mb_states = batch['states'][mb_idx]
                mb_actions = batch['actions'][mb_idx]
                mb_old_lp = batch['log_probs'][mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                mean_actions, _ = self.agent.brain(mb_states)
                values = self.value_head(mb_states).squeeze(-1)

                dist = self._get_action_distribution(mean_actions)
                new_lp = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip,
                                    1 + self.config.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_ret)
                ewc_loss = self.ewc_loss.compute_loss(self.agent.brain, torch.tensor(0.0))

                loss = (policy_loss
                        + self.config.value_loss_coef * value_loss
                        - self.config.entropy_coef * entropy
                        + ewc_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.brain.parameters()) +
                    list(self.value_head.parameters()),
                    self.config.gradient_clip
                )
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()

        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
        self.ppo_memory.clear()
        n_updates = self.config.ppo_epochs * (batch_size // mbs + 1)
        return {
            'policy_loss': total_pl / n_updates,
            'value_loss': total_vl / n_updates,
            'entropy': total_ent / n_updates,
        }

    def sleep_cycle(self) -> Dict:
        """Execute one complete sleep cycle."""
        print(f"\n  Entering Sleep at tick {self.world.tick}")

        ppo_stats = self._ppo_update()
        replay_stats = self._replay_score_and_consolidate()

        pruning_stats = self.replay_buffer.prune_by_surprise()
        pruned = pruning_stats.pruned

        self.fisher_matrix.decay_fisher(self.config.fisher_decay)
        self._update_fisher_matrix()

        scale_factor = self.agent.brain.apply_homeostatic_scaling(
            target_activity=self.config.target_activity,
            scaling_rate=self.config.scaling_rate,
        )

        self.fisher_matrix.update_optimal_weights(self.agent.brain)

        # Decay exploration (explicit schedule, NOT gradient-based)
        with torch.no_grad():
            self.action_log_std.data = torch.clamp(
                self.action_log_std.data - np.log(1 / self.config.action_std_decay),
                min=np.log(self.config.action_std_min),
            )

        self.agent.exit_sleep()
        self.total_sleep_cycles += 1
        self.last_sleep_tick = self.world.tick

        buffer_stats = self.replay_buffer.get_stats()
        current_action_std = torch.exp(self.action_log_std).mean().item()

        stats = {
            'scale_factor': scale_factor,
            'transitions_pruned': pruning_stats,
            'buffer_size': len(self.replay_buffer),
            'fisher_stats': self.fisher_matrix.get_stats(),
            'avg_surprise': buffer_stats.get('avg_surprise', 1.0),
            'median_surprise': buffer_stats.get('median_surprise', 1.0),
            'policy_loss': ppo_stats['policy_loss'],
            'value_loss': ppo_stats['value_loss'],
            'entropy': ppo_stats['entropy'],
            'action_std': current_action_std,
            'replay_stats': replay_stats,
            'drive_stats': self.drive_system.get_stats(),
            'avg_uncertainty': buffer_stats.get('avg_surprise', 1.0),
            'avg_td_error': buffer_stats.get('avg_surprise', 1.0),
            'near_consolidated': buffer_stats.get('near_consolidated', 0),
        }

        self.stats['policy_loss'] = ppo_stats['policy_loss']
        self.stats['value_loss'] = ppo_stats['value_loss']
        self.stats['entropy'] = ppo_stats['entropy']

        print(f"    Sleep done: pruned {pruned}, buffer {len(self.replay_buffer)}, "
              f"action_std {current_action_std:.4f}, "
              f"consumptions {self.drive_system.total_consumption_events}")

        return stats

    def _replay_score_and_consolidate(self) -> Dict:
        """NREM-like: Replay ALL memories, score by surprise, consolidate."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'mean_surprise': 0.0, 'consolidation_epochs': 0}

        all_surprises = []
        total_pl = total_vl = 0.0
        n_batches = 0

        for epoch in range(self.config.sleep_epochs):
            for batch_trans, batch_idx in self.replay_buffer.iter_batches(
                self.config.batch_size
            ):
                states = torch.stack([t.state for t in batch_trans]).to(self.device)
                actions = torch.stack([t.action for t in batch_trans]).to(self.device)
                rewards = torch.tensor(
                    [t.reward for t in batch_trans],
                    dtype=torch.float32, device=self.device,
                )
                rewards_norm = (rewards - self.reward_mean) / (self.reward_std + 1e-8)

                mean_actions, _ = self.agent.brain(states)
                values = self.value_head(states).squeeze(-1)

                with torch.no_grad():
                    action_err = (mean_actions - actions).pow(2).sum(dim=-1)
                    value_err = (rewards_norm - values).pow(2)
                    replay_surprise = action_err + value_err
                    for i, idx in enumerate(batch_idx):
                        self.replay_buffer.buffer[idx].replay_surprise = float(
                            replay_surprise[i].item()
                        )
                    if epoch == self.config.sleep_epochs - 1:
                        all_surprises.extend(replay_surprise.cpu().tolist())

                advantages = rewards_norm - values.detach()
                dist = self._get_action_distribution(mean_actions)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                policy_loss = -(log_probs * advantages).mean()
                value_loss = F.mse_loss(values, rewards_norm)
                ewc_loss = self.ewc_loss.compute_loss(self.agent.brain, torch.tensor(0.0))
                total_loss = policy_loss + 0.5 * value_loss + ewc_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.brain.parameters()) +
                    list(self.value_head.parameters()),
                    self.config.gradient_clip,
                )
                self.optimizer.step()
                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                n_batches += 1

        self.agent.brain.reset_hidden(batch_size=1, device=self.device)
        return {
            'mean_surprise': float(np.mean(all_surprises)) if all_surprises else 0.0,
            'std_surprise': float(np.std(all_surprises)) if all_surprises else 0.0,
            'consolidation_epochs': self.config.sleep_epochs,
            'consolidation_batches': n_batches,
            'avg_policy_loss': total_pl / max(1, n_batches),
            'avg_value_loss': total_vl / max(1, n_batches),
        }

    def _consolidation_replay(self) -> None:
        """Legacy method - now handled by _replay_score_and_consolidate."""
        pass

    def _update_fisher_matrix(self) -> None:
        """Update Fisher Information Matrix from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        batch = self.replay_buffer.sample(min(1000, len(self.replay_buffer)))
        states = torch.stack([t.state for t in batch]).to(self.device)
        dataloader = [(states[i:i+32],) for i in range(0, len(states), 32)]
        self.fisher_matrix.compute_fisher(self.agent.brain, dataloader, num_samples=1000)

    def should_sleep(self) -> bool:
        """Check if sleep should be triggered."""
        mean_activity = self.agent.brain.get_mean_activity()
        saturated = ((mean_activity < 0.1) | (mean_activity > 0.9)).float().mean()
        if saturated > self.config.sleep_trigger_saturation:
            return True
        if len(self.replay_buffer) / self.config.buffer_capacity > self.config.sleep_trigger_buffer:
            return True
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
        for key in ['total_wake_ticks', 'total_sleep_cycles', 'last_sleep_tick',
                    'reward_mean', 'reward_std', 'reward_count']:
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
        self.agent.brain.reset_hidden(batch_size=1, device=self.device)