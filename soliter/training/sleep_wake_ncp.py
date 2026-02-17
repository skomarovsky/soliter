"""
Sleep-Wake Training Loop with NCP/CfC - NO PPO!

NCP (Neural Circuit Policies) Architecture:
- Based on C. elegans connectome (biological realism)
- Continuous-time dynamics (no batch updates needed)
- Natural stability (no catastrophic forgetting)
- Online learning (no experience replay needed)

This replaces the broken PPO batch learning system with biologically-inspired
continuous learning that matches the agent's continuous existence.

Key Principles:
1. NO BATCH LEARNING - agent learns continuously during wake
2. NO PPO - use NCP's natural learning dynamics
3. Sleep = rest only (restore wakefulness, decay exploration)
4. Learning happens through CfC weight updates, not gradient descent
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.cfc_network import CfCBrain
from ..core.drive_system import DriveSystem, DriveConfig
from ..agents.soliter_agent import SoliterAgent
from ..environment import World
from ..environment.gradient_sensors import GradientSensors, GradientSensorConfig


@dataclass
class NCPTrainerConfig:
    """Configuration for NCP-based trainer."""
    
    # World parameters
    world_width: int = 200
    world_height: int = 200
    
    # Gradient sensor configuration
    gradient_config: GradientSensorConfig = None
    
    # Drive system configuration
    drive_config: DriveConfig = None
    
    # Sleep/wake cycle
    wake_duration: int = 2000  # Ticks between sleep cycles
    
    # NCP learning parameters (if we add explicit learning later)
    learning_rate: float = 0.0001  # For potential weight updates
    
    # Exploration
    initial_action_std: float = 0.8  # INCREASED: More exploration to discover consumption
    action_std_min: float = 0.05
    action_std_decay: float = 0.995  # Slower decay for continuous learning
    
    def __post_init__(self):
        if self.gradient_config is None:
            self.gradient_config = GradientSensorConfig()
        if self.drive_config is None:
            self.drive_config = DriveConfig()


class NCPSleepWakeTrainer:
    """
    Trainer using NCP/CfC architecture - NO PPO!
    
    Philosophy:
    - Agent learns continuously during wake (like real organisms)
    - Sleep is just rest (no batch updates that break the policy)
    - CfC network naturally handles temporal dynamics
    - No experience replay needed (continuous learning)
    """
    
    def __init__(
        self,
        agent: SoliterAgent,
        world: World,
        config: NCPTrainerConfig,
        device: torch.device,
    ):
        self.agent = agent
        self.world = world
        self.config = config
        self.device = device
        
        # Drive system (internal rewards)
        self.drive_system = DriveSystem(config.drive_config)
        
        # Gradient sensors (resource detection)
        self.gradient_sensors = GradientSensors(config.gradient_config)
        
        # Sensor system (proximity sensing) - FIXED!
        from ..environment import SensorSystem
        self.sensors = SensorSystem()
        
        # Exploration parameter (decays over time)
        self.action_log_std = nn.Parameter(
            torch.tensor([np.log(config.initial_action_std)] * 3, device=device)
        )
        
        # Value head for curiosity/planning (optional, simple MLP)
        self.value_head = nn.Sequential(
            nn.Linear(51, 32),  # 51 = sensor inputs
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        # CRITICAL FIX: ADD OPTIMIZER FOR LEARNING!
        # Combine brain and value head parameters
        all_params = list(self.agent.brain.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)
        
        # Learning diagnostics
        self.learning_diagnostics = {
            'total_updates': 0,
            'weight_changes': [],
            'losses': [],
            'gradient_norms': [],
        }
        
        # Store initial weights for comparison
        self.initial_weights = {
            name: param.clone().detach()
            for name, param in self.agent.brain.named_parameters()
        }
        
        # Track sleep cycles
        self.total_sleep_cycles = 0
        self.last_sleep_tick = 0
        
        # Track what was consumed this tick
        self._consumed_this_tick = None
    
    def _get_sensors(
        self,
        resources: Dict,
        drive_vector: np.ndarray,
        sensor_noise: float,
    ) -> torch.Tensor:
        """
        Construct full sensor input for the CfC brain.
        
        Uses SensorSystem which combines:
        - Proximity sensors (41 channels)
        - Gradient sensors (6 channels)  
        - Drive states (4 channels)
        
        Returns tensor of shape (51,)
        """
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
    
    def _select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using CfC network + Gaussian policy.
        
        Returns: (action, value, log_prob) for REINFORCE learning
        """
        # CfC network produces mean action (KEEP GRADIENTS!)
        mean_action, _ = self.agent.brain(state.unsqueeze(0))
        mean_action = mean_action.squeeze(0)
        
        # Create Gaussian distribution
        action_std = torch.exp(self.action_log_std)
        action_dist = torch.distributions.Normal(mean_action, action_std)
        
        # Sample action
        sampled_action = action_dist.sample()
        
        # Compute log probability for REINFORCE
        log_prob = action_dist.log_prob(sampled_action).sum()
        
        # Constrain to valid ranges
        action_constrained = torch.stack([
            torch.sigmoid(sampled_action[0]),  # velocity [0, 1]
            torch.tanh(sampled_action[1]) * 0.3,  # turn [-0.3, 0.3] rad
            torch.sigmoid(sampled_action[2]),  # sleep [0, 1]
        ])
        
        # Get value estimate (KEEP GRADIENTS!)
        value = self.value_head(state)
        
        return action_constrained, value, log_prob
    
    def _check_resource_consumption(self, resources: Dict):
        """Check if agent consumed a resource this tick."""
        self._consumed_this_tick = None
        seasonal_period = self.world.config.seasonal_period  # FIXED: use period not season
        
        # Check food
        for feeder in resources.get('feeders', []):
            if feeder.is_agent_in_consumption_range(self.agent.position):
                if feeder.can_consume():
                    amount = feeder.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('food', amount)
                        self._consumed_this_tick = 'food'
                        return
        
        # Check water (fountains, not waterers!)
        for fountain in resources.get('fountains', []):
            if fountain.is_agent_in_consumption_range(self.agent.position):
                if fountain.can_consume():
                    amount = fountain.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('water', amount)
                        self._consumed_this_tick = 'water'
                        return
        
        # Check heat
        is_night = self.world.is_night()
        for heater in resources.get('heaters', []):
            if heater.is_agent_in_consumption_range(self.agent.position, is_night):
                if heater.can_consume():
                    amount = heater.consume(self.world.tick, seasonal_period)
                    if amount > 0:
                        self.agent.consume_resource('heat', amount)
                        self._consumed_this_tick = 'heat'
                        return
    
    def wake_step(self, resources: Dict) -> Tuple[float, bool, Dict]:
        """
        Execute one wake step (no PPO updates!).
        
        Returns:
            reward: Internal reward from drive system
            done: Whether agent died
            details: Logging information
        """
        # Snapshot vitals BEFORE action
        self.drive_system.snapshot_vitals(
            self.agent.energy,
            self.agent.hydration,
            self.agent.temperature,
        )
        
        # Get drives and sensors
        drive_vector = self.drive_system.get_drive_vector(
            self.agent.energy,
            self.agent.hydration,
            self.agent.temperature,
        )
        
        sensor_noise = self.agent.get_sensor_noise()
        sensors = self._get_sensors(resources, drive_vector, sensor_noise)
        
        # Update stuck detection (track position + gradient strength)
        # Extract gradient channels from sensors
        gradient_channels = sensors[41:47]  # [food_x, food_y, water_x, water_y, heat_x, heat_y]
        # Convert to CPU for numpy operations
        gradient_channels_cpu = gradient_channels.cpu() if gradient_channels.is_cuda else gradient_channels
        gradient_strength = np.linalg.norm(gradient_channels_cpu.numpy())
        self.drive_system.update_position(self.agent.position, gradient_strength)
        
        # Select action (NCP forward pass + Gaussian sampling)
        action, value, log_prob = self._select_action(sensors)
        velocity = action[0].item()
        turn = action[1].item()
        should_sleep = action[2].item() > 0.5
        
        # Check if turning allowed (directional stability)
        allow_turning = self.agent.should_allow_turning(drive_vector)
        
        # Execute movement
        world_bounds = (self.world.config.width, self.world.config.height)
        self.agent.move(velocity, turn, allow_turning=allow_turning, 
                       world_bounds=world_bounds, dt=1.0)
        
        # Update vitals
        ambient_temp = self.world.get_ambient_temperature()
        self.agent.update_vitals(velocity, ambient_temp, turn=turn, dt=1.0)
        
        # Check resource consumption
        self._consumed_this_tick = None
        self._check_resource_consumption(resources)
        
        # Enter sleep if needed
        if should_sleep and self.agent.wakefulness < 0.3:
            self.agent.enter_sleep()
        
        # Check death
        done = not self.agent.is_alive
        
        # Compute internal reward
        reward, reward_details = self.drive_system.compute_internal_reward(
            energy=self.agent.energy,
            hydration=self.agent.hydration,
            temperature=self.agent.temperature,
            consumed_resource=self._consumed_this_tick,
            is_dead=done,
        )
        
        # NOTE: No action penalties here!
        # The DRIVE SYSTEM handles behavior through goal commitment.
        # Agent commits to one drive and sticks with it for 100 ticks.
        
        # SHAPED REWARD: Guide agent to get VERY close to resources
        if not done and resources:
            # Find closest resource (any type)
            min_distance = float('inf')
            for resource_type, resource_list in resources.items():
                for resource in resource_list:
                    dist = np.linalg.norm(self.agent.position - resource.position)
                    min_distance = min(min_distance, dist)
            
            # Proximity reward: Guide to consumption radius
            if min_distance < 30.0:  # Within detection radius
                # Gentle gradient: 0 at 30 units → 0.5 at 0 units
                proximity_bonus = 0.5 * (1.0 - min_distance / 30.0)
                reward += proximity_bonus
                
                # STRONG bonus for being VERY close (consumption range)
                if min_distance < 5.0:
                    # Strong gradient: 0 at 5 units → 3.0 at 0 units
                    consumption_distance_bonus = 3.0 * (1.0 - min_distance / 5.0)
                    reward += consumption_distance_bonus
        
        # CRITICAL FIX: ACTUALLY UPDATE NETWORK WEIGHTS!
        # Use REINFORCE policy gradient + value learning
        if not done:
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            
            # 1. VALUE LOSS: Train value head to predict rewards
            value_loss = (value.squeeze() - reward_tensor) ** 2
            
            # 2. POLICY LOSS: REINFORCE Policy Gradient
            # Advantage = how much better than expected
            advantage = reward_tensor - value.squeeze().detach()
            
            # REINFORCE: maximize log_prob weighted by advantage
            # Positive advantage → increase probability of this action
            # Negative advantage → decrease probability of this action
            policy_loss = -log_prob * advantage
            
            # 3. ENTROPY BONUS: Encourage exploration
            # Compute entropy of action distribution
            action_std = torch.exp(self.action_log_std)
            entropy = 0.5 * torch.log(2 * np.pi * np.e * (action_std ** 2)).sum()
            entropy_bonus = -0.01 * entropy  # Small bonus for high entropy
            
            # 4. TOTAL LOSS
            total_loss = policy_loss + value_loss + entropy_bonus
            
            # Backpropagate
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.agent.brain.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # DIAGNOSTICS: Track learning
            self.learning_diagnostics['total_updates'] += 1
            self.learning_diagnostics['losses'].append(total_loss.item())
            
            # Every 100 updates, check weight changes
            if self.learning_diagnostics['total_updates'] % 100 == 0:
                total_change = 0.0
                total_grad_norm = 0.0
                
                for name, param in self.agent.brain.named_parameters():
                    if name in self.initial_weights:
                        change = (param - self.initial_weights[name]).abs().mean().item()
                        total_change += change
                    
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item()
                
                self.learning_diagnostics['weight_changes'].append(total_change)
                self.learning_diagnostics['gradient_norms'].append(total_grad_norm)
                
                # Print diagnostics
                updates = self.learning_diagnostics['total_updates']
                avg_loss = np.mean(self.learning_diagnostics['losses'][-100:])
                print(f"\n  📊 Learning Diagnostics (update {updates}):")
                print(f"     Loss: {avg_loss:.4f}")
                print(f"     Weight change: {total_change:.6f}")
                print(f"     Gradient norm: {total_grad_norm:.4f}")
                
                if total_change < 0.0001:
                    print(f"     ⚠️  WARNING: Weights barely changing!")
                if total_grad_norm < 0.001:
                    print(f"     ⚠️  WARNING: Gradients very small!")
        
        return reward, done, reward_details
    
    def sleep_cycle(self) -> Dict:
        """
        Execute sleep cycle - NO LEARNING!
        
        Sleep is just rest:
        - Restore wakefulness
        - Decay exploration slightly
        - NO batch updates
        - NO PPO
        - NO experience replay
        
        The CfC network's continuous dynamics handle learning naturally.
        """
        print(f"\n  💤 Sleep at tick {self.world.tick} - NCP MODE (no batch updates)")
        
        # Decay exploration (agent becomes more confident over time)
        with torch.no_grad():
            self.action_log_std.data = torch.clamp(
                self.action_log_std.data - np.log(1 / self.config.action_std_decay),
                min=np.log(self.config.action_std_min),
            )
        
        # Exit sleep (restore wakefulness)
        self.agent.exit_sleep()
        
        # Track statistics
        self.total_sleep_cycles += 1
        self.last_sleep_tick = self.world.tick
        
        current_action_std = torch.exp(self.action_log_std).mean().item()
        
        return {
            'action_std': current_action_std,
            'sleep_cycles': self.total_sleep_cycles,
        }
    
    def print_learning_summary(self):
        """Print summary of learning diagnostics."""
        print("\n" + "=" * 80)
        print("LEARNING DIAGNOSTICS SUMMARY")
        print("=" * 80)
        
        updates = self.learning_diagnostics['total_updates']
        print(f"\nTotal weight updates: {updates}")
        
        if updates == 0:
            print("❌ NO LEARNING HAPPENED - weights were never updated!")
            return
        
        if self.learning_diagnostics['losses']:
            losses = self.learning_diagnostics['losses']
            print(f"\nLoss:")
            print(f"  Initial: {losses[0]:.4f}")
            print(f"  Final:   {losses[-1]:.4f}")
            print(f"  Average: {np.mean(losses):.4f}")
            
            if len(losses) > 100:
                early_avg = np.mean(losses[:100])
                late_avg = np.mean(losses[-100:])
                print(f"  Early avg: {early_avg:.4f}")
                print(f"  Late avg:  {late_avg:.4f}")
                improvement = early_avg - late_avg
                print(f"  Improvement: {improvement:+.4f}")
                
                if improvement < 0.01:
                    print(f"  ⚠️  Loss not decreasing - learning might not be working!")
        
        if self.learning_diagnostics['weight_changes']:
            changes = self.learning_diagnostics['weight_changes']
            print(f"\nWeight changes:")
            print(f"  Total change from initial: {changes[-1]:.6f}")
            
            if changes[-1] < 0.001:
                print(f"  ❌ Weights barely changed - learning is broken!")
            elif changes[-1] < 0.01:
                print(f"  ⚠️  Small weight changes - learning very slow")
            else:
                print(f"  ✅ Weights are updating")
        
        if self.learning_diagnostics['gradient_norms']:
            grads = self.learning_diagnostics['gradient_norms']
            print(f"\nGradient norms:")
            print(f"  Average: {np.mean(grads):.4f}")
            
            if np.mean(grads) < 0.01:
                print(f"  ⚠️  Very small gradients - might have vanishing gradient problem")
            else:
                print(f"  ✅ Gradients flowing")
        
        print("\n" + "=" * 80)
