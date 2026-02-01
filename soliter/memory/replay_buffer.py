"""
Replay Buffer with Epistemic Pruning.

The buffer stores recent experiences and prunes them based on:
1. Epistemic uncertainty (model confidence)
2. TD-error (prediction accuracy)

Experiences are removed when the model has "consolidated" them into weights.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random


@dataclass
class Transition:
    """Single experience transition."""
    state: torch.Tensor  # Sensor readings
    action: torch.Tensor  # Motor outputs
    reward: float  # Survival reward
    next_state: torch.Tensor
    done: bool
    
    # Metadata
    tick: int = 0
    uncertainty: float = 1.0  # Epistemic uncertainty
    td_error: float = 1.0  # TD-error


class ReplayBuffer:
    """
    Experience replay buffer with epistemic pruning.
    
    Unlike standard FIFO buffers, this prunes based on consolidation:
    - Keep experiences with high uncertainty (not yet learned)
    - Remove experiences with low uncertainty + low TD-error (consolidated)
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        prune_threshold_uncertainty: float = 0.1,
        prune_threshold_td_error: float = 0.05,
    ):
        self.capacity = capacity
        self.prune_threshold_uncertainty = prune_threshold_uncertainty
        self.prune_threshold_td_error = prune_threshold_td_error
        
        self.buffer: List[Transition] = []
        self.position = 0
        
        # Statistics
        self.total_added = 0
        self.total_pruned = 0
    
    def push(self, transition: Transition) -> None:
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Circular buffer: overwrite oldest
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
        self.total_added += 1
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def update_uncertainties(
        self,
        model: torch.nn.Module,
        num_samples: int = 10,
    ) -> None:
        """
        Update epistemic uncertainty for all transitions using MC Dropout.
        
        Args:
            model: The neural network (with dropout)
            num_samples: Number of forward passes for MC estimate
        """
        model.train()  # Enable dropout
        
        for transition in self.buffer:
            state = transition.state.unsqueeze(0)
            
            # Multiple forward passes
            predictions = []
            with torch.no_grad():
                for _ in range(num_samples):
                    output, _ = model(state)
                    predictions.append(output)
            
            # Calculate variance across predictions
            predictions = torch.stack(predictions)
            variance = predictions.var(dim=0).mean().item()
            
            transition.uncertainty = variance
    
    def update_td_errors(
        self,
        model: torch.nn.Module,
        gamma: float = 0.99,
    ) -> None:
        """
        Update TD-errors for all transitions.
        
        TD-error = |r + γ·V(s') - V(s)|
        
        For simplicity, we use action magnitude as proxy for value.
        """
        model.eval()
        
        for transition in self.buffer:
            if transition.done:
                td_error = abs(transition.reward)
            else:
                with torch.no_grad():
                    state_output, _ = model(transition.state.unsqueeze(0))
                    next_state_output, _ = model(transition.next_state.unsqueeze(0))
                    
                    # Simple value estimate: negative of energy expenditure
                    state_value = -state_output[0, 0].item()  # -velocity (energy cost)
                    next_state_value = -next_state_output[0, 0].item()
                    
                    td_error = abs(transition.reward + gamma * next_state_value - state_value)
            
            transition.td_error = td_error
    
    def prune_consolidated(self) -> int:
        """
        Remove consolidated experiences (low uncertainty AND low TD-error).
        
        Returns:
            Number of transitions pruned
        """
        original_size = len(self.buffer)
        
        # Keep transitions that are NOT consolidated
        self.buffer = [
            t for t in self.buffer
            if not (
                t.uncertainty < self.prune_threshold_uncertainty and
                t.td_error < self.prune_threshold_td_error
            )
        ]
        
        pruned = original_size - len(self.buffer)
        self.total_pruned += pruned
        
        # Reset position
        self.position = len(self.buffer) % self.capacity
        
        return pruned
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'total_added': self.total_added,
                'total_pruned': self.total_pruned,
            }
        
        uncertainties = [t.uncertainty for t in self.buffer]
        td_errors = [t.td_error for t in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned,
            'avg_uncertainty': np.mean(uncertainties),
            'avg_td_error': np.mean(td_errors),
            'min_uncertainty': np.min(uncertainties),
            'max_uncertainty': np.max(uncertainties),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
